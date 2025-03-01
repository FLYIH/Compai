import uuid
import orjson
import google.generativeai as genai
import re

ALLOWED_KEYS = ["age", "likes", "dislikes", "conversation_dislike"]

def is_important_information(user_query):
    """
    Uses Gemini to determine if the conversation contains important information
    """
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Construct Prompt
    prompt = f"""
    The following is the user's most recent conversation:
    "{user_query}"

    Please analyze and answer the following for each category:
    1. Age:
        - Did the user explicitly state their age or age range?
        - Example: "I am 25 years old" or "I am in my 20s" or I am 19"
    2. Likes:
        - Did the user explicitly state something they like or enjoy?
        - Example: "I like playing video games" or "I enjoy hiking"
    3. Dislikes:
        - Did the user explicitly state something they dislike or hate?
        - Example: "I hate spicy food" or "I dislike crowded places"
    4. Conversational Dislikes (conversation_dislike):
        - Did the user explicitly state they dislike certain conversational styles?
        - Examples include:
            - Direct or blunt tone
            - Overly technical jargon
            - Patronizing or condescending language
            - Too casual or too formal expressions

    - **Do not over-interpret the conversation. If the user did not explicitly state something, do not assume or infer any information.**
    - Only respond to what is clearly and explicitly stated by the user.

    - For each category, if important information is found, respond in JSON format as follows:
    {{
        "important": true,
        "key": "<category>",
        "value": "<explicit statement>"
    }}
    - If not, respond with:
    {{
        "important": false
    }}
    - Ensure to specify `key` as follows:
        - "age" for age-related statements
        - "likes" for things the user likes or enjoys
        - "dislikes" for things the user dislikes or hates
        - "conversation_dislike" for style-related dislikes, and provide the exact details in `value`. 

    - Your response should be concise and accurate, focusing only on information relevant to the user's preferences.
    """



    try:
        result = model.generate_content(prompt)
        # print("\n🔍 [DEBUG] Gemini Response:", result.text)
        cleaned_text = result.text.strip()

        # ====== Pre-Processing: Remove Markdown Code Block Indicators ======
        cleaned_text = re.sub(r"```json\s*", "", cleaned_text)
        cleaned_text = re.sub(r"```", "", cleaned_text)
        if cleaned_text.startswith("{") and cleaned_text.count("{") > 1:
            cleaned_text = f"[{cleaned_text}]"
            cleaned_text = cleaned_text.replace("}{", "},{")
        # ====== Layer 1: Direct JSON Parsing ======
        try:
            info_dict = orjson.loads(cleaned_text.encode('utf-8'))
            # print("\n✅ [DEBUG] JSON Parsed Successfully (Layer 1)")
            return info_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] Layer 1 JSON Parsing Failed:", e)

        # ====== Layer 2: Basic Cleaning and Retry ======
        basic_fixed_text = cleaned_text.replace("\n", "").replace("\t", "").strip()
        try:
            info_dict = orjson.loads(basic_fixed_text.encode('utf-8'))
            print("\n✅ [DEBUG] JSON Parsed Successfully (Layer 2)")
            return info_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] Layer 2 JSON Parsing Failed:", e)

        # ====== Layer 3: Advanced Cleaning and Retry ======
        advanced_fixed_text = re.sub(r"'", '"', cleaned_text)
        advanced_fixed_text = re.sub(r",\s*}", "}", advanced_fixed_text)
        advanced_fixed_text = re.sub(r",\s*]", "]", advanced_fixed_text)
        advanced_fixed_text = re.sub(r"(\w+):", r'"\1":', advanced_fixed_text)

        if advanced_fixed_text.count("{") > advanced_fixed_text.count("}"):
            advanced_fixed_text += "}"
        elif advanced_fixed_text.count("[") > advanced_fixed_text.count("]"):
            advanced_fixed_text += "]"

        try:
            info_dict = orjson.loads(advanced_fixed_text.encode('utf-8'))
            print("\n✅ [DEBUG] JSON Parsed Successfully (Layer 3)")
            return info_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] Layer 3 JSON Parsing Failed:", e)

        # ====== Final Layer: Return Raw Text for Debugging ======
        print("\n📝 [DEBUG] Returning Raw Text for Manual Inspection.")
        print(f"\n🔍 [DEBUG] Raw Text:\n{cleaned_text}")
        return {
            "important": False,
            "raw_text": cleaned_text
        }

    except Exception as e:
        print(f"⚠️ [ERROR] Unexpected error: {e}")
        return {
            "important": False,
            "error": str(e)
        }

def auto_update_memory(user, user_query, memory_collection):
    """
    Automatically detects and updates memory
    """
    # Use Gemini to determine if important information is present
    important_info = is_important_information(user_query)

    if isinstance(important_info, list):
        for info in important_info:
            process_memory(info, user, memory_collection)
    else:
        process_memory(important_info, user, memory_collection)


def process_memory(important_info, user, memory_collection):
    """
    Processes each important information object for memory update
    """
    if important_info.get("important"):
        key = important_info.get("key").strip()
        value = important_info.get("value")

        if key not in ALLOWED_KEYS:
            print("\n🔍 [INFO] No important information, memory not updated.")
            return

        if isinstance(value, dict):
            value = orjson.dumps(value)
        elif isinstance(value, list):
            value = ", ".join(value)
        elif isinstance(value, str):
            value = value.strip()
        else:
            print(f"⚠️ [WARN] Unexpected value type: {type(value)} - Skipping update.")
            return
        
        add_or_update_memory(user, key, value, memory_collection)
        print(f"\n✅ [INFO] Updated important information - {key}: {value}")
    else:
        print("\n🔍 [INFO] No important information, memory not updated.")


def add_or_update_memory(user, key, value, memory_collection):
    """
    Adds or updates the user's memory with specific rules for each key:
    - age: Always overwrite the existing value.
    - likes, dislikes, conversation_dislike: Accumulate unique values.
    """
    # === Key-specific Behavior ===
    if key == "age":
        print(f"🔄 [INFO] Updating age to {value} (Overwriting previous value)")
        memory_collection.delete(where={
            "$and": [
                {"user": user},
                {"key": key}
            ]
        })

    existing_memories = memory_collection.get(
        where={
            "$and": [
                {"user": user},
                {"key": key}
            ]
        }
    )

    if existing_memories["documents"]:
        for metadata in existing_memories["metadatas"]:
            if metadata.get("value") == value:
                print(f"🔍 [INFO] Value already exists, skipping: {key} -> {value}")
                return

    doc_uuid = f"mem_{uuid.uuid4()}"
    memory_collection.add(
        documents=[f"{key}:{value}"],
        embeddings=[[0.0]],
        metadatas=[{
            "user": user,
            "key": key,
            "value": value,
            "id": doc_uuid
        }],
        ids=[doc_uuid]
    )
    print(f"✅ [INFO] Added or updated memory - {key}: {value}")


def get_user_memory(user, memory_collection):
    """
    Retrieves all memory associated with the user
    """
    results = memory_collection.get(where={"user": user})
    memories = {}

    if not results or "metadatas" not in results:
        print(f"⚠️ [WARN] No memory found for user: {user}")
        return memories

    for metadata in results["metadatas"]:
        key = metadata.get("key", "").strip()
        value = metadata.get("value", "").strip()
        if key and value:
            if key in memories:
                memories[key].append(value)
            else:
                memories[key] = [value]

    print(f"✅ [INFO] Retrieved memory for user: {user} -> {memories}")
    return memories

def apply_memory_to_prompt(prompt, user, memory_collection):
    """
    Applies memory to the prompt for personalized and context-aware conversation
    """
    memories = get_user_memory(user, memory_collection)
    
    # Age
    if "age" in memories:
        prompt += f"\n- The user is {memories['age'][0]} years old, so consider topics relevant to their age group."
    
    # Likes
    if "likes" in memories:
        likes_str = ", ".join(memories["likes"])
        prompt += f"\n- The user is interested in {likes_str}. Feel free to bring up these topics naturally or relate them to the current conversation."
    
    # Dislikes
    if "dislikes" in memories:
        dislikes_str = ", ".join(memories["dislikes"])
        prompt += f"\n- The user dislikes {dislikes_str}. "
    
    # Conversational Style
    if "conversation_dislike" in memories:
        style_str = ", ".join(memories["conversation_dislike"])
        prompt += f"\n- Conversation dislike : {style_str}, match the user's preferences."
    
    return prompt