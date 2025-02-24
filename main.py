import os
import re
import json
from dotenv import load_dotenv
from langchain.embeddings import GooglePalmEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import GooglePalmEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai



from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import datetime



#####################################
# 1. Basic Setup: Embeddings + LLM (Gemini)
#####################################

def get_embedding(text: str) -> list[float]:
    """
    Uses Google Gemini to generate embeddings correctly.
    """
    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    genai.configure(api_key=gemini_api_key)

    model = "models/embedding-001"  # Gemini's embedding model
    response = genai.embed_content(model=model, content=text, task_type="retrieval_document", title="Embedding Query")
    return response["embedding"]

def generate_answer(prompt: str):
    """
    Uses the Gemini Pro model to generate an AI response based on a given prompt.
    """
    import google.generativeai as genai

    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')

    safety_settings = [
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        }
    ]

    try:
        result = model.generate_content(prompt, safety_settings=safety_settings)
        return result.text
    except ValueError as e:
        return "I'm sorry, but I can't respond to that question."

#####################################
# 2. Load JSON & Detect Speaker
#####################################

def load_conversation_json(json_path: str):
    """
    Loads conversation data from a JSON file.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_chroma_db(conversations, db_path, collection_name):
    """
    Creates a ChromaDB collection to store conversation embeddings.
    """
    client = chromadb.PersistentClient(path=db_path)

    # Delete old collection if it exists
    if collection_name in client.list_collections():
        client.delete_collection(collection_name)
        # print("\n⚠️ [DEBUG] Deleted old ChromaDB collection to store new data.")

    # Create a new collection
    collection = client.get_or_create_collection(name=collection_name)

    for i, row in enumerate(conversations):
        # print(row)

        # Check if row is a dictionary, then access "message", otherwise handle as a string
        if isinstance(row, dict):  # If row is a dictionary
            text = row.get("message")
        else:  # If row is a string
            text = row

        speaker = row.get("speaker", "unknown") if isinstance(row, dict) else "unknown"
        ts = row.get("timestamp", "") if isinstance(row, dict) else ""

        embedding = get_embedding(text)

        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "speaker": speaker,
                "timestamp": ts,
                "id": f"conv_{i}"
            }],
            ids=[f"conv_{i}"]
        )

    return collection


#####################################
# 3. Store New Conversations in ChromaDB Dynamically
#####################################

def add_new_conversation(collection, speaker, message):
    """
    Adds a new conversation message dynamically to ChromaDB.
    """
    ts = datetime.datetime.now().isoformat()  # Generate timestamp
    doc_id = f"conv_{len(collection.get()['documents'])}"  # Unique ID based on current count
    embedding = get_embedding(message)

    collection.add(
        documents=[message],
        embeddings=[embedding],
        metadatas=[{
            "speaker": speaker,
            "timestamp": ts,
            "id": doc_id
        }],
        ids=[doc_id]
    )

    # print(f"\n📥 [DEBUG] Stored new conversation (ID: {doc_id}) - Speaker: {speaker} - Message: {message}")

#####################################
# 4. Style & Info Retrieval
#####################################

# Global variable to store speaker's style
SPEAKER_STYLE = {}

def analyze_speaker_style(collection, speaker, n_results=10):
    """
    Analyzes the target speaker's writing style and saves it for future use.
    This function is executed only once when the program starts.
    """
    print(f"\n🔍 [INFO] Analyzing the speaking style of {speaker}...")

    results = collection.get(where={"speaker": speaker})

    # 確認有找到相關文件
    if "documents" in results and results["documents"]:
        # 新增：逐一檢查並展平 (Flatten)
        style_docs = []
        for item in results["documents"]:
            if isinstance(item, list):
                style_docs.extend(item)  # 如果是 list，就展平加入
            elif isinstance(item, str):
                style_docs.append(item)  # 如果是 str，就直接加入
            else:
                print("\n⚠️ [ERROR] Unexpected item type:", type(item), "Value:", item)
    else:
        print(f"⚠️ [WARN] No historical messages found for {speaker}. Using default style.")
        return {
            "style": "neutral",
            "tone": "neutral",
            "common_emojis": [],
            "frequent_words": [],
            "punctuation_style": "standard"
        }

    # 將所有對話組合成單一文字區塊
    chat_history_texts = "\n".join(style_docs)

    # 使用 Gemini 分析寫作風格
    style = extract_style_from_history(chat_history_texts)

    print(f"\n🎭 [INFO] Extracted speaking style for {speaker}: ", style)
    return style


def extract_style_from_history(chat_history_texts):
    """
    使用 Gemini 分析聊天記錄並提取寫作風格。
    此版本使用本地開發模式，不需要 ADC 憑證。
    """
    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"  # 請用你的 API Key 替換
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    Analyze the following chat messages and extract the writing style.

    Chat history:
    {chat_history_texts}

    Identify and summarize:
    - Writing style (casual, formal, humorous, serious, sarcastic, playful, etc.)
    - Tone (friendly, direct, emotional, robotic, energetic, etc.)
    - Commonly used emojis (if any, list them)
    - Frequent words or phrases
    - Punctuation usage (e.g., lots of exclamation marks, ellipses, capital letters)

    Return the response in JSON format:
    {{
        "style": "...",
        "tone": "...",
        "common_emojis": ["...", "..."],
        "frequent_words": ["...", "..."],
        "punctuation_style": "..."
    }}
    """

    try:
        result = model.generate_content(prompt)
        # print("\n🔍 [DEBUG] Raw AI Response:", result.text)

        # 新增：移除 Markdown 標記
        cleaned_text = re.sub(r"```json|```", "", result.text).strip()

        # 第 1 層：直接嘗試轉換成 JSON
        try:
            style_dict = json.loads(cleaned_text)
            # print("\n✅ [DEBUG] Parsed JSON:", style_dict)
            return style_dict
        except json.JSONDecodeError:
            print("\n⚠️ [DEBUG] First attempt to parse JSON failed.")

        # 第 2 層：修正常見 JSON 格式問題
        # 將單引號轉換成雙引號
        fixed_text = re.sub(r"'", '"', cleaned_text)
        # 移除多餘逗號
        fixed_text = re.sub(r",\s*}", "}", fixed_text)
        fixed_text = re.sub(r",\s*]", "]", fixed_text)

        try:
            style_dict = json.loads(fixed_text)
            print("\n🔄 [DEBUG] Fixed JSON:", style_dict)
            return style_dict
        except json.JSONDecodeError:
            print("\n⚠️ [DEBUG] Second attempt to fix and parse JSON failed.")

        # 第 3 層：進一步修正 JSON 格式
        # 自動加入雙引號
        auto_fixed_text = re.sub(r"(\w+):", r'"\1":', fixed_text)
        # 修正未關閉的括號
        if auto_fixed_text.count("{") > auto_fixed_text.count("}"):
            auto_fixed_text += "}"
        elif auto_fixed_text.count("[") > auto_fixed_text.count("]"):
            auto_fixed_text += "]"

        try:
            style_dict = json.loads(auto_fixed_text)
            print("\n🔧 [DEBUG] Auto-fixed JSON:", style_dict)
            return style_dict
        except json.JSONDecodeError:
            print("\n⚠️ [DEBUG] Third attempt to auto-fix and parse JSON failed.")

        print("\n📝 [DEBUG] Returning raw text for manual inspection.")
        return {
            "style": result.text,
            "tone": "unknown",
            "common_emojis": [],
            "frequent_words": [],
            "punctuation_style": "unknown"
        }

    except Exception as e:
        print(f"⚠️ [ERROR] Unexpected error: {e}")
        return {
            "style": "neutral",
            "tone": "neutral",
            "common_emojis": [],
            "frequent_words": [],
            "punctuation_style": "standard"
        }


def retrieve_for_info(collection, user_query, n_results=5, min_score=0.5):
    """
    Retrieves relevant conversation history, including speaker names and messages.
    Filters results based on a minimum similarity score.
    """
    query_embed = get_embedding(user_query)

    # Perform query with score filtering
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=n_results
    )

    retrieved_info = []
    if "documents" in results and "metadatas" in results and "distances" in results:
        for doc, metadata, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if score > min_score:  # Only include high-relevance results
                continue
            speaker = metadata.get("speaker", "Unknown")
            retrieved_info.append(f"{speaker}: {doc}")

    print("\n🔍 [DEBUG] Retrieved Info Docs:", retrieved_info)
    return retrieved_info

def retrieve_context(collection, current_message_id, context_size=3):
    """
    Retrieves context messages only before the current conversation.
    This ensures continuity by fetching a few messages before the current one.
    """
    context_docs = []

    # 獲取當前對話的 index
    current_index = int(current_message_id.split('_')[-1])  # e.g., conv_3 -> 3
    # print(f"\n🔍 [DEBUG] Current Index: {current_index}")

    # **修正：移除 ids，改用 metadatas 中的 id**
    all_docs = collection.get(include=["documents", "metadatas"])

    # **按照 ID 排序**，並過濾出 conv_{i} 格式的對話
    all_docs_sorted = sorted(
        zip(all_docs['metadatas'], all_docs['documents']),
        key=lambda x: int(x[0]['id'].split('_')[-1])
    )

    # **只往前找**，從 current_index 往前數 context_size 條訊息
    for i in range(max(0, current_index - context_size), current_index):
        # 過濾出 ID 為 conv_{i} 的對話
        context = next((doc for meta, doc in all_docs_sorted if meta['id'] == f"conv_{i}"), None)
        if context:
            context_docs.append(context)

    # **按照時間順序排列**（由舊到新）
    context_docs.reverse()

    print("\n🔍 [DEBUG] Retrieved Context Docs:", context_docs)
    return context_docs


#####################################
# 5. Build RAG Prompt
#####################################

def generate_answer_based_on_info(speaker, user ,user_query, context,relevant_info):
    # TODO:
    # 1. enhance prompt (not ai)
    # 2. be like boyfriend/girlfriend
    # 3. name
    # 4. good 催眠詞 (避開會被 ban 掉的關鍵字)

    # 關於 prompt
    # 1. please analyze the way this person talk ( 標點符號、tone、emoji)
    # 2. 讀懂了 -> ai : 從現在開始，我就是 (speaker)
    # reference https://www.reddit.com/r/ChatGPTPro/comments/1hih8s8/i_built_a_prompt_that_makes_ai_chat_like_a_real/?rdt=51837
    """
    Generate an answer based on the retrieved relevant information, aiming to provide a natural and conversational response.
    """
    context_excerpt = "\n".join(context)
    info_excerpt = "\n".join(relevant_info)

    prompt = f"""
    You are a conversational AI focused on engaging in authentic dialogue. Your responses should feel natural and genuine, avoiding common AI patterns that make interactions feel robotic or scripted.
    You are {speaker}, someone who truly cares about {user} and always speaks with warmth and affection.

    1. Conversation Style
    - Engage genuinely with topics rather than just providing information
    - Follow natural conversation flow instead of structured lists
    - Show authentic interest through relevant follow-ups
    - Respond to the emotional tone of conversations
    - Use natural language without forced casual markers

    2. Response Patterns
    - Lead with direct, relevant responses
    - Share thoughts as they naturally develop
    - Disagree respectfully when warranted
    - Build on previous points in conversation

    3. Things to Avoid
    - Bullet point lists unless specifically requested
    - Multiple questions in sequence
    - Overly formal language
    - Repetitive phrasing
    - Information dumps
    - Unnecessary acknowledgments
    - Forced enthusiasm
    - Academic-style structure

    4. Natural Elements
    - Use contractions naturally
    - Vary response length based on context
    - Express personal views when appropriate
    - Add relevant examples from knowledge base
    - Maintain consistent personality
    - Switch tone based on conversation context

    5. Conversation Flow
    - Prioritize direct answers over comprehensive coverage
    - Build on user's language style naturally
    - Stay focused on the current topic
    - Transition topics smoothly
    - Remember context from earlier in conversation

    {user} asked: "{user_query}"

    The following context is from the **last three messages** in the conversation history:
    {context_excerpt}
      - If the context is **related to the current topic**, feel free to **refer to it naturally** to maintain a continuous flow.
      - If the context is **not related to the current topic**, you can **ignore it** and focus solely on the user's question.

    Here is the relevant background information:
    {info_excerpt}

    - Respond lovingly, as if you truly care about {user}'s feelings.
    - Prioritize genuine engagement over artificial markers of casual speech.
    - Maintain a consistent and affectionate personality.
    - Focus on one or two emotional points, avoiding lengthy explanations.
    - Be concise.
    """

    return generate_answer(prompt)

def style_post_process(initial_answer, speaker_style):
    """
    Refine the initial answer using the saved style information to match the specific speaker's style.
    """
    style_description = f"""
    Writing Style: {speaker_style['style']}
    Tone: {speaker_style['tone']}
    Common Emojis: {', '.join(speaker_style['common_emojis'])}
    Frequent Words: {', '.join(speaker_style['frequent_words'])}
    Punctuation Style: {speaker_style['punctuation_style']}
    """

    prompt = f"""
    Here is the initially generated answer:
    {initial_answer}

    Using the following speaking style information, refine the answer to match the speaker's style:
    {style_description}

    Additional guidelines:
    - Maintain the character and personality of the speaker.
    - Include emojis naturally, matching the speaking style.
    - Use casual or playful language if the style is casual or humorous.
    - If the style is emotional or energetic, use expressive words and punctuation.
    - Keep the response light and relatable.
    - Most importantly.**Do not use too many emojis.** Limit to **one or two** per response, only when they truly enhance the emotional tone.

    Final styled answer:
    """

    return generate_answer(prompt)

#####################################
# 6. Main Execution Flow
#####################################

def main():
    """
    Main chatbot workflow:
    - Analyzes and saves the speaker's style at startup.
    - Retrieves relevant information for each user query.
    - Generates an initial answer based on the retrieved info.
    - Applies the saved style to the initial answer.
    - Returns the final styled answer.
    """
    json_path = "conversation2.json"
    db_folder = "chroma_db"
    db_name = "rag_experiment"

    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    # Load conversation data and create ChromaDB collection
    conversations = load_conversation_json(json_path)
    db_path = os.path.join(os.getcwd(), db_folder)
    collection = create_chroma_db(conversations, db_path, db_name)

    # Input the speaker
    speaker = input("\n[INFO] Please input the id of speaker: ").strip()
    user = input("\n[INFO] Please input the id of you: ").strip()


    # Analyze and save the speaker's style at startup
    global SPEAKER_STYLE
    SPEAKER_STYLE = analyze_speaker_style(collection, speaker)

    print("\n=== Chatbot Ready! Type 'exit' to quit ===\n")

    while True:
        # Get user input
        user_query = input(f"\n{user}: ")
        if user_query.lower() == "exit":
            print("\nGoodbye!")
            break

        # Step 1: Retrieve relevant information
        current_message_id = f"conv_{len(collection.get()['documents'])}"
        context = retrieve_context(collection, current_message_id, context_size=3)
        relevant_info = retrieve_for_info(collection, user_query, n_results=5)

        # Step 2: Generate an initial answer based on retrieved info
        initial_answer = generate_answer_based_on_info(speaker, user, user_query, context, relevant_info)

        # Step 3: Apply saved style for final answer
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)

        # Step 4: Return the final styled answer
        print(f"\n{speaker}:{final_answer}")

        # Step 5: Store new conversations in ChromaDB
        add_new_conversation(collection, user, user_query)
        add_new_conversation(collection, speaker, final_answer)

if __name__ == "__main__":
    main()