import re
import orjson
import google.generativeai as genai
from .embedding_handler import get_embedding

def analyze_speaker_style(collection, speaker, n_results=10):
    """
    Analyzes the target speaker's writing style and saves it for future use.
    This function is executed only once when the program starts.
    """
    print(f"\n🔍 [INFO] Analyzing the speaking style of {speaker}...")

    results = collection.get(where={"speaker": speaker})

    if "documents" not in results or not results["documents"]:
        print(f"⚠️ [WARN] No historical messages found for {speaker}. Using default style.")
        return {
            "style": "neutral",
            "tone": "neutral",
            "common_emojis": [],
            "frequent_words": [],
            "punctuation_style": "standard"
        }

    style_docs = []
    for item in results["documents"]:
        if isinstance(item, list):
            style_docs.extend(item)
        elif isinstance(item, str):
            style_docs.append(item)
        else:
            print("⚠️ [ERROR] Unexpected item type:", type(item), "Value:", item)

    chat_history_texts = "\n".join(style_docs)
    style = extract_style_from_history(chat_history_texts)

    print(f"\n🎭 [INFO] Extracted speaking style for {speaker}: ", style)
    return style


def extract_style_from_history(chat_history_texts):
    model = genai.GenerativeModel('gemini-1.5-flash')

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
        cleaned_text = re.sub(r"```json|```", "", result.text).strip()

        try:
            style_dict = orjson.loads(cleaned_text.encode('utf-8'))
            return style_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] First attempt to parse JSON failed:", e)

        fixed_text = re.sub(r"'", '"', cleaned_text)
        fixed_text = re.sub(r",\s*}", "}", fixed_text)
        fixed_text = re.sub(r",\s*]", "]", fixed_text)

        try:
            style_dict = orjson.loads(fixed_text.encode('utf-8'))
            return style_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] Second attempt to fix and parse JSON failed:", e)

        auto_fixed_text = re.sub(r"(\w+):", r'"\1":', fixed_text)
        if auto_fixed_text.count("{") > auto_fixed_text.count("}"):
            auto_fixed_text += "}"
        elif auto_fixed_text.count("[") > auto_fixed_text.count("]"):
            auto_fixed_text += "]"

        try:
            style_dict = orjson.loads(auto_fixed_text.encode('utf-8'))
            return style_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] Third attempt to auto-fix and parse JSON failed:", e)

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
            if score > min_score:
                continue
            speaker = metadata.get("speaker", "Unknown")
            retrieved_info.append(f"{speaker}: {doc}")

    print("\n🔍 [DEBUG] Retrieved Info Docs:", retrieved_info)
    return retrieved_info