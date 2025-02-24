import os
import re
import json
from dotenv import load_dotenv
from langchain.embeddings import GooglePalmEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

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

    model = "models/embedding-001"
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document",
        title="Embedding Query"
    )
    return response["embedding"]

def generate_answer(prompt: str):
    """
    Uses the Gemini Pro model to generate an AI response based on a given prompt.
    """
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
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in client.list_collections():
        client.delete_collection(collection_name)
        # print("\n⚠️ [DEBUG] Deleted old ChromaDB collection to store new data.")

    # Create a new collection
    collection = client.get_or_create_collection(name=collection_name)

    for i, row in enumerate(conversations):
        # print(row)

        # 依 JSON 結構取 speaker, message, timestamp
        text = row.get("message", "")
        speaker = row.get("speaker", "unknown")
        ts = row.get("timestamp", "")

        embedding = get_embedding(text)

        doc_id = f"conv_{i}"
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"speaker": speaker, "timestamp": ts, "id": f"conv_{i}"}],
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
        metadatas=[{"speaker": speaker, "timestamp": ts}],
        ids=[doc_id]
    )

    # print(f"\n📥 [DEBUG] Stored new conversation (ID: {doc_id}) - Speaker: {speaker} - Message: {message}")

#####################################
# 4. Style & Info Retrieval
#####################################

# Global variable to store speaker's style
SPEAKER_STYLE = {}

def analyze_speaker_style(collection, speaker="crush", n_results=10):
    """
    Analyzes the target speaker's writing style and saves it for future use.
    This function is executed only once when the program starts.
    """
    print(f"\n🔍 [INFO] Analyzing the speaking style of {speaker}...")

    results = collection.get(where={"speaker": speaker})

    # 確認有找到相關文件
    if "documents" not in results or not results["documents"]:
        print(f"⚠️ [WARN] No historical messages found for {speaker}. Using default style.")
        return {
            "style": "neutral",
            "tone": "neutral",
            "common_emojis": [],
            "frequent_words": [],
            "punctuation_style": "standard"
        }

    # documents 可能是一個2維 list，要攤平
    style_docs = []
    for item in results["documents"]:
        # 有時候會是 list of strings
        if isinstance(item, list):
            style_docs.extend(item)
        elif isinstance(item, str):
            style_docs.append(item)
        else:
            print("⚠️ [ERROR] Unexpected item type:", type(item), "Value:", item)

    # 將所有對話組合成單一文字
    chat_history_texts = "\n".join(style_docs)
    style_json = extract_style_from_history(chat_history_texts)

    # 解析 JSON 結果，如果失敗就回傳預設
    try:
        style_data = json.loads(style_json)
        return style_data
    except json.JSONDecodeError:
        print("⚠️ [ERROR] Failed to parse style JSON. Using default style.")
        return {
            "style": "neutral",
            "tone": "neutral",
            "common_emojis": [],
            "frequent_words": [],
            "punctuation_style": "standard"
        }


def extract_style_from_history(chat_history_texts):
    """
    Uses Gemini to analyze chat history and extract writing style traits.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["chat_history"],
        template="""
        Analyze the following chat messages and extract the writing style.

        Chat history:
        {chat_history}

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
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(chat_history=chat_history_texts)

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

#####################################
# 5. Build RAG Prompt
#####################################

def generate_answer_based_on_info(speaker, user ,user_query, relevant_info):
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
    info_excerpt = "\n".join(relevant_info)

    prompt = f"""
    Your name is {speaker}.
    Here is the relevant conversation history:
    {info_excerpt}

    {user} asked this question: "{user_query}"

    Please respond as naturally as possible, just like a close friend would.
    Keep the tone friendly, relaxed, and genuine.
    If the conversation history doesn't directly answer the question,
    make a reasonable guess or respond in a friendly and relatable way, just like a human would.

    Avoid stating that you can't answer the question due to lack of information.
    Instead, respond naturally, using context or general knowledge where appropriate.
    The goal is to keep the conversation flowing smoothly and naturally.
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

    Refined answer:
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
    json_path = "conversation.json"
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
        relevant_info = retrieve_for_info(collection, user_query, n_results=5)

        # Step 2: Generate an initial answer based on retrieved info
        initial_answer = generate_answer_based_on_info(speaker, user, user_query, relevant_info)

        # Step 3: Apply saved style for final answer
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)

        # Step 4: Return the final styled answer
        print(f"\n{speaker}:{final_answer}")

        # Step 5: Store new conversations in ChromaDB
        add_new_conversation(collection, user, user_query)
        add_new_conversation(collection, speaker, final_answer)

if __name__ == "__main__":
    main()