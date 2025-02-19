import os
import json
from langchain.embeddings import GooglePalmEmbeddings
import chromadb
from chromadb.config import Settings
import google.generativeai as genai


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
    """Generate answer using Gemini Pro API"""
    # gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    result = model.generate_content(prompt)
    return result.text

#####################################
# 2. Load JSON & Detect Speaker
#####################################

def load_conversation_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_chroma_db(conversations, db_path, collection_name):
    # Ensure we create a fresh ChromaDB instance
    client = chromadb.PersistentClient(path=db_path)

    # Delete the old collection if it exists
    if collection_name in client.list_collections():
        client.delete_collection(collection_name)
        print("\n⚠️ [DEBUG] Deleted old ChromaDB collection to store new data.")

    # Create a new collection
    collection = client.get_or_create_collection(name=collection_name)

    for i, row in enumerate(conversations):
        doc_id = f"conv_{i}"
        text = row["message"]
        speaker = row.get("speaker", "unknown")
        ts = row.get("timestamp", "")

        embedding = get_embedding(text)

        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "speaker": speaker,
                "timestamp": ts
            }],
            ids=[doc_id]
        )

        # print(f"\n📥 [DEBUG] Inserted into ChromaDB (ID: {doc_id}): {text}")

    return collection


#####################################
# 3. Style & Info
#####################################

def retrieve_for_style(collection, speaker="crush", n_results=3):
    """
    檢索某個 speaker 的對話，用來模仿語氣
    - 可使用固定 query (例如 "example of crush's style")，
      再加上 where={"speaker": speaker} 過濾
    """
    query_text = "example of style"
    query_embed = get_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=n_results,
        where={"speaker": speaker}
    )
    # 取出 documents
    style_docs = results["documents"][0] if "documents" in results else []
    print("\n🔍 [DEBUG] Retrieved Style Docs:", style_docs)
    return style_docs

def retrieve_for_info(collection, user_query, n_results=5):
    """
    根據使用者問題 (user_query)，檢索整個對話紀錄(不分 speaker)
    以取得最相關的內容
    """
    query_embed = get_embedding(user_query)
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=n_results
    )
    info_docs = results["documents"][0] if "documents" in results else []
    print("\n🔍 [DEBUG] Retrieved Info Docs:", info_docs)
    return info_docs

#####################################
# 4. 建立 RAG Prompt
#####################################

def build_final_prompt(user_query, style_texts, info_texts):
    style_excerpt = "\n".join(style_texts)
    info_excerpt = "\n".join(info_texts)

    prompt = f"""
You are now going to mimic the speaking style of "crush."
Below are messages previously spoken by crush (as style examples):
{style_excerpt}

Below are relevant messages from the overall conversation related to the question:
{info_excerpt}

User's question: {user_query}

Please respond in crush's tone and provide a complete answer based on the above information.
"""
    return prompt

#####################################
# 5. Main 流程
#####################################

def main():
    # 準備基本資料
    json_path = "/conversation.json"  # 你的 JSON 對話檔
    db_folder = "chroma_db"
    db_name = "rag_experiment"

    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    # 讀取對話
    conversations = load_conversation_json(json_path)

    # 建立 ChromaDB，並把所有對話的 embedding 加入
    db_path = os.path.join(os.getcwd(), db_folder)
    collection = create_chroma_db(conversations, db_path, db_name)

    chat_history = []
    print("\n=== Chatbot Ready! Type 'exit' to quit ===\n")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == "exit":
            print("\nGoodbye!")
            break

        style_docs = retrieve_for_style(collection, speaker="crush", n_results=3)
        info_docs = retrieve_for_info(collection, user_query, n_results=5)

        if style_docs or info_docs:
            final_prompt = build_final_prompt(user_query, style_docs, info_docs)

            # 呼叫 Gemini 產生回答
            answer = generate_answer(final_prompt)

            # 顯示 AI 回應
            print("\nAI (Crush Style):", answer)

            # 儲存對話紀錄
            chat_history.append({"user": user_query, "ai": answer})
        else:
            print("No relevant content found in the database.")

if __name__ == "__main__":
    main()
