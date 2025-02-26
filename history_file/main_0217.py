import os
import json
from dotenv import load_dotenv
from langchain_community.embeddings import GooglePalmEmbeddings
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
    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")
    
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    result = model.generate_content(prompt)
    return result.text

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

def retrieve_for_style(collection, speaker="crush", n_results=3):
    """
    Retrieves past messages spoken by a specific speaker for style imitation.
    """
    query_text = "example of style"
    query_embed = get_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=n_results,
        where={"speaker": speaker}
    )

    style_docs = results["documents"][0] if "documents" in results else []
    return style_docs

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
            if score < min_score:  # Only include high-relevance results
                continue
            speaker = metadata.get("speaker", "Unknown")
            retrieved_info.append(f"{speaker}: {doc}")

    print("\n🔍 [DEBUG] Retrieved Info Docs:", retrieved_info)
    return retrieved_info

#####################################
# 5. Build RAG Prompt
#####################################

def build_final_prompt(user_query, user, speaker, style_texts, info_texts):
    """
    Constructs a prompt for Gemini using retrieved speaker-tagged messages.
    """
    style_excerpt = "\n".join(style_texts)
    info_excerpt = "\n".join(info_texts)

    prompt = f"""
You are now going to mimic the speaking style of "{speaker}".
Below are messages previously spoken by {speaker} (as style examples):
{style_excerpt}

Below is the relevant conversation history:
{info_excerpt}

{user}'s question: {user_query}

Please respond in {speaker}'s tone and provide a complete answer based on the above information.
"""
    return prompt

#####################################
# 6. Main Execution Flow
#####################################

def main():
    """
    Runs the chatbot system:
    - Loads conversation data from JSON.
    - Stores the conversation embeddings in ChromaDB.
    - Allows real-time chat while dynamically updating ChromaDB.
    """
    json_path = "conversation1.json"
    db_folder = "chroma_db"
    db_name = "rag_experiment"

    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    # Load conversation data and detect the main speaker
    conversations = load_conversation_json(json_path)
    db_path = os.path.join(os.getcwd(), db_folder)
    collection = create_chroma_db(conversations, db_path, db_name)

    print("\n=== Chatbot Ready! Type 'exit' to quit ===\n")

    while True:
        # Get user input
        user_query = input("\nYou: ")
        if user_query.lower() == "exit":
            print("\nGoodbye!")
            break

        # Retrieve past messages for style and relevant information
        style_docs = retrieve_for_style(collection, speaker="crush", n_results=3)
        info_docs = retrieve_for_info(collection, user_query, n_results=5)

        if style_docs or info_docs:
            # Build the prompt using retrieved conversation data (need to modigy user and other)
            final_prompt = build_final_prompt(user_query, "you","crush", style_docs, info_docs)

            # Generate a response using Gemini AI
            ai_response = generate_answer(final_prompt)

            # Print the AI-generated response
            print("\nAI (Crush Style):", ai_response)

            # Store the new conversation dynamically in ChromaDB
            add_new_conversation(collection, "you", user_query)  # User message
            add_new_conversation(collection, "crush", ai_response)  # AI response

        else:
            print("No relevant content found in the database.")

if __name__ == "__main__":
    main()
