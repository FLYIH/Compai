from datetime import datetime
import uuid
from .embedding_handler import get_embedding, get_embeddings_batch

def create_chroma_db(conversations, collection):
    """
    Batch add data from `conversations` into the existing `collection`.
    Using UUID for doc_id and storing timestamp for sorting.
    """
    batch_size = 10
    texts, metadatas, ids, embeddings = [], [], [], []

    for i, row in enumerate(conversations):
        text = row["message"] if isinstance(row, dict) else row
        speaker = row.get("speaker", "unknown") if isinstance(row, dict) else "unknown"
        ts = row.get("timestamp", "") if isinstance(row, dict) else ""
        
        if not ts:
            ts = datetime.now().isoformat()

        doc_uuid = f"conv_{uuid.uuid4()}"

        texts.append(text)
        ids.append(doc_uuid)
        metadatas.append({
            "speaker": speaker,
            "timestamp": ts,
            "id": doc_uuid  
        })

        if len(texts) == batch_size or i == len(conversations) - 1:
            batch_embeddings = get_embeddings_batch(texts)
            filtered_texts, filtered_ids, filtered_metadatas, filtered_embeddings = [], [], [], []

            for idx, emb in enumerate(batch_embeddings):
                if emb:
                    filtered_texts.append(texts[idx])
                    filtered_ids.append(ids[idx])
                    filtered_metadatas.append(metadatas[idx])
                    filtered_embeddings.append(emb)
                else:
                    print(f"\n⚠️ [WARN] Skipping text with no embedding: {texts[idx]}")

            if filtered_texts:
                collection.add(
                    documents=filtered_texts,
                    embeddings=filtered_embeddings,
                    metadatas=filtered_metadatas,
                    ids=filtered_ids
                )
            
            texts, ids, metadatas, embeddings = [], [], [], []
    return collection

def add_new_conversation(collection, speaker, message):
    """
    Adds a new conversation message dynamically to ChromaDB.
    Using UUID for doc_id, timestamp for sorting.
    """

    ts = datetime.now().isoformat()
    doc_uuid = f"conv_{uuid.uuid4()}"
    embedding = get_embedding(message)

    collection.add(
        documents=[message],
        embeddings=[embedding],
        metadatas=[{
            "speaker": speaker,
            "timestamp": ts,
            "id": doc_uuid
        }],
        ids=[doc_uuid]
    )