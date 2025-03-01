import google.generativeai as genai

def get_embedding(text: str) -> list[float]:
    return get_embeddings_batch([text])[0]

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    model = "models/embedding-001"
    responses = []

    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [text for text in batch if text and text.strip()]

        if not batch:
            print("\n⚠️ [WARN] Empty batch encountered, skipping...")
            continue

        try:
            response = genai.embed_content(
                model=model,
                content=batch,
                task_type="retrieval_document"
            )

            if isinstance(response, dict) and "embedding" in response:
                embeddings = response["embedding"]

                valid_texts = []
                valid_embeddings = []

                for idx, embedding in enumerate(embeddings):
                    if embedding:
                        valid_texts.append(batch[idx])
                        valid_embeddings.append(embedding)
                    else:
                        print(f"\n⚠️ [WARN] Empty embedding for text: {batch[idx]}")

                if valid_embeddings:
                    responses.extend(valid_embeddings)
                else:
                    print("\n⚠️ [WARN] No valid embeddings in this batch.")

            else:
                print("\n⚠️ [WARN] Unexpected response format:", response)

        except Exception as e:
            print("\n❌ [ERROR] Error while getting embeddings:", e)
            print("\n⚠️ [DEBUG] Problematic batch:", batch)

    return responses