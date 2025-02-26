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
import uuid, datetime
from collections import deque
# source venv/bin/activate
#####################################
# 1. Basic Setup: Embeddings + LLM (Gemini)
#####################################

conversation_buffer = deque(maxlen=3)

def get_embedding(text: str) -> list[float]:
    """
    單行版本的 get_embedding，但改用批次版本以提升效能
    """
    return get_embeddings_batch([text])[0]

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    使用批次處理獲取 embeddings，減少 API 請求次數，提升速度。
    """
    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    genai.configure(api_key=gemini_api_key)

    model = "models/embedding-001"
    responses = []

    # 每次處理 10 條，避免超過 API 限制
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # 過濾掉空字串和只有空白符號的內容
        batch = [text for text in batch if text and text.strip()]

        # 如果 batch 全部都是空的，則跳過
        if not batch:
            print("\n⚠️ [WARN] Empty batch encountered, skipping...")
            continue

        try:
            response = genai.embed_content(
                model=model,
                content=batch,
                task_type="retrieval_document"
            )

            # 檢查回傳格式
            if isinstance(response, dict) and "embedding" in response:
                embeddings = response["embedding"]

                # 修正：只保留成功生成 embedding 的內容
                valid_texts = []
                valid_embeddings = []

                # 檢查每一個 embedding
                for idx, embedding in enumerate(embeddings):
                    if embedding:  # 如果 embedding 不為空，則保留
                        valid_texts.append(batch[idx])
                        valid_embeddings.append(embedding)
                    else:
                        print(f"\n⚠️ [WARN] Empty embedding for text: {batch[idx]}")

                # 若有任何有效的 embedding，才加入 response
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


def generate_answer(prompt: str):
    """
    Uses the Gemini Pro model to generate an AI response based on a given prompt.
    """
    import google.generativeai as genai

    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

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

def load_conversation_json_in_chunks(json_path: str, chunk_size=100):
    """
    分批讀取 JSON，減少記憶體佔用。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

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
            ts = datetime.datetime.now().isoformat()

        # 使用 UUID 做 doc_id
        doc_uuid = f"conv_{uuid.uuid4()}"

        texts.append(text)
        ids.append(doc_uuid)
        metadatas.append({
            "speaker": speaker,
            "timestamp": ts,
            "id": doc_uuid  
        })

        # 批次執行
        if len(texts) == batch_size or i == len(conversations) - 1:
            batch_embeddings = get_embeddings_batch(texts)
            filtered_texts, filtered_ids, filtered_metadatas, filtered_embeddings = [], [], [], []

            for idx, emb in enumerate(batch_embeddings):
                if emb:  # 有效 embedding
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

#####################################
# 3. Store New Conversations in ChromaDB Dynamically
#####################################

def add_new_conversation(collection, speaker, message):
    """
    Adds a new conversation message dynamically to ChromaDB.
    Using UUID for doc_id, timestamp for sorting.
    """

    ts = datetime.datetime.now().isoformat()
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

    # print(f"\n[DEBUG] Stored new conversation (ID: {doc_uuid}) - Speaker: {speaker} - Message: {message}")

def add_to_buffer(speaker, message):
    """
    Add the message to in-memory conversation buffer
    """
    conversation_buffer.append((speaker, message))

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
    gemini_api_key = "AIzaSyDTid8X9cbe_iO9soS0IfuO9OLmvToY4KU"
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")

    genai.configure(api_key=gemini_api_key)
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

def get_last_n_messages():
    """
    Return the conversation buffer as a list of (speaker, message)
    """
    return list(conversation_buffer)
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
    # print(f"\n🔍 [DEBUG] Context Excerpt:\n{context_excerpt}")
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

    # 初始化 ChromaDB
    db_path = os.path.join(os.getcwd(), db_folder)
    client = chromadb.PersistentClient(path=db_path)
    
    # 如果 collection 已存在，先刪除重建
    if db_name in client.list_collections():
        client.delete_collection(db_name)
    
    # 創建 Collection (持續更新)
    global collection
    collection = client.get_or_create_collection(name=db_name)

    # 每個 chunk 都呼叫 populate_collection
    for chunk in load_conversation_json_in_chunks(json_path, 100):
        create_chroma_db(chunk, collection)

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

        add_to_buffer(user, user_query)
        add_new_conversation(collection, user, user_query)

        # Step 1: Retrieve relevant information
        # current_message_id = f"conv_{len(collection.get()['documents'])}"
        context_pairs = get_last_n_messages()  # e.g. [(speaker, msg), (speaker2, msg2), ...]
        # 把對話拼成文字
        context_texts = [f"{spk}: {msg}" for spk, msg in context_pairs]
        relevant_info = retrieve_for_info(collection, user_query, n_results=5)

        # Step 2: Generate an initial answer based on retrieved info
        initial_answer = generate_answer_based_on_info(speaker, user, user_query, context_texts, relevant_info)

        # Step 3: Apply saved style for final answer
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)
        
        # Step 4: Return the final styled answer
        print(f"\n{speaker}:{final_answer}")

        # Step 5: Store new conversations in ChromaDB
        add_to_buffer(speaker, final_answer)
        add_new_conversation(collection, speaker, final_answer)

if __name__ == "__main__":
    main()