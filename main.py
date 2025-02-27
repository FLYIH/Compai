import os
import re
import orjson
from dotenv import load_dotenv
import telebot
from datetime import datetime
from collections import deque

# === Gemini Imports ===
import google.generativeai as genai
from chromadb.config import Settings
import chromadb
import uuid


# === Buffer 與 ChromaDB 操作函式 ===
def get_embedding(text: str) -> list[float]:
    return get_embeddings_batch([text])[0]

def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
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
                    if embedding:
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
    使用 orjson 加快 JSON 讀取速度
    """
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())
    
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

#####################################
# 3. Store New Conversations in ChromaDB Dynamically
#####################################

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

    # print(f"\n[DEBUG] Stored new conversation (ID: {doc_uuid}) - Speaker: {speaker} - Message: {message}")

def add_to_buffer(speaker, message):
    """
    Add the message to in-memory conversation buffer
    """
    conversation_buffer.append((speaker, message))

#####################################
# 4. Style & Info Retrieval
#####################################

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

    chat_history_texts = "\n".join(style_docs)
    style = extract_style_from_history(chat_history_texts)

    print(f"\n🎭 [INFO] Extracted speaking style for {speaker}: ", style)
    return style


def extract_style_from_history(chat_history_texts):
    """
    使用 Gemini 分析聊天記錄並提取寫作風格。
    此版本使用本地開發模式，不需要 ADC 憑證。
    """

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

        # 第 1 層：直接嘗試轉換成 JSON
        try:
            style_dict = orjson.loads(cleaned_text.encode('utf-8'))
            return style_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] First attempt to parse JSON failed:", e)

        # 第 2 層：修正常見 JSON 格式問題
        fixed_text = re.sub(r"'", '"', cleaned_text)
        fixed_text = re.sub(r",\s*}", "}", fixed_text)
        fixed_text = re.sub(r",\s*]", "]", fixed_text)

        try:
            style_dict = orjson.loads(fixed_text.encode('utf-8'))
            return style_dict
        except Exception as e:
            print("\n⚠️ [DEBUG] Second attempt to fix and parse JSON failed:", e)

        # 第 3 層：進一步修正 JSON 格式
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

def get_user_memory(user):
    """
    根據使用者 ID 檢索所有記憶
    """
    results = memory_collection.get(where={"user": user})
    memories = {}

    for metadata in results["metadatas"]:
        key = metadata.get("key", "")
        value = metadata.get("value", "")
        if key and value:
            if key in memories:
                memories[key].append(value)
            else:
                memories[key] = [value]

    return memories

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

def get_last_n_messages():
    """
    Return the conversation buffer as a list of (speaker, message)
    """
    return list(conversation_buffer)
#####################################
# 5. Build RAG Prompt
#####################################
def apply_memory_to_prompt(prompt, user):
    """
    Apply user memory to the prompt in English.
    """
    memories = get_user_memory(user)
    
    # Apply age
    if "age" in memories:
        prompt += f"\n- The user {user} is {memories['age'][0]} years old."

    # Apply likes and dislikes
    if "like" in memories:
        prompt += f"\n- The user likes: {', '.join(memories['like'])}."
    if "avoid_start" in memories:
        avoid_pattern = "|".join(memories["avoid_start"])
        prompt += f"\n- Avoid starting the response with: {avoid_pattern}."

    # Apply personal background information
    if "job" in memories:
        prompt += f"\n- The user works as a {memories['job'][0]}."
    if "nationality" in memories:
        prompt += f"\n- The user is from {memories['nationality'][0]}."
    if "residence" in memories:
        prompt += f"\n- The user lives in {memories['residence'][0]}."

    return prompt


def generate_answer_based_on_info(speaker, user ,user_query, context,relevant_info):
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


# === 初始化設定 ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# === 狀態管理 ===
conversation_buffer = deque(maxlen=3)
SPEAKER_ID = None
USER_ID = None
SPEAKER_STYLE = {}

# === ChromaDB 初始化 ===
json_path = "conversation/conversation2.json"
db_folder = "chroma_db"
db_name = "rag_experiment"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

db_path = os.path.join(os.getcwd(), db_folder)
client = chromadb.PersistentClient(path=db_path)

# 如果 collection 已存在，先刪除重建
if db_name in client.list_collections():
    client.delete_collection(db_name)

collection = client.get_or_create_collection(name=db_name)
for chunk in load_conversation_json_in_chunks(json_path, 100):
        create_chroma_db(chunk, collection)

# === Telegram Bot 指令處理 ===

@bot.message_handler(commands=['start'])
def get_started(message):
    bot.send_message(message.chat.id, "Hello! I am Compai! 💕\nPlease use \"/setspeaker <id>\" and \"/setuser <id>\" to input the id of speaker and user.")

@bot.message_handler(commands=['setspeaker'])
def set_speaker(message):
    global SPEAKER_ID
    try:
        SPEAKER_ID = message.text.split(" ")[1]
        bot.send_message(message.chat.id, f"Speaker ID set to {SPEAKER_ID}")
    except IndexError:
        bot.send_message(message.chat.id, "Please provide a speaker ID. Usage: /setspeaker <id>")

@bot.message_handler(commands=['setuser'])
def set_user(message):
    global USER_ID
    try:
        USER_ID = message.text.split(" ")[1]
        bot.send_message(message.chat.id, f"User ID set to {USER_ID}")
    except IndexError:
        bot.send_message(message.chat.id, "Please provide a user ID. Usage: /setuser <id>")

@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, "Use /setspeaker and /setuser to set the conversation roles.\nThen just type your message to start chatting!")
# === 主要訊息處理邏輯 ===
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    global SPEAKER_ID, USER_ID
    global SPEAKER_STYLE
    if not SPEAKER_ID or not USER_ID:
        bot.send_message(message.chat.id, "Please set both speaker and user IDs using /setspeaker and /setuser.")
        return

    if not SPEAKER_STYLE:
        SPEAKER_STYLE = analyze_speaker_style(collection, SPEAKER_ID)
        bot.send_message(message.chat.id, "Analyzing speaker style... Please wait a moment. 💕")
        return  # 初始化後先跳出，下一次進入迴圈才會開始對話

    user_query = message.text
    if user_query.lower() == "exit":
        bot.send_message(message.chat.id, "Goodbye! 👋")
        return


    add_to_buffer(USER_ID, user_query)
    add_new_conversation(collection, USER_ID, user_query)

    context_pairs = get_last_n_messages()
    context_texts = [f"{spk}: {msg}" for spk, msg in context_pairs]
    relevant_info = retrieve_for_info(collection, user_query, n_results=5)

    # 生成 Gemini 回應
    try:
        initial_answer = generate_answer_based_on_info(SPEAKER_ID, USER_ID, user_query, context_texts, relevant_info)
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)
        bot.send_message(message.chat.id, initial_answer)
        add_to_buffer(SPEAKER_ID, initial_answer)
        add_new_conversation(collection, SPEAKER_ID, initial_answer)
    except Exception as e:
        bot.send_message(message.chat.id, f"An error occurred: {e}")


# 啟動 Telegram Bot
bot.infinity_polling()
