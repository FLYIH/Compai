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

# === ChromaDB 初始化 ===
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
    if not SPEAKER_ID or not USER_ID:
        bot.send_message(message.chat.id, "Please set both speaker and user IDs using /setspeaker and /setuser.")
        return

    user_query = message.text
    add_to_buffer(USER_ID, user_query)
    add_new_conversation(collection, USER_ID, user_query)

    context_pairs = get_last_n_messages()
    context_texts = [f"{spk}: {msg}" for spk, msg in context_pairs]
    relevant_info = retrieve_for_info(collection, user_query, n_results=5)

    # 生成 Gemini 回應
    try:
        initial_answer = generate_answer_based_on_info(SPEAKER_ID, USER_ID, user_query, context_texts, relevant_info)
        bot.send_message(message.chat.id, initial_answer)
        add_to_buffer(SPEAKER_ID, initial_answer)
        add_new_conversation(collection, SPEAKER_ID, initial_answer)
    except Exception as e:
        bot.send_message(message.chat.id, f"An error occurred: {e}")

# === Buffer 與 ChromaDB 操作函式 ===

def add_to_buffer(speaker, message):
    conversation_buffer.append((speaker, message))

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

def add_new_conversation(collection, speaker, message):
    ts = datetime.now().isoformat()
    doc_uuid = f"conv_{uuid.uuid4()}"
    embedding = [0.0] * 768  # 這裡可以換成你的 get_embedding(message) 函式

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

def get_last_n_messages():
    return list(conversation_buffer)

def retrieve_for_info(collection, user_query, n_results=5, min_score=0.5):
    """
    假設有一個 get_embedding() 函式能將 user_query 轉成向量
    """
    query_embed = [0.0] * 768  # 這裡可以換成你的 get_embedding(user_query) 函式
    results = collection.query(query_embeddings=[query_embed], n_results=n_results)
    retrieved_info = []
    if "documents" in results and "metadatas" in results and "distances" in results:
        for doc, metadata, score in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if score > min_score:
                continue
            speaker = metadata.get("speaker", "Unknown")
            retrieved_info.append(f"{speaker}: {doc}")
    return retrieved_info

# 啟動 Telegram Bot
bot.infinity_polling()
