import os
from dotenv import load_dotenv
import telebot
from collections import deque
import google.generativeai as genai
from chromadb.config import Settings
import chromadb
from collections import deque
from gemini_handler import generate_answer, add_new_conversation, retrieve_for_info, get_last_n_messages, generate_answer_based_on_info, style_post_process, analyze_speaker_style, add_to_buffer

# === 初始化設定 ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# === 狀態管理 ===
global conversation_buffer
conversation_buffer = deque(maxlen=3)
SPEAKER_ID = None
USER_ID = None
SPEAKER_STYLE = {}

# === ChromaDB 初始化 ===
db_folder = "chroma_db"
db_name = "rag_experiment"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
db_path = os.path.join(os.getcwd(), db_folder)
client = chromadb.PersistentClient(path=db_path)

global collection
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
    global SPEAKER_STYLE
    if not SPEAKER_ID or not USER_ID:
        bot.send_message(message.chat.id, "Please set both speaker and user IDs using /setspeaker and /setuser.")
        return

    if not SPEAKER_STYLE:
        bot.send_message(message.chat.id, "Analyzing speaker style... Please wait a moment. 💕")
        SPEAKER_STYLE = analyze_speaker_style(collection, SPEAKER_ID)
        return 
    user_query = message.text
    if user_query.lower() == "exit":
        bot.send_message(message.chat.id, "Goodbye! 👋")
        return


    add_to_buffer(USER_ID, user_query, conversation_buffer)
    add_new_conversation(collection, USER_ID, user_query)

    context_pairs = get_last_n_messages(conversation_buffer)
    context_texts = [f"{spk}: {msg}" for spk, msg in context_pairs]
    relevant_info = retrieve_for_info(collection, user_query, n_results=5)

    # 生成 Gemini 回應
    try:
        initial_answer = generate_answer_based_on_info(SPEAKER_ID, USER_ID, user_query, context_texts, relevant_info)
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)
        bot.send_message(message.chat.id, final_answer)
        add_to_buffer(SPEAKER_ID, initial_answer,conversation_buffer)
        add_new_conversation(collection, SPEAKER_ID, initial_answer)
    except Exception as e:
        bot.send_message(message.chat.id, f"An error occurred: {e}")


# 啟動 Telegram Bot
bot.infinity_polling()
