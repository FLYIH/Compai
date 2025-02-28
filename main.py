import os
from dotenv import load_dotenv
import telebot
from collections import deque
import google.generativeai as genai
from chromadb.config import Settings
import chromadb
from collections import deque
from format_converter import TelegramChatConverter
from gemini_handler import generate_answer, add_new_conversation, retrieve_for_info, get_last_n_messages, generate_answer_based_on_info, style_post_process, analyze_speaker_style, add_to_buffer, load_conversation_json_in_chunks, create_chroma_db

# === Intialization Settings ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# === Stage Management ===
global conversation_buffer
conversation_buffer = deque(maxlen=3)
SPEAKER_ID = None
USER_ID = None
SPEAKER_STYLE = {}

# === Chat History Folder ===
CHAT_HISTORY_FOLDER = "conversation"

# === ChromaDB Initialization ===
json_path = "conversation/chat_history.json"
db_folder = "chroma_db"
db_name = "rag_experiment"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)
db_path = os.path.join(os.getcwd(), db_folder)
client = chromadb.PersistentClient(path=db_path)

global collection
collection = client.get_or_create_collection(name=db_name)


# === Telegram Bot Command Handlers ===
# === Setup command handlers ===
@bot.message_handler(commands=['start'])
def get_started(message):
    bot.send_message(message.chat.id, "Hello! I am Compai! 💕\nPlease send me the Telegrma chat history file \"result.json\" to start.")
    bot.send_message(message.chat.id, "You can download it in the chatroom you want > advanced > export chat history > format: machine-readable JSON > Save > Export")

# === Receive chat history file and analyze the speaker style ===
@bot.message_handler(content_types=['document'])
def command_handle_document(message):
    file_name = message.document.file_name

    if not file_name.endswith(".json"):
        bot.send_message(message.chat.id, "Please upload a JSON file (result.json).")
        return

    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = os.path.join(CHAT_HISTORY_FOLDER, file_name)
    with open(file_path, 'wb') as f:
        f.write(downloaded_file)
    CHAT_HISTORY = True

    bot.send_message(message.chat.id, 'Document received! Analyzing chat history... 💕')
    converter = TelegramChatConverter(file_path, json_path)
    SPEAKER_ID = converter.convert()
    USER_ID = message.chat.username

    if not SPEAKER_STYLE:
        bot.send_message(message.chat.id, "Analyzing speaker style... Please wait a moment. 💕")
        SPEAKER_STYLE = analyze_speaker_style(collection, SPEAKER_ID)
        for chunk in load_conversation_json_in_chunks(json_path, 100):
            create_chroma_db(chunk, collection)
        return 

# === Main message handler ===
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    global SPEAKER_ID, USER_ID, CHAT_HISTORY
    global SPEAKER_STYLE
    if not CHAT_HISTORY:
        bot.send_message(message.chat.id, "Please send the chat history file first.")
        return

    # if not SPEAKER_STYLE:
    #     bot.send_message(message.chat.id, "Analyzing speaker style... Please wait a moment. 💕")
    #     SPEAKER_STYLE = analyze_speaker_style(collection, SPEAKER_ID)
    
    user_query = message.text
    if user_query.lower() == "exit":
        bot.send_message(message.chat.id, "Goodbye! 👋")
        return

    add_to_buffer(USER_ID, user_query, conversation_buffer)
    add_new_conversation(collection, USER_ID, user_query)

    context_pairs = get_last_n_messages(conversation_buffer)
    context_texts = [f"{spk}: {msg}" for spk, msg in context_pairs]
    relevant_info = retrieve_for_info(collection, user_query, n_results=5)

    # Generate Gemini answer
    try:
        initial_answer = generate_answer_based_on_info(SPEAKER_ID, USER_ID, user_query, context_texts, relevant_info)
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)
        bot.send_message(message.chat.id, final_answer)
        add_to_buffer(SPEAKER_ID, initial_answer,conversation_buffer)
        add_new_conversation(collection, SPEAKER_ID, initial_answer)
    except Exception as e:
        bot.send_message(message.chat.id, f"An error occurred: {e}")

# Start Telegram bot
bot.infinity_polling()
