import os
from dotenv import load_dotenv
import telebot
from collections import deque
import google.generativeai as genai
from chromadb.config import Settings
import chromadb
from format_converter import TelegramChatConverter
from gemini_handler import (
    generate_answer, 
    add_new_conversation, 
    retrieve_for_info, 
    get_last_n_messages, 
    generate_answer_based_on_info, 
    style_post_process, 
    analyze_speaker_style, 
    add_to_buffer, 
    load_conversation_json_in_chunks, 
    create_chroma_db
)

# === Initialization Settings ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")

genai.configure(api_key=GEMINI_API_KEY)
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode=None)

# === State Management ===
global conversation_buffer
conversation_buffer = deque(maxlen=3)
SPEAKER_ID = None
USER_ID = None
SPEAKER_STYLE = {}
CHAT_HISTORY = False

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
@bot.message_handler(commands=['start'])
def get_started(message):
    """
    Start command to initiate the bot.
    """
    bot.send_message(message.chat.id, "Hello! I am Compai! 💕\nPlease send me the Telegram chat history file \"result.json\" to start.")
    bot.send_message(message.chat.id, "You can download it in the chatroom you want > advanced > export chat history > format: machine-readable JSON > Save > Export")

# === Receive chat history file and analyze the speaker style ===
@bot.message_handler(content_types=['document'])
def command_handle_document(message):
    """
    Handle the uploaded chat history JSON file.
    Parse it, convert the format, and analyze the speaker style.
    """
    global CHAT_HISTORY, SPEAKER_ID, USER_ID, SPEAKER_STYLE, collection  # Declare global variables

    file_name = message.document.file_name
    bot.send_message(message.chat.id, f"Received document: {file_name}")

    # Validate file type
    if not file_name.endswith(".json"):
        bot.send_message(message.chat.id, "Please upload a JSON file (result.json).")
        return

    # Download file from Telegram
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = os.path.join(CHAT_HISTORY_FOLDER, file_name)

    # Save the file locally
    with open(file_path, 'wb') as f:
        f.write(downloaded_file)

    # Mark chat history as available
    CHAT_HISTORY = True
    bot.send_message(message.chat.id, 'Document received! Analyzing chat history... 💕')

    # Convert format using TelegramChatConverter
    converter = TelegramChatConverter(file_path, json_path)

    converter.convert()
    
    # Set Speaker ID and User ID
    SPEAKER_ID = converter.getSPEAKER_ID()
    USER_ID = converter.getUSER_ID()
    bot.send_message(message.chat.id, f"Speaker ID: {SPEAKER_ID}\nUser ID: {USER_ID}")

    # Analyze speaker style and create ChromaDB
    if not SPEAKER_STYLE:
        bot.send_message(message.chat.id, "Analyzing speaker style... Please wait a moment. 💕")
        
        # Load conversation history and create ChromaDB
        for chunk in load_conversation_json_in_chunks(json_path, 100):
            create_chroma_db(chunk, collection)
        SPEAKER_STYLE = analyze_speaker_style(collection, SPEAKER_ID)
        bot.send_message(message.chat.id, f"Speaker style analyzed successfully! 💕")

# === Main message handler ===
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    """
    Handle user messages and generate Gemini answers.
    """
    global CHAT_HISTORY, SPEAKER_ID, USER_ID, SPEAKER_STYLE  # Declare global variables

    # Check if chat history is available
    if not CHAT_HISTORY:
        bot.send_message(message.chat.id, "Please send the chat history file first.")
        return

    user_query = message.text
    if user_query.lower() == "exit":
        bot.send_message(message.chat.id, "Goodbye! 👋")
        return

    # Add to buffer and ChromaDB
    add_to_buffer(USER_ID, user_query, conversation_buffer)
    add_new_conversation(collection, USER_ID, user_query)

    # Retrieve context and relevant info
    context_pairs = get_last_n_messages(conversation_buffer)
    context_texts = [f"{spk}: {msg}" for spk, msg in context_pairs]
    relevant_info = retrieve_for_info(collection, user_query, n_results=5)

    # Generate Gemini answer
    try:
        initial_answer = generate_answer_based_on_info(SPEAKER_ID, USER_ID, user_query, context_texts, relevant_info)
        final_answer = style_post_process(initial_answer, SPEAKER_STYLE)
        bot.send_message(message.chat.id, final_answer)
        
        # Update buffer and ChromaDB
        add_to_buffer(SPEAKER_ID, final_answer, conversation_buffer)
        add_new_conversation(collection, SPEAKER_ID, final_answer)
    except Exception as e:
        bot.send_message(message.chat.id, f"An error occurred: {e}")

# === Start Telegram bot ===
bot.infinity_polling()
