import os
import telebot
from datetime import datetime
from test import generate_answer
import json

TOKEN = os.getenv("TG_BOT_TOKEN")

bot = telebot.TeleBot(TOKEN, parse_mode=None)

@bot.message_handler(commands=['start'])
def get_started(message):
    bot.send_message(message.chat.id, "Hello! I am Compai!\nPlease use \"/setspeaker\" and \"setuser\" to input the id of speaker and user")

@bot.message_handler(commands=['setspeaker'])
def set_speaker(message):
    speaker = message.text.split(" ")[1]
    
    bot.send_message(message.chat.id, f"Speaker ID set to {speaker}")
    print(f"DEBUG: Speaker ID set to {speaker}")
    



@bot.message_handler(commands=['help'])
def send_welcome(message):
    bot.reply_to(message, f"Howdy, how are you doing?")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    """
    # TODO: 
    # 實作邏輯判斷，讓 Bot 可以對特定語句反應
    # 在 reply_to() 第二個參數填入不同回傳訊息
    """
    # if message.text == "Hello":
    #     bot.reply_to(message, f"Hello, how are you doing?")
    # else:
    #     bot.send_message(message.chat.id, message.text)
        # bot.reply_to(message, )

    result = generate_answer(message.text)
    bot.send_message(message.chat.id, result)

# @bot.message_handler(commands=['add'])
# def add_expense(message):
#     try:
#         command_parts = message.text.split(maxsplit=3)
#         if len(command_parts) < 4:
#             bot.reply_to(message, "Usage: /add <item_name> <cost> <item_type>")
#             return
#         _, item_name, cost, item_type = command_parts
#         # cost = int(cost)
#         accounting_book.add_entry(item_name, cost, item_type)
#         bot.reply_to(message, f"Added: {item_name} costing {cost} as {item_type}")
#     except ValueError:
#         bot.reply_to(message, "Invalid cost value. Please enter a number for the cost.")



# Command to show all expenses
# @bot.message_handler(commands=['history'])
# def show_expenses(message):
#     if not accounting_book.accouting_book:
#         bot.reply_to(message, "No expenses recorded.")
#         return
#     response = "Expenses:\n"
#     response += accounting_book.get_history_str()
#     bot.reply_to(message, response)

# # export accounting data to a json file
# @bot.message_handler(commands=['export'])
# def export_data(message):
#     try:
#         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         to
#         with open(json_filename, "a") as file:
#             json.dump(accounting_book.to_dict(), file, indent=4)
#         bot.reply_to(message, f"Data successfully exported to {json_filename}")
#     except Exception as e:
#         bot.reply_to(message, f"An error occurred while exporting data: {e}")


bot.infinity_polling()