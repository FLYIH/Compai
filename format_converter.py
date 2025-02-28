import json
import os

class TelegramChatConverter:
    global SPEAKER_ID, USER_ID
    def __init__(self, input_file, output_file):
        """
        Initialize with input and output file paths.
        """
        self.input_file = input_file
        self.output_file = output_file

    def getSPEAKER_ID(self):
        global SPEAKER_ID
        return SPEAKER_ID
    
    def getUSER_ID(self):
        global USER_ID
        return USER_ID

    def load_json_file(self):
        """
        Load the JSON file from the specified input path.
        Returns the data as a Python dictionary.
        """
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None

    def convert_format(self, data):
        """
        Convert the Telegram JSON format to simplified conversation format.
        Returns a list of simplified conversation dictionaries.
        """
        global SPEAKER_ID, USER_ID
        simplified_conversations = []

        # Check if 'messages' key exists in the data
        if "name" in data:
            SPEAKER_ID = data["name"]

        if "messages" in data:
            for msg in data["messages"]:
                # Only process type 'message' and text as a string
                if msg.get("type") == "message" and isinstance(msg.get("text"), str):
                    simplified_conversations.append({
                        "speaker": msg.get("from", "Unknown"),
                        "message": msg.get("text", ""),
                        "timestamp": msg.get("date", "")
                    })
                    if msg.get("from", "Unknown") != SPEAKER_ID and SPEAKER_ID != "Unknown":
                        USER_ID = msg.get("from", "Unknown")
 
        return simplified_conversations

    def save_to_json(self, data):
        """
        Save the converted data as a JSON file in the specified output path.
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"✅ Successfully converted and saved as {self.output_file}")
        except Exception as e:
            print(f"Error saving JSON file: {e}")

    def convert(self):
        """
        Main function to convert the Telegram JSON file into the simplified format.
        - Loads the JSON file
        - Converts the format
        - Saves the converted data to the output file
        """
        data = self.load_json_file()
        if not data:
            print("❌ Failed to load JSON file.")
            return
        
        converted_data = self.convert_format(data)
        
        self.save_to_json(converted_data)

