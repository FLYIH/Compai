# Gemini Chatbot with RAG and ChromaDB
This project implements a Retrieval-Augmented Generation (RAG) Chatbot using Gemini AI, ChromaDB, and Google Palm Embeddings, integrated seamlessly with Telegram Bot. It provides a natural and engaging conversational experience with an affectionate and emotionally resonant communication style, making it feel like a caring conversational partner.

---

> [Presentation Slide](https://www.canva.com/design/DAGfoSN9y6A/i_AvmPdiUjaudQNl5wpehA/edit?utm_content=DAGfoSN9y6A&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Features
#### 1. Embeddings & LLM:

- Utilizes Google Palm Embeddings for efficient vector representation of text.
- Leverages Gemini Pro model (gemini-1.5-flash) for natural language generation.

#### 2. RAG (Retrieval-Augmented Generation):

- Retrieves relevant information from conversation history using ChromaDB.
- Combines retrieved context with LLM-generated responses for accurate and context-aware interactions.

#### 3. Dynamic Memory Buffer:

- Maintains a buffer of the last three messages for seamless conversational flow.
- Stores new conversations dynamically in ChromaDB.

#### 4. Speaker Style Analysis:

- Analyzes writing style, tone, emoji usage, and frequent words.
- Adapts the generated answers to match the speaker's style for a more personalized conversation.

#### 5. Emotionally Intelligent Responses:

- Generates warm and affectionate answers, making the chatbot feel like a caring conversational partner.
- Mimics conversational patterns and emotional tones based on the speaker's historical messages.

#### 6. Telegram Integration:

Seamlessly integrates with Telegram Bot for user-friendly input and output.

Allows easy configuration of speaker and user roles using /setspeaker and /setuser commands.

Real-time messaging with continuous context-awareness.

## How It Works
#### 1. Load Conversation History:

- Loads JSON conversations in chunks using orjson for fast processing.
#### 2. ChromaDB Initialization:

- Initializes a ChromaDB collection and dynamically adds conversation data with vector embeddings.
#### 3. Speaker Style Analysis:

- Analyzes the speaker's writing style, tone, and emoji usage to personalize responses.
#### 4. Generate Response:

- Uses RAG technique to retrieve relevant context from ChromaDB.
- Generates an initial answer using Gemini Pro model.
- Post-processes the answer to match the speaker's style for an emotionally resonant response.
---
## Installation
#### 1. Clone the repository:
```
git clone https://github.com/FLYIH/Companion-bot.git
cd Compai
```
#### 2. Create a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#### 3. Install required dependencies:
```
pip install -r requirements.txt
```
## API Key Setup
This project uses Google Gemini API and Telegram Bot Token for seamless integration.

- Gemini API Key: Obtain from Google Cloud.

- Telegram Bot Token: Create a bot using BotFather on Telegram.

- create a ``.env`` file in the root directory:
    ```
    GEMINI_API_KEY=your_gemini_api_key_here
    TG_BOT_TOKEN=your_telegram_bot_token_here
    ```

## Running the Chatbot
#### 1. Activate the virtual environment:
```
source venv/bin/activate
```
#### 2. Run the chatbot:
```
python main.py
```
#### 3. Interact with the Chatbot on Telegram:

- Set the Speaker and User IDs:
    ```
    /setspeaker <speaker_id>
    /setuser <user_id>
    ```
- Start chatting:
    - Type your message to initiate a conversation.

    - The chatbot will respond with context-aware and emotionally resonant replies.

- End the conversation: Type exit.

