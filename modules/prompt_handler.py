import google.generativeai as genai
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