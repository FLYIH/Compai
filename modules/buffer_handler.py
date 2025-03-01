def add_to_buffer(speaker, message, conversation_buffer):
    """
    Add the message to in-memory conversation buffer
    """
    conversation_buffer.append((speaker, message))

def get_last_n_messages(conversation_buffer):
    """
    Return the conversation buffer as a list of (speaker, message)
    """
    return list(conversation_buffer)