import orjson
from collections import deque

def load_conversation_json_in_chunks(json_path: str, chunk_size=100, buffer=None):
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())

    if buffer is not None and isinstance(buffer, deque):
        last_five = data[-5:] if len(data) >= 5 else data
        for item in last_five:
            speaker = item.get("speaker", "unknown")
            message = item.get("message", "")
            buffer.append((speaker, message))
        # print("\n🔍 [DEBUG] conversation_buffer initialized with last 5 messages:", list(buffer))

    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

