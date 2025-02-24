from bs4 import BeautifulSoup
import json
from datetime import datetime

# 載入 HTML 檔案
# change file name
file_path = 'message_1.html'
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# 使用 BeautifulSoup 解析 HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 擷取對話內容
messages = []
conversations = soup.find_all(class_='_a6-p')

for conversation in conversations:
    # 擷取講話的人
    speaker_tag = conversation.find_previous(class_='_a6-i')
    speaker = speaker_tag.get_text().strip() if speaker_tag else 'Unknown'
    
    # 擷取訊息內容
    message_content = conversation.find('div').get_text().strip()
    
    # 擷取時間
    timestamp_tag = conversation.find_next(class_='_a6-o')
    timestamp_str = timestamp_tag.get_text().strip() if timestamp_tag else 'Unknown'
    
    # 格式化時間
    try:
        timestamp = datetime.strptime(timestamp_str, '%b %d, %Y %I:%M %p').isoformat()
    except:
        timestamp = 'Unknown'
    
    # 加入 messages 清單
    messages.append({
        "speaker": speaker,
        "message": message_content,
        "timestamp": timestamp
    })

# 轉換成 JSON 格式
json_output = json.dumps(messages, indent=2, ensure_ascii=False)

# 存成 JSON 檔案
output_file = 'conversation3.json'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_output)

print(f"已成功轉換成 JSON 檔案：{output_file}")
