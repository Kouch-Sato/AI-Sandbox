import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("文字数:", len(raw_text))
raw_text = raw_text

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
## 空白を削除
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print("語彙数:", vocab_size)