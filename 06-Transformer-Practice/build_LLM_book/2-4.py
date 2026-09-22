# endoftextやunknownを辞書に追加する
import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
## 空白を削除
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
all_words.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_words)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self._str_to_int = vocab
        self._int_to_str = {integer: string for string, integer in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        ## 空白を削除
        preprocessed = [token.strip() for token in preprocessed if token.strip()]
        preprocessed = [
            token if token in self._str_to_int
            else "<|unk|>" 
            for token in preprocessed
        ]

        return [self._str_to_int[token] for token in preprocessed]

    def decode(self, ids):
        text = " ".join([self._int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r' \1 ', text)
        return text

tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
ids = tokenizer.encode(text)

print(tokenizer.decode(ids))
