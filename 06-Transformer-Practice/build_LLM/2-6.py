import tiktoken
from torch.utils.data import Dataset, DataLoader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the kouch."
text = " <|endoftext|> ".join((text1, text2))

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

context_size = 100
x = enc_text[:context_size]
y = enc_text[1:context_size + 1]

print(x)
print(y)

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self._input_ids = []
        self._target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > context_length, "テキストの長さが短し！！"

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i: i + context_length]
            target_chunk = token_ids[i + 1: i + context_length + 1]
            self._input_ids.append(input_chunk)
            self._target_ids.append(target_chunk)

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, index):
        return self._input_ids[index], self._target_ids[index]
