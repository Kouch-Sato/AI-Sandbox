import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self._input_ids = []
        self._target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > context_length, "テキストの長さが短し！！"

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i: i + context_length]
            target_chunk = token_ids[i + 1: i + context_length + 1]
            self._input_ids.append(torch.tensor(input_chunk))
            self._target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, index):
        return self._input_ids[index], self._target_ids[index]

def create_dataloader(text, context_length=10, stride=2, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(
        text, 
        tokenizer, 
        context_length, 
        stride
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )

    return dataloader

dataloader = create_dataloader(raw_text, context_length=10, stride=2, batch_size=1, shuffle=False)

data_iterator = iter(dataloader)
input, target = next(data_iterator)
tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.decode(input[0].tolist()))
print(tokenizer.decode(target[0].tolist()))

# ちゃんと一個ずれてる！
# I HAD always thought Jack Gisburn rather
#  HAD always thought Jack Gisburn rather a