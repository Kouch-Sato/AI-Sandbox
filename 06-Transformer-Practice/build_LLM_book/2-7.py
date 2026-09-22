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

context_length = 4
dataloader = create_dataloader(raw_text, batch_size=8, context_length=context_length, stride=context_length, shuffle=False)

data_iterator = iter(dataloader)
input, target = next(data_iterator)

print(input.shape)
# torch.Size([8, 4])

vocab_size = 50257
output_dim = 256

torch.manual_seed(123)

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # Embedding(50257, 256)
token_embeddings = token_embedding_layer(input) # torch.Size([8, 4, 256])

position_embedding_layer = torch.nn.Embedding(context_length, output_dim) # Embedding(4, 256)
position_ids = torch.arange(context_length)
position_embeddings = position_embedding_layer(position_ids) #torch.Size([4, 256])

# broadcastingによって、positonが[8, 4, 256]として扱われる
input_embeddings = token_embeddings + position_embeddings
print(input_embeddings.shape)
