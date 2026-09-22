import torch
import torch.nn as nn
import tiktoken
from model import GPTModel, TransformerBlock
from layers import FeedForward

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort makes you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
logits = model(batch)

sum = 0
for p in model.parameters():
    print(p.shape, p.numel())
    sum += p.numel()

print(sum)