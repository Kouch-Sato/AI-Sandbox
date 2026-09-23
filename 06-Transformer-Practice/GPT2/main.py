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
txt1 = "I love Jobs because"
txt2 = "Today, we are"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
logits = model(batch)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond) # [2, 4, 50257]

        logits = logits[:, -1, :] # [2, 50257]
        next_index_probas = torch.softmax(logits, dim=-1) # Probabilities
        next_index = torch.argmax(next_index_probas, dim=-1, keepdim=True)

        idx = torch.cat((idx, next_index), dim=-1)

    return idx
        
model.eval()

out = generate_text_simple(
    model=model,
    idx=batch,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

for indeces in out:
    decoded_text = tokenizer.decode(indeces.tolist())
    print(decoded_text)
