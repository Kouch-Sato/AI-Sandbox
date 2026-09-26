import torch
import torch.nn as nn
import tiktoken
from model import GPTModel, TransformerBlock
from layers import FeedForward
from gpt_download import download_and_load_gpt2
from load_weights import load_weights_into_gpt 

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

model_configs = {
    "gpt2-small": {
        "emb_dim": 768,
        "n_layers": 12, 
        "n_heads": 12, 
        "qkv_bias": True,
    },
    "gpt2-medium": {
        "emb_dim": 1024,
        "n_layers": 24, 
        "n_heads": 16, 
        "qkv_bias": True,
    }
}

NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs["gpt2-medium"])

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "I love Jobs because"
txt2 = "Today, we are"
txt3 = "The meaning of life"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch.append(torch.tensor(tokenizer.encode(txt3)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(NEW_CONFIG)
model.eval()

settings, params = download_and_load_gpt2(model_size="355M", models_dir="gpt2")
load_weights_into_gpt(model, params)

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

out = generate_text_simple(
    model=model,
    idx=batch,
    max_new_tokens=10,
    context_size=NEW_CONFIG["context_length"]
)

for indices in out:
    decoded_text = tokenizer.decode(indices.tolist())
    print(decoded_text)
