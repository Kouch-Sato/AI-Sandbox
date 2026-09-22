import torch
import torch.nn as nn
from layers import LayerNorm, FeedForward

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb_layer = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_emb_layer = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb_layer = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(config) for i in range(config["n_layers"])]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embs = self.token_emb_layer(in_idx)
        position_embs = self.position_emb_layer(torch.arange(seq_len, device=in_idx.device))

        x = token_embs + position_embs
        x = self.drop_emb_layer(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x
