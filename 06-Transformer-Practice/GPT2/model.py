import torch
import torch.nn as nn
from layers import LayerNorm, FeedForward
from attention import MultiHeadAttention

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb_layer = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_emb_layer = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb_layer = nn.Dropout(config["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for i in range(config["n_layers"])]
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

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"], 
            context_length=config["context_length"],
            dropout=config["drop_rate"],
            num_heads=config["n_heads"], 
            bias=config["qkv_bias"],
        )

        self.feedforward = FeedForward(config)
        self.layer_norm_1 = LayerNorm(config["emb_dim"])
        self.layer_norm_2 = LayerNorm(config["emb_dim"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
