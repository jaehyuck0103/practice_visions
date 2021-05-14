import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, dim_feedforward, n_transformer_layers, dropout
    ):
        super().__init__()

        self.embed_layer = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_transformer_layers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        # Init weights
        self.embed_layer.weight.data.uniform_(-0.1, 0.1)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src, src_mask):  # src: (L, B), src_mask: (L, L)
        src = self.embed_layer(src) * math.sqrt(self.d_model)  # (L, B, Embedding)
        src = self.pos_encoder(src)  # (L, B, Embedding)
        output = self.transformer_encoder(src, src_mask)  # (L, B, Embedding)
        output = self.decoder(output)  # (L, B, vocab_size)
        return output
