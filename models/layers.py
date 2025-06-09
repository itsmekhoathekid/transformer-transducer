import torch
import torch.nn as nn
import math
from attention import MultiHeadAttentionBlock

class FeedForwardBlock(nn.Module):
    """
    A feed-forward block with two linear layers and a ReLU activation in between.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        residual = x
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x + residual  # Add the residual connection

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
    def get_pe(self, seq_len: int) -> torch.Tensor:
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, self.d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        # x is of shape (batch, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.get_pe(seq_len).to(x.device)
        x = x + pe
        return x

class TransformerTransducerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.left_size = left_size
        self.right_size = right_size

        self.attention = MultiHeadAttentionBlock(d_model, h, p_dropout)
        self.feed_forward = FeedForwardBlock(d_model, ff_size, p_dropout)
        self.lnom = nn.LayerNorm(d_model)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.lnom(x)
        residual = x 

        x = self.pe(x)
        x = self.attention(x, x, x, mask)

        x = x + residual
        x = self.feed_forward(x)
        x = self.dropout(x)
        return x






