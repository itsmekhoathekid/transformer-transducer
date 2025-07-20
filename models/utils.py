import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttentionBlock


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

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

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class Self_Attention_Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float,
    ) -> None:
        super().__init__()

        self.attention = MultiHeadAttentionBlock(d_model, h, p_dropout)
        self.feed_forward = FeedForwardBlock(d_model, ff_size, p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.residual_connections = nn.ModuleList(
            ResidualConnection(d_model, p_dropout) for _ in range(2)
        )
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        x = self.residual_connections[0](x, lambda x: self.attention(x, x, x, mask))
        x = self.residual_connections[1](x, lambda x : self.feed_forward(x))

        return x

def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len


def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)

import logging
def logg(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )