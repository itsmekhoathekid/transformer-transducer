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
        print(f"Input shape: {x.shape}")
        print(f"Mask shape: {mask.shape if mask is not None else 'None'}")
        x = self.attention(x, x, x, mask)

        x = x + residual
        x = self.feed_forward(x)
        x = self.dropout(x)
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

# def test_transformer_transducer_layer():
#     # Tham số mô hình
#     d_model = 256
#     ff_size = 1024
#     h = 4
#     left_size = 16
#     right_size = 16
#     p_dropout = 0.1

#     # Input giả
#     batch_size = 2
#     seq_len = 50
#     x = torch.randn(batch_size, seq_len, d_model)

#     input_lengths = torch.tensor([50, 40])
#     mask = get_mask_from_lens(input_lengths, max_len=seq_len)

#     # Tạo lớp
#     layer = TransformerTransducerLayer(
#         d_model=d_model,
#         ff_size=ff_size,
#         h=h,
#         left_size=left_size,
#         right_size=right_size,
#         p_dropout=p_dropout,
#     )

#     # Chạy forward
#     out = layer(x, mask)

#     # In kết quả
#     print("✅ Input shape:", x.shape)
#     print("✅ Output shape:", out.shape)

#     # Kiểm tra cơ bản
#     assert out.shape == x.shape, "Output shape mismatch"
#     print("✅ Test passed!")

# if __name__ == "__main__":
#     test_transformer_transducer_layer()