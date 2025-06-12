import torch
import torch.nn as nn
import math
from .utils import TransformerTransducerLayer, calc_data_len, get_mask_from_lens



class TransformerTransducerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.dec_layers = nn.ModuleList(
            [
                TransformerTransducerLayer(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    left_size=left_size,
                    right_size=right_size,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ]
        )
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Passes the input `x` through the decoder layers.

        Args:
            x (Tensor): The input tensor of shape [B, M]
            mask (Tensor): The input boolean mask of shape [B, M], where it's True
            if there is no padding.

        Returns:
            Tensor: The encoded text of shape [B, M, d_model].
        """
        lengths = mask.sum(dim=-1)
        out = self.emb(x)

        for layer in self.dec_layers:
            out = layer(out, mask)
        return out, lengths


