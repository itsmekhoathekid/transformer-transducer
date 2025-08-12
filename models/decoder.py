import torch
import torch.nn as nn
import math
from .utils import Self_Attention_Block, PositionalEncoding, PositionalEmbedding



class TransformerTransducerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float,
        joint_size: int
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model)
        self.dec_layers = nn.ModuleList(
            [
                Self_Attention_Block(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    p_dropout=p_dropout,
                    d_ff2=joint_size
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
        out_emb = self.emb(x)
        out = self.pe(out_emb)

        for layer in self.dec_layers:
            out = layer(out, mask)
        return out, lengths


