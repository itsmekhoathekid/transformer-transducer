import torch
import torch.nn as nn
import math


class TransformerTransducerEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        left_size: int,
        right_size: int,
        p_dropout: float,
        stride: int = 1,
        kernel_size: int = 1,
    ):
        pass