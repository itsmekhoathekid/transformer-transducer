import torch
import torch.nn as nn
import math
from .utils import Self_Attention_Block, calc_data_len, get_mask_from_lens, PositionalEncoding, ConvolutionFrontEnd, PositionalEmbedding


class TransformerTransducerEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float,
        joint_size: int
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=d_model)
        self.pe = PositionalEncoding(d_model)
        # self.pe = PositionalEmbedding(d_model)
        

        self.layers = nn.ModuleList(
            [
                Self_Attention_Block(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    p_dropout=p_dropout,
                    d_ff2= joint_size
                )
                for _ in range(n_layers)
            ]
        )

        # self.frontend = ConvolutionFrontEnd(
        #     in_channels=1,
        #     num_blocks=3,
        #     num_layers_per_block=2,
        #     out_channels=[8, 16, 32],
        #     kernel_sizes=[3, 3, 3],
        #     strides=[1, 2, 2],
        #     residuals=[True, True, True],
        #     activation=nn.ReLU,        
        #     norm=nn.BatchNorm2d,            
        #     dropout=0.1,
        # )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # x = x.unsqueeze(1)  # [batch, channels, time, features]
        # # print("x shape before frontend:", x.shape)  # [batch, 1, time, features]
        # x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        # # print("x shape after frontend:", x.shape)
        # x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        # x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]


        lengths = torch.sum(mask, dim=1)
        out = self.linear(x)  # (batch, seq_len, d_model)
        mask = mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            out = layer(out, mask)
        
        
        return out, lengths