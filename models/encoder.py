import torch
import torch.nn as nn
import math
from .utils import Self_Attention_Block, calc_data_len, get_mask_from_lens, PositionalEncoding, ConvolutionFrontEnd


class TransformerTransducerEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        p_dropout: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=d_model)
        # self.pe = PositionalEncoding(d_model)
        

        self.layers = nn.ModuleList(
            [
                Self_Attention_Block(
                    d_model=d_model,
                    ff_size=ff_size,
                    h=h,
                    p_dropout=p_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.frontend = ConvolutionFrontEnd(
            in_channels=1,
            num_blocks=3,
            num_layers_per_block=2,
            out_channels=[8, 16, 32],
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2],
            residuals=[True, True, True],
            activation=nn.ReLU,        
            norm=nn.BatchNorm2d,            
            dropout=0.1,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        x = x.unsqueeze(1)  # [batch, channels, time, features]
        # print("x shape before frontend:", x.shape)  # [batch, 1, time, features]
        x, mask = self.frontend(x, mask)  # [batch, channels, time, features]
        # print("x shape after frontend:", x.shape)
        x = x.transpose(1, 2).contiguous()   # batch, time, channels, features
        x = x.reshape(x.shape[0], x.shape[1], -1) # [batch, time, C * features]


        lengths = mask.sum(dim=-1)
        x = self.linear(x)  # (batch, seq_len, d_model)
        # out = self.pe(x)

        for layer in self.layers:
            out = layer(out, mask)
        return out, lengths

# def test_encoder():
#     # --- Tham số mô hình ---
#     in_features = 80
#     d_model = 256
#     ff_size = 1024
#     h = 4
#     n_layers = 6
#     left_size = 16
#     right_size = 16
#     p_dropout = 0.1
#     stride = 2
#     kernel_size = 3

#     # --- Dummy input ---
#     batch_size = 2
#     max_seq_len = 100
#     input_lengths = torch.tensor([100, 80])

#     # Tạo dummy input (batch, time, in_features)
#     x = torch.randn(batch_size, max_seq_len, in_features)
#     mask = get_mask_from_lens(input_lengths, max_len=max_seq_len)

#     # --- Tạo encoder và chạy thử ---
#     encoder = TransformerTransducerEncoder(
#         in_features=in_features,
#         d_model=d_model,
#         ff_size=ff_size,
#         h=h,
#         n_layers=n_layers,
#         left_size=left_size,
#         right_size=right_size,
#         p_dropout=p_dropout,
#         stride=stride,
#         kernel_size=kernel_size,
#     )

#     # --- Forward ---
#     out, out_lens = encoder(x, mask)

#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {out.shape}")
#     print(f"Output lengths: {out_lens}")

#     # --- Check ---
#     assert out.shape[0] == batch_size
#     assert out.shape[2] == d_model
#     assert len(out_lens) == batch_size
#     assert all(out_lens[i] <= (max_seq_len // stride + 1) for i in range(batch_size))  # kiểm tra đã co lại đúng

#     print("✅ Test passed!")

# # Gọi thử hàm test
# if __name__ == "__main__":
#     test_encoder()
