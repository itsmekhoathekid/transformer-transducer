import torch
import torch.nn as nn
import math
from utils import TransformerTransducerLayer, calc_data_len, get_mask_from_lens

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
        super().__init__()
        self.pre_net = nn.Conv1d(
            in_channels=in_features,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.layers = nn.ModuleList(
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
        # x is of shape (batch, seq_len, in_features)
        lengths = mask.sum(dim=-1)
        out = x.transpose(-1, -2)
        out = self.pre_net(out)
        out = out.transpose(-1, -2)

        lengths = calc_data_len(
            result_len=out.shape[1],
            pad_len=x.shape[1] - lengths,
            data_len=lengths,
            kernel_size=self.pre_net.kernel_size[0],
            stride=self.pre_net.stride[0],
        )
        mask = get_mask_from_lens(lengths=lengths, max_len=out.shape[1])
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
