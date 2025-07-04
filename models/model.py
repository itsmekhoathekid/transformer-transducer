import torch
from torch import Tensor, nn
from .encoder import TransformerTransducerEncoder
from .decoder import TransformerTransducerDecoder
from torch.nn import functional as F

class TransformerTransducer(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_classes: int,
        n_layers: int,
        n_dec_layers: int,
        d_model: int,
        ff_size: int,
        h: int,
        joint_size: int,
        enc_left_size: int,
        enc_right_size: int,
        dec_left_size: int,
        dec_right_size: int,
        p_dropout: float,
        stride: int = 1,
        kernel_size: int = 1,
        masking_value: int = -1e15,
    ) -> None:
        super().__init__()
        self.encoder = TransformerTransducerEncoder(
            in_features=in_features,
            n_layers=n_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            left_size=enc_left_size,
            right_size=enc_right_size,
            p_dropout=p_dropout,
            stride=stride,
            kernel_size=kernel_size,
        )
        self.decoder = TransformerTransducerDecoder(
            vocab_size=n_classes,
            n_layers=n_dec_layers,
            d_model=d_model,
            ff_size=ff_size,
            h=h,
            left_size=dec_left_size,
            right_size=dec_right_size,
            p_dropout=p_dropout,
        )
        # self.audio_fc = nn.Linear(in_features=d_model, out_features=joint_size)
        # self.text_fc = nn.Linear(in_features=d_model, out_features=joint_size)
        self.fc1 = nn.Linear(in_features = joint_size, out_features = d_model)
        self.fc2 = nn.Linear(in_features = d_model, out_features = n_classes)
        self.tanh = nn.Tanh()
        # self.join_net = nn.Linear(in_features=joint_size, out_features=n_classes)

    def _join(self, encoder_out: Tensor, deocder_out: Tensor, training=True) -> Tensor:
        if encoder_out.dim() == 3 or deocder_out.dim() == 3:
            seq_lens = encoder_out.size(1)
            target_lens = deocder_out.size(1)

            encoder_out = encoder_out.unsqueeze(2)
            deocder_out = deocder_out.unsqueeze(1)

            encoder_out = encoder_out.repeat(1, 1, target_lens, 1)
            deocder_out = deocder_out.repeat(1, seq_lens, 1, 1)

        output = torch.cat((encoder_out, deocder_out), dim=-1)
        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)
        
        output = F.log_softmax(output, dim=-1)
        return output
    
    def forward(
        self,
        speech: Tensor,
        speech_mask: Tensor,
        text: Tensor,
        text_mask: Tensor,
        *args,
        **kwargs
    ):
        """Passes the input to the model

        Args:

            speech (Tensor): The input speech of shape [B, M, d]

            speech_mask (Tensor): The speech mask of shape [B, M]

            text (Tensor): The text input of shape [B, N]

            text_mask (Tensor): The text mask of shape [B, N]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple of 3 tensors where the first
            is the predictions of shape [B, M, N, C], the last two tensor are
            the speech and text length of shape [B]
        """
        B, U = text.shape
        
        # Nếu muốn pad 0 vào cuối 
        pad_col = torch.full((B, 1),
                             fill_value=0,
                             dtype=text.dtype,
                             device=text.device)
        text_padded = torch.cat([text, pad_col], dim=1)
        pad_mask = torch.zeros((B, 1),
                               dtype=text_mask.dtype,
                               device=text_mask.device)
        mask_padded = torch.cat([text_mask, pad_mask], dim=1)


        speech, speech_len = self.encoder(speech, speech_mask)

        text, text_len = self.decoder(text_padded, mask_padded)
        # speech = self.audio_fc(speech)
        # text = self.text_fc(text)
        result = self._join(encoder_out=speech, deocder_out=text)
        speech_len, text_len = (
            speech_len.to(speech.device),
            text_len.to(speech.device),
        )


        return result, speech_len, text_len

