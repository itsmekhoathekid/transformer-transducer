import unittest
from model import TransformerTransducer
from torch import Tensor
import torch
from torch import nn
from torch.nn import functional as F

class TestTransformerTransducer(unittest.TestCase):
    def setUp(self):
        # Model hyperparameters
        self.in_features = 80      # Fbank features
        self.n_classes = 100       # vocabulary size
        self.n_layers = 2
        self.n_dec_layers = 2
        self.d_model = 64
        self.ff_size = 128
        self.h = 4
        self.joint_size = 128
        self.enc_left_size = 1
        self.enc_right_size = 1
        self.dec_left_size = 1
        self.dec_right_size = 1
        self.p_dropout = 0.1

        self.model = TransformerTransducer(
            in_features=self.in_features,
            n_classes=self.n_classes,
            n_layers=self.n_layers,
            n_dec_layers=self.n_dec_layers,
            d_model=self.d_model,
            ff_size=self.ff_size,
            h=self.h,
            joint_size=self.joint_size,
            enc_left_size=self.enc_left_size,
            enc_right_size=self.enc_right_size,
            dec_left_size=self.dec_left_size,
            dec_right_size=self.dec_right_size,
            p_dropout=self.p_dropout
        )

        # Fake data
        self.batch_size = 2
        self.time_steps = 50
        self.token_len = 20

        self.speech = torch.randn(self.batch_size, self.time_steps, self.in_features)
        self.speech_mask = torch.ones(self.batch_size, self.time_steps).bool()

        self.text = torch.randint(1, self.n_classes, (self.batch_size, self.token_len))
        self.text_mask = torch.ones(self.batch_size, self.token_len).bool()

    def test_forward_shape(self):
        output, speech_len, text_len = self.model(
            speech=self.speech,
            speech_mask=self.speech_mask,
            text=self.text,
            text_mask=self.text_mask
        )[:3]  # Bỏ qua các giá trị khác nếu model trả về nhiều hơn

        self.assertEqual(output.shape, (self.batch_size, self.time_steps, self.token_len + 1, self.n_classes))
        self.assertEqual(speech_len.shape, (self.batch_size,))
        self.assertEqual(text_len.shape, (self.batch_size,))

if __name__ == '__main__':
    unittest.main()