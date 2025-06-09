from layers import TransformerTransducerLayer
import unittest

class TestTransformerTransducerLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 256
        self.ff_size = 1024
        self.n_heads = 4
        self.left_size = 0
        self.right_size = 0
        self.dropout = 0.1

        self.layer = TransformerTransducerLayer(
            d_model=self.d_model,
            ff_size=self.ff_size,
            h=self.n_heads,
            left_size=self.left_size,
            right_size=self.right_size,
            p_dropout=self.dropout,
        )

    def test_output_shape(self):
        x = torch.randn(2, 100, self.d_model)  # (batch, seq_len, d_model)
        mask = torch.ones(2, 100).bool()       # mask shape: (batch, seq_len)
        y = self.layer(x, mask=None)           # mask optional
        self.assertEqual(y.shape, x.shape)

    def test_no_nan(self):
        x = torch.randn(2, 100, self.d_model)
        y = self.layer(x)
        self.assertFalse(torch.isnan(y).any())

    def test_requires_grad(self):
        x = torch.randn(2, 100, self.d_model, requires_grad=True)
        y = self.layer(x)
        y.mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)

if __name__ == '__main__':
    unittest.main()