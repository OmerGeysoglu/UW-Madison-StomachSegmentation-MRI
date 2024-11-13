import torch
import unittest
import os
import sys
# Add the model directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/')))
from unet import UNet

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.model = UNet(in_channels=1, out_channels=1)
        self.input_tensor = torch.randn(1, 1, 572, 572)  # Example input tensor

    def test_output_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, (1, 1, 560, 560))  # Expected output shape because of padding

    def test_forward_pass(self):
        try:
            output = self.model(self.input_tensor)
            self.assertIsInstance(output, torch.Tensor)
        except Exception as e:
            self.fail(f"Forward pass failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()