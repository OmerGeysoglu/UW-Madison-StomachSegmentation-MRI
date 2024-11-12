import torch
import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/')))
from blocks import DoubleConv, DownSampling, UpSampling, CropAndConcat


class TestBlocks(unittest.TestCase):
    def test_double_conv_forward(self):
        # Define input tensor with shape (batch_size, in_channels, height, width)
        x = torch.randn(1, 3, 64, 64)
        
        # Initialize the DoubleConv module
        model = DoubleConv(in_channels=3, out_channels=64)
        
        # Perform a forward pass
        output = model.forward(x)
        
        # Check the output shape
        self.assertEqual(output.shape, (1, 64, 64, 64), f"Expected shape (1, 64, 64, 64), but got {output.shape}")

    def test_down_sampling_forward(self):
        # Define input tensor with shape (batch_size, in_channels, height, width)
        x = torch.randn(1, 64, 64, 64)
        
        # Initialize the DownSampling module
        model = DownSampling()
        
        # Perform a forward pass
        output = model.forward(x)
        
        # Check the output shape
        self.assertEqual(output.shape, (1, 64, 32, 32), f"Expected shape (1, 64, 32, 32), but got {output.shape}")

    def test_up_sampling_forward(self):
        # Define input tensor with shape (batch_size, in_channels, height, width)
        x = torch.randn(1, 64, 32, 32)
        
        # Initialize the UpSampling module
        model = UpSampling(in_channels=64, out_channels=32)
        
        # Perform a forward pass
        output = model.forward(x)
        
        # Check the output shape
        self.assertEqual(output.shape, (1, 32, 64, 64), f"Expected shape (1, 32, 64, 64), but got {output.shape}")

    def test_crop_and_concat_forward(self):
        # Define input tensors with different shapes
        x1 = torch.randn(1, 64, 64, 64)
        x2 = torch.randn(1, 32, 32, 32)
        
        # Initialize the CropAndConcat module
        model = CropAndConcat()
        
        # Perform a forward pass
        output = model.forward(x1, x2)
        
        # Check the output shape
        self.assertEqual(output.shape, (1, 96, 64, 64), f"Expected shape (1, 96, 64, 64), but got {output.shape}")

if __name__ == "__main__":
    unittest.main()
