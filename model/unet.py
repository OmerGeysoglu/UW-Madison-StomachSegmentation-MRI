import os
import sys

import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/')))
from blocks import DoubleConv, DownSampling, UpSampling, CropAndConcat

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        '''
        Initializes the U-Net model, defining the encoder, decoder, and other layers.

        Args:
        - in_channels (int): Number of input channels (1 for scan images).
        - out_channels (int): Number of output channels (1 for binary segmentation masks).
        
        Function:
        - CBR (in_channels, out_channels): Helper function to create a block of Convolution-BatchNorm-ReLU layers. 
        (This function is optional to use)
        '''
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128, dropout_prob=0.1),
            DoubleConv(128, 256, dropout_prob=0.1),
            DoubleConv(256, 512, dropout_prob=0.3) 
        ])
        self.down = DownSampling()
        self.middle = DoubleConv(512, 1024)
        self.crop_and_concat = CropAndConcat()

        self.up_sampling = nn.ModuleList([
            UpSampling(1024, 512),
            UpSampling(512, 256),
            UpSampling(256, 128),
            UpSampling(128, 64)
        ])

        self.decoder = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64)
        ])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    
    def forward(self, x):
        '''
        Defines the forward pass of the U-Net, performing encoding, bottleneck, and decoding operations.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        '''

        # Encoding
        encoder_outputs = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encoder_outputs.append(x)
            x = self.down(x)
        
        # Middle
        x = self.middle(x)

        # Decoding
        for i in range(len(self.up_sampling)):
            x = self.up_sampling[i](x)
            x = self.crop_and_concat(x, encoder_outputs[-(i+1)])
            x = self.decoder[i](x)
        
        # Final Convolution
        output = self.final_conv(x)
        
        return output

