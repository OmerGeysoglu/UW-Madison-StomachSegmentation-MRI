import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_prob: float = 0.0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        return x

class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(x)
        return x

class UpSampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpSampling, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.upconv(x)
        return x

class CropAndConcat(nn.Module):
    def __init__(self):
        super(CropAndConcat, self).__init__()

    def forward(self, x1, x2):
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        padding = (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        x2 = F.pad(x2, padding)
        x = torch.cat([x1, x2], dim=1)
        return x