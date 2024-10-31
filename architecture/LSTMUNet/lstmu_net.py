from torch import nn
import torch

from architecture.LSTMUNet.deconv import DeConvNet
from  architecture.LSTMUNet.custom_unet import CustomUNet


class LSTMUNet(nn.Module):
    def __init__(self, sequence_length):
        super(LSTMUNet, self).__init__()
        center_nn = LSTMUNetHelper()
        self.unet = CustomUNet(sequence_length, sequence_length, center_nn)

    def forward(self, x):
        x = self.unet(x)
        return x


class LSTMUNetHelper(nn.Module):
    def __init__(self):
        super(LSTMUNetHelper, self).__init__()

        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(64, 64)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = self.lstm(x)
        return x.reshape(-1, 1024, 8, 8)


class WindSpeedLSTMDeConv(nn.Module):
    def __init__(self, seq_length, de_conv_dims):
        super(WindSpeedLSTMDeConv, self).__init__()
        self.seq_length = seq_length
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(500, 500, dtype=torch.float32)
        self.de_conv = DeConvNet(1, de_conv_dims)

    def forward(self, x):
        x = self.flatten(x)
        x, _ = self.lstm(x)
        return self.de_conv(x.reshape(-1, self.seq_length, 10, 50))