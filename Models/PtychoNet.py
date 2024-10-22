import torch
import torch.nn as nn
import numpy as np

def C(in_channel, out_channel):
  return nn.Sequential(
    nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
    nn.LeakyReLU(negative_slope=0.2, inplace=True)
  )

def Ct(in_channel, out_channel):
  return nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)

def C_BN(in_channel, out_channel):
  return nn.Sequential(
    nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(out_channel),
    nn.LeakyReLU(negative_slope=0.2, inplace=True)
  )

def Ct_BN(in_channel, out_channel):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
    nn.BatchNorm2d(out_channel),
    nn.ReLU(inplace=True),
  )

class PtychoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_1 = C(1, 64)
        self.encode_2 = C_BN(64, 128)
        self.encode_3 = C_BN(128, 256)
        self.encode_4 = C_BN(256, 512)
        self.decode_1 = Ct_BN(512, 256)
        self.decode_2 = Ct_BN(256, 128)
        self.decode_3 = Ct_BN(128, 64)
        self.decode_4 = Ct(64, 2)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encode_1(x)
        x = self.encode_2(x)
        x = self.encode_3(x)
        x = self.encode_4(x)
        x = self.sigmoid(x)
        x = self.decode_1(x)
        x = self.decode_2(x)
        x = self.decode_3(x)
        x = self.decode_4(x)
        x_updated = torch.empty_like(x) 
        x_updated[:,0] = self.tanh(x[:,0]) * np.pi  # phase: -1 to 1
        x_updated[:,1] = self.sigmoid(x[:,1]) # amp: 0 to 1
        return x_updated