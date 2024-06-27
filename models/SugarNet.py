import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.constants import HISTORY_STEPS, FUTURE_STEPS

alpha = 0.6

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pre_length = FUTURE_STEPS
        self.seq_length = HISTORY_STEPS
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.n_features = args.enc_in
        self.embed_size = 128
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        # Convolution layer for dimension extended input
        self.ffc = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=24, padding='same'),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=3, stride=3))
        # Convolution layer for raw input without dimension extension
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2))
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.leak = nn.Sequential(
            nn.Linear(32*64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, FUTURE_STEPS)
        )
        self.fuse = nn.Sequential(
           nn.Linear(2*FUTURE_STEPS, FUTURE_STEPS)
        )
        self.fuse_no_extension = nn.Sequential(
           nn.Linear(128, 64),
           nn.Linear(64, 32),
           nn.Linear(32, FUTURE_STEPS)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # B*C*I*1 x 1*D = B*C*I*D
        y = self.embeddings
        return x * y

    def forward(self, x, x_dec):
      if self.args.dim_extension:
        tf = self.forward_time_domain(x)
        tf = torch.squeeze(tf, dim=1)
        ff = self.forward_freq_domain(x)
        return self.fuse(torch.cat((tf, ff), dim=1))
      
      tf = self.forward_time_domain_no_extension(x)
      tf = torch.squeeze(tf, dim=1)
      ff = self.forward_freq_domain_no_extension(x)

      return self.fuse_no_extension(torch.cat((tf, ff), dim=1))

    def forward_time_domain_no_extension(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape

        y = self.fc(x.permute(0, 2, 1))
        y, _ = self.lstm(y.permute(0, 2, 1))

        return y

    def forward_freq_domain_no_extension(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape
        inputD = x.device
       
        #print(f"x {x.shape}")
        fft = torch.fft.rfft(x, dim=1, norm='ortho')
  
        exp = torch.zeros((B, fft.shape[1]*2, C), device=inputD)
        ci = 0
        for x in range(fft.shape[1]):
            exp[:, ci, :] = fft[:, x, :].real
            exp[:, ci+1, :] = fft[:, x, :].imag
            ci = ci + 2

        y = self.fc(exp.permute(0, 2, 1))
        y, _ = self.lstm(y.permute(0, 2, 1))
        #print(f"y {y.shape}")
        y = torch.squeeze(y, dim=1)
        split = y.shape[1]//2+1
        real = y[:, :split]
        #print(f"real {real.shape}")
        imag = y[:, split:]
        #print(f"imag {imag.shape}")
        imag = torch.cat((torch.zeros((B, 1), device=inputD), imag, torch.zeros((B, 1), device=inputD)), dim=1)
        y = torch.stack((real, imag), dim=2)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        #print(f"rfft {y.shape}")
        y = torch.fft.irfft(y, dim=1, norm="ortho")
        return y

    def forward_freq_domain(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape
        inputD = x.device
       
        x = self.tokenEmb(x)
        # [B, C, I, D]

        fft = torch.fft.rfft(x, dim=2, norm='ortho')

        # fft.shape[2] = I//2+1
        exp = torch.zeros((fft.shape[0], fft.shape[1], fft.shape[2]*2, fft.shape[3]), device=inputD)
        
        ci = 0
        for y in range(fft.shape[2]):
          exp[:, :, ci, :] = fft[:, :, y, :].real
          exp[:, :, ci+1, :] = fft[:, :, y, :].imag
          ci = ci + 2

        # if I is even, imaginary parts at Freq = 0  and Freq = I/2 is 0
        # if I is odd, only Freq = 0 has 0 imaginary part
        exp = torch.cat((exp[:, :, :1, :], exp[:, :, 2:, :]), dim=2)

        if I%2 == 0:
          split = I//2+1
          exp = torch.cat((exp[:, :, :split, :], exp[:, :, split+1:, :]), dim=2)

        exp = exp.reshape(B, C, -1)
        # [B, C, I*D]
        y = self.ffc(exp)
        #batch, out_channel, L_out
        y, _ = self.lstm(y.permute(0, 2, 1))
        y = self.leak(y.reshape(y.shape[0], -1))
        y = torch.squeeze(y, dim=1)
        split = y.shape[1]//2+1
        real = y[:, :split]
        #print(f"real {real.shape}")
        imag = y[:, split:]
       # print(f"imag {imag.shape}")
        imag = torch.cat((torch.zeros((B, 1), device=inputD), imag, torch.zeros((B, 1), device=inputD)), dim=1)
        y = torch.stack((real, imag), dim=2)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        y = torch.fft.irfft(y, dim=1, norm="ortho")
        return y

    def forward_time_domain(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape

        x = self.tokenEmb(x)
        #print(f"x {x.shape}")

        x = x.reshape(B, C, -1)

        #print(f"after reshape {x.shape}")

       # y = self.fc(x.permute(0, 2, 1))
        y = self.ffc(x)
        #print(f"after conv {y.shape}")
        y, _ = self.lstm(y.permute(2, 0, 1))
        #print(f"time lstm {y.shape}")
       # y, _ = self.lstm(y.permute(0, 2, 1))
        y = self.leak(y.reshape(y.shape[0], -1))

        return y