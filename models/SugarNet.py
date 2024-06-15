import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.constants import HISTORY_STEPS, FUTURE_STEPS

alpha = 0.6

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.pre_length = FUTURE_STEPS
        self.seq_length = HISTORY_STEPS
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.n_features = args.enc_in

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2))
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.dense = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 32),
            nn.Linear(32, FUTURE_STEPS)
        )

        self.fuse = nn.Sequential(
           nn.Linear(128, 64),
           nn.Linear(64, 32),
           nn.Linear(32, FUTURE_STEPS)
        )

    def forward(self, x, x_dec):
      tf = self.forward_time_domain(x)
      tf = torch.squeeze(tf, dim=1)
      ff = self.forward_freq_domain(x)
      return self.fuse(torch.cat((tf, ff), dim=1))

    def forward_freq_domain(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape
        inputD = x.device
       
        #print(f"x {x.shape}")
        fft = torch.fft.rfft(x, dim=1, norm='ortho')
  
        exp = torch.zeros((B, fft.shape[1]*2, C), device=inputD)
        for b in range(B):
          real = fft[b, :, :].real
          imag = fft[b, :, :].imag
          ci = 0
          for x in range(fft.shape[1]):
            exp[b, ci, :] = fft[b, x, :].real
            exp[b, ci+1, :] = fft[b, x, :].imag
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

        y = self.dense(y.reshape(B, -1))
        
        split = y.shape[1]//2+1
        real = y[:, :split]
        imag = y[:, split:]
        imag = torch.cat((torch.zeros((B, 1), device=inputD), y[:, split:], torch.zeros((B, 1), device=inputD)), dim=1)
        y = torch.stack((real, imag), dim=2)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        forecast = torch.fft.irfft(y, n=FUTURE_STEPS, dim=1, norm="ortho")
       
        return forecast

    def forward_time_domain(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape

        y = self.fc(x.permute(0, 2, 1))
        y, _ = self.lstm(y.permute(0, 2, 1))

        return y
       
        y = self.dense(y.reshape(B, -1))

        return y