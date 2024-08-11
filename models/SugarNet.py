import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

alpha = 0.6

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pre_length = self.args.pred_len
        self.seq_length = self.args.seq_len
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.n_features = args.enc_in
        self.embed_size = 128
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        # Convolution layer for dimension extended input
        self.ffct = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=24, padding='same'),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=3, stride=3))
        self.ffcf = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=24, padding='same'),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=3, stride=3))
        # Convolution layer for raw input without dimension extension
        self.fct = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2))
        self.fcf = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=8, kernel_size=12, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=6, padding='same'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.MaxPool1d(kernel_size=2))
        self.lstmt = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstmf = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.time_only = nn.Sequential(
            nn.Linear(32*64, 64),
        )
        self.fuse = nn.Sequential(
           nn.Linear(2*self.pre_length, self.pre_length)
        )
        self.fuse_no_extension = nn.Sequential(
           nn.Linear(128, 64),
           nn.Linear(64, 32),
           nn.Linear(32, self.pre_length)
        )
        self.time_or_frequency_only = nn.Sequential(
           nn.Linear(64, 128),
           nn.Linear(128, 32),
           nn.Linear(32, self.pre_length)
        )
        self.leakt = nn.Sequential(
            nn.Linear(32*64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.args.pred_len)
        )
        self.leakf = nn.Sequential(
            nn.Linear(32*64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.args.pred_len)
        )
        self.leakDiaTrendf = nn.Sequential(
            nn.Linear(6144, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.args.pred_len)
        )
        self.leakDiaTrendt = nn.Sequential(
            nn.Linear(6144, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.args.pred_len)
        )
        self.leak10f = nn.Sequential(
            nn.Linear(3072, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.args.pred_len)
        )
        self.leak10t = nn.Sequential(
            nn.Linear(3072, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.args.pred_len)
        )

    #def leakModule(self, inputsize):
      #if self.leak is None:
    #  self.leak = nn.Sequential(
    #        nn.Linear(inputsize, 128),
    #        nn.LeakyReLU(),
    #        nn.Linear(128, self.pre_length)
    #  )
    #  return self.leak

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # B*C*I*1 x 1*D = B*C*I*D
        y = self.embeddings
        return x * y

    def forward(self, x, x_dec):
      timefunc = self.forward_time_domain
      freqfunc = self.forward_freq_domain
      fusefunc = self.fuse
      if self.args.dim_extension==False:
        timefunc = self.forward_time_domain_no_extension
        freqfunc = self.forward_freq_domain_no_extension
        fusefunc = self.fuse_no_extension
      
      tf = None
      ff = None
      #time only OR both
      if self.args.mode==2 or self.args.mode==1:
          tf = timefunc(x)
          tf = torch.squeeze(tf, dim=1)
      if self.args.mode==3 or self.args.mode==1:
          ff = freqfunc(x)
      
      if tf!=None and ff!=None:
        #print(f"tf {tf.shape}")
        #print(f"ff {ff.shape}")
        return fusefunc(torch.cat((tf, ff), dim=1))
      if tf!=None:
        return self.time_or_frequency_only(tf)
      if self.args.dim_extension:
        return ff
      return self.time_or_frequency_only(ff)

    def forward_time_domain_no_extension(self, x):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape

        y = self.fcf(x.permute(0, 2, 1))
        y, _ = self.lstmf(y.permute(0, 2, 1))

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

        y = self.fcf(exp.permute(0, 2, 1))
        y, _ = self.lstmf(y.permute(0, 2, 1))
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
        y = self.ffcf(exp)
        #batch, out_channel, L_out
        #print(f"freq after conv {y.shape}")
        y, _ = self.lstmf(y.permute(0, 2, 1))
        #print(f"FREQ before leak {y.reshape(y.shape[0], -1).shape}")

        if self.args.sampling_rate==5:
          y = self.leakDiaTrendf(y.reshape(y.shape[0], -1))
        elif self.args.sampling_rate==15:
          y = self.leakf(y.reshape(y.shape[0], -1))
        else:
          y = self.leak10f(y.reshape(y.shape[0], -1))
       
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
        x = x.reshape(B, C, -1)

        #print(f"after reshape {x.shape}")

       # y = self.fc(x.permute(0, 2, 1))
        y = self.ffct(x)
        #output shape of conv layer [batch, channel, dynamic]
        #dynamic depends on input seq length and conv filters
        
        #input shape of lstm [sequence_length, batch_size, channel]
        y, _ = self.lstmt(y.permute(2, 0, 1))
        #output shape of lstm [sequence_length, batch_size, hidden_size]
   
        # time only
        if self.args.mode==2:
          y = self.time_only(y.reshape(y.shape[0], -1))
        else:
          y = y.permute(1, 0, 2)
          #print(f"TIME before leak {y.shape}")
          #print(f"permute {y.permute(1, 0, 2).shape}")
          #print(f"permute {y.reshape(y.shape[0], -1).shape}")
          if self.args.sampling_rate==5:
            y = self.leakDiaTrendt(y.reshape(y.shape[0], -1))
          elif self.args.sampling_rate==15:
            y = self.leakt(y.reshape(y.shape[0], -1))
          else:
            y = self.leak10t(y.reshape(y.shape[0], -1))

        return y