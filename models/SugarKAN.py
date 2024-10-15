import torch
#from torch._C import T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

alpha = 0.6

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, 
    scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, 
    base_activation=nn.SiLU, 
    grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0) # (batch, in, coeff)
        splines = self.b_splines(x).permute(1, 0, 2) # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0) # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2) # (batch, in, out)
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin)
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1), grid, grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.pre_length = self.args.pred_len
        self.seq_length = self.args.seq_len
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.n_features = args.enc_in
        if self.args.no_dim_extension==True:
          self.embed_size = 1
        else:
          self.embed_size = args.embed_size
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.grid = 5
        self.k = 3
        
        self.freqcore = nn.Sequential(
              KANLinear(args.feature_size*args.seq_len*self.embed_size, 2*self.args.pred_len),
              KANLinear(2*self.args.pred_len, self.args.pred_len))
        #self.timecore = nn.Sequential(
        #    KANLinear(args.feature_size*args.seq_len*self.embed_size, 2*self.args.pred_len),
        #    KANLinear(2*self.args.pred_len, self.args.pred_len))

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # B*C*I*1 x 1*D = B*C*I*D
        y = self.embeddings
        return x * y


    def forward(self, x, x_dec = None):
      freqfunc = self.forward_freq_domain_no_extension

      # x: [Batch, Input length, Channel]
      B, I, C = x.shape
      
      if self.args.no_dim_extension==False:
        x = self.tokenEmb(x)
        freqfunc = self.forward_freq_domain
     
      tf = None
      ff = None
      #time only OR both
      
      return  freqfunc(x, B, I)

    def forward_freq_domain_no_extension(self, x, B, I):
        # x: [Batch, Input length, Channel]
        B, I, C = x.shape
        inputD = x.device
       
        fft = torch.fft.rfft(x, dim=1, norm='ortho') 
        exp = torch.zeros((B, fft.shape[1]*2, C), device=inputD)
        ci = 0
        for x in range(fft.shape[1]):
            exp[:, ci, :] = fft[:, x, :].real
            exp[:, ci+1, :] = fft[:, x, :].imag
            ci = ci + 2

        exp = torch.cat((exp[:, :1, :], exp[:, 2:, :]), dim=1)

        if I%2 == 0:
          split = I//2+1
          exp = torch.cat((exp[:, :split, :], exp[:, split+1:, :]), dim=1)

        y = self.freqcore(exp.reshape(B, -1))

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

    def forward_freq_domain(self, x, B, I):
       
        inputD = x.device

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
        
        # [B, C, I*D]
        y = self.freqcore(exp.reshape(B, -1))
       
        y = y.squeeze(-1)
        split = y.shape[1]//2+1
        real = y[:, :split]
   
        imag = y[:, split:]
   
        imag = torch.cat((torch.zeros((B, 1), device=inputD), imag, torch.zeros((B, 1), device=inputD)), dim=1)
        y = torch.stack((real, imag), dim=2)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        y = torch.fft.irfft(y, dim=1, norm="ortho")
        return y

   # def forward_time_domain(self, x, B):
   #     y = self.timecore(x.reshape(B, -1))

   #     return y.squeeze(-1)