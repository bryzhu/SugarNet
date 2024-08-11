#data loader
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Dataset_Regular(Dataset):
    def __init__(self, x, y=None, size=None, scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
      
        self.scale = scale
        self.df_data = x

        self.__read_data__()

    def __read_data__(self):
        data = self.df_data.values

        self.data_x = self.df_data.values
        self.data_y = self.df_data.values

        if self.scale:
          self.scaler = StandardScaler()
          self.data_x = self.scaler.fit_transform(self.data_x)
          self.data_y = self.scaler.fit_transform(self.data_y)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # Y is [B, C, Pred_len]
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, x):
      if x is not None:
        if self.scale==True:
          return self.scaler.inverse_transform(x)
      
      return x

class Dataset_Delta(Dataset):
    def __init__(self, x, y, size=None, 
                 target_name='glucose_level', scale=True):
    #def __init__(self, data, args, scale=True):
        self.seq_len = size[0]
        self.pred_len = size[1]
      
        self.scale = scale
        self.df_data = x
        self.target = y

        self.__read_data__()

    def __read_data__(self):
        data = self.df_data.values

        self.data_x = self.df_data.values
        #print(self.df_data.columns)
        self.data_y = self.target.values

        if self.scale:
          self.x_scaler = StandardScaler()
          self.y_scaler = StandardScaler()
          self.data_x = self.x_scaler.fit_transform(self.data_x)
          self.data_y = self.y_scaler.fit_transform(self.data_y)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end-1
        r_end = r_begin + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, x = None, y = None):
      if x is not None:
        if self.scale==True:
          return self.x_scaler.inverse_transform(x)
        return x
      if self.scale==True:
        return self.y_scaler.inverse_transform(y)

      return y