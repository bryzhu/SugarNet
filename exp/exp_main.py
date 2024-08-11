import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataset import Dataset
from utils.constants import VERBOSE, LEARNING_RATE, BATCH, FREQ
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, lag_target, adjust_learning_rate
from models import DLinear, iTransformer, FreTS, SugarNet, FGN, TimeMixer, FiLM, PatchTST, FEDformer
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from datetime import timedelta

import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

alpha = 1.0

class Exp_Main(Exp_Basic):
    def __init__(self, args, model_name):
        super(Exp_Main, self).__init__(args, model_name)

    def _build_model(self):
        model_dict = {
            'FGN': FGN,
            'DLinear': DLinear,
            'iTransformer': iTransformer,
            'FreTS': FreTS,
            'SugarNet': SugarNet,
            'TimeMixer': TimeMixer,
            'FiLM': FiLM,
            'PatchTST': PatchTST,
            'FEDformer': FEDformer,
        }
        
        model = model_dict[self.name].Model(self.args).float()
        return model

    def _get_data(self, x, y):
        data_set, data_loader = data_provider(x, y, self.args.seq_len, self.args.pred_len, delta=self.args.delta_forecast)
        #data_set, data_loader = data_provider(data, args)
        return data_set, data_loader

    def _select_optimizer(self):
      if self.name=='SugarPal':
        model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)
      else:
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
      return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, data, epochs, features, verbose=VERBOSE, mode='pretrain', id=-1):
      model_path = '/content/drive/MyDrive/research/diabetes/models'
      patience = 3
      early_stopping = EarlyStopping(patience=patience, verbose=True)
      model_optim = self._select_optimizer()
      criterion = self._select_criterion()
      best_loss = torch.inf
      best_model = None
      if self.args.dim_extension:
        ext = "ext"
      else:
        ext = "noexit"
      if self.args.delta_forecast:
        delta = "delta"
      else:
        delta = "nodelta"

      for epoch in range(epochs):
        epoch_train_loss = []
        for id in data.keys():
          patient_train_loss = []
          for df_train in data[id]:
            df_train['glucose_level'] = df_train['glucose_level'].interpolate('linear')
            df_train.dropna(inplace=True)
          
            X_train, Y_train = lag_target(df_train[features], self.args.pred_len, delta=self.args.delta_forecast)
          
            train_data, train_loader = self._get_data(X_train, Y_train)

            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
              model_optim.zero_grad()
              batch_x = batch_x.float().to(self.device)
              batch_y = batch_y.float().to(self.device)

              #print(f"batch x {batch_x.shape}")
              #print(f"batch y {batch_y.shape}")
              #[B, C, pred_len] -> [B, pred_len, C]
              if batch_y.shape[2] == self.args.pred_len:
                batch_y = batch_y.permute(0, 2, 1)
              dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
              #print(f"x {batch_x.shape}")
              #print(f"y {batch_y.shape}")
              #print(f"dec inp {dec_inp.shape}")
              #print(self.args.label_len)
              dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
              outputs = self.model(batch_x, dec_inp)
              
              if len(outputs.shape)>2:
                outputs = outputs[:, -self.args.pred_len:, 0]
              
              batch_y = batch_y[:, :, 0]

              loss = criterion(outputs, batch_y)
              patient_train_loss.append(loss.item())

              loss.backward()
              model_optim.step()
            
          patient_loss = np.average(patient_train_loss)
          epoch_train_loss.append(patient_loss)
          #if VERBOSE:
          #  print(f"epoch {epoch} patient {id} loss {patient_loss}")

        train_loss = np.average(epoch_train_loss)
        #if VERBOSE:
        #print(f"epoch {epoch} overall loss {train_loss}")
        if (train_loss<best_loss):
            best_loss = train_loss
            best_model = self.model
            if mode=='transfer':
              best_model_path = model_path + '/' + f'{self.name}_{self.args.data}.{ext}.{delta}.{mode}.{id}.checkpoint.pth'
            else:
              best_model_path = model_path + '/' + f'{self.name}_{self.args.data}.{ext}.{delta}.{mode}.checkpoint.pth'
            if VERBOSE:
              print(f"save {best_model_path} at epoch {epoch}")
            torch.save(self.model.state_dict(), best_model_path)

        if epoch == epochs - 1:
              print("Epoch: {0}, final Train Loss: {1:.7f}".format(
                    epoch + 1, train_loss))
        early_stopping(train_loss)

        if early_stopping.early_stop:
              #print("Early break at Epoch: {0}, Train Loss: {1:.7f}".format(
              #      epoch + 1, train_loss))
              break

        adjust_learning_rate(model_optim, epoch + 1, self.name)

      return best_model, best_loss

    def load_model(self, path):
      self.model.load_state_dict(torch.load(path))

    def test(self, pid, features, data):
      for sdata in data:
        X_test, Y_test = lag_target(sdata[features], self.args.pred_len, delta=self.args.delta_forecast)

        test_data, test_loader = self._get_data(x = X_test, y = Y_test)

        #features_used = features.remove("Date")

        preds = []
        trues = []
        inputx = []

        self.model.eval()
        folder_path = '/content/drive/MyDrive/research/diabetes/results'

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):

                #print(f"testing {test_data[i]}")
                #batch_x = batch_x[features_used].float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                #print(f"x {batch_x.shape}")
                #print(f"y {batch_y.shape}")
                #print(f"label {label.shape}")

                if batch_y.shape[2] == self.args.pred_len:
                  batch_y = batch_y.permute(0, 2, 1)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                outputs = self.model(batch_x, dec_inp)

                # SugarNet outputs BG forecast only, but other baseline models may output multivariate forecast
                if len(outputs.shape)>2:
                  outputs = outputs[:, :, 0]
                
                #print(f"pred for {label.min()} to {label.max()}")
                pred = outputs.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)

        times = {}

        for i in range(self.args.pred_len):
          times[(i+1)*self.args.sampling_rate] = i

        rmape = []
        rrmse = []

        if self.args.delta_forecast==True:
          start = self.args.seq_len-1
          end = start + len(preds)
          if preds.shape[1]>self.args.pred_len:
            preds = preds[:, -self.args.pred_len:]
          preds = test_data.inverse_transform(y = preds)

          for time_idx, time in times.items():
            pred_col = f"pred_cgm_{time}"
           
            X_test[pred_col] = X_test['glucose_level'][start:end] + preds[:, time]
         
            X_test[pred_col] = X_test[pred_col].shift(time+1)
          
          X_test = X_test.dropna()
          
          for time in range(0, self.args.pred_len):
            pred_col = f"pred_cgm_{time}"
            
            mae, mse, rmse, mape, mspe = metric(X_test[pred_col], X_test['glucose_level'])
            csv = f"{folder_path}/{pid}_{time}_{self.name}.csv"
            X_test[[pred_col, 'glucose_level']].to_csv(csv)
            if VERBOSE:
              print('horizon {} mape:{}, rmse:{}'.format(time, mape, rmse))
            rmape.append(mape)
            rrmse.append(rmse)
        else:
          start = self.args.seq_len
          for time_idx, time in times.items():
            s = start + time
            end = s + len(preds)
            results = X_test.copy(deep=True)
            results.iloc[s:end, results.columns.get_loc('glucose_level')] = preds[:, time]
            results = test_data.inverse_transform(results[s:end])
            mae, mse, rmse, mape, mspe = metric(results[:, 0], X_test['glucose_level'][s:end])
            rmape.append(mape)
            rrmse.append(rmse)

        return rmape, rrmse