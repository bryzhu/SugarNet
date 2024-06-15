import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from utils.constants import VERBOSE, LEARNING_RATE, BATCH, FUTURE_STEPS, FREQ
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, lag_target, adjust_learning_rate
from models import DLinear, iTransformer, FreTS, SugarNet, FGN, TimeMixer, FiLM, PatchTST
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
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
            'PatchTST': PatchTST
        }
        model = model_dict[self.name].Model(self.args).float()
        return model

    def _get_data(self, x, y):
        data_set, data_loader = data_provider(x, y, delta=self.args.delta_forecast)
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

    def save_checkpoint(self, val_loss, model, path):
       # print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)

    def train(self, data, epochs, features, verbose=VERBOSE):
      patience = 3
      early_stopping = EarlyStopping(patience=patience, verbose=True)
      model_optim = self._select_optimizer()
      criterion = self._select_criterion()
      best_loss = torch.inf
      best_model = None

      for epoch in range(epochs):
        #print(f"TRAIN epoch {epoch}")
        train_loss = []
        for id in data.keys():
          df_train = data[id]
          df_train['glucose_level'] = df_train['glucose_level'].interpolate('linear')
          df_train.dropna(inplace=True)
          
          X_train, Y_train = lag_target(df_train[features], delta=self.args.delta_forecast)
          
          train_data, train_loader = self._get_data(X_train, Y_train)

          self.model.train()

          epoch_time = time.time()
          for i, (batch_x, batch_y) in enumerate(train_loader):
              model_optim.zero_grad()
              batch_x = batch_x.float().to(self.device)
              batch_y = batch_y.float().to(self.device)

              #[B, C, pred_len] -> [B, pred_len, C]
              if batch_y.shape[2] == self.args.pred_len:
                batch_y = batch_y.permute(0, 2, 1)
              dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
              dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
              outputs = self.model(batch_x, dec_inp)
              
              if len(outputs.shape)>2:
                outputs = outputs[:, -self.args.pred_len:, 0]
              
              batch_y = batch_y[:, :, 0]

              loss = criterion(outputs, batch_y)
              loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
              loss_auxi = loss_auxi.mean().abs()

              loss = alpha*loss + (1-alpha)*loss_auxi

              train_loss.append(loss.item())

              loss.backward()
              model_optim.step()

        epoch_train_loss = np.average(train_loss)
        if VERBOSE:
          print(f"epoch {epoch} loss {epoch_train_loss}")
        if (epoch_train_loss<best_loss):
            best_loss = epoch_train_loss
            best_model = self.model

        if epoch == epochs - 1:
              print("Epoch: {0}, Train Loss: {1:.7f}".format(
                    epoch + 1, epoch_train_loss))
        early_stopping(epoch_train_loss)

        if early_stopping.early_stop:
              print("Early break at Epoch: {0}, Train Loss: {1:.7f}".format(
                    epoch + 1, epoch_train_loss))
              break

        adjust_learning_rate(model_optim, epoch + 1, self.name)
      #  self.model.load_state_dict(torch.load(best_model_path))

      return best_model, best_loss

    def load_model(self, path):
      self.model.load_state_dict(torch.load(path))

    def test(self, features, data):
        X_test, Y_test = lag_target(data[features], delta=self.args.delta_forecast)

        test_data, test_loader = self._get_data(x = X_test, y = Y_test)

     #  print(f"loader {len(test_loader)}")
        preds = []
        trues = []
        inputx = []

        self.model.eval()
        folder_path = '/content/drive/MyDrive/research/FreTS/results'

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):

                #print(f"testing {test_data[i]}")
                batch_x = batch_x.float().to(self.device)
                
                batch_y = batch_y.float().to(self.device)

                if batch_y.shape[2] == self.args.pred_len:
                  batch_y = batch_y.permute(0, 2, 1)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                outputs = self.model(batch_x, dec_inp)

                if len(outputs.shape)>2:
                  outputs = outputs[:, :, 0]

                pred = outputs.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                #inputx.append(batch_x.detach().cpu().numpy())

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)

        times = {
          FREQ: 0,
          2*FREQ: 1,
          3*FREQ: 2,
          4*FREQ: 3,
          5*FREQ: 4,
          6*FREQ: 5,
          7*FREQ: 6,
          8*FREQ: 7,
        }

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
          
          for time in range(0, FUTURE_STEPS):
            pred_col = f"pred_cgm_{time}"
            mae, mse, rmse, mape, mspe = metric(X_test[pred_col], X_test['glucose_level'])
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