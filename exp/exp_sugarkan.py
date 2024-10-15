import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data.dataset import Dataset
from utils.constants import VERBOSE, LEARNING_RATE, BATCH, FREQ
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, lag_target, adjust_learning_rate
from models import DLinear, iTransformer, FreTS, SugarNet, FGN, TimeMixer, FiLM, PatchTST, FEDformer, gru, SugarKAN, SugarKAN_LSTM, gru_ext_both, SugarKAN_MLP, SugarKAN_pykan
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from datetime import timedelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.buffer import Buffer

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
        if args.exp_mode==0:
          self.exp_mode = "KAN"
        elif args.exp_mode==1:
          self.exp_mode = "MLP"
        
        self.buffer_size = 64
        self.erbuffer = Buffer(self.buffer_size, self.device)

    def _build_model(self):
      model_dict = {
            'FGN': FGN,
            'DLinear': DLinear,
            'iTransformer': iTransformer,
            'FreTS': FreTS,
            'SugarNet': gru_ext_both,
            'TimeMixer': TimeMixer,
            'FiLM': FiLM,
            'PatchTST': PatchTST,
            'FEDformer': FEDformer,
            'SugarKAN':SugarKAN,
            'SugarMLP':SugarKAN_MLP,
            'slstm': SugarKAN_LSTM,
            'pykan': SugarKAN_pykan,
        }
      model = model_dict[self.model_name].Model(self.args).float()
      return model

    def _get_data(self, x, y):
        data_set, data_loader = data_provider(x, y, self.args.seq_len, self.args.pred_len, delta=self.args.delta_forecast)
        #data_set, data_loader = data_provider(data, args)
        return data_set, data_loader

    def _select_optimizer(self):
      if self.name=='SugarNet' or self.name=='SugarKAN' or self.name=='SugarMLP':
        #model_optim = optim.Adam([
            #{'params': self.model.kanf.parameters(), 'lr': 0.001},  # Special learning rate for component1
            #{'params': self.model.kant.parameters(), 'lr': 0.0001},  # Default learning rate for component2
        #], lr=0.001, weight_decay=1e-4)
        model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
      else:
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        
      return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, data, epochs, features, verbose=False, train_mode='pretrain', id=-1, save=False):
      model_path = f'/content/drive/MyDrive/research/diabetes/models/{self.model_name}'
      if not os.path.exists(model_path):
          os.makedirs(model_path)
      patience = 5
      early_stopping = EarlyStopping(patience=patience, verbose=True)
      model_optim = self._select_optimizer()
      criterion = self._select_criterion()
      scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=5, verbose=True)
      best_loss = torch.inf
      split = 0.8
      best_model = None
      if self.args.no_dim_extension:
        ext = "noext"
      else:
        ext = self.args.embed_size
      if self.args.delta_forecast:
        delta = "delta"
      else:
        delta = "nodelta"

      #num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
      #print(f"{self.model_name} size {num_trainable_params}")
      
      for epoch in range(epochs):
      
        epoch_train_loss = []
        epoch_val_loss = []
      
        self.model.train()
        val_data = []
        for id in data.keys():
          patient_train_loss = []
          for df in data[id]:
            df['glucose_level'] = df['glucose_level'].interpolate('linear')
            df.dropna(inplace=True)
            df_train = df[:int(len(df)*0.8)].copy(deep=True)
            if len(df_train) < self.args.seq_len + self.args.pred_len:
              continue
            df_val = df[int(len(df)*0.8):].copy(deep=True)
            val_data.append(df_val)
            X_train, Y_train = lag_target(df_train[features], self.args.pred_len, delta=self.args.delta_forecast)  
            train_data, train_loader = self._get_data(X_train, Y_train)

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

              if self.name=='SugarKAN':
                outputs = self.model(batch_x)
              else:
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
        
  
        self.model.eval()
  
        with torch.no_grad():
          for df_val in val_data:
            X_val, Y_val = lag_target(df_val[features], self.args.pred_len, delta=self.args.delta_forecast) 
            if len(df_val) < self.args.seq_len + self.args.pred_len:
              continue 
            val_data, val_loader = self._get_data(X_val, Y_val)
            
            for i, (batch_x, batch_y) in enumerate(val_loader):
              batch_x = batch_x.float().to(self.device)
              batch_y = batch_y.float().to(self.device)

              if batch_y.shape[2] == self.args.pred_len:
                  batch_y = batch_y.permute(0, 2, 1)
              dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
              dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
              outputs = self.model(batch_x, dec_inp)
              # SugarNet outputs BG forecast only, but other baseline models may output multivariate forecast
              if len(outputs.shape)>2:
                  outputs = outputs[:, :, 0]
  
              batch_y = batch_y[:, :, 0]
              val_loss = criterion(outputs, batch_y)
              epoch_val_loss.append(val_loss.item())

        # Print the epoch, training loss, and validation loss
        train_loss = np.average(epoch_train_loss)
        val_loss = np.average(epoch_val_loss)
        if verbose:
          print(f"epoch {epoch} train loss {train_loss} val loss {val_loss}")

        # Step the scheduler
        scheduler.step(val_loss)
        early_stopping(val_loss)
     
        if (val_loss<best_loss):
            best_loss = val_loss
            best_model = self.model
            if train_mode=='transfer':
              best_model_path = model_path + '/' + f'{self.name}_{self.args.data}.{self.exp_mode}.{self.mode}.{self.ext}.{delta}.{train_mode}.{id}.checkpoint.pth'
            else:
              best_model_path = model_path + '/' + f'{self.name}_{self.args.data}.{self.exp_mode}.{self.mode}.{self.ext}.{delta}.{train_mode}.checkpoint.pth'
            if save:
              print(f"save {best_model_path} at epoch {epoch}")
              torch.save(self.model.state_dict(), best_model_path)

        if epoch == epochs - 1 and VERBOSE:
              print("Epoch: {0}, final Train Loss: {1:.7f}".format(
                    epoch + 1, train_loss))
        

        if early_stopping.early_stop:
              break

      return best_model, best_loss

    def trainWithBuffer(self, data, epochs, features, verbose=False, train_mode='pretrain', id=-1, save=False):
      model_path = f'/content/drive/MyDrive/research/diabetes/models/{self.model_name}'
      if not os.path.exists(model_path):
          os.makedirs(model_path)
      patience = 5
      early_stopping = EarlyStopping(patience=patience, verbose=True)
      model_optim = self._select_optimizer()
      criterion = self._select_criterion()
      scheduler = ReduceLROnPlateau(model_optim, mode='min', factor=0.5, patience=5, verbose=True)
      best_loss = torch.inf
      split = 0.8
      best_model = None
      if self.args.no_dim_extension:
        ext = "noext"
      else:
        ext = self.args.embed_size
      if self.args.delta_forecast:
        delta = "delta"
      else:
        delta = "nodelta"

      #num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
      #print(f"{self.model_name} size {num_trainable_params}")
      
      for epoch in range(epochs):
        if verbose:
          print(f"epoch {epoch}")
       
        epoch_train_loss = []
        epoch_val_loss = []
      
        self.model.train()
        val_data = []
        for id in data.keys():
          patient_train_loss = []
          for df in data[id]:
            df['glucose_level'] = df['glucose_level'].interpolate('linear')
            df.dropna(inplace=True)
            df_train = df[:int(len(df)*0.8)].copy(deep=True)
            if len(df_train) < self.args.seq_len + self.args.pred_len:
              continue
            df_val = df[int(len(df)*0.8):].copy(deep=True)
            val_data.append(df_val)
            X_train, Y_train = lag_target(df_train[features], self.args.pred_len, delta=self.args.delta_forecast)  
            train_data, train_loader = self._get_data(X_train, Y_train)

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

              inputs = batch_x.clone()
              trueOutput = batch_y.clone()
              if not self.erbuffer.is_empty():
                    # Strategy 50/50
                    # From batch of 64 (dataloader) to 64 + 64 (dataloader + replay)
                    buf_input, buf_output, _ = self.erbuffer.get_data(BATCH)
                    batch_x = torch.cat((batch_x, torch.stack(buf_input)))
                    batch_y = torch.cat((batch_y, torch.stack(buf_output)))

              if self.name.startswith('Sugar'):
                outputs = self.model(batch_x)
              else:
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device) 
                outputs = self.model(batch_x, dec_inp)
             
              if len(outputs.shape)>2:
                outputs = outputs[:, -self.args.pred_len:, 0]
              
              batch_y = batch_y[:, :, 0]

              loss = criterion(outputs, batch_y)
              patient_train_loss.append(loss.item())

              loss.backward()
              
              model_optim.step()

              if epoch == 0:
                    self.erbuffer.add_data(examples=inputs.to(self.device), labels=trueOutput.to(self.device))

          patient_loss = np.average(patient_train_loss)
          epoch_train_loss.append(patient_loss)
            
        self.model.eval()
  
        with torch.no_grad():
          for df_val in val_data:
            X_val, Y_val = lag_target(df_val[features], self.args.pred_len, delta=self.args.delta_forecast) 
            if len(df_val) < self.args.seq_len + self.args.pred_len:
              continue 
            val_data, val_loader = self._get_data(X_val, Y_val)
            
            for i, (batch_x, batch_y) in enumerate(val_loader):
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
  
              batch_y = batch_y[:, :, 0]
              val_loss = criterion(outputs, batch_y)
              epoch_val_loss.append(val_loss.item())

        # Print the epoch, training loss, and validation loss
        train_loss = np.average(epoch_train_loss)
        val_loss = np.average(epoch_val_loss)
        if verbose:
          print(f"epoch {epoch} train loss {train_loss} val loss {val_loss}")

        # Step the scheduler
        scheduler.step(val_loss)
        early_stopping(val_loss)
                    
          #if VERBOSE:
          #  print(f"epoch {epoch} patient {id} loss {patient_loss}")
     
        if (val_loss<best_loss):
            best_loss = val_loss
            best_model = self.model
            if train_mode=='transfer':
              best_model_path = model_path + '/' + f'{self.name}_{self.args.data}.{self.exp_mode}.{self.mode}.{self.ext}.{delta}.{train_mode}.{id}.checkpoint.pth'
            else:
              best_model_path = model_path + '/' + f'{self.name}_{self.args.data}.{self.exp_mode}.{self.mode}.{self.ext}.{delta}.{train_mode}.checkpoint.pth'
            if save:
              print(f"save {best_model_path} at epoch {epoch}")
              torch.save(self.model.state_dict(), best_model_path)

        if epoch == epochs - 1 and VERBOSE:
              print("Epoch: {0}, final Train Loss: {1:.7f}".format(
                    epoch + 1, train_loss))
        

        if early_stopping.early_stop:
              break

      return best_model, best_loss

    def load_model(self, path):
      self.model.load_state_dict(torch.load(path))

    def test(self, pid, features, data, datatype):
      for sdata in data:
        X_test, Y_test = lag_target(sdata[features], self.args.pred_len, delta=self.args.delta_forecast)

        test_data, test_loader = self._get_data(x = X_test, y = Y_test)

        #features_used = features.remove("Date")

        preds = []
        trues = []
        inputx = []

        self.model.eval()
        folder_path = '/content/drive/MyDrive/research/diabetes/results/'+datatype+'/'+self.model_name

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
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
            csv = f"{folder_path}/{pid}_{time}_{self.name}.{self.exp_mode}.{self.mode}.{self.ext}.{self.delta}.csv"
            #print(f"save to {csv}")
            #X_test[[pred_col, 'glucose_level']].to_csv(csv)
            if VERBOSE:
              print('horizon {} mape:{}, rmse:{}'.format(time, mape, rmse))
            rmape.append(mape)
            rrmse.append(rmse)
        else:
          start = self.args.seq_len
          for time_idx, time in times.items():
            pred_col = f"pred_cgm_{time}"
            csv = f"{folder_path}/{pid}_{time}_{self.name}.{self.exp_mode}.{self.mode}.{self.ext}.{self.delta}.csv"
            #print(f"save to {csv}")
            s = start + time
            end = s + len(preds)
            results = X_test.copy(deep=True)
            #print(X_test.columns)
            results.iloc[s:end, results.columns.get_loc('glucose_level')] = preds[:, time]
            calced_results = test_data.inverse_transform(results[s:end])        
            results[pred_col] = X_test['glucose_level']
            results[pred_col][s:end] = calced_results[:, 0]
            mae, mse, rmse, mape, mspe = metric(calced_results[:, 0], X_test['glucose_level'][s:end])
            # Restore back to original BG values.
            results.iloc[s:end, results.columns.get_loc('glucose_level')] = X_test['glucose_level'][s:end]
            #results[[pred_col, 'glucose_level']][s:end].to_csv(csv)
            
            rmape.append(mape)
            rrmse.append(rmse)
          #start = self.args.seq_len
          #for time_idx, time in times.items():
          #  s = start + time
          #  end = s + len(preds)
          #  results = X_test.copy(deep=True)
          #  results.iloc[s:end, results.columns.get_loc('glucose_level')] = preds[:, time]
          #  results = test_data.inverse_transform(results[s:end])
          #  mae, mse, rmse, mape, mspe = metric(results[:, 0], X_test['glucose_level'][s:end])
          #  rmape.append(mape)
          #  rrmse.append(rmse)

        return rmape, rrmse