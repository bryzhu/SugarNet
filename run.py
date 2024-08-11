import argparse
import os
import torch
from exp.exp_main import Exp_Main
from utils.diatrend_subsampling import readDiaTrend
from utils.tools import loadPreTrainingData, loadIndividualLearningData, visual, calcClark, visualRMSE, readVirtual
import random
import numpy as np

model_path = '/content/drive/MyDrive/research/diabetes/models'
transfer_path = '/content/drive/MyDrive/research/diabetes/transfer_models'

def pretraining(args, epochs=1, name="SugarNet", epochsRun=0, pretrain_map=None):
    global model_path
    exp = Exp_Main(args, name)
    if epochsRun>0:
      #/content/drive/MyDrive/research/diabetes/models/PatchTST_DiaTrend.ext.nodelta.pretrain.checkpoint.pth
      if args.dim_extension:
        ext = "ext"
      else:
        ext = "noexit"
      if args.delta_forecast:
        delta = "delta"
      else:
        delta = "nodelta"
      pre_model_path = model_path + '/' + f'{name}_{args.data}.{ext}.{delta}.pretrain.checkpoint.pth'
      print(f"Continue training, load from {pre_model_path}")
      exp.model.load_state_dict(torch.load(pre_model_path))
    print(pretrain_map.keys())
    model, loss = exp.train(data=pretrain_map, epochs = epochs-epochsRun, features=FEATURES, mode='pretrain')

def transfer_learn_model(args, all_train_df=None, all_test_df=None, name="SugarNet", test_only = False):
  global model_path
  global transfer_path
  if args.dim_extension:
      ext = "ext"
  else:
      ext = "noexit"
  if args.delta_forecast:
      delta = "delta"
  else:
      delta = "nodelta"
  
  base_model_path = model_path + '/' + f'{name}_{args.data}.{ext}.{delta}.pretrain.checkpoint.pth'

  rmape = []
  rrmse = []

  for id in all_train_df.keys():
    exp = Exp_Main(args, name)

    transfer_model_path = transfer_path + '/' + f'{name}_{id}_transfer.checkpoint.pth'
    df_test = all_test_df[id]

    if test_only:
      exp.model.load_state_dict(torch.load(transfer_model_path))
    else:
      exp.model.load_state_dict(torch.load(base_model_path))

        #df_train['glucose_level'] = df_train['glucose_level'].interpolate('linear')

      #print(f"transfer learn {id}")
      train = {}
      train[id] = all_train_df[id]
      model, _ = exp.train(data=train, epochs = args.learn_epochs, features=FEATURES, verbose=False, mode='transfer', id=id)

      print(f"save {transfer_model_path}")
      torch.save(model.state_dict(), transfer_model_path)
      
    df_test = all_test_df[id]
    features=FEATURES
    #features.append("Date")
    mape, rmse = exp.test(pid=id, data=df_test, features=FEATURES)
    rmape.append(mape)
    rrmse.append(rmse)

   # print(f"result for {id}: mape: {mape}, rmse {rmse}")

  return np.mean(np.array(rmape), axis=0), np.mean(np.array(rrmse), axis=0)

# Example of generating figures
def visualAnsSummary():
  #calcClark(['SugarNet'], [2069], "T2D", draw=True)
  visual(['SugarNet'], [2078])
  visualRMSE("T1D")

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SugarNet')

    # basic config
    parser.add_argument('--pretraining_epochs', type=int, default=50,
                        help='Number of epochs for pretraining')
    parser.add_argument('--learn_epochs', type=int, default=50,
                        help='Number of epochs for transfer learning')
    parser.add_argument('--delta_forecast', type=bool, default=False, help='generate delta forecast')
    parser.add_argument('--dim_extension', type=bool, default=True, help='enable dimension extension')
    parser.add_argument('--mode', type=int, default=1, help='1. time+freq 2. time 3. freq')
    parser.add_argument('--sampling_rate', type=int, default=15, help='5 or 15 min')
    parser.add_argument('--feature_size', type=int, default=5, help='5')

    # data loader
    parser.add_argument('--data', type=str, default='DiaTrend', help='dataset type: T1D, T2D, DiaTrend.basal, DiaTrend, virtual')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=6, help='start token length')
    parser.add_argument('--pred_len', type=int, default=8, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Hourly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=5, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=5, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=1, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='conv',
                        help='down sampling method, only support avg, max, conv')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    args = parser.parse_args()

    data_path='/content/drive/MyDrive/research/diabetes/datasets/Shanghai_T1DM'
    pre_training = set(range(1001, 1013))
    individual_learning = [1002, 1006]
    
    if args.data=='virtual':
      data_path='/content/drive/MyDrive/research/diabetes/datasets/virtual'
      FEATURES=['glucose_level', 'bolus', 'carbs']
      basal = False
      args.sampling_rate = 5
      args.seq_len = int(60*3/args.sampling_rate)
      args.label_len = int(args.seq_len/2)
      args.pred_len = int(60*2/args.sampling_rate)

      pretrain_map, train_map, test_map  = readVirtual(path=data_path, 
        history=args.seq_len, 
        future=args.pred_len, 
        features=FEATURES, 
        freq=args.sampling_rate,  # Only pass `freq` here
        includeBasal=basal)
      args.enc_in = len(FEATURES)
      args.dec_in = len(FEATURES)
      args.c_out = len(FEATURES)
    elif args.data=='DiaTrend' or args.data=='DiaTrend.basal':
      data_path='/content/drive/MyDrive/research/diabetes/datasets/DiaTrend'
      
      #FREQ=5
      FEATURES=['glucose_level', 'bucket', 'bolus', 'carbs']
      basal = False

      args.sampling_rate = 5
      args.seq_len = int(60*3/args.sampling_rate)
      args.label_len = int(args.seq_len/2)
      args.pred_len = int(60*2/args.sampling_rate)

      if args.data=='DiaTrend.basal':
        FEATURES.append('basal')
        basal = True
      else:
        FEATURES.append('iob')

      pretrain_map, train_map, test_map  = readDiaTrend(path=data_path, 
        history=args.seq_len, 
        future=args.pred_len, 
        features=FEATURES, 
        freq=args.sampling_rate,  # Only pass `freq` here
        includeBasal=basal)
      args.enc_in = len(FEATURES)
      args.dec_in = len(FEATURES)
      args.c_out = len(FEATURES)
    else:
      if args.data=='T2D':
        data_path='/content/drive/MyDrive/research/diabetes/datasets/Shanghai_T2DM'
        pre_training = set(range(2000, 2099))
        individual_learning = [2001, 2014, 2015, 2017, 2055, 2078, 2074, 2069]
        FEATURES = ['glucose_level', 'carbs', 'bolus', 'basal', 'bolus_long', 'meds']
        args.enc_in = len(FEATURES)
        args.dec_in = len(FEATURES)
        args.c_out = len(FEATURES)
    
      total = 0
      for i in individual_learning:
          pre_training.remove(i)
      pretrain_map, pretest_map, t = loadPreTrainingData(pre_training, data_path, FEATURES)
      total += t
      train_map, test_map, t = loadIndividualLearningData(individual_learning, data_path, args.data, FEATURES)
      total += t
    
      print(f"total {total} points")

    #'SugarNet', 'PatchTST', 'FreTS', 'DLinear', 'iTransformer', 'FGN', 'FiLM', 'TimeMixer', 'FEDformer'
    models = ['FEDformer']
    #models = ['SugarNet']

    for name in models:
      print(f"run {name} extension = {args.dim_extension} delta = {args.delta_forecast}")
      if args.mode==2:
        print("time")
      elif args.mode==1:
        print("time + freq")
      else:
        print("freq")
      if name != 'SugarNet':
        args.learning_rate = 0.0001
      
      if args.sampling_rate==15:
        indices = [1, 3, 5, 7]
      elif args.sampling_rate==10:
        indices = [2, 5, 8, 11]
      else:
        indices = [5, 11, 17, 23]

      if args.data=='virtual':
        #patients = ['adolescent', 'child', 'adult']
        patients = ['adult']

        for patient in patients:
          print(f"pretrain {patient}")
          pretraining(args, epochs=args.pretraining_epochs, name=name, epochsRun=0, pretrain_map = pretrain_map[patient])
          print(f"transfer {patient}")
          tmape, trmse = transfer_learn_model(args, all_train_df = train_map[patient], all_test_df = test_map[patient], name=name, test_only=False)
          print(tmape)
          print(trmse)
          print(f"{name}\nmape {tmape[indices]} {tmape[indices].mean()}\nrmse {trmse[indices]} {trmse[indices].mean()}")
      else:
        pretraining(args, epochs=args.pretraining_epochs, name=name, epochsRun=0, pretrain_map=pretrain_map)
        tmape, trmse = transfer_learn_model(args, all_train_df = train_map, all_test_df = test_map, name=name, test_only=False)
        print(tmape)
        print(trmse)
        print(f"{name}\nmape {tmape[indices]} {tmape[indices].mean()}\nrmse {trmse[indices]} {trmse[indices].mean()}")
