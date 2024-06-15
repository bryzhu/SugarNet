import argparse
import os
import torch
from exp.exp_main import Exp_Main
from utils.constants import FEATURES
from utils.tools import loadPreTrainingData, loadIndividualLearningData
import random
import numpy as np

def pretraining(args, epochs=1, name="SugarNet"):
    exp = Exp_Main(args, name)
    print(f"{FEATURES}")
    model, loss = exp.train(data=pretrain_map, epochs = epochs, features=FEATURES)
    best_model_path = model_path + '/' + f'{name}_{args.data}.checkpoint.pth'
    print(f"save {best_model_path}")
    torch.save(model.state_dict(), best_model_path)

def transfer_learn_model(all_train_df, all_test_df, args, name="SugarPal"):
  best_model_path = model_path + '/' + f'{name}_{args.data}.checkpoint.pth'
  rmape = []
  rrmse = []

  for id in all_train_df.keys():
    exp = Exp_Main(args, name)

    exp.model.load_state_dict(torch.load(best_model_path))
    df_train = all_train_df[id]
    df_train['glucose_level'] = df_train['glucose_level'].interpolate('linear')

    print(f"transfer learn patient {id}")
    train = {}
    train[id] = df_train
    exp.train(data=train, epochs = args.learn_epochs, features=FEATURES, verbose=False)

    df_test = all_test_df[id]
    mape, rmse = exp.test(data=df_test, features=FEATURES)
    rmape.append(mape)
    rrmse.append(rmse)

   # print(f"result for {id}: mape: {mape}, rmse {rmse}")

  return np.mean(np.array(rmape), axis=0), np.mean(np.array(rrmse), axis=0)

if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='SugarPal')

    # basic config
    parser.add_argument('--pretraining_epochs', type=int, default=50,
                        help='Number of epochs for pretraining')
    parser.add_argument('--learn_epochs', type=int, default=50,
                        help='Number of epochs for transfer learning')
    parser.add_argument('--delta_forecast', type=bool, default=True, help='generate delta forecast')

    # data loader
    parser.add_argument('--data', type=str, default='T1D', help='dataset type')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
   # parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=6, help='start token length')
    parser.add_argument('--pred_len', type=int, default=8, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Daily', help='subset for M4')
    #parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

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
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
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
    
    model_path = '/content/drive/MyDrive/research/diabetes/FREQ_DOMAIN/models'

    #'SugarNet', 'PatchTST', 'FreTS', 'DLinear', 'iTransformer', 'FGN', 'FiLM', 'TimeMixer'
    models = ['SugarNet']

    for name in models:
      print(f"run {name}")
      if name != 'SugarNet':
        args.learning_rate = 0.0001
      pretraining(args, epochs=args.pretraining_epochs, name=name)
      mape, rmse = transfer_learn_model(train_map, test_map, args, name=name)
      print(f"run {name} result mape {mape} \n rmse {rmse}")
