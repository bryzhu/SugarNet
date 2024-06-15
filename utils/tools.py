import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from utils.constants import FREQ, FEATURES, FUTURE_STEPS

plt.switch_backend('agg')

MAX_CARBS=200

from pandas._libs.tslibs.offsets import Hour
# Data source: https://www.nature.com/articles/s41597-023-01940-7
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
import matplotlib.dates as mdates

composite={'bean sprouts', 'bitter melon', 'lotus root', 'water bamboo', 'potato leaves', 'bergamot melon', 'bean curd','bamboo shoot',
            'snow pea', 'milk cereal', 'green bean'}
protein={'pork', 'egg', 'duck', 'fish', 'meat', 'chicken', 'lamb', 'beef', 'shrimp', 'goose', 'bacon', 'crab', 'pomfret', 'crucian', 'omelette',
         'mutton', 'hairtail'}
vege={'vegetable', 'cabbage', 'asparagus', 'cowpea', 'fungus', 'pepper', 'onion', 'tomato',
      'carrot', 'cucumber', 'broccoli', 'amaranth', 'lettuce', 'celery',
       'scallion', 'gourd', 'zucchini', 'radish', 'mushroom', 'eggplant', 'spinach', 'apple', 'bean', 'taro', 'leek', 'bean sprout',
      'plum', 'cauliflower', 'watermelon', 'tofu', 'kelp', 'basil'}
carbs={'grain', 'bread', 'corn', 'rice', 'potato', 'yam', 'milk', 'pancake', 'pie', 'noodle', 'glucose', 'peanut', 'coffee', 'oatcake',
       'cereal', 'vermicelli', 'biscuit', 'cake', 'dough stick', 'porridge', 'burger', 'wonton', 'dumpling', 'baijiu', 'crust'}
# caramel treat (萨其马)
# brown daisy
#GREEN beans?  nuts = RAW ALMONDS?
#coffee ?
#oatcakes (oat cookied)
#cereal = REGULAR OR INSTANT OATS, COOKED WITH WATER?
#BAIJIU
#milk cereal

carbsPerOunce = {'pork':0, 'egg':0.3, 'duck':0, 'fish':0, 'chicken':0, 'lamb':0, 'beef':0, 'cabbage':1.4, 'asparagus':1.2, 'cowpea':5.9, 'snow pea':2,
                 'fungus':1.5, 'pepper':1.8, 'onion':2.9, 'tomato':1.1, 'yam':7.8, 'carrot':2.3, 'cucumber':1, 'broccoli':2, 'lettuce':0.8, 'lotus root':4.5,
                 'celery':1.1, 'bean sprouts':3, 'bitter melon':1.2, 'scallion':2.1, 'gourd':1.2, 'potato leaves':2.1, 'zucchini':1.1, 'radish':1, 'mushroom':1.5,
                 'eggplant':2.5, 'bread':13.9, 'corn':5.9, 'rice':8, 'potato':6.1, 'grain':7, 'vegetable':2, 'meat':0, 'amaranth':2, 'water bamboo':0.5, 'bergamot melon':3.5,
                 'spinach':1.1, 'pancake':10.4, 'pie':10.5, 'milk':1.6, 'bean':2.2, 'shrimp':0, 'goose':0, 'cauliflower':1.2, 'apple':3.9, 'noodle':7.1, 'bamboo shoot':0.5,
                 'bacon':0.4, 'glucose':28, 'nuts':5.6, 'taro':9.8, 'peanut':6.1, 'bean curd':2.4, 'leek':2.2, 'bean sprout':3, 'coffee':6, 'plum':3.2, 'oatcake':16.9,
                 'watermelon':2.1, 'cereal':3.4, 'vermicelli':22.9, 'crab':0.3, 'tofu':3, 'kelp':2.7, 'cereal':3.4, 'pomfret':0, 'biscuit':15.7, 'basil':1.2, 'baijiu':0,
                 'crucian':0, 'tea':0, 'omelette':0.2, 'burger':6.8, 'mutton':0, 'cake':16.4, 'milk cereal':2.8, 'wonton':6.2, 'dumpling':5.6, 'dough stick':11, 'amaranth':18.8,
                 'porridge':3.6, 'hairtail':0, 'green bean':2, 'crust':17}

proteinPerOunce = {'pork':7.3, 'egg':3.6, 'duck':5.4, 'fish':7.4, 'chicken':8.8, 'lamb':6.3, 'beef':7.7, 'cabbage':0.3, 'asparagus':0.7, 'cowpea':2.2, 'snow pea':0.9,
                   'fungus':0.6, 'pepper':0.3, 'onion':0.4, 'tomato':0.2, 'yam':0.4, 'carrot':0.2, 'cucumber':0.2, 'broccoli':0.7, 'lettuce':0.3, 'lotus root':0.4,
                   'celery':0.2, 'bean sprouts':1.2, 'bitter melon':0.2, 'scallion':0.5, 'gourd':0.2, 'potato leaves':0.7, 'zucchini':0.2, 'radish':0.2, 'mushroom':0.6,
                   'eggplant':0.2, 'bread':2.6, 'corn':1, 'rice':0.8, 'potato':0.7, 'spinach':0.8, 'pancake':1.5, 'pie':0.7, 'milk':1, 'bean':0.5, 'shrimp':5.9, 'goose':7.1,
                   'cauliflower':0.5, 'apple':0.1, 'noodle':0.3, 'bamboo shoot':0.4, 'bacon':10.5, 'glucose':0, 'nuts':6, 'taro':0.1, 'peanut':6.7, 'bean curd':1.2, 'leek':0.2,
                   'bean sprout':1.2, 'coffee':4, 'plum':0.2, 'oatcake':1.5, 'watermelon':0.2, 'cereal':0.7, 'vermicelli':1.5, 'crab':6.3, 'tofu':4.9, 'kelp':0.5, 'cereal':0.7,
                   'pomfret':2.7, 'biscuit':2.2, 'basil':0.7, 'baijiu':0, 'crucian':5.2, 'tea':0, 'omelette':3, 'burger':4.7, 'mutton':7, 'cake':1.7, 'milk cereal':1.5,
                   'wonton':2.6, 'dumpling':2.5, 'dough stick':1.8, 'amaranth':0.5, 'porridge':1.3, 'hairtail':5.6}

fatPerOunce = {'pork':5.9, 'egg':3, 'duck':8, 'fish':0.8, 'chicken':1, 'lamb':8.4, 'beef':4.9, 'cabbage':0.1, 'asparagus':0, 'cowpea':0.2, 'snow pea':0,
               'fungus':0.1, 'pepper':0, 'onion':0, 'tomato':0, 'yam':0, 'carrot':0, 'cucumber':0, 'broccoli':0.1, 'lettuce':0, 'lotus root':0, 'celery':0, 'bean sprouts':0,
               'bitter melon':0, 'scallion':0, 'gourd':0, 'potato leaves':0, 'zucchini':0, 'radish':0, 'mushroom':0.1, 'eggplant':0, 'bread':0.9, 'corn':0.4, 'rice':0,
               'potato':0, 'spinach':0, 'pancake':0.7, 'pie':3.5, 'milk':1, 'bean':0, 'shrimp':0.3, 'goose':6.2, 'cauliflower':0.1, 'apple':0, 'noodle':0, 'bamboo shoot':0,
               'bacon':11.8, 'glucose':0, 'nuts':14.4, 'taro':0, 'peanut':14.1, 'bean curd':2.1, 'leek':0, 'bean sprout':0, 'coffee':4, 'plum':0, 'oatcake':6.9, 'watermelon':0,
               'cereal':0.4, 'vermicelli':0, 'crab':0.4, 'tofu':5.7, 'kelp':0.2, 'cereal':0.4, 'pomfret':0.9, 'biscuit':1.5, 'basil':0.2, 'baijiu':0, 'crucian':0.7, 'tea':0,
               'omelette':3.2, 'burger':4, 'mutton':6, 'cake':0.1, 'milk cereal':0.3, 'wonton':1, 'dumpling':1.2, 'dough stick':3.2, 'amaranth':0, 'porridge':0.7, 'hairtail':2.4,
               }

def findShortTermInsulin(b):
  b = str(b)

  if b.lower().find('novolin') != -1 or b.lower().find('humulin') != -1 or b.lower().find('gansulin') != -1:
    iu = re.findall(' [0-9]+ IU', b)
    if iu:
      return (float)(iu[0].split()[0])
  return 0

def align_series(df):
    interest_index = df[~df['glucose_level'].isna()].index

    df = df.resample('1T').ffill()

    df = df.fillna(method='ffill').fillna(0)
    df = df.resample(f'{FREQ}T').first().shift(-1).dropna()

    return df, interest_index

def findLongTermInsulin(b):
  b = str(b)

  if b.lower().find('detemir') != -1 or b.lower().find('degludec') != -1 or b.lower().find('glargine'):
    iu = re.findall(' [0-9]+ IU', b)
    if iu:
      return (float)(iu[0].split()[0])

  return 0

def lookupCarbs(items, weight):
  sum = 0
  if len(items) != 0:
    weight /= len(items)
    for i in items:
      sum += carbsPerOunce[i]*weight/28.3495
  return sum

def calcCarbs(tcomp, tveg, tprotein, tcarb, weight, unit):
  proteinw = 0
  vegw = 0
  carbw = 0
  tveg = tcomp + tveg
  if len(tprotein) != 0 and len(tcarb) != 0 and len(tveg) != 0:
    carbw = 0.45 * weight
    proteinw = 0.3 * weight
    vegw = 0.25 * weight
  elif len(tprotein) != 0 and len(tcarb) != 0:
    proteinw = 0.5 * weight
    carbw = 0.5 * weight
  elif len(tprotein) != 0 and len(tveg) != 0:
    proteinw = 0.7 * weight
    vegw = 0.3 * weight
  elif len(tcarb) != 0 and len(tveg) != 0:
    vegw = 0.5 * weight
    carbw = 0.5 * weight
  elif len(tprotein) != 0:
    proteinw = weight
  elif len(tcarb) != 0:
    carbw = weight
  else:
    vegw = weight

  #print("carbw: "+str(carbw)+" proteinw: "+str(proteinw)+" vegw: "+str(vegw))
  tmp = lookupCarbs(tprotein, proteinw) + lookupCarbs(tveg, vegw) + lookupCarbs(tcarb, carbw)
  #print(tmp)
  return tmp

def findFood(dish, types):
  result = []
  arr = dish.split(" ")
  for t in types:
    if t in dish:
      dish = dish.replace(t, '')
      #print(t)
      result.append(t)
  return (result, dish)

def findUnits(str):
  units = ["g", "piece", "cup", "ml", "bag", "loaf", "bowl"]

  for u in units:
    pattern = " *[0-9]+ *"+u
   # print(pattern)
    reg = re.findall(pattern, str)
    if len(reg) > 0:
      return u

  return ""

def calculateCarbs(b):
  b = str(b)
  totalCarb = 0
  if b.lower()=='nan' or b.lower()=='data not available' or b.isspace():
    return 0
  items = b.split('\n')

  for i in items:
    if i.isspace():
      continue
    iu = re.findall(' *[0-9]+', i) # g/piece/cup/ml/bag/loaf
    #ten, cup
    total = 0
    if (iu):
   #  number = iu[0].split(' ')
    #  removeChar = []
     # weight = 0

      t = i.lower()
      this_composite, t = findFood(t, composite)
      this_veges, t = findFood(t, vege)
      #print(f"vege {this_veges}")
      this_protein, t = findFood(t, protein)
      #print(f"protein {this_protein}")
      this_carb, t = findFood(t, carbs)
      #print(f"carbs {this_carb}")

      weight = (float)(iu[0])
      unit = findUnits(t)
      tmp = calcCarbs(this_composite, this_veges, this_protein, this_carb, weight, unit)
     # print("total carb: "+str(tmp))
      totalCarb += tmp
  
  return totalCarb/MAX_CARBS

def convertToTimeBucket(cur):
  hour = cur.hour
  min = cur.minute
  bucket = (hour * 60 + min)/FREQ
  total = 24 * 60 / FREQ
  
  return int(bucket)

def fill(data, feature, start, end, bucket, bucket_basal):
  period = int(24*60/FREQ)
  while start < end:
    #first copy bucket till the end
    num = min(end - start, period - bucket)
    #print(f"{start} {num} {bucket}-{bucket+num}")
    #data.loc[start:start+num, feature] = bucket_basal[bucket:bucket+num]
    data.iloc[start:start+num, data.columns.get_loc(feature)] = bucket_basal[bucket:bucket+num]
    #print(data.iloc[start:start+num].basal)
    #then copy 0, bucket-1
    start = start + num
    bucket = (bucket + num)%period

  return data[feature]

def fillBasal(data, first_pos, last_pos, days, feature='basal', bucket='bucket'):
  period = int(24*60/FREQ)
  t = []
  for day in range(1, days+1):
    t.append(data.shift(-day*period))

  bucket_basal = np.full(period, 0.0)
  for pos in range(first_pos, first_pos+period):
    mp = {}
    bt = int(data[bucket][pos])
    for i in range(0, days):
      if t[i]['basal'][pos] in mp:
        mp[t[i]['basal'][pos]] += 1
      else:
        mp[t[i]['basal'][pos]] = 1

   # print(f"bt {bt}")
   # print(f"mp {mp}")
    bucket_basal[bt] = max(mp, key=mp.get)

  data[feature] = fill(data, feature, 0, first_pos, data[bucket][0], bucket_basal)
  if last_pos < len(data)-1:
    data[feature] = fill(data, feature, last_pos+1, len(data)-1, data[bucket][last_pos+1], bucket_basal)

  return data

# 0.014 min^-1
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7968367/
def calcEffectiveBolus(df, col):
  nonzero = df[df[col] != 0].index
  for i in nonzero:
    j = i+1
    start = df[col][i]
   # print(f"initial {i} {start}")
    while j<len(df) and df[col][j]==0:
      effective = start - (j-i)*FREQ*BOLUS_RATE
      if effective>0:
        df.loc[j, col] = effective
       # print(f"{j} {effective}")
      else:
        break
      j += 1
  return df

#https://github.com/openaps/oref0/blob/master/lib/iob/calculate.js#L83
def calcEffectiveBolusAgain(df, col):
  end = 180 # 3 hours
  peak = 75
  tau = peak * (1 - peak / end) / (1 - 2 * peak / end);  # time constant of exponential decay
  a = 2 * tau / end;                                     # rise time factor
  S = 1 / (1 - a + (1 + a) * math.exp(-end / tau));      # auxiliary scale factor

      #  iobContrib = treatment.insulin * (1 - S * (1 - a) * ((Math.pow(minsAgo, 2) / (tau * end * (1 - a)) - minsAgo / tau - 1) * Math.exp(-minsAgo / tau) + 1));

  nonzero = df[df[col] != 0].index
  for i in nonzero:
    j = i+1
    start = df[col][i]
    print(f"initial {i} {start}")
    while j<len(df) and df[col][j]==0:
      minsAgo = j-i
      activityContrib = start * (S / math.pow(tau, 2)) * minsAgo * (1 - minsAgo / end) * math.exp(-minsAgo / tau)
      effective = start * (1 - S * (1 - a) * ((math.pow(minsAgo, 2) / (tau * end * (1 - a)) - minsAgo / tau - 1) * math.exp(-minsAgo / tau) + 1))
      if effective>0:
        df[col][j] = effective
        print(f"{j} {effective}")
      else:
        break
      j += 1
  return df

meds = {'acarbose', 'metformin', 'dapagliflozin', 'liraglutide', 'voglibose', 'pioglitazone', 'sitagliptin', 'gliclazide', 'canagliflozin',
        'repaglinide', 'glimepiride','empagliflozin', 'linagliptin', 'gliquidone'}

def findMeds(b):
  b = str(b).lower()
  if b=='nan':
    return
  units = ['mg', 'g']
  sum = 0

  for m in meds:
    if b.find(m) != -1:
      #sitagliptinphosphate/metforminhydrochloride tablets 50 mg / 500 mg
      pattern = m+".*/"
      if re.findall(pattern, b):
        for u in units:
          pattern = " +(\d+\.\d+|\d+) *"+u+" */"
          iu = re.findall(pattern, b)
          if iu:
            num = (float)(iu[0])
            if u == 'mg':
              num /= 1000
            sum += num
            break
          continue
      else:
        pattern = "/"+m+".* "
        if re.findall(pattern, b):
          for u in units:
            pattern = " +(\d+\.\d+|\d+) *"+u+" */"
            iu = re.findall(pattern, b)
            if iu:
              num = (float)(iu[0])
              if u == 'mg':
                num /= 1000
              sum += num
              break
          continue
      for u in units:
        pattern = m+" +(\d+\.\d+|\d+) *"+u
        iu = re.findall(pattern, b)
        if iu:
          #print(f"iu {iu}")
          #print(f"iu {iu[0]}")
          #num = (float)(iu[0].split()[1])
          num = (float)(iu[0])
          if u == 'mg':
            num /= 1000
          sum += num
          break

  if sum == 0:
    print(f"~find any meds {b}")
    for m in meds:
      if b.find(m) != -1:
        print(f"at least has {m}")
        for u in units:
          pattern = m+" +[0-9]+ *"+u
          iu = re.findall(pattern, b)
          print(f"find {iu} too")
          if iu:
       #   print(f"iu {iu[0]}")
            num = (float)(iu[0].split()[1])
            print(f"num {num}")
            if u == 'mg':
              num /= 1000
            sum += num
            break
    return 0.0
  return sum

def readParseData(file, features):
  data=pd.read_excel(file)
  data.rename(columns={"CGM (mg / dl)":"glucose_level", "Dietary intake":"meal", "CSII - bolus insulin (Novolin R, IU)":"bolus", "CSII - basal insulin (Novolin R, IU / H)":"basal"
        , "Insulin dose - s.c.":"bolus_pen", "Insulin dose - i.v.":"bolus_iv", "Non-insulin hypoglycemic agents":"meds"}, inplace=True)

  if "meds" in features:
    data['meds'] = data['meds'].apply(findMeds)
    
  data['bolus_long_pen'] = data['bolus_pen'].apply(findLongTermInsulin)
  data['bolus_short_pen'] = data['bolus_pen'].apply(findShortTermInsulin)
  data['bolus_short_iv'] = data['bolus_iv'].apply(findShortTermInsulin)
  data['bolus_long_iv'] = data['bolus_iv'].apply(findLongTermInsulin)

  data['bolus'] = pd.to_numeric(data['bolus'], downcast='float', errors='coerce')
  data['bolus'].fillna(0, inplace=True)

  #check if these columns are merged appropriately (TODO)
  data['bolus'] = data.loc[:, ['bolus', 'bolus_short_pen', 'bolus_short_iv'] ].sum(axis=1)
  #data = calcEffectiveBolus(data, 'bolus')
  data['bolus_long'] = data.loc[:, ['bolus_long_pen', 'bolus_long_iv'] ].sum(axis=1)
  #data = calcEffectiveBolus(data, 'bolus_long')
  data['carbs'] = data['meal'].apply(calculateCarbs)
  data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y %H:%M:%S')

  data.index = data['Date']
  data['bucket'] = data['Date']
  data.sort_index(inplace=True)

  # Note that basal must fillna AFTER series is aligned.
  #print(data['basal'])
  data.basal = pd.to_numeric(data['basal'], downcast='float', errors='coerce')
  nonzero = data['basal'].fillna(0).to_numpy().nonzero()[0]
  data['bucket'] = data.index.to_pydatetime()
  data['bucket'] = data['bucket'].apply(convertToTimeBucket)
  totalT = 24 * 60 / FREQ
  data['cos']  =  np.cos(2 * np.pi * data['bucket'] / totalT)

  if nonzero.any():
    first_pos = nonzero[0]
    last_pos = nonzero[len(nonzero)-1]

    selected_categories = data.Date[::48].copy()

    delta = data.index[last_pos] - data.index[first_pos]

    if delta.days > 0:
      data = fillBasal(data, first_pos, last_pos, delta.days)

  data, _ = align_series(data)

  return data[features]

PRETRAIN_SPLIT = 1.0

def loadPreTrainingData(ids, data_path, features):
  train_map = {}
  test_map = {}
  sum = 0

  for file in os.listdir(data_path):

    id=file.split('_')[0]
    if int(id) not in ids:
      continue

    data = readParseData(f'{data_path}/{file}', features)
    sum += len(data)
    split = int(len(data)*PRETRAIN_SPLIT)
    train = data[:split].copy(deep=True)
    test = data[split:].copy(deep=True)
    train_map[id] = train
    test_map[id] = test

 # print(f"total {sum} pretraining")
  return (train_map, test_map, sum)
  #print(data.loc[data['carbs']>0, :])

def lag_target(df, lag=FUTURE_STEPS, target_col='glucose_level', delta=True):
    # Get delta
    if delta==True:
      target_df = pd.concat([-df[target_col].diff(periods=-i) for i in range(1, lag + 1)], axis=1).dropna(axis=0)
      df = df.iloc[:len(target_df)]
    else:
      target_df = df

    return df, target_df
    
def loadIndividualLearningData(ids, data_path, datatype, features):
  train_map = {}
  valid_map = {}
  test_map = {}
  sum = 0

  for file in os.listdir(data_path):
    id=file.split('_')[0]
    dataId=file.split('_')[1]
    if int(id) not in ids:
      continue

    data = readParseData(f'{data_path}/{file}', features)
    sum += len(data)
    if datatype=='T2D':
      if id == 2069: # subject 2069 has 3 periods
        if dataId=='0' or dataId=='1':
          if id in train_map:
            train_map[id] = pd.concat([train_map[id], data])
          else:
            train_map[id] = data
        else:
          test_map[id] = data
      else:
        if dataId=='0':
          train_map[id] = data
        else:
          test_map[id] = data
    else: #T1D
      if dataId=='0' or dataId=='1':
        if id in train_map:
          train_map[id] = pd.concat([train_map[id], data])
        else:
          train_map[id] = data
      else:
        test_map[id] = data

  #print(f"total {sum} learning")
  return (train_map, test_map, sum)

def adjust_learning_rate(optimizer, epoch, name):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
   # if args.lradj == 'type1':
    learning_dict = {
      'SugarNet': 0.001,
      'FreTS': 0.0001,
      'DLinear': 0.0001,
      'iTransformer': 0.0001,
      'FGN': 0.0001,
      'TimeMixer': 0.0001,
      'FiLM': 0.0001,
      'PatchTST': 0.0001
    }
    lr_adjust = {epoch: learning_dict[name] * (0.5 ** ((epoch - 1) // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = val_loss
        if score < self.best_score:
          self.best_score = score
          self.counter = 0
        else:
          if score<self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')