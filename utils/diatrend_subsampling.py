import os
from typing_extensions import dataclass_transform

import numpy as np
from pandas._libs.lib import is_bool
from pandas._libs.tslibs.period import IncompatibleFrequency
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
from utils.constants import FREQ, VERBOSE
#from utils.metrics import metric
from matplotlib.dates import DateFormatter, HourLocator
from datetime import datetime, timedelta
import re

plt.switch_backend('agg')

transferSplit = 0.85
transferId = 6
FREQ = 5

def convertToTimeBucket(cur):
  hour = cur.hour
  minu = cur.minute
  sec = cur.second
  bucket = (hour * 3600 + minu*60 + sec)/(5*60)
  return int(bucket)

def add_timedelta(row, freq):
    base_time = row['Date'].replace(hour=0, minute=0, second=0, microsecond=0)
    return base_time + timedelta(minutes=row['bucket'] * freq)

def bucketize(data, col):
  data['Date'] = pd.to_datetime(data['Date'])
  data = data.dropna(subset=['Date']).sort_values(by='Date')
  # Create the 'bucket' column by converting 'Date' to Python datetime objects
  data['bucket'] = data['Date'].apply(lambda x: x.to_pydatetime())
  data['bucket'] = data['bucket'].apply(convertToTimeBucket)

  # Apply the function to each row in the DataFrame
  data['bTime'] = data.apply(add_timedelta, axis=1, freq=FREQ)

  data['day'] = data['Date'].apply(lambda x: x.to_pydatetime().date())

  return data

def closest_join(df1, df2, basal=None): 
    result = pd.merge(df1, df2, on=['day', 'bucket', 'bTime'], how='left', suffixes=('_df1', '_df2'))
    result = result.fillna(0)
   
    if basal is not None:
      result = pd.merge(result, basal, on=['day', 'bucket', 'bTime'], suffixes=('_df2', '_df3'))
    
    return result

def handleMeals(data):
  #print(data[data['day']=='2021-11-15'])
  meals = data[data['carbs'] > 0].groupby('day')['carbs'].count()
  # Filter out the days with less than 2 non-zero carbs
  filtered_dates = meals[meals >= 2].index

  data = data[data['day'].isin(filtered_dates)]
  return data

def subsampling(data, scale):
  if scale>1:
    data['newbucket'] = data['bucket']//scale
    data['meals'] = data.groupby(['day', 'newbucket'])['carbs'].transform('sum')
    data['carbs'] = data['meals']
  result = []
  for i in range(scale):
    result.append(data[data['bucket']%scale==i])
  return result
  
def readDiaTrend(path, history, future, freq=5, features=None, includeBasal=True):
  # Too little food information
  ignore = [2, 29]
  basicp = [30, 31, 36, 38, 39, 42, 45, 46, 47, 50, 51, 52, 53]
  #basicp = [30]
  transferp = [37, 49, 54]
  #transferp = [37]
  os.chdir(path)
  files = glob.glob("*.xlsx") + glob.glob("*.xls")
  patients = {}
  pretrain_map = {}
  train_map = {}
  test_map = {}

  total = 0
  raw = 0
  transferTotal = 0
  transferTrain = 0
  files.sort()
  for fi in files:
    match = re.search(r'\d+', fi)
    if match:
      id = match.group()
      id = int(id)
      patients[id] = []
    else:
      print("No number found.")
      continue

    if id in ignore:
      continue

    if includeBasal and id not in basicp and id not in transferp:
      continue
    sheets = pd.ExcelFile(fi).sheet_names

    if includeBasal:
      if 'Basal' not in sheets:
        continue
    elif 'Basal' in sheets:
      continue

    basal=None
    hasCarbs = 0
    for sh in sheets:
      if sh=='CGM':
        cgm = pd.read_excel(fi, sheet_name=sh)
        cgm.rename(columns={"mg/dl":"glucose_level", "date":"Date"}, inplace=True)

        print(f"{id} raw cgm {len(cgm)}")
        raw += len(cgm)
        cgm = cgm.groupby('Date')['glucose_level'].mean().reset_index()
        cgm = cgm.sort_values(by='Date')
        cgm = bucketize(cgm, "Date")

      elif sh=='Bolus':
        bolus = pd.read_excel(fi, sheet_name=sh)
        bolus.rename(columns={"normal":"bolus", "date":"Date", "carbInput":"carbs"}, inplace=True)
        bolus = bucketize(bolus, "Date")

        #Handle multiple entries of bolus within same bucket
        dat = bolus.groupby(['bucket', 'day', 'bTime'])['Date'].mean().reset_index()
        val = bolus.groupby(['bucket', 'day', 'bTime'])[['bolus', 'carbs']].sum().reset_index()
        iob = None
        if 'insulinOnBoard' in bolus.columns:
          bolus.rename(columns={"insulinOnBoard":"iob"}, inplace=True)
          iob = bolus.groupby(['bucket', 'day', 'bTime'])['iob'].mean().reset_index()

        bolus = pd.merge(dat, val, on=['bucket', 'day', 'bTime'])
        if iob is not None:
          bolus = pd.merge(bolus, iob, on=['bucket', 'day', 'bTime'])

        #non_zero_meals_count = bolus[bolus['carbs'] > 0].groupby('day')['carbs'].count()
        # Filter out the days with less than 2 non-zero carbs
        #filtered_dates = non_zero_meals_count[non_zero_meals_count >= 3].index
        #print(f"before {len(bolus)}")
        #bolus = bolus[bolus['day'].isin(filtered_dates)]
        #print(f"after {len(bolus)}")
      else:
        basal = pd.read_excel(fi, sheet_name=sh)
        basal.rename(columns={"date":"Date", "rate":"basal"}, inplace=True)
        
        dur = basal.groupby('Date')['duration'].max().reset_index()
        
        basal = pd.merge(basal, dur, on=['Date', 'duration'], how="right")
        basal = basal[['Date','basal']].dropna()

        basal = basal.sort_values(by='Date')
        basal = basal.set_index('Date').resample(f'1T').ffill()
        basal = basal.resample(f'5T').first().shift(-1).reset_index()
        
        basal = bucketize(basal, "Date")

    data = closest_join(cgm, bolus, basal=basal)
   
    #print(f"before meals {len(data)}")
    data = handleMeals(data)
    #if len(data)==0:
    #  print("skip due to empty data set")
    #  continue
    #print(f"after meals {len(data)}")

    split = data.iloc[int(len(data)*transferSplit)]['Date_df1']
    # sampling per basic_freq*scale min
    scale = int(freq/FREQ)
    threshold = timedelta(minutes=20)
    
    data = subsampling(data, scale)
    if scale>1:
      threshold = timedelta(minutes=3*scale*FREQ)

    sum=0
    pret=0
    tran=0
    tes=0
    # Patient id has 3 groups based on buckets
    if (includeBasal and id in transferp) or (not includeBasal and id % transferId==0):
      train_map[id] = []
      test_map[id] = []
      testBatch = -1
      for i in range(3):
        if len(data[i][data[i]['Date_df1']==split])>0:
          testBatch = i
          break
      if testBatch==-1:
        print("don't find testbatch??")
    else:
      pretrain_map[id] = []

    for i in range(scale):
      thisbatch = data[i]
      thisbatch = thisbatch.sort_values(by='bTime').reset_index(drop=True)
      thisbatch['timediff'] = thisbatch['bTime'].diff()
      thisbatch['group'] = (thisbatch['timediff'] > threshold).cumsum()
      sets = [group for _, group in thisbatch.groupby('group')]

      for _, group in enumerate(sets, start = 1):
        if (len(group)<1.2*(history+future)):
            continue

        sum += len(group)
        group['glucose_level'] =group['glucose_level'].interpolate(method='linear', limit_direction='forward')
        
        group['Date'] = group['bTime']

        if (includeBasal and id in transferp) or (not includeBasal and id % transferId==0):
            if group.Date_df1.max()<=split:
              train_map[id].append(group)
              tran += len(group)
              #print(f"patient {id} train {len(group)}")
            else:
              test_map[id].append(group)
              tes += len(group)
        else:
            pretrain_map[id].append(group)
            pret += len(group)

    if (includeBasal and id in transferp) or (not includeBasal and id % transferId==0):
      transferTotal += sum
      transferTrain += tran
      print(f"pretrain {pret} transfer train {tran} testing {tes}")

    total += sum
  
  print(f"raw {raw} total {total} transfer total {transferTotal} train {transferTrain}")

  return pretrain_map, train_map, test_map