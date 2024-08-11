import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import glob
from utils.constants import FREQ, FEATURES, VERBOSE
#from utils.metrics import metric
from matplotlib.dates import DateFormatter, HourLocator
from datetime import datetime, timedelta
import re

plt.switch_backend('agg')

transferSplit = 5
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
  meals = data[['day', 'carbs']].groupby('day').sum()

  days_with_no_meals = meals[meals['carbs'] < 1]
  #if len(days_with_no_meals) > 0:
  #  print('no meals')
  #  print(days_with_no_meals)
  for day, _ in days_with_no_meals.items():
      data = data[data['day'] != day]

  meal_count = data[data['carbs']>0].groupby('day').count()
  one_meal_a_day = meal_count[meal_count==1]
  #if len(one_meal_a_day)>0:
  #  print('one meal a day')
  #  print(one_meal_a_day)
  return data

def downSampling(data, freq):
  data['newbucket'] = data['bucket']/3
  data['meals'] = data[['carbs']].groupby('newbucket').sum()
  data['mealtime'] = data[data['carbs']>0]['bucket']%3
  return data

def readDiaTrend(path, history, future, features, freq=5, includeBasal=True):
  # Too little food information
  ignore = [29]
  #basicp = [30, 31, 36, 38, 39, 42, 45, 46, 47, 49, 50, 51, 52]
  basicp = [30]
  transferp = [37, 53, 54]
  os.chdir(path)
  files = glob.glob("*.xlsx") + glob.glob("*.xls")
  patients = {}
  pretrain_map = {}
  train_map = {}
  test_map = {}

  total = 0
  train = 0
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
    
    if id not in basicp and id not in transferp:
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
        cgm = cgm.groupby('Date')['glucose_level'].mean().reset_index()
        cgm = cgm.sort_values(by='Date')
        cgm = bucketize(cgm, "Date")

      elif sh=='Bolus':
        bolus = pd.read_excel(fi, sheet_name=sh)
        bolus.rename(columns={"normal":"bolus", "date":"Date", "carbInput":"carbs"}, inplace=True)
        bolus = bucketize(bolus, "Date")

        dat = bolus.groupby(['bucket', 'day', 'bTime'])['Date'].mean().reset_index()
        val = bolus.groupby(['bucket', 'day', 'bTime'])[['bolus', 'carbs']].sum().reset_index()
        bolus = pd.merge(dat, val, on=['bucket', 'day', 'bTime'])
      else:
        basal = pd.read_excel(fi, sheet_name=sh)
        basal.rename(columns={"date":"Date", "rate":"basal"}, inplace=True)
        
        dur = basal.groupby('Date')['duration'].max().reset_index()
        
        basal = pd.merge(basal, dur, on=['Date', 'duration'], how="right")
        basal = basal[['Date','basal']].dropna()

        basal = basal.sort_values(by='Date')
        basal = basal.set_index('Date').resample(f'1T').ffill()
        basal = basal.resample(f'{FREQ}T').first().shift(-1).reset_index()
        
        basal = bucketize(basal, "Date")

    data = closest_join(cgm, bolus, basal=basal)

    #print(data.columns)

    #meal = data[data['carbs']==0]
    data = handleMeals(data)

    fewcgm = data[data['glucose_level']<30]
    if len(fewcgm)>0:
      print("fewcgm")
      print(fewcgm)

    data = data.sort_values(by='bTime').reset_index(drop=True)

    data = downSampling(data, 15)
    print("down.csv")

    threshold = timedelta(minutes=30)
    data['timediff'] = data['bTime'].diff()
    data['group'] = (data['timediff'] > threshold).cumsum()
    sets = [group for _, group in data.groupby('group')]

    # Print each set
    threshold = timedelta(minutes=5)
    sum=0
    
    for i, group in enumerate(sets, start = 1):
        if (len(group)<1.5*(history+future)):
            continue

        group['glucose_level'] =group['glucose_level'].interpolate(method='linear', limit_direction='forward')
        
        group['Date'] = group['bTime']

        if 'basal' in group.columns:
          group['basal'] = group['basal'].ffill()
        
        y = group[group['carbs']>0]
        hasCarbs += len(y)

        patients[id].append(group)
        sum += len(group)

    total += sum
    
    #These are transferrec learnt and tested patients
    running = 0
    if (includeBasal and id in transferp) or (not includeBasal and id % transferSplit==0):
      train_map[id] = []
      test_map[id] = []
      for i in range(len(patients[id]) - 1, -1, -1):
        #print(f"group {i}, {len(patients[id][i])}")
        running += len(patients[id][i])
        #print(running)
        if running/sum >= 0.25:
          break
      train_map[id] = patients[id][:i]
      test_map[id] = patients[id][i:]  
    else:
      train += sum
      pretrain_map[id] = patients[id] 
    
    print(f"{id} {sum} {sum/(12*24)} days eat {hasCarbs}")
  
  print(f"total {total} pre-train {train} {train/total} transfer {total-train}")

  return pretrain_map, train_map, test_map