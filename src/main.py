import numpy as np
import pandas as pd
import os, sys
import gc
import shap
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from utils.data.dataset import Dataset, to_dataset
from utils.utilities import *
from preprocessing.balancer import Transformer
from preprocessing.encoder import OneHotEncoder, RollingTargetEncoder
from preprocessing.normalizer import StandardScaler, RobustScaler

PLOT_PATH = Path('./src/plot')
OUTPUT_DIR = Path('../data/processed')
RAW_PATH = Path('./data/engineered')
CHKP_PATH = Path('./src/checkpoints')

'''
--LOAD DATA--
'''
print('LOADING DATA...')
train_df = pd.read_csv(RAW_PATH / 'train_engineered.csv', low_memory=False)
val_df = pd.read_csv(RAW_PATH / 'val_engineered.csv', low_memory=False)
test_df = pd.read_csv(RAW_PATH / 'test_engineered.csv', low_memory=False)
print('DONE LOADING.')
print('--------------------------------')

'''
--PREPROCESSING PIPELINE--
'''
print('--PREPROCESSING PIPELINE--')

train_ds = to_dataset(train_df, 'arr_delay')
val_ds = to_dataset(val_df, 'arr_delay')
test_ds = to_dataset(test_df, 'arr_delay')

del train_df, val_df, test_df
gc.collect()

# Feature Selection
print('FEATURE SELECTION:')
corr_drop = [
    'crs_arr_time',
    'dew_point_2m'
]
print(f'Droping {corr_drop}...')
train_ds.x = train_ds.x.drop(columns=corr_drop, errors='ignore')
val_ds.x = val_ds.x.drop(columns=corr_drop, errors='ignore')
test_ds.x = test_ds.x.drop(columns=corr_drop, errors='ignore')

# One-hot Encoding
print('ONE-HOT ENCODING...')
low_card, high_card = analyze_cardinality(train_ds.x)
ohe = OneHotEncoder(drop_first=False)
train_ohe = ohe.fit_transform(train_ds, cols=low_card)
val_ohe = ohe.transform(val_ds)
test_ohe = ohe.transform(test_ds)

# Rolling Target Encoding
print('ROLLING TARGET ENCODING...')
te = RollingTargetEncoder()
train_te = te.fit_transform(train_ohe, cols=high_card, smoothing=10.)
val_te = te.transform(val_ohe, smoothing=10.)
test_te = te.transform(test_ohe, smoothing=10.)

# Transform Target
print('TRANSFORMING TARGET...')
tf = Transformer()
tf.fit(train_te)
train_bal, weights = tf.transform(train_te, qbins=30, retweights=True)
val_bal = tf.transform(val_te)
test_bal = tf.transform(test_te)

# # Normalizer
# print('NORMALIZING DATA...')
# norm = RobustScaler()
# norm.fit(train_bal)
# train_norm = norm.transform(train_bal)
# val_norm = norm.transform(val_bal)
# test_norm = norm.transform(test_bal)

print('DONE PREPROCESSING.')
print('--------------------------------')

print('Length of train, val, test')
print(len(train_bal), len(val_bal), len(test_bal))
print('Number of features:', train_bal.x.columns.size)

# Save Dataset
print('SAVING DATASET...')
os.makedirs(CHKP_PATH, exist_ok=True)
with open(CHKP_PATH / 'transformer.pkl', 'wb') as f:
    pickle.dump(tf, f)

with open(CHKP_PATH / 'weights.pkl', 'wb') as f:
    pickle.dump(weights, f)

train_final = pd.concat([train_bal.x.reset_index(drop=True), train_bal.y], axis=1)
val_final = pd.concat([val_bal.x.reset_index(drop=True), val_bal.y], axis=1)
test_final = pd.concat([test_bal.x.reset_index(drop=True), test_bal.y], axis=1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
train_final.to_csv(OUTPUT_DIR / 'train.csv', index=False)
val_final.to_csv(OUTPUT_DIR / 'val.csv', index=False)
test_final.to_csv(OUTPUT_DIR / 'test.csv', index=False)
print('SAVED DATASET SUCCESSFULLY.')