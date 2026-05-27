import numpy as np
import pandas as pd
import os, sys
from pathlib import Path
from lightgbm import LGBMRegressor

from utils.data.dataset import Dataset, to_dataset
from utils.utilities import *
from preprocessing.balancer import GroupCat, Transformer
from preprocessing.filler import fill_missing
from preprocessing.encoder import OHE, TargetEncoder
from preprocessing.filter import quantile_boundary
from preprocessing.normalizer import StandardScaler

leakage_cols = [
    'dep_time', 'dep_delay', 'dep_delay_new',

    'arr_time',

    'actual_elapsed_time', 'air_time',

    'taxi_in', 'taxi_out',
    'wheels_on', 'wheels_off',

    'carrier_delay', 'weather_delay', 'nas_delay',
    'security_delay', 'late_aircraft_delay'
]

redundant_cols = [
    'origin_state_nm',
    'origin_city_name',
    'dest_state_nm',
    'dest_city_name',
    'op_carrier_fl_num',
    'cancellation_code',
    'cancelled',
    'diverted',
    'fl_date',
    'year',
    'crs_dep_time',
    'dep_datetime',
    'weather_hour',
    'datetime'
]

# --LOAD DATA AND DTYPE CONVERTER--
RAW_PATH = Path('../data/raw')

df_dt = pd.read_csv('../data/flight_data_2024_data_dictionary.csv')
df1 = pd.read_csv(RAW_PATH / 'flight_24.csv', low_memory=False)
df2 = pd.read_csv(RAW_PATH / 'flight_25.csv', low_memory=False)

dtypes = dict(zip(df_dt['column'], df_dt['dtype']))

check_schema(df1, df2)
df1 = dtype_converter(df1, dtypes)
df2 = dtype_converter(df2, dtypes)
df = pd.concat([df1, df2], axis=0)

'''
--DATA ENGINEERING--
'''


'''
--PREPROCESSING PIPELINE--
'''
# Basic Filter
df = drop_missing(df, thresh=0.3)
df = df.dropna(subset=['arr_delay'])
df = df.drop(columns=leakage_cols, errors='ignore')
df = df.drop(columns=redundant_cols, errors='ignore')

df = obj2cat(df)

ds = to_dataset(df, 'arr_delay')

# Handle Outliers
ds = quantile_boundary(ds, 'arr_delay', lower=0.001, upper=0.999)
ds = ds[~(ds.x['distance'] < 15)]
ds = ds[~((ds['distance'] / ds['crs_elapsed_time'] * 60 > 400) & (ds['crs_elapsed_time'] < 30))]

# Convert weather code
ds = encode_weather(ds, 'weather_code')

for col in ds.x.select_dtypes(include=['object', 'string']).columns:
    ds.x[col] = ds.x[col].astype('category')

# Fill Missing
group = ['month', 'op_unique_carrier']
ds = fill_missing(ds, group=group)

# Group Rare Categories
cat_cols = ['op_unique_carrier', 'origin', 'dest']
min_pct = [0.02, 0.002, 0.002] # REASON WHY CHOOSE THESE NUMBERS

gr = GroupCat()
gr.fit(ds, cat_cols, min_pct=min_pct)
ds = gr.transform(ds)

# Feature Selection
corr_drop = [
    'crs_arr_time',
    'dew_point_2m'
]
ds.x = ds.x.drop(columns=corr_drop, errors='ignore')

# One-hot Encoding
low_card, high_card = analyze_cardinality(ds.x)
ds = OHE(ds, columns=low_card, drop_first=False)

# Split Data
train_ds, val_ds, test_ds = train_val_test_split(ds, random_state=21)

# Target Encoding
enc = TargetEncoder()
enc.fit(train_ds, high_card, smoothing=10.)

train_enc = enc.transform(train_ds)
val_enc = enc.transform(val_ds)
test_enc = enc.transform(test_ds)

# Balancer
tf = Transformer()
tf.fit(train_enc)
train_bal, weights = tf.transform(train_enc, retweights=True)
val_bal = tf.transform(val_enc)
test_bal = tf.transform(test_enc)

# # Normalizer
# stdz = StandardScaler()
# stdz.fit(train_bal)
# train_norm = stdz.transform(train_bal)
# val_norm = stdz.transform(val_bal)
# test_norm = stdz.transform(test_bal)

# Save Dataset
train_final = pd.concat([train_bal.x.reset_index(drop=True), train_bal.y], axis=1)
val_final = pd.concat([val_bal.x.reset_index(drop=True), val_bal.y], axis=1)
test_final = pd.concat([test_bal.x.reset_index(drop=True), test_bal.y], axis=1)

OUTPUT_DIR = Path("../data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_final.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_final.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_final.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    train_final.x,
    train_final.y,
    sample_weight=weights
)

y_pred_tf = model.predict(val_bal.x)
y_pred = tf.inverse_transform(y_pred_tf).ravel()
y_true = val_ds.y.to_numpy()

# Metrics
print('--MEAN SQUARED ERROR--')
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.2f}')

print('--MEAN ABSOLUTE ERROR')
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae:.2f}')

print('||COMPARE TO BASELINE||')
print('--NAIVE PREDICTION (y_pred=0)--')
y_pred_naive = np.zeros(y_true.shape[0], dtype='float64')
rmse_naive = root_mean_squared_error(y_true, y_pred_naive)
mae_naive = mean_absolute_error(y_true, y_pred_naive)
print(f'RMSE: {rmse_naive:.2f}')
print(f'MAE: {mae_naive:.2f}')