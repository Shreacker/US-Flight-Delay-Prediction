import numpy as np
import pandas as pd
import os, sys
import gc
import shap
from pathlib import Path
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from utils.data.dataset import Dataset, to_dataset
from utils.utilities import *
from preprocessing.balancer import GroupCat, Transformer
from preprocessing.filler import fill_missing
from preprocessing.encoder import OHE, TargetEncoder
from preprocessing.filter import quantile_boundary
from preprocessing.normalizer import StandardScaler, RobustScaler

leakage_cols = [
    'dep_time', 'dep_delay_new',
    'dep_delay',

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

PLOT_PATH = Path('./src/plot')

'''
--LOAD DATA--
'''
print('LOADING DATA...')
RAW_PATH = Path('./data/raw/post_feat_engineer')
df = pd.read_csv(RAW_PATH / 'flight_data_2425_fe.csv', low_memory=False)
print('DONE LOADING.')
print('--------------------------------')

'''
--PREPROCESSING PIPELINE--
'''
print('--PREPROCESSING PIPELINE--')
# Basic Filter
print('BASIC FILTERING...')
df = drop_missing(df, thresh=0.3)
df = df.dropna(subset=['arr_delay'])
# df = df.dropna(subset=['dep_delay'])
df = df.drop(columns=leakage_cols, errors='ignore')
df = df.drop(columns=redundant_cols, errors='ignore')

df = obj2cat(df)

ds = to_dataset(df, 'arr_delay')

# Handle Outliers
print('HANDLING OUTLIERS...')
ds = quantile_boundary(ds=ds, col='arr_delay', lower=0.001, upper=0.999)
# ds = quantile_boundary(ds=ds, col='dep_delay', lower=0.001, upper=0.999)
ds = ds[~(ds.x['distance'] < 15)]
ds = ds[~((ds.x['distance'] / ds.x['crs_elapsed_time'] * 60 > 400) & (ds.x['crs_elapsed_time'] < 30))]

for col in ds.x.select_dtypes(include=['object', 'string']).columns:
    ds.x[col] = ds.x[col].astype('category')

# Handle Duplicates
print('HANDLING EXACT DUPLICATES...')
df.drop_duplicates(inplace=True)

# Fill Missing
print('FILLING MISSING...')
group = ['month', 'op_unique_carrier']
ds = fill_missing(ds, group=group)

# Group Rare Categories
print('GROUPING RARE CATEGORIES')
cat_cols = ['op_unique_carrier', 'origin', 'dest']
min_pct = [0.02, 0.002, 0.002] # REASON WHY CHOOSE THESE NUMBERS

gr = GroupCat()
gr.fit(ds, cat_cols, min_pct=min_pct)
ds = gr.transform(ds)

# Feature Selection
print('FEATURE SELECTION:')
corr_drop = [
    'crs_arr_time',
    'dew_point_2m'
]
print(f'Droping {corr_drop}...')
ds.x = ds.x.drop(columns=corr_drop, errors='ignore')

# One-hot Encoding
print('ONE-HOT ENCODING...')
low_card, high_card = analyze_cardinality(ds.x)
ds = OHE(ds, columns=low_card, drop_first=False)

# Split Data
print('SPLITTING DATA...')
train_ds, val_ds, test_ds = train_val_test_split(ds, random_state=21)

# Target Encoding
print('TARGET ENCODING...')
enc = TargetEncoder()
enc.fit(train_ds, high_card, smoothing=10.)

train_enc = enc.transform(train_ds)
val_enc = enc.transform(val_ds)
test_enc = enc.transform(test_ds)

# Transform Target
print('TRANSFORMING TARGET...')
tf = Transformer()
tf.fit(train_enc)
train_bal, weights = tf.transform(train_enc, retweights=True)
val_bal = tf.transform(val_enc)
test_bal = tf.transform(test_enc)

# # Normalizer
# print('NORMALIZING DATA...')
# norm = RobustScaler()
# norm.fit(train_bal)
# train_norm = norm.transform(train_bal)
# val_norm = norm.transform(val_bal)
# test_norm = norm.transform(test_bal)

print('DONE PREPROCESSING.')
print('--------------------------------')

# Save Dataset
print('SAVING DATASET...')
train_final = pd.concat([train_bal.x.reset_index(drop=True), train_bal.y], axis=1)
val_final = pd.concat([val_bal.x.reset_index(drop=True), val_bal.y], axis=1)
test_final = pd.concat([test_bal.x.reset_index(drop=True), test_bal.y], axis=1)

# OUTPUT_DIR = Path("../data/processed")
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# train_final.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
# val_final.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
# test_final.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
# print('SAVED DATASET SUCCESSFULLY.')
# print('--------------------------------')

del df, ds, train_enc, val_enc, test_enc
gc.collect()

print('--EVALUATION--')

X_sample = val_bal.x.sample(
    10000,
    random_state=21
)

# LGBM Regressor
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
    train_bal.x,
    train_bal.y,
    sample_weight=weights
)

y_pred_tf = model.predict(val_bal.x)
y_pred = tf.inverse_transform(y_pred_tf).ravel()
y_true = val_ds.y.to_numpy()

# Metrics
print('\n||RESULTS OF LGBM||')
print('--ROOT MEAN SQUARED ERROR--')
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.2f}')

print('--MEAN ABSOLUTE ERROR')
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae:.2f}')

print('--R2 SCORE--')
r2 = r2_score(y_true, y_pred)
print(f'R2: {r2:.2f}')

# Feature Importance
print('--FEATURE IMPORTANCE--')
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH / 'LGBM_fi_bar.png', dpi=300, bbox_inches='tight')
plt.close()

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH / 'LGBM_fi_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# XGBoost Regressor
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=2000,
    max_depth=8,
    min_child_weight=50,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

model.fit(
    train_bal.x,
    train_bal.y,
    sample_weight=weights,
    eval_set=[(val_bal.x, val_bal.y)],
    verbose=100
)

y_pred_tf = model.predict(val_bal.x)
y_pred = tf.inverse_transform(y_pred_tf).ravel()
y_true = val_ds.y.to_numpy()

# Metrics
print('\n||RESULTS OF XGBOOST||')
print('--ROOT MEAN SQUARED ERROR--')
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.2f}')

print('--MEAN ABSOLUTE ERROR')
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae:.2f}')

print('--R2 SCORE--')
r2 = r2_score(y_true, y_pred)
print(f'R2: {r2:.2f}')

# Feature Importance
print('--FEATURE IMPORTANCE--')
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH / 'XGB_fi_bar.png', dpi=300, bbox_inches='tight')
plt.close()

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig(PLOT_PATH / 'XGB_fi_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# BASELINE
print('\n||COMPARE TO BASELINE||')
print('\n--NAIVE PREDICTION (y_pred=0)--')
y_pred_naive = np.zeros(y_true.shape[0], dtype='float64')
rmse_naive = root_mean_squared_error(y_true, y_pred_naive)
mae_naive = mean_absolute_error(y_true, y_pred_naive)
r2_naive = r2_score(y_true, y_pred_naive)
print(f'RMSE: {rmse_naive:.2f}')
print(f'MAE: {mae_naive:.2f}')
print(f'R2: {r2_naive:.2f}')

print('\n--MEAN PREDICTION (y_pred=mean(y_true)--')
y_pred_mean = np.full(y_true.shape[0], y_true.mean())
rmse_mean = root_mean_squared_error(y_true, y_pred_mean)
mae_mean = mean_absolute_error(y_true, y_pred_mean)
r2_mean = r2_score(y_true, y_pred_mean)
print(f'RMSE: {rmse_mean:.2f}')
print(f'MAE: {mae_mean:.2f}')
print(f'R2: {r2_mean:.2f}')

