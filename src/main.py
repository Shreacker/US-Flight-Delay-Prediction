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
from preprocessing.balancer import Transformer
from preprocessing.encoder import OneHotEncoder, RollingTargetEncoder
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
RAW_PATH = Path('./data/engineered')
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

# Save Dataset
print('SAVING DATASET...')
train_final = pd.concat([train_bal.x.reset_index(drop=True), train_bal.y], axis=1)
val_final = pd.concat([val_bal.x.reset_index(drop=True), val_bal.y], axis=1)
test_final = pd.concat([test_bal.x.reset_index(drop=True), test_bal.y], axis=1)

OUTPUT_DIR = Path("../data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_final.to_csv(OUTPUT_DIR / 'train.csv', index=False)
val_final.to_csv(OUTPUT_DIR / 'val.csv', index=False)
test_final.to_csv(OUTPUT_DIR / 'test.csv', index=False)
print('SAVED DATASET SUCCESSFULLY.')
print('--------------------------------')

del train_df, val_df, test_df, train_ds, train_te, val_te, test_te
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
y_true = tf.inverse_transform(val_bal.y.to_numpy()).ravel()

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

# ---Feature Importance---
# print('--FEATURE IMPORTANCE--')
# explainer = shap.Explainer(model)
# shap_values = explainer(X_sample)

# shap.plots.bar(shap_values, show=False)
# plt.tight_layout()
# plt.savefig(PLOT_PATH / 'LGBM_fi_bar.png', dpi=300, bbox_inches='tight')
# plt.close()

# shap.plots.beeswarm(shap_values, show=False)
# plt.tight_layout()
# plt.savefig(PLOT_PATH / 'LGBM_fi_beeswarm.png', dpi=300, bbox_inches='tight')
# plt.close()

# XGBoost Regressor
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1300,
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

# ---Feature Importance---
# print('--FEATURE IMPORTANCE--')
# explainer = shap.Explainer(model)
# shap_values = explainer(X_sample)

# shap.plots.bar(shap_values, show=False)
# plt.tight_layout()
# plt.savefig(PLOT_PATH / 'XGB_fi_bar.png', dpi=300, bbox_inches='tight')
# plt.close()

# shap.plots.beeswarm(shap_values, show=False)
# plt.tight_layout()
# plt.savefig(PLOT_PATH / 'XGB_fi_beeswarm.png', dpi=300, bbox_inches='tight')
# plt.close()

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
