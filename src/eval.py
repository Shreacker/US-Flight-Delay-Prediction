import numpy as np
import pandas as pd
import os, sys
import gc
import shap
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

from utils.data.dataset import Dataset, to_dataset
from utils.utilities import *
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

PLOT_PATH = Path('./src/plot')
DATA_PATH = Path('../data/processed')
CHKP_PATH = Path('./src/checkpoints')

'''
--LOAD DATA--
'''
print('LOADING DATA...')
train_df = pd.read_csv(DATA_PATH / 'train.csv', low_memory=False)
val_df = pd.read_csv(DATA_PATH / 'val.csv', low_memory=False)
test_df = pd.read_csv(DATA_PATH / 'test.csv', low_memory=False)
print('DONE LOADING.')
print('--------------------------------')

train_ds = to_dataset(train_df, 'arr_delay')
val_ds = to_dataset(val_df, 'arr_delay')
test_ds = to_dataset(test_df, 'arr_delay')

del train_df, val_df, test_df
gc.collect()

print('LOADING TRANSFORMER...')
with open(CHKP_PATH / 'transformer.pkl', 'rb') as f:
    tf = pickle.load(f)

print('LOADING WEIGHTS...')
with open(CHKP_PATH / 'weights.pkl', 'rb') as f:
    weights = pickle.load(f)

'''
--EVALUATION--
'''
print('--EVALUATION--')

X_sample = val_ds.x.sample(
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
    train_ds.x,
    train_ds.y,
    sample_weight=weights
)

y_pred_tf = model.predict(val_ds.x)
y_pred = tf.inverse_transform(y_pred_tf).ravel()
y_true = tf.inverse_transform(val_ds.y.to_numpy()).ravel()

# Metrics
print('\n||RESULTS OF LGBM||')
print('--ROOT MEAN SQUARED ERROR--')
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.2f}')

print('--MEAN ABSOLUTE ERROR')
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae:.2f}')

# ---Feature Importance---
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
    train_ds.x,
    train_ds.y,
    sample_weight=weights,
    eval_set=[(val_ds.x, val_ds.y)],
    verbose=100
)

y_pred_tf = model.predict(val_ds.x)
y_pred = tf.inverse_transform(y_pred_tf).ravel()
y_true = tf.inverse_transform(val_ds.y.to_numpy()).ravel()

# Metrics
print('\n||RESULTS OF XGBOOST||')
print('--ROOT MEAN SQUARED ERROR--')
rmse = root_mean_squared_error(y_true, y_pred)
print(f'RMSE: {rmse:.2f}')

print('--MEAN ABSOLUTE ERROR')
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae:.2f}')

# ---Feature Importance---
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
print(f'RMSE: {rmse_naive:.2f}')
print(f'MAE: {mae_naive:.2f}')

print('\n--MEAN PREDICTION (y_pred=mean(y_true)--')
y_pred_mean = np.full(y_true.shape[0], y_true.mean())
rmse_mean = root_mean_squared_error(y_true, y_pred_mean)
mae_mean = mean_absolute_error(y_true, y_pred_mean)
print(f'RMSE: {rmse_mean:.2f}')
print(f'MAE: {mae_mean:.2f}')
