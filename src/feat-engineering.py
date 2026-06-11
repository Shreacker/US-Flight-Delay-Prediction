import numpy as np
import pandas as pd
import os, sys
import gc
from sklearn.preprocessing import PowerTransformer

sys.path.insert(0, '../')
from utils.utilities import *
from utils.data.dataset import Dataset, to_dataset
from preprocessing.filler import MissingValueImputer
from preprocessing.filter import QuantileBoundaryFilter
from preprocessing.balancer import GroupCat
from preprocessing.engineer import FeatureEngineer

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
    # 'year',
    'crs_dep_time',
    'dep_datetime',
    'weather_hour',
    'datetime'
]

'''
--LOAD DATA AND DTYPE CONVERTER--
'''
RAW_PATH = Path('./data/raw/post_weather_crawl')

print('LOADING DATA...')
df_dt = pd.read_csv('./data/flight_data_2024_data_dictionary.csv')
df1 = pd.read_csv(RAW_PATH / 'flight_24_weather.csv', low_memory=False)
df2 = pd.read_csv(RAW_PATH / 'flight_25_weather.csv', low_memory=False)
print('DONE LOADING.')
print('|----------------|')

dtypes = dict(zip(df_dt['column'], df_dt['dtype']))

check_schema(df1, df2)
print('CONVERTING DATA TYPES...')
df1 = dtype_converter(df1, dtypes)
df2 = dtype_converter(df2, dtypes)
df = pd.concat([df1, df2], axis=0, ignore_index=True)
print('DONE CONVERTING.')
print('|----------------|')

del df1, df2
gc.collect()

'''
DATA CLEANING
'''
print('---DATA CLEANING---')
df = df.reset_index(drop=True)
fl_date = df['fl_date']

# Basic Filter
print('BASIC FILTERING...')
df = drop_missing(df, thresh=0.3)
df = df.dropna(subset=['arr_delay'])
df = df.drop(columns=leakage_cols, errors='ignore')
df = df.drop(columns=redundant_cols, errors='ignore')

df = obj2cat(df)

# Handle Duplicates
print('HANDLING EXACT DUPLICATES...')
df.drop_duplicates(inplace=True)

ds = to_dataset(df, 'arr_delay')

# Split Data
print('SPLITTING DATA...')
fl_date = fl_date.loc[ds.x.index]
train_ds, val_ds, test_ds = time_split(
                                ds,
                                fl_date,
                                val_size=0.15,
                                test_size=0.15,
                                random_state=21
                            )

# Fill Missing
print('FILLING MISSING...')
group = ['month', 'op_unique_carrier']
filler = MissingValueImputer(group)
train_filled = filler.fit_transform(train_ds)
val_filled = filler.transform(val_ds)
test_filled = filler.transform(test_ds)

# Handle Outliers
def remove_outliers(ds: Dataset):
    ds = ds[~(ds.x['distance'] < 15)]
    ds = ds[ds.x['crs_elapsed_time'] > 0]
    ds = ds[ds.x['distance'] > 0]
    ds = ds[~((ds.x['distance'] / ds.x['crs_elapsed_time'] * 60 > 400) & (ds.x['crs_elapsed_time'] < 30))]
    return ds

print('HANDLING OUTLIERS...')
filter = QuantileBoundaryFilter('arr_delay', lower=0.001, upper=0.999)
train_filtered = filter.fit_transform(train_filled)
val_filtered = filter.transform(val_filled)
test_filtered = filter.transform(test_filled)

train_filtered = remove_outliers(train_filtered)
val_filtered = remove_outliers(val_filtered)
test_filtered = remove_outliers(test_filtered)

# Group Rare Categories
print('GROUPING RARE CATEGORIES...')
cat_cols = ['op_unique_carrier', 'origin', 'dest']
coverage = [0.98, 0.999, 0.999]

gr = GroupCat()
gr.fit(train_filtered, cat_cols, coverage=coverage)
train_grouped = gr.transform(train_filtered)
val_grouped = gr.transform(val_filtered)
test_grouped = gr.transform(test_filtered)

# Transform Skewed Columns
train_grouped.x['wind_speed_10m'] = np.log1p(train_grouped.x['wind_speed_10m'])
val_grouped.x['wind_speed_10m'] = np.log1p(val_grouped.x['wind_speed_10m'])
test_grouped.x['wind_speed_10m'] = np.log1p(test_grouped.x['wind_speed_10m'])

boxcox_cols = ['distance', 'crs_elapsed_time']
bc = PowerTransformer(method='box-cox', standardize=True)
train_grouped.x[boxcox_cols] = bc.fit_transform(train_grouped.x[boxcox_cols])
val_grouped.x[boxcox_cols] = bc.transform(val_grouped.x[boxcox_cols])
test_grouped.x[boxcox_cols] = bc.transform(test_grouped.x[boxcox_cols])

'''
--DATA ENGINEERING--
'''
print('|----------------|')
print('---FEATURE ENGINEERING---')
eng = FeatureEngineer()
train_engineered = eng.fit_transform(train_grouped)
val_engineered = eng.transform(val_grouped)
test_engineered = eng.transform(test_grouped)

train_final = pd.concat([train_engineered.x, train_engineered.y], axis=1)
val_final   = pd.concat([val_engineered.x, val_engineered.y], axis=1)
test_final  = pd.concat([test_engineered.x, test_engineered.y], axis=1)

print('Size of train, val, test:')
print(len(train_final), len(val_final), len(test_final))
print('Number of features:', train_final.columns.size - 1)
print('All features:')
print(train_final.columns)

print('|----------------|')
print('SAVING DATASET...')
OUTPUT_DIR = Path('./data/engineered')
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_final.to_csv(OUTPUT_DIR / 'train_engineered.csv', index=False)
val_final.to_csv(OUTPUT_DIR / 'val_engineered.csv', index=False)
test_final.to_csv(OUTPUT_DIR / 'test_engineered.csv', index=False)
print('SAVED SUCCESSFULLY!')