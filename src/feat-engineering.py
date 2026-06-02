import numpy as np
import pandas as pd
import os, sys
import gc
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, '../')
from utils.utilities import *
from utils.data.dataset import Dataset, to_dataset

'''
--LOAD DATA AND DTYPE CONVERTER--
'''
RAW_PATH = Path('./data/raw')

print('Loading all DataFrames...')
df_dt = pd.read_csv('./data/flight_data_2024_data_dictionary.csv')
df1 = pd.read_csv(RAW_PATH / 'flight_24.csv', low_memory=False)
df2 = pd.read_csv(RAW_PATH / 'flight_25.csv', low_memory=False)
print('Done loading.')
print('|----------------|')

dtypes = dict(zip(df_dt['column'], df_dt['dtype']))

check_schema(df1, df2)
print('Converting data types...')
df1 = dtype_converter(df1, dtypes)
df2 = dtype_converter(df2, dtypes)
df = pd.concat([df1, df2], axis=0)
print('Done converting.')
print('|----------------|')

del df1, df2
gc.collect()

'''
--DATA ENGINEERING--
'''
# Convert weather code
df, new_col = encode_weather(df, 'weather_code')
df[new_col] = df[new_col].astype('category')
df = pd.get_dummies(df, columns=[new_col])

# Season
month = df['month'].values
season_conditions = [
    np.isin(month, [3, 4, 5]),      # Spring
    np.isin(month, [6, 7, 8]),      # Summer
    np.isin(month, [9, 10, 11]),    # Fall
]
season_choices = [1, 2, 3] # Default = 0 (Winter)
df['season'] = np.select(season_conditions, season_choices, default=0).astype(np.int8)

# Time of day
hour = df['hour'].values
tod_conditions = [
    (hour >= 5) & (hour < 12),      # Morning
    (hour >= 12) & (hour < 17),     # Afternoon
    (hour >= 17) & (hour < 21),     # Evening
]
tod_choices = [1, 2, 3] # Default = 0 (Night)
df['time_of_day'] = np.select(tod_conditions, tod_choices, default=0).astype(np.int8)

rush_hours = [7, 8, 9, 17, 18]
df['is_rush_hour'] = df['hour'].isin(rush_hours).astype('bool')

# Weather
is_storm = df.get('weather_Storm', 0) == 1
is_snow = df.get('weather_Snow_Ice', 0) == 1
is_rain = df.get('weather_Rain', 0) == 1

wind_speed = df.get('wind_speed_10m', 0)
cloud_cover = df.get('cloud_cover', 0)
temperature = df.get('temperature_2m', 0)
humidity = df.get('relative_humidity_2m', 0)

cond_storm = is_storm | ((is_rain | is_snow) & (wind_speed > 30) & (temperature <= 0))
cond_snow_fog = (is_snow & (temperature < -10) & (humidity > 90))
cond_rain = is_rain
cond_cloudy = cloud_cover > 50

conditions = [cond_storm, cond_snow_fog, cond_rain, cond_cloudy]
choices = [4, 3, 2, 1]

df['weather_severity_score'] = np.select(conditions, choices, default=0)

df['is_bad_weather'] = (df['weather_severity_score'] >= 2).astype('bool')

# df['wind_cloud_interaction'] = df['wind_speed_10m'] * df['cloud_cover']
# df['wind_risk'] = np.log1p(df['wind_speed_10m'])

df['rush_hour_x_weather'] = df['is_rush_hour'].astype('int8') * (df['weather_severity_score'] ** 2)

df['wind_exposure'] = df['wind_speed_10m'] * np.log1p(df['crs_elapsed_time'])

df['wind_dir_sin'] = np.sin(np.radians(df['wind_direction_10m']))
df['wind_dir_cos'] = np.cos(np.radians(df['wind_direction_10m']))

# Previous Delay Influence
df['scheduled_dt'] = pd.to_datetime(
    df[['year', 'month', 'day_of_month', 'hour']].rename(columns={'day_of_month': 'day'})
)
df['is_delayed'] = (df['dep_delay'] > 15).astype(int)

hourly_stats = df.groupby(['origin', 'scheduled_dt']).agg(
    total_flights=('is_delayed', 'count'),
    delayed_flights=('is_delayed', 'sum')
).reset_index()

hourly_stats['target_dt'] = hourly_stats['scheduled_dt'] + pd.DateOffset(hours=1)
hourly_stats = hourly_stats.rename(columns={
    'total_flights': 'prev_hour_total',
    'delayed_flights': 'prev_hour_delayed'
}).drop(columns=['scheduled_dt'])

df = df.merge(
    hourly_stats,
    left_on=['origin', 'scheduled_dt'],
    right_on=['origin', 'target_dt'],
    how='left'
)

df['prev_hour_delay_rate'] = df['prev_hour_delayed'] / df['prev_hour_total']
df['prev_hour_delay_rate'] = df['prev_hour_delay_rate'].fillna(-1).astype('float32')

trash_cols = ['scheduled_dt', 'is_delayed', 'target_dt', 'prev_hour_total', 'prev_hour_delayed']
df.drop(columns=trash_cols, inplace=True, errors='ignore')

print(df.shape)
print(df.columns)

print('SAVING DATASET...')
OUTPUT_DIR = './data/raw/post_feat_engineer'
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(os.path.join(OUTPUT_DIR, 'flight_data_2425_fe.csv'), index=False)