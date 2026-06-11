import numpy as np
import pandas as pd

from utils.data.dataset import Dataset
from utils.utilities import *

class FeatureEngineer:
    def __init__(
            self,
            congestion_col: str = 'origin_congestion',
            wind_col: str = 'wind_speed_10m',
    ):
        self.congestion_col = congestion_col
        self.wind_col = wind_col

    def fit(self, ds: Dataset) -> "FeatureEngineer":
        X, _ = ds[:]
        
        X = self._compute_static_features(X)
        self.congestion_threshold_ = X[self.congestion_col].quantile(0.75)
        self.wind_threshold_ = X[self.wind_col].quantile(0.75)
        return self

    def transform(self, ds: Dataset) -> Dataset:
        if not hasattr(self, 'congestion_threshold_'):
            raise RuntimeError('Call fit() before transform().')

        X, y = ds[:]
        X = self._compute_static_features(X)
        X = self._compute_threshold_features(X)
        X = self._compute_lag_features(X, y)
        return Dataset(X.reset_index(drop=True), y.reset_index(drop=True))

    def fit_transform(self, ds: Dataset) -> Dataset:
        return self.fit(ds).transform(ds)

    # ------------------------------------------------------------------

    def _compute_static_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Weather encoding
        X, new_col = encode_weather(X, 'weather_code')
        X[new_col] = X[new_col].astype('category')
        X = pd.get_dummies(X, columns=[new_col])

        # Season
        month = X['month'].values
        X['season'] = np.select(
            [np.isin(month, [3, 4, 5]),
             np.isin(month, [6, 7, 8]),
             np.isin(month, [9, 10, 11])],
            [1, 2, 3],
            default=0
        ).astype(np.int8)

        # Time of day
        hour = X['hour'].values
        X['time_of_day'] = np.select(
            [(hour >= 5)  & (hour < 12),
             (hour >= 12) & (hour < 17),
             (hour >= 17) & (hour < 21)],
            [1, 2, 3],
            default=0
        ).astype(np.int8)

        # Weather severity
        is_rush  = X['hour'].isin([7, 8, 9, 10, 17, 18])
        is_storm = X.get('weather_Storm', 0) == 1
        is_snow  = X.get('weather_Snow_Ice', 0) == 1
        is_rain  = X.get('weather_Rain', 0) == 1

        wind_speed  = X.get('wind_speed_10m', 0)
        cloud_cover = X.get('cloud_cover', 0)
        temperature = X.get('temperature_2m', 0)
        humidity    = X.get('relative_humidity_2m', 0)

        X['weather_severity_score'] = np.select(
            [is_storm | ((is_rain | is_snow) & (wind_speed > 30) & (temperature <= 0)),
             is_snow & (temperature < -10) & (humidity > 90),
             is_rain,
             cloud_cover > 50],
            [4, 3, 2, 1],
            default=0
        )
        X['is_bad_weather']      = (X['weather_severity_score'] >= 2).astype('bool')
        X['rush_hour_x_weather'] = is_rush.astype('int8') * (X['weather_severity_score'] ** 2)

        # Wind / direction
        X['wind_exposure'] = X['wind_speed_10m'] * np.log1p(X['crs_elapsed_time'])
        X['wind_dir_sin']  = np.sin(np.radians(X['wind_direction_10m']))
        X['wind_dir_cos']  = np.cos(np.radians(X['wind_direction_10m']))

        # Cyclical hour
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

        # Congestion
        X['origin_congestion'] = (
            X.groupby(['origin', 'hour'])['origin']
            .transform('count')
            .astype('int32')
        )
        X['dest_congestion'] = (
            X.groupby(['dest', 'hour'])['dest']
            .transform('count')
            .astype('int32')
        )

        # Route risk
        X['weather_dist_risk'] = (X['weather_severity_score'] * X['distance']).astype('float32')
        X['wind_route_risk']   = (X['wind_speed_10m'] * X['distance']).astype('float32')
        X['expected_speed']    = (X['distance'] / (X['crs_elapsed_time'] + 1)).astype('float32')

        # Peak / weekend flags
        X['is_peak_hour'] = (
            ((X['hour'] >= 6) & (X['hour'] <= 11)) |
            ((X['hour'] >= 16) & (X['hour'] <= 19))
        ).astype('bool')
        X['is_weekend_rush'] = X['day_of_week'].isin([1, 4, 5, 7]).astype('bool')

        return X

    def _compute_threshold_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X['wind_risk'] = (X[self.wind_col] > self.wind_threshold_).astype('bool')

        X['severe_ops_risk'] = (
            (X['is_bad_weather'] == 1) &
            (X['is_peak_hour'] == 1) &
            (X[self.congestion_col] > self.congestion_threshold_)
        ).astype('bool')

        X['hub_pressure'] = (
            (X['is_peak_hour'] == 1) &
            (X[self.congestion_col] > self.congestion_threshold_)
        ).astype('bool')

        X['long_haul_weather_risk'] = (
            (X['distance'] > 1500) &
            (X['is_bad_weather'] == 1)
        ).astype('bool')

        X['weekend_peak_risk'] = (
            (X['is_weekend_rush'] == 1) &
            (X['is_peak_hour'] == 1) &
            (X[self.congestion_col] > self.congestion_threshold_)
        ).astype('bool')

        X['adverse_weather_interaction'] = (
            (X['wind_risk'] == 1) &
            (X['is_bad_weather'] == 1)
        ).astype('bool')

        return X

    def _compute_lag_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X['scheduled_dt'] = pd.to_datetime(
            X[['year', 'month', 'day_of_month', 'hour']]
            .rename(columns={'day_of_month': 'day'})
        )
        X['is_delayed'] = (y > 15).astype(int)

        hourly_stats = (
            X.groupby(['origin', 'scheduled_dt'])
            .agg(total_flights=('is_delayed', 'count'),
                 delayed_flights=('is_delayed', 'sum'))
            .reset_index()
        )
        hourly_stats['target_dt'] = hourly_stats['scheduled_dt'] + pd.DateOffset(hours=1)
        hourly_stats = hourly_stats.rename(columns={
            'total_flights': 'prev_hour_total',
            'delayed_flights': 'prev_hour_delayed'
        }).drop(columns=['scheduled_dt'])

        X = X.merge(
            hourly_stats,
            left_on=['origin', 'scheduled_dt'],
            right_on=['origin', 'target_dt'],
            how='left'
        )
        X = X.reset_index(drop=True)

        X['airport_operational_stress'] = (
            X['prev_hour_delayed'] / X['prev_hour_total']
        ).fillna(-1).astype('float32')

        X.drop(columns=['scheduled_dt', 'is_delayed', 'target_dt',
                        'prev_hour_total', 'prev_hour_delayed', 'year'],
               inplace=True, errors='ignore')
        
        return X