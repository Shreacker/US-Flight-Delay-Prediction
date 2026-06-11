import numpy as np
import pandas as pd
import sys
import tarfile
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.../')
from .data.dataset import Dataset

def check_schema(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
):
    cols_1 = set(df1.columns)
    cols_2 = set(df2.columns)

    diff_1 = list(cols_1 - cols_2)
    if diff_1:
        raise KeyError(f'Missing columns in df2: {diff_1}')
    
    diff_2 = list(cols_2 - cols_1)
    if diff_2:
        raise KeyError(f'Missing columns in df1: {diff_2}')

def dtype_converter(
        df: pd.DataFrame,
        dtype_dict: dict
    ):
    for col in df.columns:
        if col == 'fl_date':
            continue
        if df[col].isna().sum() != 0 and (dtype_dict[col] == 'int64' or dtype_dict[col] == 'int32'):
            df[col] = df[col].astype('Int64')
        else:
            df[col] = df[col].astype(dtype_dict[col])

    return df

def drop_missing(
        df: pd.DataFrame,
        thresh: float | int | None = 0.5,
):
    n = len(df)
    na_cols = []
    for idx, val in df.isna().sum().items():
        if val > (thresh * n):
            na_cols.append(idx)

    return df.drop(columns=na_cols, errors='ignore')

def obj2cat(dataF: pd.DataFrame):
    df = dataF.copy()

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype('category')

    return df

def bool2int(dataF: pd.DataFrame):
    df = dataF.copy()

    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype('int8')

    return df

def analyze_cardinality(
        df: pd.DataFrame | None = None,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input should be a DataFrame.')
    
    cat_cols = df.select_dtypes(include=['category', 'object', 'string']).columns
    low_card = []
    high_card = []

    n_unique = df.loc[:, cat_cols].nunique()
    mask = n_unique <= 20
    low_card.extend(mask[mask].index.tolist())
    high_card.extend(mask[~mask].index.tolist())

    return low_card, high_card

def train_val_test_split(
        ds: Dataset,
        val_size: float | None = 0.15,
        test_size: float | None = 0.15,
        **kwargs
):
    assert abs(val_size + test_size) < 1. + 1e-6, "Sum of validation size and test size is greater than 1"
    train_size = 1. - val_size - test_size

    X, y = ds[:]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - train_size),
        **kwargs
    )

    val_ratio = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio),
        **kwargs
    )

    return Dataset(X_train, y_train), Dataset(X_val, y_val), Dataset(X_test, y_test)

def time_split(
        ds: Dataset,
        time_series: pd.Series | None = None,
        val_size: float | None = None,
        test_size: float | None = 0.15,
        **kwargs
):
    if time_series is None:
        raise ValueError('time_series cannot be None!')
    assert time_series.index.isin(ds.x.index).all(), 'time_series index doesn\'t match ds.x index'

    if val_size is None and test_size is not None:
        train_size = 1. - test_size
    elif val_size is not None and test_size is None:
        test_size = val_size
        val_size = None
        train_size = 1. - test_size
    else:
        assert abs(val_size + test_size) < 1. + 1e-6, "Sum of validation size and test size is greater than 1"
        train_size = 1. - test_size - val_size

    if not pd.api.types.is_datetime64_any_dtype(time_series):
        time_series = pd.to_datetime(time_series, format='mixed')
    
    sorted_positions = np.argsort(time_series.values)

    n = len(ds)
    i_train = int(train_size * n)
    
    if val_size is None:
        train_ds = ds.iloc[sorted_positions[:i_train]]
        test_ds = ds.iloc[sorted_positions[i_train:]]

        return train_ds, test_ds

    else:
        i_val = int((train_size + val_size) * n)

        train_ds = ds.iloc[sorted_positions[:i_train]]
        val_ds = ds.iloc[sorted_positions[i_train:i_val]]
        test_ds = ds.iloc[sorted_positions[i_val:]]

        return train_ds, val_ds, test_ds

def entropy(x: pd.Series):
    if len(x) == 0:
        return 0
    
    _, counts = np.unique(x, return_counts=True)
    probs = counts / len(x)

    return -np.sum(probs * np.log2(probs + 1e-9))

def targz(path: Path):
    archive_path = path.with_name(path.stem + '.tar.gz')
    with tarfile.open(archive_path, 'w:gz') as tar:
        tar.add(path, arcname=path.name)

def encode_weather(dataF: pd.DataFrame, weather_col: str = None):
    if weather_col is None:
        raise KeyError('Weather code column is needed.')
    
    df = dataF.copy()
    df['weather'] = df[weather_col].apply(_map_wmo_code)
    df = df.drop(columns=weather_col)

    return df, 'weather'

def _map_wmo_code(code):
    if pd.isna(code): return 'Unknown'

    if code <= 3: 
        return 'Clear_Cloudy'     
    elif code in [45, 48]: 
        return 'Fog'              
    elif (code >= 51 and code <= 67) or (code in [80, 81, 82]): 
        return 'Rain'             
    elif (code >= 71 and code <= 77) or (code in [85, 86]): 
        return 'Snow_Ice'         
    elif code >= 95: 
        return 'Storm'            
    else: 
        return 'Other'