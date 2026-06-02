import numpy as np
import pandas as pd

from utils.data.dataset import Dataset

class TargetEncoder:
    def __init__(self):
        self.mappings = {}

    def fit(
            self,
            ds: Dataset,
            cols: list | np.ndarray | None = None,
            smoothing: float | None = 0.0
    ):
        if cols is None or not cols:
            cols = ds.x.select_dtypes(include='category').columns.tolist()
        
        X = ds.x.copy()
        X['__target__'] = ds.y
        self.cols = cols

        for col in cols:
            mean = X['__target__'].mean()
            stats = X.groupby(col, observed=True)['__target__'].agg(['mean', 'count'])
            smooth = (stats['count'] * stats['mean'] + smoothing * mean) / (stats['count'] + smoothing)

            self.mappings[col] = (smooth, mean)

        return self
    
    def transform(
            self,
            ds: Dataset,
    ):
        ds_enc = ds.copy()

        for col in self.cols:
            smooth, mean = self.mappings[col]
            ds_enc = Dataset(ds_enc.x.copy(), ds_enc.y.copy())
            ds_enc.x[col] = ds_enc.x[col].astype('object').map(smooth).fillna(mean)

        return ds_enc

def OHE(
        ds: Dataset,
        columns: list | np.ndarray | None = None,
        drop_first: bool = False,
        **kwargs
):
    if columns is None:
        print('low_card list is empty.')
        return ds
    
    X, y = ds[:]
    X = pd.get_dummies(X, columns=columns, drop_first=drop_first, **kwargs)

    return Dataset(X, y)