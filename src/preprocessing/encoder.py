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

class RollingTargetEncoder:
    def __init__(self):
        self.stats = {}
        self.cols = None
        self.global_mean = None

    def fit_transform(
            self,
            ds: Dataset,
            cols: list | np.ndarray | None = None,
            smoothing: float | None = 0.0
    ):
        if cols is None or not cols:
            cols = ds.x.select_dtypes(include='category').columns.tolist()

        self.cols = cols
        self.global_mean = ds.y.mean()

        ds_enc = ds.copy()

        for col in cols:
            temp = pd.DataFrame({
                col: ds.x[col],
                'target': ds.y
            })

            cumulative_sum = (
                temp.groupby(col)['target'].cumsum().shift(1).fillna(0)
            )
            cumulative_count = temp.groupby(col).cumcount()

            encoded = (cumulative_sum + smoothing * self.global_mean) \
                    / (cumulative_count + smoothing)

            ds_enc.x[col] = encoded

            final_stats = (
                temp.groupby(col)['target'].agg(['sum', 'count'])
            )

            self.stats[col] = final_stats

        return ds_enc
    
    def transform(
            self,
            ds: Dataset,
            smoothing: float | None = 0.0
    ):
        ds_enc = ds.copy()

        for col in self.cols:
            stats = self.stats[col]

            mapping = (stats['sum'] + smoothing) \
                    / (stats['count'] + smoothing)
            
            ds_enc.x[col] = (
                ds_enc.x[col].astype('object').map(mapping).fillna(self.global_mean)
            )

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

class OneHotEncoder:
    def __init__(self, drop_first=False, **kwargs):
        self.drop_first = drop_first
        self.kwargs = kwargs

        self.cols = None
        self.feature_names = None

    def fit(
            self,
            ds: Dataset,
            cols: list | np.ndarray | None = None
    ):
        if cols is None or not cols:
            self.columns = []
            self.feature_names = ds.x.columns.tolist()
            return self
        
        self.cols = list(cols)

        X_dummy = pd.get_dummies(
            ds.x,
            columns=self.cols,
            drop_first=self.drop_first,
            **self.kwargs
        )

        self.feature_names = X_dummy.columns.tolist()

        return self
    
    def transform(
            self,
            ds: Dataset
    ):
        X = pd.get_dummies(
            ds.x,
            columns=self.cols,
            drop_first=self.drop_first,
            **self.kwargs
        )

        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0

        X = X.reindex(columns=self.feature_names, fill_value=0)

        return Dataset(X, ds.y.copy())
    
    def fit_transform(
            self,
            ds: Dataset,
            cols: list | np.ndarray | None = None
    ):
        self.fit(ds, cols)
        return self.transform(ds)