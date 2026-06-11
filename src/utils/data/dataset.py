import numpy as np
import pandas as pd

class _LocIndexer:
    def __init__(self, dataset):
        self._ds = dataset

    def __getitem__(self, i):
        x = self._ds.x.loc[i]
        y = self._ds.y.loc[i]
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return Dataset(x, y)
        return x, y

class _iLocIndexer:
    def __init__(self, dataset):
        self._ds = dataset

    def __getitem__(self, i):
        x = self._ds.x.iloc[i]
        y = self._ds.y.iloc[i]
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return Dataset(x, y)
        return x, y

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        if isinstance(i, (pd.Series, np.ndarray)) and i.dtype == bool:
            return Dataset(self.x[i], self.y[i])
        return self.x[i], self.y[i]
    
    def copy(self, deep=True):
        if deep:
            return Dataset(self.x.copy(), self.y.copy())
        else:
            return Dataset(self.x, self.y) 
        
    def reset_index(self, **kwargs):
        self.x = self.x.reset_index(**kwargs)
        self.y = self.y.reset_index(**kwargs)

        return Dataset(self.x, self.y)

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def iloc(self):
        return _iLocIndexer(self)
    
def to_dataset(df, target):
    mask = ~df.columns.isin([target])
    x = df.loc[:, mask]
    y = df[target]

    return Dataset(x, y)