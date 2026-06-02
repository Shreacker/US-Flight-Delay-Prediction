import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod
from sklearn.preprocessing import PowerTransformer

from utils.data.dataset import Dataset

class BaseBalancer(ABC):
    @abstractmethod
    def fit(self, ds: Dataset):
        ...

    @abstractmethod
    def transform(self, ds: Dataset) -> Dataset:
        ...

class GroupCat(BaseBalancer):
    def __init__(self):
        self.cat_dict = dict()
        self.thresh_dict = dict()

    def fit(
            self,
            ds: Dataset,
            cols: np.ndarray | list | None = None,
            min_freq: np.ndarray | list | None = None,
            min_pct: np.ndarray | list | None = None
    ):
        if cols is None:
            raise ValueError
        self.cols = cols

        if min_pct is not None:
            if len(min_pct) != len(cols):
                raise ValueError(f'Expecting min_pct of len {len(cols)} (got {len(min_pct)})')
            
            threshold = min_pct * len(ds)

        elif min_freq is not None:
            if len(min_freq) != len(cols):
                raise ValueError(f'Expecting min_freq of len {len(cols)} (got {len(min_freq)})')
            
            threshold = min_freq

        else:
            raise ValueError('Specify either min_pct or min_freq.')
        
        X, y = ds[:]

        self.thresh_dict = dict(zip(cols, threshold))
        for col in cols:
            counts = X[col].value_counts()
            keep_cat = counts[counts >= self.thresh_dict[col]].index
            self.cat_dict[col] = keep_cat

        return self
    
    def transform(
            self,
            ds: Dataset
    ):
        X, y = ds[:]

        for col in self.cols:
            if X[col].dtype != 'category':
                X[col] = X[col].astype('category')

            keep_cat = self.cat_dict[col]

            X[col] = X[col].cat.add_categories(['Other'])
            X.loc[~X[col].isin(keep_cat), col] = 'Other'
            X[col] = X[col].cat.remove_unused_categories()

        return Dataset(X, y)

class Transformer(BaseBalancer):
    def __init__(self, method='yeo-johnson', **kwargs):
        self.pt = PowerTransformer(method=method, **kwargs)

    def fit(
            self,
            ds: Dataset
    ):
        X, y = ds[:]

        self.pt = self.pt.fit(y.to_numpy().reshape(-1, 1))

    def transform(
            self,
            ds: Dataset,
            qbins: int | None = 1,
            retweights: bool = False
    ):
        X, y = ds[:]
        name = y.name

        y = self.pt.transform(y.to_numpy().reshape(-1, 1))
        y = pd.Series(y.squeeze(-1), name=name)
        bins = pd.qcut(y, q=qbins, duplicates='drop')
        freq = bins.value_counts()

        if not retweights:
            return Dataset(X, y)
        
        else:
            weights = 1 / freq.reindex(bins).to_numpy()
            weights = weights.astype('float64')
            weights = weights / weights.mean()

            return Dataset(X, y), weights
        
    def inverse_transform(
            self,
            y_pred: np.ndarray | list | None = None
    ):
        assert y_pred is not None

        return self.pt.inverse_transform(y_pred.reshape(-1, 1))