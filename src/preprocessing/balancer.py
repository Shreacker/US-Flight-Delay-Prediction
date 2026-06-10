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
        self.coverage_dict = dict()

    def fit(
            self,
            ds: Dataset,
            cols: np.ndarray | list | None = None,
            coverage: np.ndarray | list | None = None
    ):
        if cols is None:
            raise ValueError('cols cannot be None!')
        
        if coverage is None:
            raise ValueError('coverage cannot be None!')

        if len(cols) != len(coverage):
            raise ValueError(
                f'Expected coverage length {len(cols)}'
                f'(got {len(coverage)})'
            )
        
        self.cols = cols
        self.coverage_dict = dict(zip(cols, coverage))
        
        X, y = ds[:]

        for col in self.cols:
            counts = X[col].value_counts(normalize=True).sort_values(ascending=False)
            cum_pct = counts.cumsum()
            coverage = self.coverage_dict[col]
            n_keep = (cum_pct < coverage).sum() + 1
            n_keep = min(n_keep, len(counts))
            keep_cat = counts.index[:n_keep]

            self.cat_dict[col] = set(keep_cat)

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

            if 'Other' not in X[col].cat.categories:
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

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

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