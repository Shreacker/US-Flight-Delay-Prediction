import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

from utils.data.dataset import Dataset

def fill_missing(
        ds: Dataset,
        group: list | np.ndarray | None = None
):
    X, y = ds[:]

    for col in tqdm(X.columns):
        if X[col].isna().sum() == 0:
            continue

        if isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].cat.add_categories(['Unknown'])
            X[col] = X[col].fillna('Unknown')

        else:
            global_median = X[col].median()
            group_median = X.groupby(group, observed=True)[col]\
                            .transform('median')
            
            X[col] = X[col].fillna(group_median)
            X[col] = X[col].fillna(global_median)

    return Dataset(X, y)

class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            group: list | np.ndarray | None = None
    ):
        self.group = group

    def fit(
            self,
            ds: Dataset
    ):
        X = ds.x.copy()
        self.fill_values_: dict = {}

        for col in tqdm(X.columns):
            if X[col].isna().sum() == 0:
                continue

            if isinstance(X[col].dtype, pd.CategoricalDtype):
                self.fill_values_[col] = {'type': 'categorical'}
            else:
                group_medians = None
                if self.group:
                    group_medians = (
                        X.groupby(self.group, observed=True)[col].median()
                    )
                self.fill_values_[col] = {
                    'type': 'numeric',
                    'global_median': X[col].median(),
                    'group_medians': group_medians,
                }

        return self
    
    def transform(
            self,
            ds: Dataset
    ) -> Dataset:
        X, y = ds[:]

        for col, fill_info in self.fill_values_.items():
            if col not in X.columns or X[col].isna().sum() == 0:
                continue
            if fill_info['type'] == 'categorical':
                if 'Unknown' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(['Unknown'])
                X[col] = X[col].fillna('Unknown')
            else:
                if fill_info['group_medians'] is not None:
                    group_median = X[self.group].join(
                        fill_info['group_medians'].rename('_gm'),
                        on=self.group
                    )['_gm']
                    X[col] = X[col].fillna(group_median)
                X[col] = X[col].fillna(fill_info['global_median'])

        return Dataset(X, y)
    
    def fit_transform(
            self,
            ds: Dataset
    ) -> Dataset:
        self.fit(ds)
        return self.transform(ds)