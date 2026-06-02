import numpy as np
import sys
from abc import ABC, abstractmethod

from utils.data.dataset import Dataset

class BaseNormalizer(ABC):
    @abstractmethod
    def fit(self, ds: Dataset):
        ...

    @abstractmethod
    def transform(self, ds: Dataset) -> Dataset:
        ...

class StandardScaler(BaseNormalizer):
    def __init__(self):
        self.mean = None
        self.STDV = None

    def fit(
            self,
            ds: Dataset,
            eps: float | None = 1e-3
    ):
        X, y = ds[:]

        self.mean = X.mean()
        self.STDV = X.std().clip(lower=eps)

        return self
    
    def transform(
            self,
            ds: Dataset
    ):
        if self.mean is None or self.STDV is None:
            raise ValueError("The scaler hasn't been fit on any dataset")
        
        X, y = ds[:]

        for i, col in enumerate(X.columns):
            if X[col].dtype == 'bool':
                continue
            else:
                X[col] = (X[col] - self.mean[i]) / self.STDV[i]
        
        return Dataset(X, y)
    
class RobustScaler(BaseNormalizer):
    def __init__(self):
        self.median = None
        self.IQR = None

        self.MIN_SCALE = 1e-2

    def fit(
            self,
            ds: Dataset,
            eps: float | None = 0.0
    ):
        X, y = ds[:]

        self.numeric_cols = X.select_dtypes(
            include=['number'],
            exclude=['bool']
        ).columns

        X_num = X[self.numeric_cols]

        Q1 = X_num.quantile(0.25)
        Q3 = X_num.quantile(0.75)

        self.IQR = Q3 - Q1
        scale = X_num.abs().median()
        self.median = X_num.median()

        self.IQR = np.maximum(
            self.IQR,
            np.maximum(eps * scale, self.MIN_SCALE)
        )

        return self
    
    def transform(
            self,
            ds: Dataset
    ):
        if self.median is None or self.IQR is None:
            raise ValueError
        
        X, y = ds[:]
        
        X[self.numeric_cols] = (
            X[self.numeric_cols] - self.median
        ) / self.IQR

        return Dataset(X, y)