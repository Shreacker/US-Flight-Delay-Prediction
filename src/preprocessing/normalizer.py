import numpy as np
import sys
from abc import ABC, abstractmethod

sys.path.insert(0, '../')
from ..utils.data.dataset import Dataset

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