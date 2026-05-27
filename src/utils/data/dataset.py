import numpy as np
import pandas as pd

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
    
def to_dataset(df, target):
    mask = ~df.columns.isin([target])
    x = df.loc[:, mask]
    y = df[target]

    return Dataset(x, y)