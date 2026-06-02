import numpy as np
import pandas as pd
from tqdm import tqdm

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