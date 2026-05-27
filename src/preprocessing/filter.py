import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import sys

sys.path.insert(0, '../')
from ..utils.data.dataset import Dataset
from ..utils.utilities import entropy

class BaseFilterMethod(ABC):
    @abstractmethod
    def __init__(self):
        self.score_dict = dict()

    @abstractmethod
    def score(self, ds: Dataset) -> dict[str, float]:
        ...

class InformationGain(BaseFilterMethod):
    def score(
            self,
            ds: Dataset,
            bins: int = 10
    ):
        X, y = ds[:]

        for col in X.columns:
            if (X[col].dtype.kind in 'bifc') and len(np.unique(X[col])) > bins:
                binned_x = pd.qcut(X[col], bins, duplicates='drop')
                X_0 = binned_x
                ds_0 = Dataset(X_0, y)
            else:
                X_0 = X[col]
                ds_0 = Dataset(X_0, y)

            self.score_dict[col] = self._ig(ds_0)

        return dict(sorted(self.score_dict.items(), key=lambda item: item[1], reverse=True))

    def _ig(
            self,
            ds: Dataset
    ):
        X, y = ds[:]

        ent_Y = entropy(y)
        ent_YX = 0.

        for x in np.unique(X):
            mask = X == x
            group_y = y[mask]
            ent_YX += (len(group_y)/len(y)) * entropy(group_y)

        return ent_Y - ent_YX
    
class BaseScoreCombiner(ABC):
    @abstractmethod
    def combine(self, scores: list[dict[str, float]]) -> list[int]:
        ...

    def _cut(self, scores: dict[int, float]) -> list[int]:
        if self.top_k:
            return sorted(scores, key=scores.get, reverse=True)[:self.top_k]
        if self.threshold:
            return [k for k, v in scores.items() if v >= self.threshold]
        raise ValueError('Specify either top_k or threshold.')

class MeanCombiner(BaseScoreCombiner):
    def __init__(
            self,
            top_k: int | None = 1,
            threshold: float | None = None
    ):
        self.top_k = top_k
        self.threshold = threshold
    
    def combine(
            self,
            scores: list[dict[str, float]]
    ):
        keys = scores[0].keys()
        merged = {k: np.mean([s[k] for s in scores]) for k in keys}
        return self._cut(merged)
    
class IntersectCombiner(BaseScoreCombiner):
    def __init__(
            self,
            top_k: int | None = 1,
            min_agreement: int | None = 1
    ):
        self.top_k = top_k
        self.min_agreement = min_agreement

    def combine(
            self,
            scores: list[dict[str, float]]
    ):
        min_agree = self.min_agreement or len(scores)

        def top_keys(s):
            return set(sorted(s, key=s.get, reverse=True)[:self.top_k])
        
        agreement = {
            k: sum(k in top_keys(s) for s in scores)
            for k in scores[0].keys()
        }
        
        return [k for k, count in agreement.items() if count >= min_agree]
    
class Filter:
    def __init__(
            self,
            methods: list[BaseFilterMethod],
            combiner: BaseScoreCombiner = None,
    ):
        self.methods = methods
        self.combiner = combiner

    def fit_select(
            self,
            ds: Dataset
    ) -> Dataset:
        scores = [m.score(ds) for m in self.methods]

        self.selected = (
            scores[0]
            if len(scores) == 1
            else self.combiner.combine(scores)
        )

        return self.apply(ds)
    
    def apply(
            self,
            ds: Dataset,
    ) -> Dataset:
        X, y = ds.copy()
        X = X.loc[:, self.selected]

        return Dataset(X, y)

def quantile_boundary(
        ds: Dataset | None = None,
        df: pd.DataFrame | None = None,
        col: str | None = None,
        lower: float | None = None,
        upper: float | None = None
):
    if ds is not None and df is not None:
        raise ValueError('Pass only either Dataset or DataFrame argument at a time.')
    
    if lower is None or upper is None:
        raise ValueError
    
    if col is None:
        raise ValueError

    if ds:
        if col in ds.x.columns:
            feat = ds.x[col]
        elif col == ds.y.name:
            feat = ds.y
        else:
            raise KeyError(f'Invalid feature name: {col}.')
    elif df:
        feat = df[col]

    mask = _make_quantile_mask(feat, lower, upper)
    
    if ds:
        ds = ds[mask]
        return ds

    elif df:
        df = df[mask]
        return df

def _make_quantile_mask(
        feat: pd.Series | None = None,
        lower: float | None = None,
        upper: float | None = None,
):
    lower = feat.quantile(lower)
    upper = feat.quantile(upper)
    mask = (feat.between(lower, upper) | feat.isna())

    return mask