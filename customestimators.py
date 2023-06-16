
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
class NumericCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self = self
    def fit(self, X, y=None):
        print("NumericCleaner.fit called")
        return self
    def transform(self, X):
        print("NumericCleaner.transform called")
        X["localHour"] = X["localHour"].fillna(-99)
        X.loc[X.loc[:,"localHour"] == -1, "localHour"] = -99
        return X

class CategoricalCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self = self
    def fit(self, X, y=None):
        print("CategoricalCleaner.fit called")
        return self
    def transform(self, X):
        print("CategoricalCleaner.transform called")
        X = X.fillna(value={"cardType":"U","cvvVerifyResult": "N"})
        return X
