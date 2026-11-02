"""Train-fitted, serializable transformations for the Adult dataset."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
CATEGORICAL = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
DERIVED = ['capital_net', 'hours_per_age']


class MissingNormalizer(TransformerMixin, BaseEstimator):
    """Convert OpenML nulls and literal '?' in a copy, never in raw storage."""
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        result = X.copy(deep=True)
        for col in CATEGORICAL:
            result[col] = result[col].astype(object)
            result.loc[result[col].isna() | result[col].eq('?'), col] = np.nan
        return result


class FrameImputer(TransformerMixin, BaseEstimator):
    """Keep DataFrame column names through train-only median/mode imputation."""
    def fit(self, X, y=None):
        self.numeric_ = SimpleImputer(strategy='median').fit(X[NUMERIC])
        self.categorical_ = SimpleImputer(strategy='most_frequent').fit(X[CATEGORICAL])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        result = X.copy(deep=True)
        result[NUMERIC] = self.numeric_.transform(X[NUMERIC])
        result[CATEGORICAL] = self.categorical_.transform(X[CATEGORICAL])
        return result


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Apply training 1st/99th percentiles, including to held-out rows."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError('Clipping quantiles must satisfy 0 <= lower < upper <= 1.')
        self.lower_bounds_ = X[NUMERIC].quantile(self.lower)
        self.upper_bounds_ = X[NUMERIC].quantile(self.upper)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        result = X.copy(deep=True)
        result[NUMERIC] = X[NUMERIC].clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return result


class DerivedFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self)
        result = X.copy(deep=True)
        if self.enabled:
            denominator = result['age'] - 16
            if (denominator <= 0).any():
                raise ValueError('hours_per_age requires age > 16.')
            result['capital_net'] = result['capital-gain'] - result['capital-loss']
            result['hours_per_age'] = result['hours-per-week'] / denominator
        return result


def make_preprocessor(engineered=False, lower=0.01, upper=0.99):
    numbers = NUMERIC + (DERIVED if engineered else [])
    return Pipeline([
        ('missing', MissingNormalizer()),
        ('impute', FrameImputer()),
        ('clip', QuantileClipper(lower=lower, upper=upper)),
        ('derive', DerivedFeatures(enabled=engineered)),
        ('encode', ColumnTransformer([
            ('numeric', StandardScaler(), numbers),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), CATEGORICAL),
        ], sparse_threshold=1.0)),
    ])
