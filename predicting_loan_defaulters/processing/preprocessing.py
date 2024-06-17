from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import os, sys
from imblearn.over_sampling import KMeansSMOTE

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from predicting_loan_defaulters.config import config


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.variables_to_drop, axis=1)
        return X

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X, y=None):
        self.label_dict = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index
            self.label_dict[var] = {k: i for i, k in enumerate(t, 0)}
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X


class ClassImbalance(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_strategy='auto', cluster_balance_threshold=0.1, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.cluster_balance_threshold = cluster_balance_threshold
        self.random_state = random_state
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for ClassImbalance transform method.")
        
        smote = KMeansSMOTE(
            sampling_strategy=self.sampling_strategy,
            cluster_balance_threshold=self.cluster_balance_threshold,
            random_state=self.random_state,
            )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled