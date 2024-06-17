from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import os, sys
from imblearn.over_sampling import KMeansSMOTE, SMOTE
from sklearn.cluster import KMeans
import numpy as np

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


# class ClassImbalance(BaseEstimator, TransformerMixin):
#     def __init__(self, sampling_strategy='auto', cluster_balance_threshold=0.1, random_state=42):
#         self.sampling_strategy = sampling_strategy
#         self.cluster_balance_threshold = cluster_balance_threshold
#         self.random_state = random_state
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         if y is None:
#             raise ValueError("y cannot be None for ClassImbalance transform method.")
        
#         smote = KMeansSMOTE(
#             sampling_strategy=self.sampling_strategy,
#             cluster_balance_threshold=self.cluster_balance_threshold,
#             random_state=self.random_state,
#             )
#         X_resampled, y_resampled = smote.fit_resample(X, y)
#         return X_resampled, y_resampled

class ClassImbalance(BaseEstimator, TransformerMixin):
    """
    Class that applies SMOTE oversampling to address class imbalance.

    Args:
        sampling_strategy (str, optional): Sampling strategy for SMOTE (default: 'auto').
        cluster_balance_threshold (float, optional): Threshold for cluster balance (default: 0.1, used with KMeansSMOTE).
        random_state (int, optional): Random seed for reproducibility (default: 42).
        use_kmeans (bool, optional): Flag to enable KMeans clustering before SMOTE (default: False).
        k (int, optional): Number of clusters for KMeans (default: 5, used with use_kmeans=True).
    """

    def __init__(self, sampling_strategy='auto', cluster_balance_threshold=0.1, random_state=42, use_kmeans=False, k=5):
        self.sampling_strategy = sampling_strategy
        self.cluster_balance_threshold = cluster_balance_threshold
        self.random_state = random_state
        self.use_kmeans = use_kmeans
        self.k = k
        self.smote = None
        self.kmeans = None  # Initialize if using KMeans

    def fit(self, X, y):
        """
        Fits the SMOTE sampler (and optionally KMeans model) to the data.

        Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or np.ndarray): Target labels.

        Returns:
        self: The fitted ClassImbalance object.
        """
        if self.use_kmeans:
            self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
            self.kmeans.fit(X)
        self.smote = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        return self

    def transform(self, X, y=None):
        """
        Transforms features (X) by applying SMOTE oversampling.

        Args:
        X (pd.DataFrame or np.ndarray): Feature data.
        y (pd.Series or np.ndarray, optional): Target labels (ignored).

        Returns:
        tuple: (X_resampled, y_resampled): The oversampled features and target labels.
        """
        if y is not None:
            from warnings import warn
            warn("y argument is ignored in transform method of ClassImbalance")

        X_resampled = []
        y_resampled = []

        if self.use_kmeans:
            # Perform KMeans clustering on new data points
            new_cluster_labels = self.kmeans.predict(X)

            for i in range(len(X)):
                cluster = new_cluster_labels[i]
                sample_x = X.iloc[i]

                # Apply SMOTE for imbalanced classes within the cluster
                oversampled_x, oversampled_y = self.smote.fit_resample(sample_x.reshape(1, -1), np.array([1]))  # Dummy target for SMOTE
                X_resampled.extend(oversampled_x)
                y_resampled.extend(oversampled_y)
        else:
            # Apply SMOTE directly to the entire data
            X_resampled, y_resampled = self.smote.fit_resample(X, y)

        return np.concatenate(X_resampled), np.concatenate(y_resampled)