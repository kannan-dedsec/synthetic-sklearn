"""Custom transformer for removing outliers from a dataset.

This module provides a custom transformer that can be used to identify and
remove outliers from a dataset based on the interquartile range (IQR).
The transformer implements the fit and transform methods from scikit-learn's
BaseEstimator and TransformerMixin classes.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Transformer to remove outliers from a dataset based on IQR."""

    def __init__(self, threshold: float = 1.5):
        """
        Parameters:
            threshold (float): The IQR multiplier to determine outliers.
        """
        self.threshold = threshold
        self.lower_bound: Optional[np.ndarray] = None
        self.upper_bound: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "OutlierRemover":
        """Fit the transformer to the data by calculating the IQR.

        Parameters:
            X (pd.DataFrame): The input data.
            y (Optional[np.ndarray]): Target values (ignored).

        Returns:
            OutlierRemover: The fitted transformer.
        """
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1

        self.lower_bound = q1 - self.threshold * iqr
        self.upper_bound = q3 + self.threshold * iqr

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data.

        Parameters:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: The data with outliers removed.
        """
        if self.lower_bound is None or self.upper_bound is None:
            raise RuntimeError("You must fit the transformer before calling transform.")

        mask = (X >= self.lower_bound) & (X <= self.upper_bound)
        filtered_data = X[mask].dropna()

        return filtered_data.reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit the transformer and then remove outliers.

        Parameters:
            X (pd.DataFrame): The input data.
            y (Optional[np.ndarray]): Target values (ignored).

        Returns:
            pd.DataFrame: The data with outliers removed.
        """
        return self.fit(X, y).transform(X)