"""Feature scaling utilities for real-time inference."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.common.logging import get_logger

logger = get_logger(__name__)


class StreamingScaler:
    """Wrapper around sklearn MinMaxScaler for streaming data.

    Supports loading pre-fitted scalers and transforming single samples.
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted

    def fit(self, data: np.ndarray) -> None:
        """Fit the scaler on training data.

        Args:
            data: 2D array of shape (n_samples, n_features)
        """
        self.scaler.fit(data)
        self._is_fitted = True
        logger.info("Scaler fitted", n_features=len(self.feature_names))

    def transform(self, features: dict[str, float]) -> list[float]:
        """Transform a single sample.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            List of scaled values in feature_names order
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Load or fit first.")

        # Build feature vector in correct order
        vector = []
        for name in self.feature_names:
            value = features.get(name, 0.0)
            # Handle NaN/Inf
            if not np.isfinite(value):
                value = 0.0
            vector.append(value)

        # Transform
        scaled = self.scaler.transform([vector])[0]
        return scaled.tolist()

    def inverse_transform(self, scaled: list[float]) -> dict[str, float]:
        """Inverse transform scaled values back to original scale.

        Args:
            scaled: List of scaled values

        Returns:
            Dictionary of feature name -> original value
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Load or fit first.")

        original = self.scaler.inverse_transform([scaled])[0]
        return dict(zip(self.feature_names, original))

    def save(self, scaler_path: Path, feature_list_path: Path) -> None:
        """Save scaler and feature list to disk.

        Args:
            scaler_path: Path to save pickle file
            feature_list_path: Path to save feature names JSON
        """
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        feature_list_path.parent.mkdir(parents=True, exist_ok=True)

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        with open(feature_list_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(
            "Scaler saved",
            scaler_path=str(scaler_path),
            feature_list_path=str(feature_list_path),
        )

    @classmethod
    def load(cls, scaler_path: Path, feature_list_path: Path) -> "StreamingScaler":
        """Load scaler and feature list from disk.

        Args:
            scaler_path: Path to scaler pickle file
            feature_list_path: Path to feature names JSON

        Returns:
            Loaded StreamingScaler instance
        """
        with open(feature_list_path) as f:
            feature_names = json.load(f)

        instance = cls(feature_names)

        with open(scaler_path, "rb") as f:
            instance.scaler = pickle.load(f)

        instance._is_fitted = True

        logger.info(
            "Scaler loaded",
            scaler_path=str(scaler_path),
            n_features=len(feature_names),
        )

        return instance


def fit_scaler_on_dataframe(
    df: "pd.DataFrame",
    feature_names: list[str],
) -> StreamingScaler:
    """Fit a scaler on a pandas DataFrame.

    Args:
        df: DataFrame with feature columns
        feature_names: List of feature column names

    Returns:
        Fitted StreamingScaler
    """
    import pandas as pd

    # Extract feature columns
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")

    data = df[feature_names].values

    # Handle NaN/Inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StreamingScaler(feature_names)
    scaler.fit(data)

    return scaler

