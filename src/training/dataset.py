"""Dataset and data loading utilities for LSTM training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.config import Settings
from src.common.logging import get_logger
from src.feature_service.indicators import StreamingIndicators
from src.feature_service.scaler import StreamingScaler, fit_scaler_on_dataframe

logger = get_logger(__name__)


class StockSequenceDataset(Dataset):
    """PyTorch Dataset for stock sequence data.

    Creates overlapping windows of features for LSTM training.
    Labels are binary: 1 if price goes up, 0 if down.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 60,
    ):
        """Initialize the dataset.

        Args:
            features: 2D array of shape (n_samples, n_features)
            labels: 1D array of shape (n_samples,) with binary labels
            sequence_length: Number of timesteps per sequence
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length

        # Calculate valid indices (need sequence_length + 1 for label)
        self.valid_indices = len(features) - sequence_length

        if self.valid_indices <= 0:
            raise ValueError(
                f"Not enough data: {len(features)} samples for sequence length {sequence_length}"
            )

    def __len__(self) -> int:
        return self.valid_indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its label.

        Args:
            idx: Index

        Returns:
            Tuple of (sequence, label)
            - sequence: (sequence_length, n_features)
            - label: scalar (0 or 1)
        """
        # Get sequence of features
        sequence = self.features[idx : idx + self.sequence_length]

        # Label is whether price went up after the sequence
        label = self.labels[idx + self.sequence_length - 1]

        return sequence, label


def compute_features_for_dataframe(
    df: pd.DataFrame,
    settings: Settings,
) -> Tuple[pd.DataFrame, list[str]]:
    """Compute technical indicators for a DataFrame of OHLCV data.

    Args:
        df: DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        settings: Application settings

    Returns:
        Tuple of (DataFrame with features, list of feature names)
    """
    # Process each symbol separately to maintain temporal ordering
    symbols = df["symbol"].unique()
    all_features = []

    for symbol in symbols:
        symbol_df = df[df["symbol"] == symbol].copy().sort_values("timestamp")

        # Create indicator calculator
        calc = StreamingIndicators(settings.features)

        # Compute features row by row
        feature_rows = []
        for _, row in symbol_df.iterrows():
            features = calc.compute(
                open_price=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            feature_row = {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "close": row["close"],
                **features,
            }
            feature_rows.append(feature_row)

        symbol_features_df = pd.DataFrame(feature_rows)

        # Skip warmup period
        warmup = calc.warmup_period
        symbol_features_df = symbol_features_df.iloc[warmup:].reset_index(drop=True)

        all_features.append(symbol_features_df)

    # Combine all symbols
    features_df = pd.concat(all_features, ignore_index=True)

    # Get feature names (exclude metadata columns)
    feature_names = [
        col for col in features_df.columns if col not in ["timestamp", "symbol", "close"]
    ]

    logger.info(
        "Computed features",
        n_samples=len(features_df),
        n_features=len(feature_names),
        feature_names=feature_names,
    )

    return features_df, feature_names


def create_labels(df: pd.DataFrame) -> np.ndarray:
    """Create binary labels for price direction prediction.

    Label = 1 if close[t+1] > close[t], else 0

    Args:
        df: DataFrame with 'close' column

    Returns:
        Binary labels array
    """
    # Shift close prices to get next close
    next_close = df["close"].shift(-1)

    # Binary label: 1 if price goes up
    labels = (next_close > df["close"]).astype(float)

    # Fill last value (no next price)
    labels.iloc[-1] = 0

    return labels.values


def prepare_data(
    data_path: Path,
    settings: Settings,
    symbols: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[str], StreamingScaler]:
    """Prepare data for training.

    Args:
        data_path: Path to OHLCV data directory or file
        settings: Application settings
        symbols: Optional list of symbols to filter

    Returns:
        Tuple of (features array, labels array, feature names, fitted scaler)
    """
    # Load OHLCV data
    if data_path.is_file():
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        csv_files = list(data_path.glob("*.csv"))
        dfs = []
        for f in csv_files:
            if f.name != "combined.csv":
                dfs.append(pd.read_csv(f, parse_dates=["timestamp"]))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(data_path / "combined.csv", parse_dates=["timestamp"])

    # Filter symbols if specified
    if symbols:
        df = df[df["symbol"].isin(symbols)]

    logger.info("Loaded data", rows=len(df), symbols=df["symbol"].unique().tolist())

    # Compute features
    features_df, feature_names = compute_features_for_dataframe(df, settings)

    # Create labels
    labels = create_labels(features_df)

    # Extract feature matrix
    feature_matrix = features_df[feature_names].values

    # Handle NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit scaler on features
    scaler = fit_scaler_on_dataframe(
        pd.DataFrame(feature_matrix, columns=feature_names),
        feature_names,
    )

    # Scale features
    scaled_features = scaler.scaler.transform(feature_matrix)

    logger.info(
        "Prepared data",
        n_samples=len(scaled_features),
        n_features=len(feature_names),
        label_balance=f"{labels.mean():.2%} positive",
    )

    return scaled_features, labels, feature_names, scaler


def create_data_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    settings: Settings,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders.

    Args:
        features: Scaled feature matrix
        labels: Binary labels
        settings: Application settings

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_samples = len(features)
    train_size = int(n_samples * settings.training.train_split)
    val_size = int(n_samples * settings.training.val_split)

    # Time-series split (no shuffling to preserve temporal order)
    train_features = features[:train_size]
    train_labels = labels[:train_size]

    val_features = features[train_size : train_size + val_size]
    val_labels = labels[train_size : train_size + val_size]

    test_features = features[train_size + val_size :]
    test_labels = labels[train_size + val_size :]

    # Create datasets
    train_dataset = StockSequenceDataset(
        train_features,
        train_labels,
        settings.features.sequence_length,
    )
    val_dataset = StockSequenceDataset(
        val_features,
        val_labels,
        settings.features.sequence_length,
    )
    test_dataset = StockSequenceDataset(
        test_features,
        test_labels,
        settings.features.sequence_length,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,  # Shuffle within train set is OK
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(
        "Created data loaders",
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        test_batches=len(test_loader),
    )

    return train_loader, val_loader, test_loader

