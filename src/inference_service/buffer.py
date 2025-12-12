"""Rolling buffer for maintaining feature sequences per symbol."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from src.common.logging import get_logger

logger = get_logger(__name__)


class FeatureBuffer:
    """Rolling buffer for maintaining feature sequences.

    Maintains a fixed-size window of feature vectors for each symbol,
    suitable for feeding into the LSTM model.
    """

    def __init__(self, sequence_length: int, n_features: int):
        """Initialize the buffer.

        Args:
            sequence_length: Number of timesteps to maintain
            n_features: Number of features per timestep
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.buffer: deque[list[float]] = deque(maxlen=sequence_length)

    def append(self, features: list[float]) -> None:
        """Add a feature vector to the buffer.

        Args:
            features: Feature vector of length n_features
        """
        if len(features) != self.n_features:
            logger.warning(
                "Feature dimension mismatch",
                expected=self.n_features,
                got=len(features),
            )
            # Pad or truncate
            if len(features) < self.n_features:
                features = features + [0.0] * (self.n_features - len(features))
            else:
                features = features[: self.n_features]

        self.buffer.append(features)

    def is_ready(self) -> bool:
        """Check if buffer has enough data for inference.

        Returns:
            True if buffer is full
        """
        return len(self.buffer) >= self.sequence_length

    def get_sequence(self) -> np.ndarray:
        """Get the current sequence as a numpy array.

        Returns:
            Array of shape (sequence_length, n_features)
        """
        if not self.is_ready():
            raise ValueError("Buffer not ready")

        return np.array(list(self.buffer), dtype=np.float32)

    def get_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        """Get the current sequence as a PyTorch tensor.

        Args:
            device: Device to place tensor on

        Returns:
            Tensor of shape (1, sequence_length, n_features)
        """
        arr = self.get_sequence()
        tensor = torch.FloatTensor(arr).unsqueeze(0)  # Add batch dimension

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class SymbolBufferManager:
    """Manager for multiple per-symbol feature buffers."""

    def __init__(self, sequence_length: int, n_features: int):
        """Initialize the manager.

        Args:
            sequence_length: Sequence length for each buffer
            n_features: Number of features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.buffers: dict[str, FeatureBuffer] = {}

    def get_buffer(self, symbol: str) -> FeatureBuffer:
        """Get or create a buffer for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            FeatureBuffer for the symbol
        """
        if symbol not in self.buffers:
            self.buffers[symbol] = FeatureBuffer(
                self.sequence_length, self.n_features
            )

        return self.buffers[symbol]

    def append(self, symbol: str, features: list[float]) -> None:
        """Append features to a symbol's buffer.

        Args:
            symbol: Stock symbol
            features: Feature vector
        """
        buffer = self.get_buffer(symbol)
        buffer.append(features)

    def is_ready(self, symbol: str) -> bool:
        """Check if a symbol's buffer is ready for inference.

        Args:
            symbol: Stock symbol

        Returns:
            True if buffer is full
        """
        return symbol in self.buffers and self.buffers[symbol].is_ready()

    def get_tensor(
        self, symbol: str, device: torch.device | None = None
    ) -> torch.Tensor:
        """Get sequence tensor for a symbol.

        Args:
            symbol: Stock symbol
            device: Device to place tensor on

        Returns:
            Tensor of shape (1, sequence_length, n_features)
        """
        buffer = self.get_buffer(symbol)
        return buffer.get_tensor(device)

    def clear(self, symbol: str | None = None) -> None:
        """Clear buffer(s).

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol is None:
            self.buffers.clear()
        elif symbol in self.buffers:
            self.buffers[symbol].clear()

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        return {
            "num_symbols": len(self.buffers),
            "ready_symbols": sum(1 for b in self.buffers.values() if b.is_ready()),
            "buffer_sizes": {s: len(b) for s, b in self.buffers.items()},
        }

