"""Tests for inference buffer management."""

import pytest
import numpy as np
import torch

from src.inference_service.buffer import FeatureBuffer, SymbolBufferManager


class TestFeatureBuffer:
    """Tests for FeatureBuffer class."""

    def test_create_buffer(self):
        """Test buffer creation."""
        buffer = FeatureBuffer(sequence_length=60, n_features=17)

        assert buffer.sequence_length == 60
        assert buffer.n_features == 17
        assert len(buffer) == 0

    def test_append_features(self):
        """Test appending features."""
        buffer = FeatureBuffer(sequence_length=5, n_features=3)

        for i in range(3):
            buffer.append([float(i), float(i + 1), float(i + 2)])

        assert len(buffer) == 3
        assert not buffer.is_ready()

    def test_is_ready_when_full(self):
        """Test is_ready returns True when buffer is full."""
        buffer = FeatureBuffer(sequence_length=5, n_features=3)

        for i in range(5):
            buffer.append([1.0, 2.0, 3.0])

        assert buffer.is_ready()

    def test_get_sequence(self):
        """Test getting sequence as numpy array."""
        buffer = FeatureBuffer(sequence_length=3, n_features=2)

        buffer.append([1.0, 2.0])
        buffer.append([3.0, 4.0])
        buffer.append([5.0, 6.0])

        seq = buffer.get_sequence()

        assert isinstance(seq, np.ndarray)
        assert seq.shape == (3, 2)
        assert seq[0, 0] == 1.0
        assert seq[2, 1] == 6.0

    def test_get_tensor(self):
        """Test getting sequence as PyTorch tensor."""
        buffer = FeatureBuffer(sequence_length=3, n_features=2)

        buffer.append([1.0, 2.0])
        buffer.append([3.0, 4.0])
        buffer.append([5.0, 6.0])

        tensor = buffer.get_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 2)  # Batch dimension added

    def test_rolling_window(self):
        """Test that buffer maintains rolling window."""
        buffer = FeatureBuffer(sequence_length=3, n_features=2)

        # Fill buffer
        for i in range(5):
            buffer.append([float(i), float(i)])

        seq = buffer.get_sequence()

        # Should have last 3 values
        assert seq[0, 0] == 2.0
        assert seq[2, 0] == 4.0

    def test_clear(self):
        """Test clearing buffer."""
        buffer = FeatureBuffer(sequence_length=3, n_features=2)

        buffer.append([1.0, 2.0])
        buffer.append([3.0, 4.0])

        buffer.clear()

        assert len(buffer) == 0
        assert not buffer.is_ready()

    def test_dimension_mismatch_padding(self):
        """Test that dimension mismatch is handled with padding."""
        buffer = FeatureBuffer(sequence_length=3, n_features=5)

        # Append fewer features than expected
        buffer.append([1.0, 2.0])

        assert len(buffer) == 1


class TestSymbolBufferManager:
    """Tests for SymbolBufferManager class."""

    def test_create_manager(self):
        """Test manager creation."""
        manager = SymbolBufferManager(sequence_length=60, n_features=17)

        assert manager.sequence_length == 60
        assert manager.n_features == 17

    def test_get_or_create_buffer(self):
        """Test getting or creating buffers."""
        manager = SymbolBufferManager(sequence_length=5, n_features=3)

        buffer1 = manager.get_buffer("AAPL")
        buffer2 = manager.get_buffer("AAPL")
        buffer3 = manager.get_buffer("GOOGL")

        assert buffer1 is buffer2
        assert buffer1 is not buffer3

    def test_append_to_symbol(self):
        """Test appending features to specific symbol."""
        manager = SymbolBufferManager(sequence_length=5, n_features=3)

        manager.append("AAPL", [1.0, 2.0, 3.0])
        manager.append("GOOGL", [4.0, 5.0, 6.0])

        assert len(manager.buffers["AAPL"]) == 1
        assert len(manager.buffers["GOOGL"]) == 1

    def test_is_ready_per_symbol(self):
        """Test readiness check per symbol."""
        manager = SymbolBufferManager(sequence_length=3, n_features=2)

        # Fill AAPL buffer
        for i in range(3):
            manager.append("AAPL", [1.0, 2.0])

        # Partially fill GOOGL
        manager.append("GOOGL", [1.0, 2.0])

        assert manager.is_ready("AAPL")
        assert not manager.is_ready("GOOGL")
        assert not manager.is_ready("MSFT")  # Never added

    def test_get_tensor_for_symbol(self):
        """Test getting tensor for specific symbol."""
        manager = SymbolBufferManager(sequence_length=3, n_features=2)

        for i in range(3):
            manager.append("AAPL", [float(i), float(i)])

        tensor = manager.get_tensor("AAPL")

        assert tensor.shape == (1, 3, 2)

    def test_get_stats(self):
        """Test getting buffer statistics."""
        manager = SymbolBufferManager(sequence_length=3, n_features=2)

        manager.append("AAPL", [1.0, 2.0])
        manager.append("AAPL", [1.0, 2.0])
        manager.append("AAPL", [1.0, 2.0])
        manager.append("GOOGL", [1.0, 2.0])

        stats = manager.get_stats()

        assert stats["num_symbols"] == 2
        assert stats["ready_symbols"] == 1
        assert stats["buffer_sizes"]["AAPL"] == 3
        assert stats["buffer_sizes"]["GOOGL"] == 1

    def test_clear_all(self):
        """Test clearing all buffers."""
        manager = SymbolBufferManager(sequence_length=3, n_features=2)

        manager.append("AAPL", [1.0, 2.0])
        manager.append("GOOGL", [1.0, 2.0])

        manager.clear()

        assert len(manager.buffers) == 0

    def test_clear_single_symbol(self):
        """Test clearing single symbol buffer."""
        manager = SymbolBufferManager(sequence_length=3, n_features=2)

        manager.append("AAPL", [1.0, 2.0])
        manager.append("GOOGL", [1.0, 2.0])

        manager.clear("AAPL")

        assert len(manager.buffers["AAPL"]) == 0
        assert len(manager.buffers["GOOGL"]) == 1

