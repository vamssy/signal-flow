"""Tests for technical indicator calculations."""

import pytest
import numpy as np

from src.common.config import FeaturesConfig
from src.feature_service.indicators import StreamingIndicators, RollingWindow


class TestRollingWindow:
    """Tests for RollingWindow class."""

    def test_append_and_size(self):
        """Test appending values and checking size."""
        window = RollingWindow(5)

        for i in range(3):
            window.append(float(i))

        assert len(window) == 3
        assert not window.is_full()

    def test_is_full(self):
        """Test is_full check."""
        window = RollingWindow(3)

        for i in range(3):
            window.append(float(i))

        assert window.is_full()

    def test_max_size(self):
        """Test that window maintains max size."""
        window = RollingWindow(3)

        for i in range(10):
            window.append(float(i))

        assert len(window) == 3
        assert list(window.data) == [7.0, 8.0, 9.0]

    def test_mean(self):
        """Test mean calculation."""
        window = RollingWindow(3)
        window.append(1.0)
        window.append(2.0)
        window.append(3.0)

        assert window.mean() == 2.0

    def test_std(self):
        """Test standard deviation calculation."""
        window = RollingWindow(3)
        window.append(1.0)
        window.append(2.0)
        window.append(3.0)

        assert window.std() == pytest.approx(1.0, rel=1e-5)

    def test_to_array(self):
        """Test conversion to numpy array."""
        window = RollingWindow(3)
        window.append(1.0)
        window.append(2.0)
        window.append(3.0)

        arr = window.to_array()
        assert isinstance(arr, np.ndarray)
        assert list(arr) == [1.0, 2.0, 3.0]


class TestStreamingIndicators:
    """Tests for StreamingIndicators class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return FeaturesConfig(
            sequence_length=10,
            include_returns=True,
            include_log_returns=True,
            include_volatility=True,
            volatility_window=5,
            include_volume_zscore=True,
            volume_zscore_window=5,
        )

    @pytest.fixture
    def indicators(self, config):
        """Create indicators calculator."""
        return StreamingIndicators(config)

    def test_compute_returns_features(self, indicators):
        """Test that features are computed."""
        # Generate some price data
        for i in range(100):
            price = 100 + i * 0.1 + np.random.randn() * 0.5
            features = indicators.compute(
                open_price=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
            )

        # After warmup, should have features
        assert indicators.is_warmed_up
        assert "macd" in features
        assert "rsi" in features or "returns" in features

    def test_warmup_period(self, indicators):
        """Test warmup period calculation."""
        warmup = indicators.warmup_period
        assert warmup > 0
        assert warmup >= 20  # At least BBands window

    def test_not_warmed_up_initially(self, indicators):
        """Test that indicator is not warmed up initially."""
        assert not indicators.is_warmed_up

    def test_reset(self, indicators):
        """Test resetting the indicator state."""
        # Add some data
        for i in range(50):
            indicators.compute(100.0, 101.0, 99.0, 100.0, 1000000)

        # Reset
        indicators.reset()

        assert not indicators.is_warmed_up
        assert indicators.message_count == 0

    def test_get_feature_names(self, indicators):
        """Test getting feature names list."""
        names = indicators.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert "macd" in names
        assert "rsi" in names

    def test_macd_calculation(self, indicators):
        """Test MACD is calculated correctly."""
        # Feed stable prices
        for i in range(100):
            features = indicators.compute(100.0, 101.0, 99.0, 100.0, 1000000)

        # With stable prices, MACD should be near 0
        assert "macd" in features
        assert abs(features["macd"]) < 1.0

    def test_rsi_bounds(self, indicators):
        """Test RSI stays within bounds."""
        # Feed increasing prices
        for i in range(100):
            price = 100 + i
            features = indicators.compute(price, price + 1, price - 1, price, 1000000)

        if "rsi" in features:
            assert 0 <= features["rsi"] <= 100

    def test_volume_zscore(self, indicators):
        """Test volume z-score calculation."""
        # Feed consistent volume
        for i in range(100):
            features = indicators.compute(100.0, 101.0, 99.0, 100.0, 1000000)

        if "volume_zscore" in features:
            # With consistent volume, z-score should be near 0
            assert abs(features["volume_zscore"]) < 0.5

