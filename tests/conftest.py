"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    return {
        "timestamp": "2024-01-15T10:30:00Z",
        "symbol": "AAPL",
        "open": 185.50,
        "high": 186.20,
        "low": 185.10,
        "close": 185.90,
        "volume": 1250000,
    }


@pytest.fixture
def sample_features():
    """Sample feature vector for testing."""
    return {
        "rsi": 55.2,
        "macd": 0.45,
        "macd_signal": 0.38,
        "macd_hist": 0.07,
        "bb_upper": 190.0,
        "bb_middle": 185.0,
        "bb_lower": 180.0,
        "bb_width": 0.054,
        "sma_5": 185.5,
        "sma_10": 184.0,
        "sma_20": 183.0,
        "sma_50": 180.0,
        "ema_5": 185.3,
        "ema_10": 184.2,
        "ema_20": 183.5,
        "atr": 2.5,
        "volatility": 0.015,
    }


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from src.common.config import Settings

    return Settings()

