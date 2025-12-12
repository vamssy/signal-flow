"""Tests for Pydantic message schemas."""

import pytest
from datetime import datetime, timezone

from src.common.schemas import (
    OHLCVMessage,
    FeaturesMessage,
    SignalMessage,
    SignalType,
    serialize_message,
    deserialize_ohlcv,
)


class TestOHLCVMessage:
    """Tests for OHLCVMessage schema."""

    def test_create_valid_message(self):
        """Test creating a valid OHLCV message."""
        msg = OHLCVMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
        )

        assert msg.symbol == "AAPL"
        assert msg.open == 150.0
        assert msg.close == 154.0

    def test_symbol_normalization(self):
        """Test that symbols are normalized to uppercase."""
        msg = OHLCVMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="  aapl  ",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
        )

        assert msg.symbol == "AAPL"

    def test_serialization_roundtrip(self):
        """Test JSON serialization and deserialization."""
        original = OHLCVMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="GOOGL",
            open=140.0,
            high=145.0,
            low=139.0,
            close=144.0,
            volume=500000,
        )

        # Serialize
        json_str = original.to_json()
        assert isinstance(json_str, str)

        # Deserialize
        restored = OHLCVMessage.from_json(json_str)
        assert restored.symbol == original.symbol
        assert restored.close == original.close

    def test_bytes_serialization(self):
        """Test bytes serialization for Kafka."""
        msg = OHLCVMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="MSFT",
            open=300.0,
            high=305.0,
            low=299.0,
            close=304.0,
            volume=750000,
        )

        data = msg.to_bytes()
        assert isinstance(data, bytes)

        restored = deserialize_ohlcv(data)
        assert restored.symbol == "MSFT"


class TestSignalMessage:
    """Tests for SignalMessage schema."""

    def test_create_buy_signal(self):
        """Test creating a BUY signal."""
        msg = SignalMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            signal=SignalType.BUY,
            confidence=0.85,
            predicted_direction="UP",
            raw_probability=0.92,
        )

        assert msg.signal == SignalType.BUY
        assert msg.confidence == 0.85

    def test_create_sell_signal(self):
        """Test creating a SELL signal."""
        msg = SignalMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            signal=SignalType.SELL,
            confidence=0.75,
            predicted_direction="DOWN",
            raw_probability=0.12,
        )

        assert msg.signal == SignalType.SELL

    def test_confidence_bounds(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            SignalMessage(
                timestamp=datetime.now(timezone.utc),
                symbol="AAPL",
                signal=SignalType.BUY,
                confidence=1.5,  # Invalid
                predicted_direction="UP",
                raw_probability=0.9,
            )

    def test_optional_fields(self):
        """Test optional fields are handled correctly."""
        msg = SignalMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            signal=SignalType.HOLD,
            confidence=0.5,
            predicted_direction="NEUTRAL",
            raw_probability=0.5,
            model_version="v1.0",
            latency_ms=12.5,
        )

        assert msg.model_version == "v1.0"
        assert msg.latency_ms == 12.5


class TestFeaturesMessage:
    """Tests for FeaturesMessage schema."""

    def test_create_with_features(self):
        """Test creating message with features dict."""
        msg = FeaturesMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            features={"rsi": 55.0, "macd": 0.5},
        )

        assert msg.features["rsi"] == 55.0

    def test_get_feature_vector(self):
        """Test extracting feature vector."""
        msg = FeaturesMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            features={"rsi": 55.0, "macd": 0.5},
            features_scaled=[0.55, 0.62],
        )

        # Should prefer scaled features
        vector = msg.get_feature_vector()
        assert vector == [0.55, 0.62]

    def test_get_feature_vector_unscaled(self):
        """Test feature vector falls back to unscaled."""
        msg = FeaturesMessage(
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            features={"rsi": 55.0, "macd": 0.5},
        )

        vector = msg.get_feature_vector()
        assert 55.0 in vector
        assert 0.5 in vector

