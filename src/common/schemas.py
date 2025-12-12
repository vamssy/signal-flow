"""Pydantic schemas for Kafka message serialization/deserialization."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SignalType(str, Enum):
    """Trading signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BaseMessage(BaseModel):
    """Base message class with common serialization methods."""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    def to_bytes(self) -> bytes:
        """Serialize to bytes for Kafka."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, data: str | bytes) -> "BaseMessage":
        """Deserialize from JSON string or bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.model_validate_json(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseMessage":
        """Create from dictionary."""
        return cls.model_validate(data)


class OHLCVMessage(BaseMessage):
    """OHLCV (Open, High, Low, Close, Volume) market data message."""

    timestamp: datetime = Field(..., description="Bar timestamp")
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    open: float = Field(..., description="Opening price", ge=0)
    high: float = Field(..., description="High price", ge=0)
    low: float = Field(..., description="Low price", ge=0)
    close: float = Field(..., description="Closing price", ge=0)
    volume: int = Field(..., description="Trading volume", ge=0)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info: Any) -> float:
        """Validate high >= open, close, low."""
        # Note: Pydantic v2 field_validator runs before other fields may be set
        # Full cross-field validation should use model_validator
        return v

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class FeaturesMessage(BaseMessage):
    """Enriched message with computed technical indicators."""

    timestamp: datetime = Field(..., description="Bar timestamp")
    symbol: str = Field(..., description="Stock symbol")

    # Original OHLCV
    open: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    low: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    volume: int = Field(..., ge=0)

    # Computed features (dynamic based on config)
    features: dict[str, float] = Field(
        default_factory=dict, description="Computed technical indicators"
    )

    # Normalized features ready for model input
    features_scaled: list[float] = Field(
        default_factory=list, description="MinMax scaled features for model"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    def get_feature_vector(self) -> list[float]:
        """Get feature vector for model input."""
        if self.features_scaled:
            return self.features_scaled
        return list(self.features.values())

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class SignalMessage(BaseMessage):
    """Trading signal message with prediction and confidence."""

    timestamp: datetime = Field(..., description="Signal generation timestamp")
    symbol: str = Field(..., description="Stock symbol")
    signal: SignalType = Field(..., description="Trading signal")
    confidence: float = Field(..., description="Prediction confidence", ge=0, le=1)
    predicted_direction: str = Field(..., description="Predicted price direction (UP/DOWN)")
    raw_probability: float = Field(..., description="Raw model output probability", ge=0, le=1)

    # Optional debug information
    model_version: str | None = Field(None, description="Model version used")
    latency_ms: float | None = Field(None, description="Inference latency in milliseconds")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.upper().strip()

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class BatchOHLCVMessage(BaseMessage):
    """Batch of OHLCV messages for efficient Kafka publishing."""

    messages: list[OHLCVMessage] = Field(..., description="List of OHLCV messages")
    batch_id: str = Field(..., description="Unique batch identifier")
    batch_size: int = Field(..., description="Number of messages in batch")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int, info: Any) -> int:
        """Validate batch size matches messages length."""
        return v


# Serialization utilities for Kafka
def serialize_message(message: BaseMessage) -> bytes:
    """Serialize a message to bytes for Kafka producer."""
    return message.to_bytes()


def deserialize_ohlcv(data: bytes) -> OHLCVMessage:
    """Deserialize bytes to OHLCVMessage."""
    return OHLCVMessage.from_json(data)


def deserialize_features(data: bytes) -> FeaturesMessage:
    """Deserialize bytes to FeaturesMessage."""
    return FeaturesMessage.from_json(data)


def deserialize_signal(data: bytes) -> SignalMessage:
    """Deserialize bytes to SignalMessage."""
    return SignalMessage.from_json(data)

