"""Configuration management for the stock prediction system."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class KafkaTopicsConfig(BaseModel):
    """Kafka topics configuration."""

    ohlcv: str = "market_ohlcv"
    features: str = "features"
    signals: str = "signals"


class KafkaConfig(BaseModel):
    """Kafka configuration."""

    bootstrap_servers: str = "localhost:9092"
    topics: KafkaTopicsConfig = Field(default_factory=KafkaTopicsConfig)
    consumer_group: str = "stock-prediction-group"
    auto_offset_reset: str = "earliest"


class RSIConfig(BaseModel):
    """RSI indicator configuration."""

    window: int = 14


class MACDConfig(BaseModel):
    """MACD indicator configuration."""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


class BollingerBandsConfig(BaseModel):
    """Bollinger Bands configuration."""

    window: int = 20
    num_std: int = 2


class SMAConfig(BaseModel):
    """Simple Moving Average configuration."""

    windows: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])


class EMAConfig(BaseModel):
    """Exponential Moving Average configuration."""

    windows: list[int] = Field(default_factory=lambda: [5, 10, 20])


class ATRConfig(BaseModel):
    """Average True Range configuration."""

    window: int = 14


class IndicatorsConfig(BaseModel):
    """Technical indicators configuration."""

    rsi: RSIConfig = Field(default_factory=RSIConfig)
    macd: MACDConfig = Field(default_factory=MACDConfig)
    bollinger_bands: BollingerBandsConfig = Field(default_factory=BollingerBandsConfig)
    sma: SMAConfig = Field(default_factory=SMAConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    atr: ATRConfig = Field(default_factory=ATRConfig)


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    sequence_length: int = 60
    indicators: IndicatorsConfig = Field(default_factory=IndicatorsConfig)
    include_returns: bool = True
    include_log_returns: bool = True
    include_volatility: bool = True
    volatility_window: int = 20
    include_volume_zscore: bool = True
    volume_zscore_window: int = 20


class ModelConfig(BaseModel):
    """LSTM model configuration."""

    input_dim: int = 17
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 10
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


class SignalThresholdsConfig(BaseModel):
    """Signal thresholds configuration."""

    buy: float = 0.55
    sell: float = 0.45


class InferenceConfig(BaseModel):
    """Inference configuration."""

    model_path: str = "artifacts/model.pt"
    scaler_path: str = "artifacts/scaler.pkl"
    feature_list_path: str = "artifacts/feature_list.json"
    confidence_threshold: float = 0.6
    signal_thresholds: SignalThresholdsConfig = Field(default_factory=SignalThresholdsConfig)


class MLflowConfig(BaseModel):
    """MLflow configuration."""

    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "stock-prediction-lstm"


class ReplayConfig(BaseModel):
    """Replay service configuration."""

    data_path: str = "data/ohlcv"
    replay_speed: float = 1.0
    batch_size: int = 100


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    enable_metrics: bool = True
    metrics_port: int = 8000
    log_level: str = "INFO"


class Settings(BaseSettings):
    """Application settings loaded from config file and environment."""

    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    symbols: list[str] = Field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    model_config = {"env_prefix": "STOCK_", "env_nested_delimiter": "__"}

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Load settings from a YAML configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data: dict[str, Any] = yaml.safe_load(f)

        return cls(**config_data)


@lru_cache
def get_settings(config_path: str | None = None) -> Settings:
    """Get cached settings instance.

    Args:
        config_path: Path to config file. If None, uses CONFIG_PATH env var
                    or defaults to configs/config.yaml

    Returns:
        Settings instance
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")

    path = Path(config_path)
    if path.exists():
        return Settings.from_yaml(path)

    # Return default settings if no config file
    return Settings()

