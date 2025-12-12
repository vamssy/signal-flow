"""Feature service for computing technical indicators on streaming OHLCV data."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from src.common import get_logger, get_settings, setup_logging
from src.common.config import Settings
from src.common.kafka_utils import KafkaMessageConsumer, KafkaMessageProducer
from src.common.schemas import FeaturesMessage, OHLCVMessage
from src.feature_service.indicators import StreamingIndicators
from src.feature_service.scaler import StreamingScaler

logger = get_logger(__name__)


class FeatureService:
    """Service for computing features on streaming OHLCV data."""

    def __init__(self, settings: Settings):
        self.settings = settings

        # Per-symbol indicator calculators
        self.indicators: dict[str, StreamingIndicators] = defaultdict(
            lambda: StreamingIndicators(settings.features)
        )

        # Optional pre-fitted scaler
        self.scaler: StreamingScaler | None = None

        # Metrics
        self.messages_processed = 0
        self.messages_published = 0
        self.start_time: float | None = None

    def load_scaler(self, scaler_path: Path, feature_list_path: Path) -> None:
        """Load a pre-fitted scaler for feature normalization.

        Args:
            scaler_path: Path to scaler pickle file
            feature_list_path: Path to feature names JSON
        """
        if scaler_path.exists() and feature_list_path.exists():
            self.scaler = StreamingScaler.load(scaler_path, feature_list_path)
            logger.info("Loaded pre-fitted scaler")
        else:
            logger.warning(
                "Scaler files not found, features will not be scaled",
                scaler_path=str(scaler_path),
            )

    def process_message(self, message: dict[str, Any]) -> FeaturesMessage | None:
        """Process an OHLCV message and compute features.

        Args:
            message: Raw OHLCV message dictionary

        Returns:
            FeaturesMessage with computed indicators, or None if still warming up
        """
        try:
            # Parse message
            ohlcv = OHLCVMessage.from_dict(message)
        except Exception as e:
            logger.error("Failed to parse message", error=str(e), message=message)
            return None

        # Get or create indicator calculator for this symbol
        indicator_calc = self.indicators[ohlcv.symbol]

        # Compute features
        features = indicator_calc.compute(
            open_price=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
        )

        self.messages_processed += 1

        # Skip if still in warmup period
        if not indicator_calc.is_warmed_up:
            return None

        # Scale features if scaler is available
        scaled_features: list[float] = []
        if self.scaler is not None and self.scaler.is_fitted:
            try:
                scaled_features = self.scaler.transform(features)
            except Exception as e:
                logger.warning("Failed to scale features", error=str(e))

        # Create output message
        features_message = FeaturesMessage(
            timestamp=ohlcv.timestamp,
            symbol=ohlcv.symbol,
            open=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
            features=features,
            features_scaled=scaled_features,
        )

        return features_message

    def run(
        self,
        max_messages: int | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        """Run the feature service, consuming from OHLCV topic and publishing to features topic.

        Args:
            max_messages: Maximum messages to process (None for infinite)
            timeout_ms: Kafka poll timeout in milliseconds
        """
        self.start_time = time.time()

        logger.info(
            "Starting feature service",
            input_topic=self.settings.kafka.topics.ohlcv,
            output_topic=self.settings.kafka.topics.features,
        )

        with KafkaMessageProducer(self.settings) as producer:
            with KafkaMessageConsumer(
                self.settings,
                topics=[self.settings.kafka.topics.ohlcv],
            ) as consumer:

                def handle_message(message: dict[str, Any]) -> None:
                    features_msg = self.process_message(message)

                    if features_msg is not None:
                        producer.send(
                            topic=self.settings.kafka.topics.features,
                            value=features_msg.model_dump(mode="json"),
                            key=features_msg.symbol,
                        )
                        self.messages_published += 1

                        # Progress logging
                        if self.messages_published % 100 == 0:
                            elapsed = time.time() - self.start_time
                            rate = self.messages_published / elapsed if elapsed > 0 else 0
                            logger.info(
                                "Feature service progress",
                                processed=self.messages_processed,
                                published=self.messages_published,
                                rate=f"{rate:.1f} msg/s",
                            )

                consumer.consume(
                    handler=handle_message,
                    max_messages=max_messages,
                    timeout_ms=timeout_ms,
                )

        # Final stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(
            "Feature service stopped",
            processed=self.messages_processed,
            published=self.messages_published,
            elapsed=f"{elapsed:.2f}s",
        )


@click.command()
@click.option(
    "--scaler-path",
    default=None,
    help="Path to pre-fitted scaler pickle file",
)
@click.option(
    "--feature-list-path",
    default=None,
    help="Path to feature names JSON file",
)
@click.option(
    "--max-messages",
    "-n",
    default=None,
    type=int,
    help="Maximum messages to process (default: infinite)",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config file",
)
def main(
    scaler_path: str | None,
    feature_list_path: str | None,
    max_messages: int | None,
    config: str | None,
) -> None:
    """Run the feature engineering service."""
    setup_logging(level="INFO")
    settings = get_settings(config)

    service = FeatureService(settings)

    # Load scaler if paths provided
    if scaler_path and feature_list_path:
        service.load_scaler(Path(scaler_path), Path(feature_list_path))
    elif settings.inference.scaler_path and settings.inference.feature_list_path:
        service.load_scaler(
            Path(settings.inference.scaler_path),
            Path(settings.inference.feature_list_path),
        )

    try:
        service.run(max_messages=max_messages)
    except KeyboardInterrupt:
        logger.info("Feature service interrupted")
    except Exception as e:
        logger.error("Feature service failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

