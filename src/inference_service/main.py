"""Real-time inference service for generating trading signals."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import torch

from src.common import get_logger, get_settings, setup_logging
from src.common.config import Settings
from src.common.kafka_utils import KafkaMessageConsumer, KafkaMessageProducer
from src.common.schemas import FeaturesMessage, SignalMessage, SignalType
from src.inference_service.buffer import SymbolBufferManager
from src.training.model import load_model

logger = get_logger(__name__)


class InferenceService:
    """Real-time inference service for stock prediction.

    Consumes feature messages from Kafka, maintains rolling buffers per symbol,
    runs LSTM inference, and publishes trading signals.
    """

    def __init__(
        self,
        settings: Settings,
        model_path: Path,
        device: torch.device | None = None,
    ):
        """Initialize the inference service.

        Args:
            settings: Application settings
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.settings = settings
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load model
        logger.info("Loading model", path=str(model_path))
        self.model = load_model(str(model_path), self.device)
        self.model.eval()

        model_config = self.model.get_config()
        logger.info(
            "Model loaded",
            parameters=model_config["num_parameters"],
            input_dim=model_config["input_dim"],
        )

        # Initialize buffer manager
        self.buffer_manager = SymbolBufferManager(
            sequence_length=settings.features.sequence_length,
            n_features=model_config["input_dim"],
        )

        # Metrics
        self.messages_processed = 0
        self.signals_generated = 0
        self.total_latency_ms = 0.0
        self.start_time: float | None = None

        # Model version for tracking
        self.model_version = model_path.stem

    def process_message(self, message: dict[str, Any]) -> SignalMessage | None:
        """Process a feature message and generate a signal if ready.

        Args:
            message: Raw feature message dictionary

        Returns:
            SignalMessage if inference was performed, None otherwise
        """
        inference_start = time.perf_counter()

        try:
            # Parse message
            features_msg = FeaturesMessage.from_dict(message)
        except Exception as e:
            logger.error("Failed to parse message", error=str(e))
            return None

        self.messages_processed += 1
        symbol = features_msg.symbol

        # Get feature vector (prefer scaled features)
        if features_msg.features_scaled:
            feature_vector = features_msg.features_scaled
        else:
            feature_vector = list(features_msg.features.values())

        # Add to buffer
        self.buffer_manager.append(symbol, feature_vector)

        # Check if buffer is ready for inference
        if not self.buffer_manager.is_ready(symbol):
            return None

        # Run inference
        try:
            with torch.no_grad():
                input_tensor = self.buffer_manager.get_tensor(symbol, self.device)
                probability = self.model.predict(input_tensor)
                prob_value = probability.item()
        except Exception as e:
            logger.error("Inference failed", symbol=symbol, error=str(e))
            return None

        # Calculate latency
        latency_ms = (time.perf_counter() - inference_start) * 1000
        self.total_latency_ms += latency_ms

        # Generate signal
        signal = self._generate_signal(
            symbol=symbol,
            probability=prob_value,
            timestamp=features_msg.timestamp,
            latency_ms=latency_ms,
        )

        self.signals_generated += 1

        return signal

    def _generate_signal(
        self,
        symbol: str,
        probability: float,
        timestamp: datetime,
        latency_ms: float,
    ) -> SignalMessage:
        """Generate a trading signal from model output.

        Args:
            symbol: Stock symbol
            probability: Model output probability (prob of UP)
            timestamp: Message timestamp
            latency_ms: Inference latency

        Returns:
            SignalMessage
        """
        thresholds = self.settings.inference.signal_thresholds

        # Determine signal type
        if probability >= thresholds.buy:
            signal_type = SignalType.BUY
            predicted_direction = "UP"
        elif probability <= thresholds.sell:
            signal_type = SignalType.SELL
            predicted_direction = "DOWN"
        else:
            signal_type = SignalType.HOLD
            predicted_direction = "NEUTRAL"

        # Confidence is distance from 0.5 (uncertainty)
        confidence = abs(probability - 0.5) * 2  # Scale to [0, 1]

        return SignalMessage(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal=signal_type,
            confidence=confidence,
            predicted_direction=predicted_direction,
            raw_probability=probability,
            model_version=self.model_version,
            latency_ms=latency_ms,
        )

    def run(
        self,
        max_messages: int | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        """Run the inference service.

        Args:
            max_messages: Maximum messages to process (None for infinite)
            timeout_ms: Kafka poll timeout
        """
        self.start_time = time.time()

        logger.info(
            "Starting inference service",
            input_topic=self.settings.kafka.topics.features,
            output_topic=self.settings.kafka.topics.signals,
            device=str(self.device),
        )

        with KafkaMessageProducer(self.settings) as producer:
            with KafkaMessageConsumer(
                self.settings,
                topics=[self.settings.kafka.topics.features],
                group_id="inference-service",
            ) as consumer:

                def handle_message(message: dict[str, Any]) -> None:
                    signal = self.process_message(message)

                    if signal is not None:
                        producer.send(
                            topic=self.settings.kafka.topics.signals,
                            value=signal.model_dump(mode="json"),
                            key=signal.symbol,
                        )

                        # Log signal
                        if self.signals_generated % 100 == 0:
                            self._log_progress()

                consumer.consume(
                    handler=handle_message,
                    max_messages=max_messages,
                    timeout_ms=timeout_ms,
                )

        self._log_final_stats()

    def _log_progress(self) -> None:
        """Log progress metrics."""
        elapsed = time.time() - self.start_time if self.start_time else 1
        avg_latency = (
            self.total_latency_ms / self.signals_generated
            if self.signals_generated > 0
            else 0
        )

        logger.info(
            "Inference progress",
            processed=self.messages_processed,
            signals=self.signals_generated,
            rate=f"{self.signals_generated / elapsed:.1f} sig/s",
            avg_latency_ms=f"{avg_latency:.2f}",
            buffer_stats=self.buffer_manager.get_stats(),
        )

    def _log_final_stats(self) -> None:
        """Log final statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 1
        avg_latency = (
            self.total_latency_ms / self.signals_generated
            if self.signals_generated > 0
            else 0
        )

        logger.info(
            "Inference service stopped",
            processed=self.messages_processed,
            signals=self.signals_generated,
            elapsed=f"{elapsed:.2f}s",
            throughput=f"{self.signals_generated / elapsed:.1f} sig/s",
            avg_latency_ms=f"{avg_latency:.2f}",
        )


@click.command()
@click.option(
    "--model-path",
    "-m",
    default=None,
    help="Path to trained model checkpoint (default: from config)",
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
    model_path: str | None,
    max_messages: int | None,
    config: str | None,
) -> None:
    """Run the real-time inference service."""
    setup_logging(level="INFO")
    settings = get_settings(config)

    # Resolve model path
    path = Path(model_path) if model_path else Path(settings.inference.model_path)

    if not path.exists():
        logger.error("Model not found", path=str(path))
        logger.info("Run 'python -m src.training.train' to train a model first")
        return

    # Create and run service
    service = InferenceService(settings, path)

    try:
        service.run(max_messages=max_messages)
    except KeyboardInterrupt:
        logger.info("Inference service interrupted")
    except Exception as e:
        logger.error("Inference service failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

