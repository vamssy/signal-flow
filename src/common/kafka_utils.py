"""Kafka producer and consumer utilities."""

from __future__ import annotations

import json
from typing import Any, Callable

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from src.common.config import Settings
from src.common.logging import get_logger

logger = get_logger(__name__)


def create_producer(settings: Settings) -> KafkaProducer:
    """Create a Kafka producer with JSON serialization.

    Args:
        settings: Application settings

    Returns:
        Configured KafkaProducer
    """
    producer = KafkaProducer(
        bootstrap_servers=settings.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
        max_in_flight_requests_per_connection=1,
        compression_type="gzip",
    )
    logger.info("Kafka producer created", bootstrap_servers=settings.kafka.bootstrap_servers)
    return producer


def create_consumer(
    settings: Settings,
    topics: list[str],
    group_id: str | None = None,
    auto_offset_reset: str | None = None,
) -> KafkaConsumer:
    """Create a Kafka consumer with JSON deserialization.

    Args:
        settings: Application settings
        topics: List of topics to subscribe to
        group_id: Consumer group ID (default from settings)
        auto_offset_reset: Offset reset policy (default from settings)

    Returns:
        Configured KafkaConsumer
    """
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=settings.kafka.bootstrap_servers,
        group_id=group_id or settings.kafka.consumer_group,
        auto_offset_reset=auto_offset_reset or settings.kafka.auto_offset_reset,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
    )
    logger.info(
        "Kafka consumer created",
        bootstrap_servers=settings.kafka.bootstrap_servers,
        topics=topics,
        group_id=group_id or settings.kafka.consumer_group,
    )
    return consumer


class KafkaMessageProducer:
    """High-level Kafka producer wrapper with batching and metrics."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.producer = create_producer(settings)
        self.messages_sent = 0
        self.errors = 0

    def send(
        self,
        topic: str,
        value: dict[str, Any],
        key: str | None = None,
        callback: Callable | None = None,
    ) -> None:
        """Send a message to a Kafka topic.

        Args:
            topic: Target topic
            value: Message value (will be JSON serialized)
            key: Optional message key for partitioning
            callback: Optional callback for delivery confirmation
        """
        try:
            future = self.producer.send(topic, value=value, key=key)
            if callback:
                future.add_callback(callback)
            self.messages_sent += 1
        except KafkaError as e:
            self.errors += 1
            logger.error("Failed to send message", topic=topic, error=str(e))
            raise

    def flush(self, timeout: float | None = None) -> None:
        """Flush pending messages.

        Args:
            timeout: Maximum time to wait for flush
        """
        self.producer.flush(timeout=timeout)

    def close(self) -> None:
        """Close the producer."""
        self.flush()
        self.producer.close()
        logger.info(
            "Producer closed",
            messages_sent=self.messages_sent,
            errors=self.errors,
        )

    def __enter__(self) -> "KafkaMessageProducer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class KafkaMessageConsumer:
    """High-level Kafka consumer wrapper with metrics."""

    def __init__(
        self,
        settings: Settings,
        topics: list[str],
        group_id: str | None = None,
    ):
        self.settings = settings
        self.topics = topics
        self.consumer = create_consumer(settings, topics, group_id)
        self.messages_received = 0

    def consume(
        self,
        handler: Callable[[dict[str, Any]], None],
        max_messages: int | None = None,
        timeout_ms: int = 1000,
    ) -> None:
        """Consume messages and process with handler.

        Args:
            handler: Function to process each message
            max_messages: Maximum messages to consume (None for infinite)
            timeout_ms: Poll timeout in milliseconds
        """
        try:
            while max_messages is None or self.messages_received < max_messages:
                records = self.consumer.poll(timeout_ms=timeout_ms)
                for topic_partition, messages in records.items():
                    for message in messages:
                        try:
                            handler(message.value)
                            self.messages_received += 1
                        except Exception as e:
                            logger.error(
                                "Error processing message",
                                topic=topic_partition.topic,
                                offset=message.offset,
                                error=str(e),
                            )
        except KeyboardInterrupt:
            logger.info("Consumer interrupted")
        finally:
            self.close()

    def close(self) -> None:
        """Close the consumer."""
        self.consumer.close()
        logger.info(
            "Consumer closed",
            messages_received=self.messages_received,
        )

    def __enter__(self) -> "KafkaMessageConsumer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

