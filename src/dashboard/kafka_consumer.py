"""Async Kafka consumer for dashboard WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Set

from aiokafka import AIOKafkaConsumer

from src.common import get_logger
from src.common.config import Settings

logger = get_logger(__name__)


class SignalConsumer:
    """Async Kafka consumer that broadcasts signals to WebSocket clients."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.consumer: AIOKafkaConsumer | None = None
        self.running = False
        self._callbacks: Set[Callable[[dict], Any]] = set()

    def add_callback(self, callback: Callable[[dict], Any]) -> None:
        """Add a callback to receive signal updates."""
        self._callbacks.add(callback)

    def remove_callback(self, callback: Callable[[dict], Any]) -> None:
        """Remove a callback."""
        self._callbacks.discard(callback)

    async def _notify_callbacks(self, data: dict) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks.copy():
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Callback error", error=str(e))

    async def start(self) -> None:
        """Start consuming from Kafka."""
        if self.running:
            return

        self.consumer = AIOKafkaConsumer(
            self.settings.kafka.topics.signals,
            self.settings.kafka.topics.features,
            bootstrap_servers=self.settings.kafka.bootstrap_servers,
            group_id="dashboard-consumer",
            auto_offset_reset="latest",  # Only get new messages
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True,
        )

        await self.consumer.start()
        self.running = True
        logger.info(
            "Dashboard Kafka consumer started",
            topics=[self.settings.kafka.topics.signals, self.settings.kafka.topics.features],
        )

    async def stop(self) -> None:
        """Stop the consumer."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()
            logger.info("Dashboard Kafka consumer stopped")

    async def consume(self) -> None:
        """Main consume loop - broadcasts messages to callbacks."""
        if not self.consumer:
            await self.start()

        try:
            async for msg in self.consumer:
                if not self.running:
                    break

                data = {
                    "topic": msg.topic,
                    "partition": msg.partition,
                    "offset": msg.offset,
                    "timestamp": msg.timestamp,
                    "value": msg.value,
                }

                await self._notify_callbacks(data)

        except Exception as e:
            logger.error("Consumer error", error=str(e))
            if self.running:
                # Reconnect after delay
                await asyncio.sleep(1)
                await self.consume()


class PriceConsumer:
    """Async Kafka consumer for OHLCV price data."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.consumer: AIOKafkaConsumer | None = None
        self.running = False
        self._callbacks: Set[Callable[[dict], Any]] = set()

    def add_callback(self, callback: Callable[[dict], Any]) -> None:
        """Add a callback to receive price updates."""
        self._callbacks.add(callback)

    def remove_callback(self, callback: Callable[[dict], Any]) -> None:
        """Remove a callback."""
        self._callbacks.discard(callback)

    async def _notify_callbacks(self, data: dict) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks.copy():
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Callback error", error=str(e))

    async def start(self) -> None:
        """Start consuming from Kafka."""
        if self.running:
            return

        self.consumer = AIOKafkaConsumer(
            self.settings.kafka.topics.ohlcv,
            bootstrap_servers=self.settings.kafka.bootstrap_servers,
            group_id="dashboard-price-consumer",
            auto_offset_reset="latest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True,
        )

        await self.consumer.start()
        self.running = True
        logger.info("Dashboard price consumer started")

    async def stop(self) -> None:
        """Stop the consumer."""
        self.running = False
        if self.consumer:
            await self.consumer.stop()

    async def consume(self) -> None:
        """Main consume loop."""
        if not self.consumer:
            await self.start()

        try:
            async for msg in self.consumer:
                if not self.running:
                    break

                data = {
                    "topic": msg.topic,
                    "value": msg.value,
                }

                await self._notify_callbacks(data)

        except Exception as e:
            logger.error("Price consumer error", error=str(e))
            if self.running:
                await asyncio.sleep(1)
                await self.consume()

