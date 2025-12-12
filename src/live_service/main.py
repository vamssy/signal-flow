"""Live price service that polls real-time stock data and publishes to Kafka."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import click
import yfinance as yf

from src.common import get_logger, get_settings, setup_logging
from src.common.config import Settings
from src.common.kafka_utils import KafkaMessageProducer
from src.common.schemas import OHLCVMessage

logger = get_logger(__name__)


class LivePriceService:
    """Service that polls live stock prices and publishes to Kafka."""

    def __init__(self, settings: Settings, poll_interval: int = 60):
        """Initialize the live price service.

        Args:
            settings: Application settings
            poll_interval: Seconds between price fetches
        """
        self.settings = settings
        self.poll_interval = poll_interval
        self.symbols = settings.symbols
        self.running = False

        # Stats
        self.total_messages = 0
        self.errors = 0
        self.start_time: float | None = None

    def fetch_live_prices(self) -> list[dict[str, Any]]:
        """Fetch current prices for all symbols.

        Returns:
            List of OHLCV dictionaries
        """
        prices = []

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get the most recent data (1 minute intervals, last 1 day)
                # This gives us the most current available data
                hist = ticker.history(period="1d", interval="1m")

                if hist.empty:
                    # Fallback to daily data
                    hist = ticker.history(period="1d", interval="1d")

                if not hist.empty:
                    # Get the last row (most recent)
                    latest = hist.iloc[-1]

                    price_data = {
                        "timestamp": datetime.now(timezone.utc),
                        "symbol": symbol,
                        "open": float(latest["Open"]),
                        "high": float(latest["High"]),
                        "low": float(latest["Low"]),
                        "close": float(latest["Close"]),
                        "volume": int(latest["Volume"]),
                    }
                    prices.append(price_data)

                    logger.debug(
                        "Fetched price",
                        symbol=symbol,
                        price=price_data["close"],
                    )
                else:
                    logger.warning("No data available", symbol=symbol)

            except Exception as e:
                logger.error("Failed to fetch price", symbol=symbol, error=str(e))
                self.errors += 1

        return prices

    def run(self, producer: KafkaMessageProducer) -> None:
        """Run the live price polling loop.

        Args:
            producer: Kafka producer to publish messages
        """
        self.running = True
        self.start_time = time.time()

        logger.info(
            "Starting live price service",
            symbols=self.symbols,
            poll_interval=f"{self.poll_interval}s",
            topic=self.settings.kafka.topics.ohlcv,
        )

        print("\n" + "=" * 60)
        print("  LIVE PRICE SERVICE STARTED")
        print("=" * 60)
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Poll interval: {self.poll_interval} seconds")
        print(f"  Publishing to: {self.settings.kafka.topics.ohlcv}")
        print("=" * 60)
        print("\n  Press Ctrl+C to stop\n")

        try:
            while self.running:
                cycle_start = time.time()

                # Fetch live prices
                prices = self.fetch_live_prices()

                if prices:
                    # Publish to Kafka
                    for price_data in prices:
                        message = OHLCVMessage(**price_data)
                        producer.send(
                            topic=self.settings.kafka.topics.ohlcv,
                            value=message.model_dump(mode="json"),
                            key=message.symbol,
                        )
                        self.total_messages += 1

                    producer.flush()

                    # Log status
                    elapsed = time.time() - self.start_time
                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] "
                        f"Published {len(prices)} prices | "
                        f"Total: {self.total_messages} | "
                        f"Uptime: {elapsed:.0f}s"
                    )

                    # Show current prices
                    for p in prices:
                        print(f"    {p['symbol']}: ${p['close']:.2f}")

                    print()

                # Wait for next poll
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, self.poll_interval - cycle_time)

                if sleep_time > 0:
                    logger.debug(f"Sleeping {sleep_time:.1f}s until next poll")
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Live service interrupted")
            self.running = False

        # Final stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        print("\n" + "=" * 60)
        print("  LIVE SERVICE STOPPED")
        print("=" * 60)
        print(f"  Total messages: {self.total_messages}")
        print(f"  Errors: {self.errors}")
        print(f"  Uptime: {elapsed:.0f} seconds")
        print("=" * 60 + "\n")

    def stop(self) -> None:
        """Stop the service."""
        self.running = False


@click.command()
@click.option(
    "--interval",
    "-i",
    default=60,
    help="Poll interval in seconds (default: 60)",
)
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    default=None,
    help="Symbols to track (default: from config)",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config file",
)
def main(
    interval: int,
    symbols: tuple[str, ...],
    config: str | None,
) -> None:
    """Run the live price polling service.

    This service fetches real-time stock prices and publishes them to Kafka
    for processing by the feature and inference services.

    Example:
        python -m src.live_service.main --interval 30
    """
    setup_logging(level="INFO")
    settings = get_settings(config)

    # Override symbols if provided
    if symbols:
        settings.symbols = list(symbols)

    # Create service
    service = LivePriceService(settings, poll_interval=interval)

    # Create producer and run
    try:
        with KafkaMessageProducer(settings) as producer:
            service.run(producer)
    except Exception as e:
        logger.error("Live service failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

