"""Replay service for publishing historical OHLCV data to Kafka."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import click
import pandas as pd

from src.common import get_logger, get_settings, setup_logging
from src.common.kafka_utils import KafkaMessageProducer
from src.common.schemas import OHLCVMessage

logger = get_logger(__name__)


def load_ohlcv_data(data_path: Path, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load OHLCV data from CSV files.

    Args:
        data_path: Path to data directory or file
        symbols: Optional list of symbols to filter

    Returns:
        DataFrame with OHLCV data sorted by timestamp
    """
    if data_path.is_file():
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        # Load all CSV files in directory
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        dfs = []
        for csv_file in csv_files:
            if csv_file.name == "combined.csv":
                continue  # Skip combined file, load individual ones
            try:
                df = pd.read_csv(csv_file, parse_dates=["timestamp"])
                dfs.append(df)
                logger.info("Loaded data file", file=csv_file.name, rows=len(df))
            except Exception as e:
                logger.warning("Failed to load file", file=csv_file.name, error=str(e))

        if not dfs:
            # Try combined file as fallback
            combined_path = data_path / "combined.csv"
            if combined_path.exists():
                df = pd.read_csv(combined_path, parse_dates=["timestamp"])
            else:
                raise FileNotFoundError(f"No valid data files in {data_path}")
        else:
            df = pd.concat(dfs, ignore_index=True)

    # Filter symbols if specified
    if symbols:
        df = df[df["symbol"].isin(symbols)]

    # Sort by timestamp for deterministic ordering
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Loaded OHLCV data", total_rows=len(df), symbols=df["symbol"].unique().tolist())

    return df


def generate_messages(df: pd.DataFrame) -> Iterator[OHLCVMessage]:
    """Generate OHLCVMessage objects from DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Yields:
        OHLCVMessage objects
    """
    for _, row in df.iterrows():
        yield OHLCVMessage(
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
        )


def replay_data(
    producer: KafkaMessageProducer,
    df: pd.DataFrame,
    topic: str,
    replay_speed: float = 1.0,
    batch_size: int = 100,
) -> int:
    """Replay OHLCV data to Kafka with controlled speed.

    Args:
        producer: Kafka producer
        df: DataFrame with OHLCV data
        topic: Target Kafka topic
        replay_speed: Speed multiplier (0 = as fast as possible, 1 = real-time)
        batch_size: Number of messages per batch for progress reporting

    Returns:
        Number of messages sent
    """
    messages_sent = 0
    start_time = time.time()
    last_timestamp = None

    logger.info(
        "Starting replay",
        topic=topic,
        total_messages=len(df),
        replay_speed=replay_speed,
    )

    for message in generate_messages(df):
        # Simulate real-time delays if replay_speed > 0
        if replay_speed > 0 and last_timestamp is not None:
            time_diff = (message.timestamp - last_timestamp).total_seconds()
            if time_diff > 0:
                sleep_time = time_diff / replay_speed
                # Cap sleep time to avoid very long waits
                sleep_time = min(sleep_time, 1.0)
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        # Send message
        producer.send(
            topic=topic,
            value=message.model_dump(mode="json"),
            key=message.symbol,
        )

        messages_sent += 1
        last_timestamp = message.timestamp

        # Progress reporting
        if messages_sent % batch_size == 0:
            elapsed = time.time() - start_time
            rate = messages_sent / elapsed if elapsed > 0 else 0
            logger.info(
                "Replay progress",
                messages_sent=messages_sent,
                total=len(df),
                rate=f"{rate:.1f} msg/s",
                progress=f"{100 * messages_sent / len(df):.1f}%",
            )

    # Final flush
    producer.flush()

    elapsed = time.time() - start_time
    rate = messages_sent / elapsed if elapsed > 0 else 0

    logger.info(
        "Replay complete",
        messages_sent=messages_sent,
        elapsed=f"{elapsed:.2f}s",
        rate=f"{rate:.1f} msg/s",
    )

    return messages_sent


@click.command()
@click.option(
    "--data-path",
    "-d",
    default=None,
    help="Path to OHLCV data directory or file (default: from config)",
)
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    default=None,
    help="Symbols to replay (default: all in data)",
)
@click.option(
    "--speed",
    "-x",
    default=0.0,
    help="Replay speed (0 = max speed, 1 = real-time)",
)
@click.option(
    "--batch-size",
    "-b",
    default=100,
    help="Batch size for progress reporting",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config file",
)
def main(
    data_path: str | None,
    symbols: tuple[str, ...],
    speed: float,
    batch_size: int,
    config: str | None,
) -> None:
    """Replay historical OHLCV data to Kafka."""
    setup_logging(level="INFO")
    settings = get_settings(config)

    # Resolve data path
    path = Path(data_path) if data_path else Path(settings.replay.data_path)
    if not path.exists():
        logger.error("Data path not found", path=str(path))
        logger.info("Run 'python -m src.data.download' to download sample data first")
        return

    # Filter symbols if specified
    symbol_list = list(symbols) if symbols else None

    try:
        # Load data
        df = load_ohlcv_data(path, symbol_list)

        if df.empty:
            logger.error("No data to replay")
            return

        # Create producer and replay
        with KafkaMessageProducer(settings) as producer:
            replay_data(
                producer=producer,
                df=df,
                topic=settings.kafka.topics.ohlcv,
                replay_speed=speed,
                batch_size=batch_size,
            )

    except Exception as e:
        logger.error("Replay failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

