"""Download historical OHLCV data using yfinance."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd
import yfinance as yf

from src.common import get_logger, get_settings, setup_logging

logger = get_logger(__name__)


def download_ohlcv(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1h",
) -> pd.DataFrame:
    """Download OHLCV data for a symbol.

    Args:
        symbol: Stock symbol (e.g., AAPL)
        start_date: Start date for data
        end_date: End date for data
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)

    Returns:
        DataFrame with OHLCV data
    """
    logger.info("Downloading data", symbol=symbol, start=start_date, end=end_date, interval=interval)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        logger.warning("No data returned", symbol=symbol)
        return df

    # Normalize column names
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    # Select only OHLCV columns
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # Reset index to get timestamp as column
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "timestamp", "Date": "timestamp"})

    # Add symbol column
    df["symbol"] = symbol

    # Ensure proper ordering
    df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    logger.info("Downloaded data", symbol=symbol, rows=len(df))

    return df


def save_ohlcv(df: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    """Save OHLCV data to CSV file.

    Args:
        df: DataFrame with OHLCV data
        output_dir: Output directory
        symbol: Stock symbol

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{symbol.lower()}.csv"

    df.to_csv(output_path, index=False)
    logger.info("Saved data", path=str(output_path), rows=len(df))

    return output_path


@click.command()
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    default=None,
    help="Symbols to download (default: from config)",
)
@click.option(
    "--days",
    "-d",
    default=365,
    help="Number of days of historical data",
)
@click.option(
    "--interval",
    "-i",
    default="1h",
    type=click.Choice(["1m", "5m", "15m", "30m", "1h", "1d"]),
    help="Data interval",
)
@click.option(
    "--output-dir",
    "-o",
    default="data/ohlcv",
    help="Output directory for CSV files",
)
def main(
    symbols: tuple[str, ...],
    days: int,
    interval: str,
    output_dir: str,
) -> None:
    """Download historical OHLCV data for stock symbols."""
    setup_logging(level="INFO")
    settings = get_settings()

    # Use provided symbols or fall back to config
    symbol_list = list(symbols) if symbols else settings.symbols

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting download",
        symbols=symbol_list,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval=interval,
    )

    all_data = []
    for symbol in symbol_list:
        try:
            df = download_ohlcv(symbol, start_date, end_date, interval)
            if not df.empty:
                save_ohlcv(df, output_path, symbol)
                all_data.append(df)
        except Exception as e:
            logger.error("Failed to download", symbol=symbol, error=str(e))

    # Also save combined file
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(["timestamp", "symbol"])
        combined_path = output_path / "combined.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info("Saved combined data", path=str(combined_path), rows=len(combined_df))

    logger.info("Download complete", total_symbols=len(symbol_list))


if __name__ == "__main__":
    main()

