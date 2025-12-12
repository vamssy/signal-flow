#!/usr/bin/env python
"""Benchmark script for measuring system performance."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

# Add project to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common import get_logger, get_settings, setup_logging
from src.common.schemas import OHLCVMessage, FeaturesMessage
from src.feature_service.indicators import StreamingIndicators
from src.inference_service.buffer import SymbolBufferManager
from src.training.model import load_model

logger = get_logger(__name__)


def generate_synthetic_ohlcv(n_messages: int, n_symbols: int = 5) -> list[dict]:
    """Generate synthetic OHLCV messages for benchmarking.

    Args:
        n_messages: Total number of messages to generate
        n_symbols: Number of unique symbols

    Returns:
        List of OHLCV message dictionaries
    """
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    messages = []

    base_time = datetime.now() - timedelta(days=365)
    base_prices = {s: 100.0 + np.random.randn() * 20 for s in symbols}

    for i in range(n_messages):
        symbol = symbols[i % n_symbols]
        base_price = base_prices[symbol]

        # Random walk
        change = np.random.randn() * 0.02
        base_prices[symbol] = base_price * (1 + change)

        close = base_prices[symbol]
        high = close * (1 + abs(np.random.randn() * 0.01))
        low = close * (1 - abs(np.random.randn() * 0.01))
        open_price = close * (1 + np.random.randn() * 0.005)

        messages.append({
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": int(np.random.exponential(1000000)),
        })

    return messages


def benchmark_feature_computation(n_messages: int = 10000) -> dict:
    """Benchmark feature computation performance.

    Args:
        n_messages: Number of messages to process

    Returns:
        Benchmark results
    """
    settings = get_settings()
    messages = generate_synthetic_ohlcv(n_messages)

    # Create indicator calculators per symbol
    indicators: dict[str, StreamingIndicators] = {}

    start_time = time.perf_counter()

    for msg in messages:
        symbol = msg["symbol"]
        if symbol not in indicators:
            indicators[symbol] = StreamingIndicators(settings.features)

        indicators[symbol].compute(
            open_price=msg["open"],
            high=msg["high"],
            low=msg["low"],
            close=msg["close"],
            volume=msg["volume"],
        )

    elapsed = time.perf_counter() - start_time
    rate = n_messages / elapsed

    return {
        "operation": "feature_computation",
        "messages": n_messages,
        "elapsed_seconds": elapsed,
        "messages_per_second": rate,
        "latency_ms_per_message": (elapsed / n_messages) * 1000,
    }


def benchmark_inference(
    n_messages: int = 10000,
    model_path: str = "artifacts/model.pt",
) -> dict:
    """Benchmark inference performance.

    Args:
        n_messages: Number of messages to process
        model_path: Path to trained model

    Returns:
        Benchmark results
    """
    settings = get_settings()
    path = Path(model_path)

    if not path.exists():
        logger.warning("Model not found, skipping inference benchmark")
        return {"operation": "inference", "skipped": True, "reason": "model_not_found"}

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(path), device)
    model.eval()

    config = model.get_config()
    n_features = config["input_dim"]
    sequence_length = settings.features.sequence_length

    # Generate synthetic feature sequences
    buffer_manager = SymbolBufferManager(sequence_length, n_features)

    # Pre-fill buffers
    for i in range(sequence_length * 5):
        symbol = f"SYM{i % 5}"
        features = np.random.randn(n_features).tolist()
        buffer_manager.append(symbol, features)

    # Benchmark inference
    latencies = []
    symbols = [f"SYM{i}" for i in range(5)]

    start_time = time.perf_counter()

    for i in range(n_messages):
        symbol = symbols[i % len(symbols)]

        # Add new features
        features = np.random.randn(n_features).tolist()
        buffer_manager.append(symbol, features)

        if buffer_manager.is_ready(symbol):
            infer_start = time.perf_counter()

            with torch.no_grad():
                input_tensor = buffer_manager.get_tensor(symbol, device)
                _ = model.predict(input_tensor)

            latencies.append((time.perf_counter() - infer_start) * 1000)

    elapsed = time.perf_counter() - start_time
    rate = n_messages / elapsed

    latencies_arr = np.array(latencies)

    return {
        "operation": "inference",
        "device": str(device),
        "messages": n_messages,
        "inferences": len(latencies),
        "elapsed_seconds": elapsed,
        "messages_per_second": rate,
        "inferences_per_second": len(latencies) / elapsed if elapsed > 0 else 0,
        "latency_mean_ms": float(latencies_arr.mean()) if len(latencies) > 0 else 0,
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)) if len(latencies) > 0 else 0,
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)) if len(latencies) > 0 else 0,
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)) if len(latencies) > 0 else 0,
    }


def benchmark_end_to_end(
    n_messages: int = 10000,
    model_path: str = "artifacts/model.pt",
) -> dict:
    """Benchmark end-to-end pipeline performance.

    Args:
        n_messages: Number of messages to process
        model_path: Path to trained model

    Returns:
        Benchmark results
    """
    settings = get_settings()
    path = Path(model_path)

    if not path.exists():
        logger.warning("Model not found, skipping e2e benchmark")
        return {"operation": "end_to_end", "skipped": True, "reason": "model_not_found"}

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(path), device)
    model.eval()

    config = model.get_config()
    n_features = config["input_dim"]
    sequence_length = settings.features.sequence_length

    # Generate synthetic OHLCV
    messages = generate_synthetic_ohlcv(n_messages)

    # Initialize components
    indicators: dict[str, StreamingIndicators] = {}
    buffer_manager = SymbolBufferManager(sequence_length, n_features)

    latencies = []
    signals_generated = 0

    start_time = time.perf_counter()

    for msg in messages:
        msg_start = time.perf_counter()
        symbol = msg["symbol"]

        # Feature computation
        if symbol not in indicators:
            indicators[symbol] = StreamingIndicators(settings.features)

        features = indicators[symbol].compute(
            open_price=msg["open"],
            high=msg["high"],
            low=msg["low"],
            close=msg["close"],
            volume=msg["volume"],
        )

        if not indicators[symbol].is_warmed_up:
            continue

        # Add to buffer
        feature_vector = list(features.values())
        if len(feature_vector) < n_features:
            feature_vector.extend([0.0] * (n_features - len(feature_vector)))
        feature_vector = feature_vector[:n_features]

        buffer_manager.append(symbol, feature_vector)

        # Inference
        if buffer_manager.is_ready(symbol):
            with torch.no_grad():
                input_tensor = buffer_manager.get_tensor(symbol, device)
                _ = model.predict(input_tensor)

            signals_generated += 1
            latencies.append((time.perf_counter() - msg_start) * 1000)

    elapsed = time.perf_counter() - start_time

    latencies_arr = np.array(latencies) if latencies else np.array([0])

    return {
        "operation": "end_to_end",
        "device": str(device),
        "messages": n_messages,
        "signals_generated": signals_generated,
        "elapsed_seconds": elapsed,
        "messages_per_second": n_messages / elapsed if elapsed > 0 else 0,
        "signals_per_second": signals_generated / elapsed if elapsed > 0 else 0,
        "latency_mean_ms": float(latencies_arr.mean()),
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "latency_p99_ms": float(np.percentile(latencies_arr, 99)),
    }


@click.command()
@click.option("--messages", "-n", default=10000, help="Number of messages to benchmark")
@click.option("--model-path", "-m", default="artifacts/model.pt", help="Path to model")
@click.option("--output", "-o", default=None, help="Output file for results (JSON)")
def main(messages: int, model_path: str, output: str | None) -> None:
    """Run performance benchmarks."""
    setup_logging(level="INFO")

    logger.info("Starting benchmarks", n_messages=messages)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "messages": messages,
            "model_path": model_path,
        },
        "benchmarks": {},
    }

    # Feature computation benchmark
    logger.info("Running feature computation benchmark...")
    results["benchmarks"]["features"] = benchmark_feature_computation(messages)
    logger.info("Feature benchmark complete", **results["benchmarks"]["features"])

    # Inference benchmark
    logger.info("Running inference benchmark...")
    results["benchmarks"]["inference"] = benchmark_inference(messages, model_path)
    logger.info("Inference benchmark complete", **results["benchmarks"]["inference"])

    # End-to-end benchmark
    logger.info("Running end-to-end benchmark...")
    results["benchmarks"]["end_to_end"] = benchmark_end_to_end(messages, model_path)
    logger.info("End-to-end benchmark complete", **results["benchmarks"]["end_to_end"])

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for name, benchmark in results["benchmarks"].items():
        if benchmark.get("skipped"):
            print(f"\n{name}: SKIPPED ({benchmark.get('reason')})")
            continue

        print(f"\n{name.upper()}:")
        print(f"  Messages processed: {benchmark.get('messages', 'N/A')}")
        print(f"  Throughput: {benchmark.get('messages_per_second', 0):.0f} msg/s")

        if "latency_p50_ms" in benchmark:
            print(f"  Latency p50: {benchmark['latency_p50_ms']:.2f} ms")
            print(f"  Latency p95: {benchmark['latency_p95_ms']:.2f} ms")
            print(f"  Latency p99: {benchmark['latency_p99_ms']:.2f} ms")

    print("\n" + "=" * 60)

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved", path=output)


if __name__ == "__main__":
    main()

