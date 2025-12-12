"""FastAPI server with WebSocket streaming for real-time signals."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.common import get_logger, get_settings, setup_logging
from src.common.config import Settings
from src.dashboard.kafka_consumer import SignalConsumer, PriceConsumer

logger = get_logger(__name__)

# Global state
settings: Settings | None = None
signal_consumer: SignalConsumer | None = None
price_consumer: PriceConsumer | None = None
connected_clients: set[WebSocket] = set()
consumer_task: asyncio.Task | None = None
price_consumer_task: asyncio.Task | None = None

# Stats tracking
stats = {
    "total_signals": 0,
    "signals_by_type": defaultdict(int),
    "signals_by_symbol": defaultdict(int),
    "total_confidence": 0.0,
    "last_signal": None,
    "last_prices": {},
}


async def broadcast_message(message: dict) -> None:
    """Broadcast a message to all connected WebSocket clients."""
    if not connected_clients:
        return

    message_json = json.dumps(message, default=str)
    disconnected = set()

    for client in connected_clients:
        try:
            await client.send_text(message_json)
        except Exception:
            disconnected.add(client)

    # Clean up disconnected clients
    for client in disconnected:
        connected_clients.discard(client)


async def handle_kafka_message(data: dict) -> None:
    """Handle incoming Kafka messages and broadcast to clients."""
    global stats

    topic = data.get("topic", "")
    value = data.get("value", {})

    if "signals" in topic:
        # Update stats
        stats["total_signals"] += 1
        signal_type = value.get("signal", "UNKNOWN")
        symbol = value.get("symbol", "UNKNOWN")
        confidence = value.get("confidence", 0)

        stats["signals_by_type"][signal_type] += 1
        stats["signals_by_symbol"][symbol] += 1
        stats["total_confidence"] += confidence
        stats["last_signal"] = value

        # Broadcast signal
        await broadcast_message({
            "type": "signal",
            "data": value,
            "stats": {
                "total_signals": stats["total_signals"],
                "avg_confidence": stats["total_confidence"] / stats["total_signals"],
                "signals_by_type": dict(stats["signals_by_type"]),
            },
        })

    elif "features" in topic:
        # Broadcast features/price update
        symbol = value.get("symbol", "UNKNOWN")
        stats["last_prices"][symbol] = {
            "open": value.get("open"),
            "high": value.get("high"),
            "low": value.get("low"),
            "close": value.get("close"),
            "volume": value.get("volume"),
            "timestamp": value.get("timestamp"),
        }

        await broadcast_message({
            "type": "price",
            "data": {
                "symbol": symbol,
                "price": value.get("close"),
                "timestamp": value.get("timestamp"),
                "ohlcv": stats["last_prices"][symbol],
            },
        })


async def handle_price_message(data: dict) -> None:
    """Handle incoming OHLCV messages."""
    value = data.get("value", {})
    symbol = value.get("symbol", "UNKNOWN")

    stats["last_prices"][symbol] = {
        "open": value.get("open"),
        "high": value.get("high"),
        "low": value.get("low"),
        "close": value.get("close"),
        "volume": value.get("volume"),
        "timestamp": value.get("timestamp"),
    }

    await broadcast_message({
        "type": "ohlcv",
        "data": value,
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global signal_consumer, price_consumer, consumer_task, price_consumer_task, settings

    setup_logging(level="INFO")
    settings = get_settings()

    # Initialize Kafka consumers
    signal_consumer = SignalConsumer(settings)
    signal_consumer.add_callback(handle_kafka_message)

    price_consumer = PriceConsumer(settings)
    price_consumer.add_callback(handle_price_message)

    # Start consumers in background
    try:
        await signal_consumer.start()
        await price_consumer.start()
        consumer_task = asyncio.create_task(signal_consumer.consume())
        price_consumer_task = asyncio.create_task(price_consumer.consume())
        logger.info("Dashboard server started")
    except Exception as e:
        logger.warning(f"Could not connect to Kafka: {e}")

    yield

    # Cleanup
    if signal_consumer:
        await signal_consumer.stop()
    if price_consumer:
        await price_consumer.stop()
    if consumer_task:
        consumer_task.cancel()
    if price_consumer_task:
        price_consumer_task.cancel()

    logger.info("Dashboard server stopped")


# Create FastAPI app
app = FastAPI(
    title="Stock Prediction Dashboard",
    description="Real-time trading signals dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard HTML."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Dashboard not found. Run from project root.</h1>")


@app.get("/api/symbols")
async def get_symbols():
    """Get list of available symbols."""
    return {"symbols": settings.symbols if settings else ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]}


@app.get("/api/stats")
async def get_stats():
    """Get current statistics."""
    return {
        "total_signals": stats["total_signals"],
        "avg_confidence": (
            stats["total_confidence"] / stats["total_signals"]
            if stats["total_signals"] > 0
            else 0
        ),
        "signals_by_type": dict(stats["signals_by_type"]),
        "signals_by_symbol": dict(stats["signals_by_symbol"]),
        "last_signal": stats["last_signal"],
        "last_prices": stats["last_prices"],
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "kafka_connected": signal_consumer is not None and signal_consumer.running,
        "connected_clients": len(connected_clients),
    }


@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signal streaming."""
    await websocket.accept()
    connected_clients.add(websocket)

    logger.info("WebSocket client connected", total_clients=len(connected_clients))

    # Send initial state
    await websocket.send_json({
        "type": "init",
        "data": {
            "symbols": settings.symbols if settings else [],
            "stats": {
                "total_signals": stats["total_signals"],
                "signals_by_type": dict(stats["signals_by_type"]),
            },
            "last_signal": stats["last_signal"],
            "last_prices": stats["last_prices"],
        },
    })

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Could handle client commands here (e.g., symbol filter)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        connected_clients.discard(websocket)
        logger.info("WebSocket client disconnected", total_clients=len(connected_clients))


@click.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", default=8501, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def main(host: str, port: int, reload: bool) -> None:
    """Run the dashboard server."""
    import uvicorn

    uvicorn.run(
        "src.dashboard.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

