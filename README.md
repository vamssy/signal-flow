# SignalFlow - Real-Time Stock Market Prediction System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade stock price direction prediction system using LSTM neural networks, Apache Kafka for real-time streaming, and MLflow for experiment tracking. Features a beautiful real-time dashboard for monitoring trading signals.

![Dashboard Screenshot](docs/dashboard.png)

## ✨ Key Features

- **🧠 LSTM Neural Network**: ~209K parameter model for stock price direction prediction
- **📊 17 Technical Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, volatility, and more
- **⚡ Real-Time Streaming**: Apache Kafka-based pipeline with sub-second latency
- **📈 Live Dashboard**: Beautiful web UI showing real-time BUY/SELL/HOLD signals
- **🔴 Live Price Feed**: Stream real-time stock prices from Yahoo Finance
- **📉 MLflow Integration**: Systematic experiment tracking and model versioning

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNALFLOW ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │   Yahoo Finance │
                    │   (Live Prices) │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Live Service   │───►│ Feature Service │───►│Inference Service│
│  (Price Feed)   │    │ (17 Indicators) │    │  (LSTM Model)   │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
    ┌─────────┐           ┌──────────┐          ┌──────────┐
    │market_  │           │features  │          │signals   │
    │ohlcv    │           │          │          │          │
    └─────────┘           └──────────┘          └──────────┘
              \                 |                    /
               └────────────────┼───────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Apache Kafka      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Real-Time Dashboard │
                    │   (WebSocket + Charts)│
                    └───────────────────────┘
```

## 🖥️ Dashboard Preview

The dashboard provides:

- **Real-time BUY/SELL/HOLD signals** with confidence scores
- **Live price charts** with signal markers
- **Symbol selector** for AAPL, GOOGL, MSFT, AMZN, META
- **Signal history** and statistics
- **Live mode indicator** showing real-time data status

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- 4GB+ RAM recommended

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/signalflow.git
cd signalflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Start Infrastructure

```bash
# Start Kafka, Zookeeper, and MLflow
docker-compose up -d

# Wait for services to be ready (~30 seconds)
docker-compose ps
```

### 3. Download Sample Data & Train Model

```bash
# Download 1 year of hourly data
python -m src.data.download --days 365 --interval 1h

# Train LSTM model with MLflow tracking
python -m src.training.train
```

### 4. Run the Complete System

**Option A: Live Mode (Real-Time Prices)**

```bash
# Terminal 1: Feature service
python -m src.feature_service.main

# Terminal 2: Inference service
python -m src.inference_service.main

# Terminal 3: Dashboard
python -m src.dashboard.server

# Terminal 4: Live price feed (updates every 10s)
python -m src.live_service.main --interval 10
```

**Option B: Replay Mode (Historical Data)**

```bash
# Terminal 1: Feature service
python -m src.feature_service.main

# Terminal 2: Inference service
python -m src.inference_service.main

# Terminal 3: Dashboard
python -m src.dashboard.server

# Terminal 4: Replay historical data
python -m src.replay_service.main --speed 0
```

### 5. Open Dashboard

Navigate to **http://localhost:8501** in your browser!

## 📊 Technical Indicators

| Category   | Indicators                                                   |
| ---------- | ------------------------------------------------------------ |
| Momentum   | RSI (14), MACD (12/26/9), MACD Signal, MACD Histogram        |
| Volatility | Bollinger Bands (Upper, Mid, Lower, Width), ATR, Rolling Vol |
| Trend      | SMA (5, 10, 20, 50), EMA (5, 10, 20)                         |
| Returns    | Simple Returns, Log Returns                                  |
| Volume     | Volume Z-Score                                               |

## 🧠 Model Architecture

```python
StockLSTM(
  (lstm): LSTM(17, 128, num_layers=2, batch_first=True, dropout=0.2)
  (layer_norm): LayerNorm((128,))
  (dropout): Dropout(p=0.2)
  (fc): Linear(in_features=128, out_features=1)
  (sigmoid): Sigmoid()
)
# Total Parameters: ~209,000
```

## 📁 Project Structure

```
signalflow/
├── configs/
│   └── config.yaml           # Main configuration
├── data/
│   └── ohlcv/                # OHLCV data files
├── artifacts/                # Trained model artifacts
│   ├── model.pt              # PyTorch model weights
│   ├── scaler.pkl            # MinMaxScaler
│   └── feature_list.json     # Feature names
├── src/
│   ├── common/               # Shared utilities
│   │   ├── config.py         # Configuration management
│   │   ├── logging.py        # Structured logging
│   │   ├── schemas.py        # Pydantic schemas
│   │   └── kafka_utils.py    # Kafka producer/consumer
│   ├── data/
│   │   └── download.py       # Data download utility
│   ├── live_service/         # Real-time price feed
│   ├── replay_service/       # Historical data replay
│   ├── feature_service/      # Technical indicators
│   ├── inference_service/    # LSTM inference
│   ├── dashboard/            # Web UI
│   │   ├── server.py         # FastAPI backend
│   │   ├── kafka_consumer.py # Async Kafka consumer
│   │   └── static/           # Frontend assets
│   └── training/             # Model training
│       ├── model.py          # LSTM architecture
│       ├── dataset.py        # Data loading
│       └── train.py          # Training loop
├── scripts/                  # Utility scripts
├── tests/                    # Unit tests
├── docker-compose.yml        # Infrastructure setup
├── Makefile                  # Convenience commands
└── pyproject.toml            # Python dependencies
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model configuration
model:
  input_dim: 17
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2

# Training configuration
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 50
  early_stopping_patience: 10

# Signal thresholds
inference:
  confidence_threshold: 0.6
  signal_thresholds:
    buy: 0.55
    sell: 0.45
```

## 🛠️ Make Commands

```bash
make help          # Show all commands
make install       # Install dependencies
make docker-up     # Start infrastructure
make train         # Train the model
make dashboard     # Start web dashboard
make live          # Start live price streaming
make clean         # Clean artifacts
```

## 📡 Service URLs

| Service   | URL                   | Description              |
| --------- | --------------------- | ------------------------ |
| Dashboard | http://localhost:8501 | Real-time trading UI     |
| Kafka UI  | http://localhost:8080 | Message broker dashboard |
| MLflow    | http://localhost:5001 | Experiment tracking      |

## 📨 Kafka Topics

### `market_ohlcv`

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "symbol": "AAPL",
  "open": 185.5,
  "high": 186.2,
  "low": 185.1,
  "close": 185.9,
  "volume": 1250000
}
```

### `signals`

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "symbol": "AAPL",
  "signal": "BUY",
  "confidence": 0.72,
  "predicted_direction": "UP",
  "raw_probability": 0.86,
  "latency_ms": 12.5
}
```

## 🧪 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests

# Lint
ruff check src tests

# Type checking
mypy src
```

## 🔧 Troubleshooting

### Kafka Connection Issues

```bash
docker-compose ps          # Check status
docker-compose logs kafka  # View logs
docker-compose restart kafka
```

### Model Not Found

```bash
python -m src.training.train  # Train new model
```

### Dashboard Not Loading

```bash
# Ensure all services are running
pgrep -f "feature_service" || echo "Start feature service"
pgrep -f "inference_service" || echo "Start inference service"
pgrep -f "dashboard" || echo "Start dashboard"
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Apache Kafka](https://kafka.apache.org/) - Event streaming platform
- [MLflow](https://mlflow.org/) - ML experiment tracking
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [TradingView Lightweight Charts](https://www.tradingview.com/lightweight-charts/) - Charting library
- [Alpine.js](https://alpinejs.dev/) - Frontend reactivity
- [TailwindCSS](https://tailwindcss.com/) - CSS framework

---

