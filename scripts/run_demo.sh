#!/bin/bash
# Demo script for running the complete stock prediction pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Real-Time Stock Market Prediction Demo"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Start infrastructure
echo "Step 1: Starting Kafka and MLflow..."
docker-compose up -d
echo "Waiting for services to be ready..."
sleep 15

# Check if data exists
if [ ! -f "data/ohlcv/combined.csv" ] && [ ! -f "data/ohlcv/aapl.csv" ]; then
    echo ""
    echo "Step 2: Downloading sample data..."
    python -m src.data.download --days 365 --interval 1h
else
    echo ""
    echo "Step 2: Using existing data in data/ohlcv/"
fi

# Train model if not exists
if [ ! -f "artifacts/model.pt" ]; then
    echo ""
    echo "Step 3: Training LSTM model..."
    python -m src.training.train
else
    echo ""
    echo "Step 3: Using existing model in artifacts/"
fi

echo ""
echo "Step 4: Starting services..."
echo "  - Feature service (background)"
echo "  - Inference service (background)"
echo ""

# Start feature service in background
python -m src.feature_service.main --max-messages 15000 &
FEATURE_PID=$!

# Start inference service in background
python -m src.inference_service.main --max-messages 12000 &
INFERENCE_PID=$!

# Wait a moment for services to initialize
sleep 3

echo "Step 5: Starting data replay..."
python -m src.replay_service.main --speed 0

echo ""
echo "Waiting for services to complete..."
wait $FEATURE_PID 2>/dev/null || true
wait $INFERENCE_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "View results:"
echo "  - Kafka UI: http://localhost:8080"
echo "  - MLflow UI: http://localhost:5000"
echo ""
echo "To stop infrastructure:"
echo "  docker-compose down"

