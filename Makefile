.PHONY: help install dev-install test lint format clean docker-up docker-down download train demo benchmark dashboard live

help:
	@echo "╔═══════════════════════════════════════════════════════════╗"
	@echo "║     SignalFlow - Real-Time Stock Prediction System        ║"
	@echo "╚═══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev-install   Install development dependencies"
	@echo "  make docker-up     Start infrastructure (Kafka, MLflow)"
	@echo "  make docker-down   Stop infrastructure"
	@echo ""
	@echo "Data & Training:"
	@echo "  make download      Download sample stock data"
	@echo "  make train         Train the LSTM model"
	@echo ""
	@echo "Run Services:"
	@echo "  make dashboard     Start the web dashboard"
	@echo "  make live          Start live price streaming (30s)"
	@echo "  make live-fast     Start live price streaming (10s)"
	@echo "  make demo          Run full demo with replay data"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo "  make clean         Clean up artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff check src tests --fix

docker-up:
	docker-compose up -d
	@echo "Waiting for services to start..."
	@sleep 15
	@echo ""
	@echo "✓ Services ready:"
	@echo "  • Kafka UI: http://localhost:8080"
	@echo "  • MLflow:   http://localhost:5001"

docker-down:
	docker-compose down

download:
	python -m src.data.download --days 365 --interval 1h

train:
	python -m src.training.train

demo:
	bash scripts/run_demo.sh

benchmark:
	python scripts/benchmark.py --messages 10000

# Run services individually
feature-service:
	python -m src.feature_service.main

inference-service:
	python -m src.inference_service.main

replay:
	python -m src.replay_service.main --speed 0

dashboard:
	@echo "Starting dashboard at http://localhost:8501"
	python -m src.dashboard.server

live:
	@echo "Starting live price service..."
	@echo "This will fetch real-time stock prices and stream to the dashboard"
	python -m src.live_service.main --interval 30

live-fast:
	@echo "Starting live price service (10s interval)..."
	python -m src.live_service.main --interval 10

clean:
	rm -rf artifacts/*.pt artifacts/*.pkl artifacts/*.json
	rm -rf data/ohlcv/*.csv
	rm -rf mlruns/
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

