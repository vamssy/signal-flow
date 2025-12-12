"""Training script with MLflow integration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import get_logger, get_settings, setup_logging
from src.common.config import Settings
from src.training.dataset import create_data_loaders, prepare_data
from src.training.model import StockLSTM, create_model, save_model

logger = get_logger(__name__)


def train_epoch(
    model: StockLSTM,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Tuple of (average loss, directional accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model.predict(sequences)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(labels)

        # Compute accuracy
        predictions = (outputs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(
    model: StockLSTM,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device

    Returns:
        Tuple of (loss, accuracy, precision, recall)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for sequences, labels in data_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model.predict(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

            # For precision/recall
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    return avg_loss, accuracy, precision, recall


def train_model(
    settings: Settings,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    artifacts_dir: Path,
) -> tuple[StockLSTM, dict]:
    """Train the LSTM model with early stopping and MLflow logging.

    Args:
        settings: Application settings
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to train on
        artifacts_dir: Directory to save artifacts

    Returns:
        Tuple of (trained model, metrics dict)
    """
    # Create model
    model = create_model(
        input_dim=settings.model.input_dim,
        hidden_dim=settings.model.hidden_dim,
        num_layers=settings.model.num_layers,
        dropout=settings.model.dropout,
        bidirectional=settings.model.bidirectional,
        device=device,
    )

    logger.info(
        "Created model",
        parameters=model.count_parameters(),
        config=model.get_config(),
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training state
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    patience_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(1, settings.training.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_precision, val_recall = evaluate(
            model, val_loader, criterion, device
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Log metrics
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        logger.info(
            f"Epoch {epoch}/{settings.training.epochs}",
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.2%}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.2%}",
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()

            # Save checkpoint
            save_model(
                model,
                str(artifacts_dir / "best_model.pt"),
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_accuracy": val_acc},
            )
        else:
            patience_counter += 1

        if patience_counter >= settings.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_loss, test_acc, test_precision, test_recall = evaluate(
        model, test_loader, criterion, device
    )

    logger.info(
        "Test evaluation",
        loss=f"{test_loss:.4f}",
        accuracy=f"{test_acc:.2%}",
        precision=f"{test_precision:.2%}",
        recall=f"{test_recall:.2%}",
    )

    # Log final test metrics
    mlflow.log_metrics(
        {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "best_val_accuracy": best_val_accuracy,
        }
    )

    # Save final model
    save_model(
        model,
        str(artifacts_dir / "model.pt"),
        metrics={
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
        },
    )

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "best_val_accuracy": best_val_accuracy,
        "parameters": model.count_parameters(),
    }

    return model, metrics


@click.command()
@click.option(
    "--data-path",
    "-d",
    default="data/ohlcv",
    help="Path to OHLCV data directory or file",
)
@click.option(
    "--symbols",
    "-s",
    multiple=True,
    default=None,
    help="Symbols to train on (default: from config)",
)
@click.option(
    "--artifacts-dir",
    "-a",
    default="artifacts",
    help="Directory to save model artifacts",
)
@click.option(
    "--config",
    "-c",
    default=None,
    help="Path to config file",
)
@click.option(
    "--experiment-name",
    "-e",
    default=None,
    help="MLflow experiment name (default: from config)",
)
def main(
    data_path: str,
    symbols: tuple[str, ...],
    artifacts_dir: str,
    config: str | None,
    experiment_name: str | None,
) -> None:
    """Train the LSTM stock prediction model."""
    setup_logging(level="INFO")
    settings = get_settings(config)

    # Set random seeds
    torch.manual_seed(settings.training.random_seed)
    np.random.seed(settings.training.random_seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device", device=str(device))

    # Data path
    data_dir = Path(data_path)
    if not data_dir.exists():
        logger.error("Data path not found", path=str(data_dir))
        logger.info("Run 'python -m src.data.download' to download sample data first")
        return

    # Artifacts directory
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    # Symbol list
    symbol_list = list(symbols) if symbols else settings.symbols

    # MLflow setup
    mlflow_uri = settings.mlflow.tracking_uri
    exp_name = experiment_name or settings.mlflow.experiment_name

    # Try to connect to MLflow, fall back to local if unavailable
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(exp_name)
        logger.info("Connected to MLflow", uri=mlflow_uri, experiment=exp_name)
    except Exception as e:
        logger.warning(f"MLflow server not available, using local tracking: {e}")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(exp_name)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(
            {
                "sequence_length": settings.features.sequence_length,
                "input_dim": settings.model.input_dim,
                "hidden_dim": settings.model.hidden_dim,
                "num_layers": settings.model.num_layers,
                "dropout": settings.model.dropout,
                "bidirectional": settings.model.bidirectional,
                "batch_size": settings.training.batch_size,
                "learning_rate": settings.training.learning_rate,
                "epochs": settings.training.epochs,
                "symbols": ",".join(symbol_list),
            }
        )

        # Prepare data
        logger.info("Preparing data...")
        features, labels, feature_names, scaler = prepare_data(
            data_dir, settings, symbol_list
        )

        # Update input_dim based on actual features
        settings.model.input_dim = len(feature_names)
        mlflow.log_param("actual_input_dim", len(feature_names))

        # Save feature list and scaler
        feature_list_path = artifacts_path / "feature_list.json"
        with open(feature_list_path, "w") as f:
            json.dump(feature_names, f, indent=2)

        scaler_path = artifacts_path / "scaler.pkl"
        scaler.save(scaler_path, feature_list_path)

        # Log artifacts (may fail if MLflow artifact storage is not accessible)
        try:
            mlflow.log_artifact(str(feature_list_path))
            mlflow.log_artifact(str(scaler_path))
        except Exception as e:
            logger.warning(f"Could not log artifacts to MLflow: {e}")

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            features, labels, settings
        )

        # Train model
        logger.info("Starting training...")
        model, metrics = train_model(
            settings,
            train_loader,
            val_loader,
            test_loader,
            device,
            artifacts_path,
        )

        # Log model artifact (may fail if MLflow artifact storage is not accessible)
        try:
            mlflow.log_artifact(str(artifacts_path / "model.pt"))
        except Exception as e:
            logger.warning(f"Could not log model artifact to MLflow: {e}")

        logger.info(
            "Training complete",
            parameters=model.count_parameters(),
            test_accuracy=f"{metrics['test_accuracy']:.2%}",
        )


if __name__ == "__main__":
    main()

