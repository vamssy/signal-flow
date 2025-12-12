"""LSTM model for stock price direction prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """LSTM model for predicting stock price direction.

    Architecture:
        - LSTM layers with dropout
        - Fully connected output layer
        - Sigmoid activation for binary classification (up/down)
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """Initialize the LSTM model.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_dim * self.num_directions, 1)

        # Sigmoid for probability output
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output probability, (hidden_state, cell_state))
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)

        # Take the last hidden state
        # For bidirectional, concatenate forward and backward final states
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_dim)
            # Get last layer's forward and backward hidden states
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            last_hidden = torch.cat([last_forward, last_backward], dim=1)
        else:
            # h_n shape: (num_layers, batch, hidden_dim)
            last_hidden = h_n[-1]

        # Layer normalization and dropout
        normalized = self.layer_norm(last_hidden)
        dropped = self.dropout(normalized)

        # Output layer
        logits = self.fc(dropped)
        probability = self.sigmoid(logits)

        return probability.squeeze(-1), (h_n, c_n)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions without returning hidden states.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Probability tensor of shape (batch,)
        """
        prob, _ = self.forward(x)
        return prob

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration.

        Returns:
            Dictionary of model hyperparameters
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "num_parameters": self.count_parameters(),
        }


def create_model(
    input_dim: int = 17,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    device: str | torch.device = "cpu",
) -> StockLSTM:
    """Create and initialize the LSTM model.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        device: Device to place model on

    Returns:
        Initialized StockLSTM model
    """
    model = StockLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )

    model = model.to(device)

    return model


def load_model(
    path: str,
    device: str | torch.device = "cpu",
) -> StockLSTM:
    """Load a trained model from disk.

    Args:
        path: Path to model checkpoint
        device: Device to load model onto

    Returns:
        Loaded StockLSTM model
    """
    checkpoint = torch.load(path, map_location=device)

    # Get model config from checkpoint
    config = checkpoint.get("config", {})

    model = create_model(
        input_dim=config.get("input_dim", 17),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
        bidirectional=config.get("bidirectional", False),
        device=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def save_model(
    model: StockLSTM,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict | None = None,
) -> None:
    """Save model checkpoint to disk.

    Args:
        model: Model to save
        path: Output path
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.get_config(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)

