"""Tests for LSTM model."""

import pytest
import torch
import tempfile
from pathlib import Path

from src.training.model import StockLSTM, create_model, save_model, load_model


class TestStockLSTM:
    """Tests for StockLSTM model."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return create_model(
            input_dim=17,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
        )

    def test_create_model(self, model):
        """Test model creation."""
        assert isinstance(model, StockLSTM)
        assert model.input_dim == 17
        assert model.hidden_dim == 64

    def test_forward_pass(self, model):
        """Test forward pass produces correct output shape."""
        batch_size = 8
        seq_len = 60
        input_dim = 17

        x = torch.randn(batch_size, seq_len, input_dim)
        output, (h_n, c_n) = model(x)

        assert output.shape == (batch_size,)
        assert h_n.shape == (2, batch_size, 64)  # num_layers, batch, hidden
        assert c_n.shape == (2, batch_size, 64)

    def test_predict(self, model):
        """Test predict method."""
        batch_size = 4
        x = torch.randn(batch_size, 60, 17)

        output = model.predict(x)

        assert output.shape == (batch_size,)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_count_parameters(self, model):
        """Test parameter counting."""
        params = model.count_parameters()

        assert params > 0
        assert isinstance(params, int)

    def test_get_config(self, model):
        """Test config extraction."""
        config = model.get_config()

        assert config["input_dim"] == 17
        assert config["hidden_dim"] == 64
        assert config["num_layers"] == 2
        assert "num_parameters" in config

    def test_save_and_load(self, model):
        """Test saving and loading model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            # Save
            save_model(model, str(path), epoch=5, metrics={"accuracy": 0.85})

            # Load
            loaded = load_model(str(path))

            assert isinstance(loaded, StockLSTM)
            assert loaded.input_dim == model.input_dim
            assert loaded.hidden_dim == model.hidden_dim

    def test_bidirectional_model(self):
        """Test bidirectional LSTM."""
        model = create_model(
            input_dim=17,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
        )

        x = torch.randn(4, 60, 17)
        output = model.predict(x)

        assert output.shape == (4,)

    def test_gradient_flow(self, model):
        """Test that gradients flow properly."""
        x = torch.randn(4, 60, 17, requires_grad=True)
        output = model.predict(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_eval_mode(self, model):
        """Test model behavior in eval mode."""
        model.eval()

        x = torch.randn(4, 60, 17)

        with torch.no_grad():
            output1 = model.predict(x)
            output2 = model.predict(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)

