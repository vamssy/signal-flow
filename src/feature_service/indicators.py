"""Technical indicator calculations for feature engineering."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from src.common.config import FeaturesConfig
from src.common.logging import get_logger

logger = get_logger(__name__)


class RollingWindow:
    """Efficient rolling window for streaming calculations."""

    def __init__(self, size: int):
        self.size = size
        self.data: deque[float] = deque(maxlen=size)

    def append(self, value: float) -> None:
        """Add a value to the window."""
        self.data.append(value)

    def is_full(self) -> bool:
        """Check if window is full."""
        return len(self.data) >= self.size

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array(self.data)

    def mean(self) -> float:
        """Calculate mean of window."""
        if not self.data:
            return 0.0
        return float(np.mean(self.data))

    def std(self) -> float:
        """Calculate standard deviation of window."""
        if len(self.data) < 2:
            return 0.0
        return float(np.std(self.data, ddof=1))

    def __len__(self) -> int:
        return len(self.data)


class StreamingIndicators:
    """Streaming technical indicators calculator.

    Maintains internal state for computing indicators on streaming data
    without lookahead bias.
    """

    def __init__(self, config: FeaturesConfig):
        self.config = config
        self._init_windows()

    def _init_windows(self) -> None:
        """Initialize rolling windows for each indicator."""
        # Price windows for various indicators
        self.close_window = RollingWindow(max(50, self.config.indicators.bollinger_bands.window))
        self.high_window = RollingWindow(self.config.indicators.atr.window + 1)
        self.low_window = RollingWindow(self.config.indicators.atr.window + 1)
        self.volume_window = RollingWindow(self.config.volume_zscore_window)

        # RSI components
        self.rsi_gains = RollingWindow(self.config.indicators.rsi.window)
        self.rsi_losses = RollingWindow(self.config.indicators.rsi.window)
        self.prev_close: float | None = None

        # MACD components (EMA values)
        self.ema_fast: float | None = None
        self.ema_slow: float | None = None
        self.ema_signal: float | None = None
        self.macd_values = RollingWindow(self.config.indicators.macd.signal_period)

        # EMA windows
        self.ema_values: dict[int, float | None] = {
            w: None for w in self.config.indicators.ema.windows
        }

        # Returns for volatility
        self.returns_window = RollingWindow(self.config.volatility_window)

        # Message count for warmup
        self.message_count = 0

    def reset(self) -> None:
        """Reset all internal state."""
        self._init_windows()

    def compute(
        self,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ) -> dict[str, float]:
        """Compute all technical indicators for a new bar.

        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Closing price
            volume: Trading volume

        Returns:
            Dictionary of indicator names to values
        """
        self.message_count += 1
        features: dict[str, float] = {}

        # Update price windows
        self.close_window.append(close)
        self.high_window.append(high)
        self.low_window.append(low)
        self.volume_window.append(float(volume))

        # 1. Returns
        if self.config.include_returns and self.prev_close is not None:
            returns = (close - self.prev_close) / self.prev_close
            features["returns"] = returns
            self.returns_window.append(returns)

            # 2. Log returns
            if self.config.include_log_returns and self.prev_close > 0:
                features["log_returns"] = float(np.log(close / self.prev_close))

        # 3. RSI
        if self.prev_close is not None:
            change = close - self.prev_close
            gain = max(0, change)
            loss = max(0, -change)
            self.rsi_gains.append(gain)
            self.rsi_losses.append(loss)

            if self.rsi_gains.is_full():
                avg_gain = self.rsi_gains.mean()
                avg_loss = self.rsi_losses.mean()
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features["rsi"] = 100 - (100 / (1 + rs))
                else:
                    features["rsi"] = 100.0

        # 4-6. MACD (macd, macd_signal, macd_hist)
        macd_config = self.config.indicators.macd
        alpha_fast = 2 / (macd_config.fast_period + 1)
        alpha_slow = 2 / (macd_config.slow_period + 1)
        alpha_signal = 2 / (macd_config.signal_period + 1)

        if self.ema_fast is None:
            self.ema_fast = close
            self.ema_slow = close
        else:
            self.ema_fast = alpha_fast * close + (1 - alpha_fast) * self.ema_fast
            self.ema_slow = alpha_slow * close + (1 - alpha_slow) * self.ema_slow

        macd_line = self.ema_fast - self.ema_slow
        self.macd_values.append(macd_line)

        if self.ema_signal is None:
            self.ema_signal = macd_line
        else:
            self.ema_signal = alpha_signal * macd_line + (1 - alpha_signal) * self.ema_signal

        features["macd"] = macd_line
        features["macd_signal"] = self.ema_signal
        features["macd_hist"] = macd_line - self.ema_signal

        # 7-10. Bollinger Bands (upper, middle, lower, width)
        bb_config = self.config.indicators.bollinger_bands
        if len(self.close_window) >= bb_config.window:
            recent_closes = list(self.close_window.data)[-bb_config.window:]
            bb_middle = float(np.mean(recent_closes))
            bb_std = float(np.std(recent_closes, ddof=1))

            features["bb_middle"] = bb_middle
            features["bb_upper"] = bb_middle + bb_config.num_std * bb_std
            features["bb_lower"] = bb_middle - bb_config.num_std * bb_std
            features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / bb_middle if bb_middle > 0 else 0

        # 11-14. SMA (multiple windows)
        for window in self.config.indicators.sma.windows:
            if len(self.close_window) >= window:
                recent = list(self.close_window.data)[-window:]
                features[f"sma_{window}"] = float(np.mean(recent))

        # 15-17. EMA (multiple windows)
        for window in self.config.indicators.ema.windows:
            alpha = 2 / (window + 1)
            if self.ema_values[window] is None:
                self.ema_values[window] = close
            else:
                self.ema_values[window] = alpha * close + (1 - alpha) * self.ema_values[window]
            features[f"ema_{window}"] = self.ema_values[window]

        # ATR (Average True Range)
        if len(self.high_window) >= 2 and len(self.low_window) >= 2:
            highs = list(self.high_window.data)
            lows = list(self.low_window.data)
            closes = list(self.close_window.data)

            if len(closes) >= 2:
                # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
                tr = max(
                    high - low,
                    abs(high - closes[-2]) if len(closes) >= 2 else 0,
                    abs(low - closes[-2]) if len(closes) >= 2 else 0,
                )
                features["atr"] = tr  # Simplified - would need smoothing for real ATR

        # Volatility (rolling std of returns)
        if self.config.include_volatility and self.returns_window.is_full():
            features["volatility"] = self.returns_window.std()

        # Volume Z-score
        if self.config.include_volume_zscore and self.volume_window.is_full():
            vol_mean = self.volume_window.mean()
            vol_std = self.volume_window.std()
            if vol_std > 0:
                features["volume_zscore"] = (float(volume) - vol_mean) / vol_std
            else:
                features["volume_zscore"] = 0.0

        # Update previous close
        self.prev_close = close

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names that will be computed.

        Returns:
            List of feature names in consistent order
        """
        names = []

        if self.config.include_returns:
            names.append("returns")
        if self.config.include_log_returns:
            names.append("log_returns")

        names.append("rsi")
        names.extend(["macd", "macd_signal", "macd_hist"])
        names.extend(["bb_middle", "bb_upper", "bb_lower", "bb_width"])

        for window in self.config.indicators.sma.windows:
            names.append(f"sma_{window}")

        for window in self.config.indicators.ema.windows:
            names.append(f"ema_{window}")

        names.append("atr")

        if self.config.include_volatility:
            names.append("volatility")

        if self.config.include_volume_zscore:
            names.append("volume_zscore")

        return names

    @property
    def warmup_period(self) -> int:
        """Minimum number of messages needed before indicators are valid."""
        return max(
            self.config.indicators.rsi.window,
            self.config.indicators.macd.slow_period + self.config.indicators.macd.signal_period,
            self.config.indicators.bollinger_bands.window,
            max(self.config.indicators.sma.windows),
            self.config.volatility_window,
        )

    @property
    def is_warmed_up(self) -> bool:
        """Check if enough data has been processed for valid indicators."""
        return self.message_count >= self.warmup_period

