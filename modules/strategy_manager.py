# modules/strategy_manager.py
"""Strategy manager abstraction.
Provides BaseStrategy and TrendStrategy classes that coordinate signal generation,
risk checks, and order execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseStrategy(ABC):
    """Abstract base for all strategies.
    Subclasses must implement `detect_signal` which returns a signal dict or None.
    """

    @abstractmethod
    def detect_signal(
        self,
        primary_ohlc: Any,
        confluence_ohlc: Any,
        htf_ohlc: Any,
        bid: float,
        ask: float,
    ) -> Dict[str, Any] | None:
        """Detect a trade signal.
        Returns a dictionary with at least ``direction`` ("BUY"/"SELL") and optional metadata.
        """
        ...

class TrendStrategy(BaseStrategy):
    """Concrete implementation for the EMA‑RSI trend‑following strategy.
    Orchestrates the trend_engine, volatility_filter, signal_validator, risk_manager,
    trade_scoring and exit_manager.
    """

    def __init__(self, config: Dict[str, Any], components: Dict[str, Any]):
        """Initialize with configuration and a dictionary of component instances.
        Expected keys in ``components``:
            - ``trend_engine``
            - ``volatility_filter``
            - ``signal_validator``
            - ``risk_manager``
            - ``trade_scoring``
        """
        self.config = config
        self.components = components

    def detect_signal(self, primary_ohlc, confluence_ohlc, htf_ohlc, bid, ask):
        # 1. Trend detection
        trend = self.components["trend_engine"].detect_trend(primary_ohlc, htf_ohlc)
        if not trend:
            return None
        # 2. Volatility filter
        if not self.components["volatility_filter"].allow_entry(primary_ohlc):
            return None
        # 3. Signal validation (pull‑back, RSI, etc.)
        signal = self.components["signal_validator"].validate_signal(primary_ohlc, trend)
        if not signal:
            return None
        # 4. Portfolio‑level risk check
        if not self.components["risk_manager"].can_open_position(signal):
            return None
        # 5. Score the signal
        signal["score"] = self.components["trade_scoring"].score(signal)
        return signal
