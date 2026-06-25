# modules/exit_manager.py
"""Exit manager module.
Implements dynamic trailing‑stop and profit‑lock logic.
"""

from typing import Any, Dict
import logging

log = logging.getLogger(__name__)

class ExitManager:
    def __init__(self, config: Dict[str, Any]):
        self.atr_multiplier = config.get('ATR_MULTIPLIER', 1.5)
        self.trailing_atr_factor = config.get('TRAILING_ATR_FACTOR', 2)

    def _atr(self, high: Any, low: Any, close: Any, period: int = 14):
        """Simple ATR calculation (Wilder) – returns latest ATR value."""
        import numpy as np
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        atr = np.mean(tr[-period:])
        return atr

    def trailing_stop(self, current_price: float, ohlc: Any) -> float:
        """Calculate trailing stop price based on current price minus 2 * ATR."""
        atr = self._atr(ohlc['high'], ohlc['low'], ohlc['close'])
        stop = current_price - self.trailing_atr_factor * atr
        log.debug(f"Trailing stop calculated: {stop:.4f} (price {current_price:.4f}, ATR {atr:.4f})")
        return stop

    def profit_lock(self, entry_price: float, direction: str, ohlc: Any) -> float:
        """Calculate a profit‑lock level (e.g., 1.5 * ATR from entry)."""
        atr = self._atr(ohlc['high'], ohlc['low'], ohlc['close'])
        if direction == "BUY":
            lock = entry_price + self.atr_multiplier * atr
        else:
            lock = entry_price - self.atr_multiplier * atr
        log.debug(f"Profit lock calculated: {lock:.4f} (entry {entry_price:.4f}, ATR {atr:.4f})")
        return lock
