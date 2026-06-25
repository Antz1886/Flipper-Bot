# modules/volatility_filter.py
"""Volatility filter module.
Calculates ATR and median‑ATR for the primary timeframe and decides whether market
energy is sufficient for entry.
"""

import numpy as np
from typing import Any, Dict

class VolatilityFilter:
    def __init__(self, config: Dict[str, Any]):
        self.atr_period = config.get('ATR_PERIOD', 14)
        self.median_window = config.get('ATR_MEDIAN_WINDOW', 20)

    def _atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Average True Range (ATR) using the standard Wilder method.
        Returns an array matching ``close`` length with ``nan`` for the first ``atr_period`` values.
        """
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        atr = np.empty_like(close, dtype=float)
        atr[:self.atr_period] = np.nan
        atr[self.atr_period] = np.mean(tr[:self.atr_period])
        for i in range(self.atr_period + 1, len(close)):
            atr[i] = (atr[i - 1] * (self.atr_period - 1) + tr[i - 1]) / self.atr_period
        return atr

    def allow_entry(self, ohlc: Any) -> bool:
        """Return ``True`` if the latest ATR exceeds the median of the previous ``median_window`` ATR values.
        ``ohlc`` is expected to be a mapping with ``high``, ``low`` and ``close`` list‑like sequences.
        """
        high = np.array(ohlc['high'])
        low = np.array(ohlc['low'])
        close = np.array(ohlc['close'])
        atr = self._atr(high, low, close)
        if np.isnan(atr[-1]):
            return False
        median_atr = np.nanmedian(atr[-self.median_window:])
        return atr[-1] > median_atr
