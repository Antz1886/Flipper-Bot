# modules/trend_engine.py
"""Trend detection engine.
Implements EMA and ADX based trend validation for primary (M5) and higher‑timeframe (H1) data.
"""

from typing import Any, Dict
import numpy as np

class TrendEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # EMA periods
        self.fast_ema = config.get('FAST_EMA_PERIOD', 20)
        self.slow_ema = config.get('SLOW_EMA_PERIOD', 50)
        self.htf_ema = config.get('HTF_EMA_PERIOD', 200)
        self.adx_period = config.get('ADX_PERIOD', 14)

    def _ema(self, series: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average using pandas‑like formula.
        Returns array of same length with NaN for the first `period-1` values.
        """
        if len(series) < period:
            return np.full_like(series, np.nan, dtype=float)
        alpha = 2 / (period + 1)
        ema = np.empty_like(series, dtype=float)
        ema[:period - 1] = np.nan
        ema[period - 1] = np.mean(series[:period])
        for i in range(period, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
        return ema

    def detect_trend(self, primary_ohlc: Any, htf_ohlc: Any) -> Dict[str, Any] | None:
        """Return a dict with trend direction and strength or None if no clear trend.
        `primary_ohlc` and `htf_ohlc` are expected to be dicts with `close` arrays.
        """
        # Compute EMAs for primary timeframe
        close = np.array(primary_ohlc['close'])
        fast = self._ema(close, self.fast_ema)
        slow = self._ema(close, self.slow_ema)
        # Simple trend direction
        if fast[-1] > slow[-1]:
            direction = 'UP'
        elif fast[-1] < slow[-1]:
            direction = 'DOWN'
        else:
            return None
        # HTF EMA validation
        htf_close = np.array(htf_ohlc['close'])
        htf_ema = self._ema(htf_close, self.htf_ema)
        if (direction == 'UP' and htf_close[-1] < htf_ema[-1]) or (direction == 'DOWN' and htf_close[-1] > htf_ema[-1]):
            return None
        # ADX strength (placeholder, real implementation would compute ADX)
        adx = primary_ohlc.get('adx', [0])[-1]
        if adx < self.config.get('MIN_ADX_TREND', 20):
            return None
        return {'direction': direction, 'adx': adx}
