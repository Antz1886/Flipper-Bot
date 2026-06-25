# modules/signal_validator.py
"""Signal validator module.
Validates pull‑back conditions and RSI thresholds after the trend and volatility filters.
"""

from typing import Any, Dict

class SignalValidator:
    def __init__(self, config: Dict[str, Any]):
        self.rsi_buy = config.get('RSI_BUY_THRESHOLD', 45)
        self.rsi_sell = config.get('RSI_SELL_THRESHOLD', 55)
        self.pullback_zone_percent = config.get('PULLBACK_ZONE_PCT', 0.03)  # 3% price zone for pull‑back

    def _rsi(self, ohlc: Any) -> float:
        """Placeholder for RSI calculation – expects ``ohlc`` dict with ``close`` list.
        In production this would compute a true RSI; here we simply return the last value of a pre‑computed series if present.
        """
        if 'rsi' in ohlc:
            return ohlc['rsi'][-1]
        # Fallback – assume neutral
        return 50.0

    def validate_signal(self, primary_ohlc: Any, trend: Dict[str, Any]) -> Dict[str, Any] | None:
        """Validate that a pull‑back and RSI condition are met.
        Returns a signal dict with ``direction`` and ``metadata`` or ``None`` if no signal.
        """
        direction = trend.get('direction')
        if not direction:
            return None

        # Pull‑back detection – price must re‑enter the EMA20‑EMA50 zone (simplified)
        # We approximate using the last close vs. EMA distance (assumed present in ohlc)
        close = primary_ohlc['close'][-1]
        ema_fast = primary_ohlc.get('ema_fast', [0])[-1]
        ema_slow = primary_ohlc.get('ema_slow', [0])[-1]
        zone_low = min(ema_fast, ema_slow) * (1 - self.pullback_zone_percent)
        zone_high = max(ema_fast, ema_slow) * (1 + self.pullback_zone_percent)
        in_zone = zone_low <= close <= zone_high
        if not in_zone:
            return None

        rsi = self._rsi(primary_ohlc)
        if direction == 'UP' and rsi < self.rsi_buy:
            return None
        if direction == 'DOWN' and rsi > self.rsi_sell:
            return None

        signal = {
            'direction': 'BUY' if direction == 'UP' else 'SELL',
            'metadata': {
                'rsi': rsi,
                'close': close,
                'zone_low': zone_low,
                'zone_high': zone_high,
            },
        }
        return signal
