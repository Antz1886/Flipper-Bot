# modules/trade_scoring.py
"""Trade scoring module.
Computes a numeric score (0‑100) based on indicator strengths and validates score buckets.
"""

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)

class TradeScoring:
    def __init__(self, config: Dict[str, Any]):
        # Bucket definitions from config (e.g., {"high": (90,100), ...})
        self.buckets = config.get('TRADE_SCORE_BUCKETS', {
            'high': (90, 100),
            'mid': (80, 90),
            'low': (70, 80),
        })

    def _raw_score(self, signal: Dict[str, Any]) -> int:
        """Simple heuristic: combine ADX, RSI deviation from threshold, and EMA distance.
        Returns an integer between 0 and 100.
        """
        adx = signal.get('metadata', {}).get('adx', 0)
        rsi = signal.get('metadata', {}).get('rsi', 50)
        # Distance between fast and slow EMA as a percentage of price
        ema_dist = signal.get('metadata', {}).get('ema_distance', 0)
        # Weight components (tunable)
        score = (
            min(adx / 50 * 30, 30) +          # up to 30 pts for ADX strength
            min(abs(rsi - 50) / 50 * 30, 30) + # up to 30 pts for RSI extremeness
            min(ema_dist / 0.05 * 40, 40)       # up to 40 pts for EMA separation
        )
        return int(max(0, min(100, round(score))))

    def score(self, signal: Dict[str, Any]) -> int:
        """Public method to compute and assign a score, and ensure it fits a bucket.
        Logs the bucket classification.
        """
        raw = self._raw_score(signal)
        # Find bucket
        bucket_name = None
        for name, (low, high) in self.buckets.items():
            if low <= raw <= high:
                bucket_name = name
                break
        if bucket_name is None:
            bucket_name = 'unknown'
        log.info(f"Trade scored {raw} → bucket '{bucket_name}'")
        signal['score'] = raw
        signal['score_bucket'] = bucket_name
        return raw
