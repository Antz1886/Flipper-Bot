# modules/parameter_sensitivity.py
"""Parameter sensitivity testing module.
Runs a grid‑search over configurable EMA, RSI, and ADX ranges and validates that performance metrics stay within a tolerance.
"""

import logging
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

class ParameterSensitivity:
    def __init__(self, config: Dict[str, Any], backtester_factory):
        """`backtester_factory` should be a callable that returns a Backtester instance.
        The Backtester must expose `run()` returning a dict with performance metrics.
        """
        self.config = config
        self.backtester_factory = backtester_factory
        self.tolerance = config.get('PARAM_SENS_TOLERANCE', 0.05)  # 5% allowed degradation

    def _range(self, rng: Tuple[int, int]) -> List[int]:
        lo, hi = rng
        # simple step of 5 for demonstration
        step = max(1, (hi - lo) // 5)
        return list(range(lo, hi + 1, step))

    def run(self) -> bool:
        """Execute the grid‑search and return ``True`` if all combinations meet the tolerance.
        Logs any failing parameter sets.
        """
        ema_fast_vals = self._range(self.config.get('EMA_FAST_RANGE', (15, 25)))
        ema_slow_vals = self._range(self.config.get('EMA_SLOW_RANGE', (40, 60)))
        rsi_vals = self._range(self.config.get('RSI_RANGE', (35, 45)))
        adx_vals = self._range(self.config.get('ADX_RANGE', (20, 35)))

        baseline = self.backtester_factory().run()
        baseline_pf = baseline.get('profit_factor', 1.0)
        log.info(f"Baseline profit factor: {baseline_pf:.3f}")
        all_ok = True
        for ef in ema_fast_vals:
            for es in ema_slow_vals:
                for rsi in rsi_vals:
                    for adx in adx_vals:
                        bt = self.backtester_factory()
                        # Inject temporary config overrides
                        bt.config.update({
                            'FAST_EMA_PERIOD': ef,
                            'SLOW_EMA_PERIOD': es,
                            'RSI_BUY_THRESHOLD': rsi,
                            'RSI_SELL_THRESHOLD': rsi + 10,
                            'MIN_ADX_TREND': adx,
                        })
                        result = bt.run()
                        pf = result.get('profit_factor', 0)
                        if pf < baseline_pf * (1 - self.tolerance):
                            all_ok = False
                            log.warning(
                                f"Parameter set EF={ef}, ES={es}, RSI={rsi}, ADX={adx} degraded PF to {pf:.3f}"
                            )
        return all_ok
