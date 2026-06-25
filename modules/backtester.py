# modules/backtester.py
"""Backtester module.
Runs historical simulation of the full strategy pipeline.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

class Backtester:
    def __init__(self, config: Dict[str, Any], components: Dict[str, Any]):
        """`components` should contain instantiated modules:
        trend_engine, volatility_filter, signal_validator, risk_manager, trade_scoring, exit_manager, trade_database.
        """
        self.config = config
        self.components = components
        self.results = []  # list of trade dicts for performance analysis

    def load_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Load OHLC data from CSV files located in `data/` directory.
        Returns a list of dicts ordered chronologically.
        """
        path = f"data/{symbol}.csv"
        df = pd.read_csv(path)
        # Expect columns: timestamp, open, high, low, close, volume, maybe adx, rsi, ema_fast, ema_slow
        records = df.to_dict(orient="records")
        return records

    def run(self) -> Dict[str, Any]:
        """Execute back‑test for all configured symbols.
        Returns a performance summary dict.
        """
        symbols = self.config.get('SYMBOLS', [])
        for symbol in symbols:
            ohlc_series = self.load_data(symbol)
            # Simple rolling window simulation
            for i in range(self.config.get('N_BARS', 500), len(ohlc_series)):
                primary = self._slice_window(ohlc_series, i - self.config.get('N_BARS', 500), i)
                primary_dict = {
                    'close': [x['close'] for x in primary],
                    'high': [x['high'] for x in primary],
                    'low': [x['low'] for x in primary],
                    'open': [x['open'] for x in primary],
                    'rsi': [x.get('rsi', 50.0) for x in primary],
                    'adx': [x.get('adx', 25.0) for x in primary],
                    'ema_fast': [x.get('ema_fast', x['close']) for x in primary],
                    'ema_slow': [x.get('ema_slow', x['close']) for x in primary],
                }
                confluence = None  # Not used in this simple back‑test
                htf = self._slice_window(ohlc_series, max(0, i - 5 * self.config.get('N_BARS', 500)), i)
                htf_dict = {
                    'close': [x['close'] for x in htf],
                }
                bid = primary[-1]["close"]
                ask = bid  # Simplify
                signal = self.components["strategy_manager"].detect_signal(primary_dict, confluence, htf_dict, bid, ask)
                if signal:
                    # Simulate execution and exit via exit_manager
                    entry_price = bid
                    # For simplicity, assume exit after fixed holding period
                    exit_idx = min(i + self.config.get('HOLDING_PERIOD_BARS', 20), len(ohlc_series) - 1)
                    exit_price = ohlc_series[exit_idx]["close"]
                    pnl = (exit_price - entry_price) * (1 if signal["direction"] == "BUY" else -1) * self.config.get('MULTIPLIER', 1)
                    trade = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": symbol,
                        "direction": signal["direction"],
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "stake": 1.0,  # placeholder
                        "pnl": pnl,
                        "score": signal.get("score", 0),
                    }
                    self.results.append(trade)
                    # Persist to DB if available
                    if "trade_database" in self.components:
                        self.components["trade_database"].insert_trade(trade)
        return self._summary()

    def _slice_window(self, series: List[Dict[str, Any]], start: int, end: int) -> List[Dict[str, Any]]:
        return series[start:end]

    def _summary(self) -> Dict[str, Any]:
        pnl_series = np.array([t["pnl"] for t in self.results])
        total_pnl = pnl_series.sum()
        win_rate = (pnl_series > 0).mean() if len(pnl_series) > 0 else 0
        profit_factor = (pnl_series[pnl_series > 0].sum() / -pnl_series[pnl_series < 0].sum()) if (pnl_series < 0).any() else float('inf')
        max_dd = self._max_drawdown(pnl_series)
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252) if pnl_series.std() > 0 else 0
        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "num_trades": len(self.results),
        }

    def _max_drawdown(self, pnl_series: np.ndarray) -> float:
        cumulative = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (peak + 1e-9)
        return drawdown.max() if len(drawdown) > 0 else 0.0
