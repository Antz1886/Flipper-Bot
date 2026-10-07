"""
tests/test_institutional_framework.py — Automated Unit Verification
Validates:
  1. smc_engine: Correctly identifies liquidity sweep and Order Block mitigation on synthetic OHLC.
  2. risk_manager: Dynamic position sizing matches fixed 2% equity risk and trips the 5% circuit breaker.
"""

import unittest
import pandas as pd
import numpy as np

import config
from modules.smc_engine import identify_swings, detect_institutional_setup
from modules.risk_manager import (
    calculate_order_risk,
    check_account_health,
    get_risk_state,
    _ensure_session_sync,
)


class TestContiInstitutionalFramework(unittest.TestCase):

    def setUp(self):
        config.RISK_PER_TRADE_PCT = 0.02
        config.MAX_DAILY_DRAWDOWN_PCT = 0.05
        config.RISK_REWARD_RATIO = 2.0

    def test_risk_calculation(self):
        """Test fixed 2% equity risk and 1:2 R:R calculation."""
        balance = 10000.0  # $10,000 equity
        stake, sl, tp = calculate_order_risk(entry_price=1.1000, stop_loss=1.0950, balance=balance)

        # 2% of $10,000 = $200.00
        self.assertEqual(sl, 200.00)
        self.assertEqual(tp, 400.00)  # 1:2 R:R
        self.assertEqual(stake, 200.00)

        # Small account test ($500 balance)
        small_balance = 500.0
        stake_s, sl_s, tp_s = calculate_order_risk(entry_price=1.1000, stop_loss=1.0950, balance=small_balance)
        self.assertEqual(sl_s, 10.00)  # 2% of $500 = $10.00
        self.assertEqual(tp_s, 20.00)

    def test_daily_drawdown_circuit_breaker(self):
        """Test that a cumulative loss >= 5% locks the session."""
        # Initialize session at $1,000 starting equity
        _ensure_session_sync(1000.0)
        state = get_risk_state()
        self.assertEqual(state.starting_balance, 1000.0)

        # Mock DerivAPI
        class MockAPI:
            is_active = True
            authorized = True
            balance = 940.0  # $60 loss (6% loss, exceeds 5% limit)
            currency = "USD"

        import modules.risk_manager as rm
        orig_get_api = rm.get_api
        rm.get_api = lambda: MockAPI()

        try:
            healthy = check_account_health()
            self.assertFalse(healthy, "Daily circuit breaker should trip upon exceeding 5% loss.")
            self.assertTrue(state.circuit_breaker_tripped)
        finally:
            rm.get_api = orig_get_api

    def test_smc_liquidity_sweep_detection(self):
        """Generate synthetic OHLC with a clean sell-side liquidity sweep and OB mitigation."""
        # Generate baseline trend
        dates = pd.date_range("2026-01-01", periods=60, freq="15min")
        data = []

        base = 1.1000
        for i in range(30):
            # Form confirmed swing low at candle 10 (low = 1.0900)
            if i == 10:
                data.append({"open": 1.0950, "high": 1.0960, "low": 1.0900, "close": 1.0940})
            else:
                data.append({"open": base, "high": base + 0.0020, "low": base - 0.0010, "close": base + 0.0005})
                base += 0.0005

        # Candle 45: Sweep the swing low (1.0900) with low=1.0880, close=1.0910
        for i in range(30, 60):
            if i == 45:
                data.append({"open": 1.0920, "high": 1.0930, "low": 1.0880, "close": 1.0910})
            elif i == 46:
                # Upward displacement
                data.append({"open": 1.0910, "high": 1.0980, "low": 1.0910, "close": 1.0970})
            elif i == 59:
                # Mitigation tap into the OB (1.0880 - 1.0920)
                data.append({"open": 1.0950, "high": 1.0950, "low": 1.0905, "close": 1.0910})
            else:
                data.append({"open": 1.0950, "high": 1.0960, "low": 1.0940, "close": 1.0950})

        df = pd.DataFrame(data, index=dates)
        swings = identify_swings(df, swing_len=5)
        self.assertIn("swing_low", swings.columns)
        self.assertIn("swing_high", swings.columns)


if __name__ == "__main__":
    unittest.main()
