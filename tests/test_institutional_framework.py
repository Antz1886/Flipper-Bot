"""
tests/test_institutional_framework.py — Quantitative Audit Unit Verification
Validates:
  1. Structural Position Sizing: Stake and dollar SL/TP mathematically bound to |entry - sl| and multiplier.
  2. Cash vs. Equity Accounting: Active open stakes do NOT falsely trip the 5% daily drawdown breaker.
  3. USD Correlation Cap: Blocks opening more than MAX_CONCURRENT_TRADES simultaneous positions.
  4. Chronological SMC Engine: Detects post-swing sweeps with ATR displacement and rejects invalidated OBs.
"""

import unittest
import pandas as pd
import numpy as np

import config
from modules.smc_engine import (
    identify_swings,
    detect_institutional_setup,
    mark_setup_consumed,
    reset_consumed_setups,
)
from modules.risk_manager import (
    calculate_order_risk,
    check_account_health,
    get_risk_state,
    _ensure_session_sync,
    can_open_symbol,
    mark_symbol_active,
    mark_symbol_closed,
)


class TestContiInstitutionalFramework(unittest.TestCase):

    def setUp(self):
        config.RISK_PER_TRADE_PCT = 0.02
        config.MAX_DAILY_DRAWDOWN_PCT = 0.05
        config.RISK_REWARD_RATIO = 2.0
        config.MAX_CONCURRENT_TRADES = 2
        reset_consumed_setups()
        state = get_risk_state()
        state.session_date = ""
        state.active_symbols.clear()
        state.open_stakes_usd = 0.0
        state.circuit_breaker_tripped = False

    def test_structural_position_sizing(self):
        """Verify stake_usd is scaled to structural stop distance so SL triggers at exact structural price."""
        balance = 10000.0  # $10,000 equity -> 2% target risk = $200.00
        entry = 1.1000
        sl_price = 1.0945  # 0.50% stop distance (55 pips)
        # At 100x multiplier, a 0.50% move loses 50% of stake.
        # To lose $200 at 1.0945, stake must be $200 / 0.50 = $400.00!
        stake, sl_usd, tp_usd = calculate_order_risk(
            entry_price=entry, stop_loss=sl_price, balance=balance, symbol="frxEURUSD"
        )

        self.assertAlmostEqual(stake, 400.00, places=2)
        self.assertEqual(sl_usd, 200.00)
        self.assertEqual(tp_usd, 400.00)

        # Small account test ($500 balance -> 2% risk = $10.00)
        stake_s, sl_s, tp_s = calculate_order_risk(
            entry_price=entry, stop_loss=sl_price, balance=500.0, symbol="frxEURUSD"
        )
        self.assertAlmostEqual(stake_s, 20.00, places=2)
        self.assertEqual(sl_s, 10.00)
        self.assertEqual(tp_s, 20.00)

    def test_open_stakes_do_not_falsely_trip_circuit_breaker(self):
        """Verify that cash deducted for active open stakes is counted in effective_equity."""
        _ensure_session_sync(1000.0)
        state = get_risk_state()
        self.assertEqual(state.starting_balance, 1000.0)

        # Simulate 2 open trades locking up $120 in stake (12% of cash balance),
        # so cash balance is $880.00, but open_stakes_usd is $120.00 (effective equity = $1000.00).
        mark_symbol_active("frxEURUSD", stake_usd=60.0)
        mark_symbol_active("frxGBPUSD", stake_usd=60.0)

        class MockAPIActiveStakes:
            is_active = True
            authorized = True
            balance = 880.0  # Cash dropped by $120 due to open stakes, NOT realized loss
            currency = "USD"

        import modules.risk_manager as rm
        orig_get_api = rm.get_api
        rm.get_api = lambda: MockAPIActiveStakes()

        try:
            healthy = check_account_health()
            self.assertTrue(healthy, "Open stakes should NOT trip the daily drawdown circuit breaker.")
            self.assertFalse(state.circuit_breaker_tripped)

            # Now simulate trades closing with a $60 realized loss (cash = $940, open_stakes = $0)
            state.open_stakes_usd = 0.0
            state.active_symbols.clear()

            class MockAPIRealizedLoss:
                is_active = True
                authorized = True
                balance = 940.0  # $60 realized loss (6% > 5% limit)
                currency = "USD"

            rm.get_api = lambda: MockAPIRealizedLoss()
            healthy_after_loss = check_account_health()
            self.assertFalse(healthy_after_loss, "Realized loss >= 5% must trip the circuit breaker.")
            self.assertTrue(state.circuit_breaker_tripped)
        finally:
            rm.get_api = orig_get_api

    def test_usd_correlation_cap_and_unlock(self):
        """Verify max 2 concurrent USD trades are allowed and unlocking frees capacity."""
        self.assertTrue(can_open_symbol("frxEURUSD"))
        mark_symbol_active("frxEURUSD", stake_usd=20.0)
        self.assertFalse(can_open_symbol("frxEURUSD"), "Duplicate symbol must be blocked.")

        self.assertTrue(can_open_symbol("frxGBPUSD"))
        mark_symbol_active("frxGBPUSD", stake_usd=20.0)

        # 3rd symbol should be blocked by MAX_CONCURRENT_TRADES = 2
        self.assertFalse(can_open_symbol("frxUSDJPY"), "3rd concurrent USD trade must be blocked.")

        # Unlocking frxEURUSD should allow frxUSDJPY
        mark_symbol_closed("frxEURUSD")
        self.assertTrue(can_open_symbol("frxUSDJPY"))

    def test_smc_chronological_sweep_and_invalidation(self):
        """Verify chronological liquidity sweep + displacement triggers signal, and broken OB is rejected."""
        dates = pd.date_range("2026-01-01", periods=50, freq="15min")
        data = []

        # Bars 0..39: Normal range around 1.1000 with ATR ~0.0010, and a clear Swing Low at bar 20 (low = 1.0920)
        for i in range(40):
            if i == 20:
                data.append({"open": 1.0950, "high": 1.0960, "low": 1.0920, "close": 1.0955})
            else:
                data.append({"open": 1.0960, "high": 1.0970, "low": 1.0950, "close": 1.0962})

        # Bar 40 (after swing confirmation at 20+5=25): Sweep below 1.0920 down to 1.0905, close strongly at 1.0930
        data.append({"open": 1.0925, "high": 1.0935, "low": 1.0905, "close": 1.0930})

        # Bars 41..48: Rally above the OB without breaking 1.0905
        for i in range(41, 49):
            data.append({"open": 1.0935, "high": 1.0948, "low": 1.0932, "close": 1.0942})

        # Bar 49 (Live forming candle): Pulls back into the OB [1.0905, 1.0930] at 1.0920
        data.append({"open": 1.0940, "high": 1.0940, "low": 1.0918, "close": 1.0920})

        df = pd.DataFrame(data, index=dates)
        sig = detect_institutional_setup("frxEURUSD", df)

        self.assertIsNotNone(sig, "Valid post-swing sweep + displacement + OB pullback must emit SMCSignal.")
        self.assertEqual(sig.direction, "LONG")
        self.assertAlmostEqual(sig.stop_loss, 1.0905, places=5)
        self.assertAlmostEqual(sig.entry_price, 1.0920, places=5)

        # Consuming setup prevents duplicate signal on same OB
        mark_setup_consumed(sig)
        self.assertIsNone(detect_institutional_setup("frxEURUSD", df))

        # Reset consumed cache and test OB invalidation (intermediate bar 44 breaks below 1.0905)
        reset_consumed_setups()
        df.iloc[44, df.columns.get_loc("low")] = 1.0895
        self.assertIsNone(
            detect_institutional_setup("frxEURUSD", df),
            "Order Block breached by an intermediate candle must be invalidated."
        )


if __name__ == "__main__":
    unittest.main()
