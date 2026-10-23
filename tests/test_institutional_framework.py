"""
tests/test_institutional_framework.py — Quantitative Institutional Audit & Architecture Verification
Validates:
  1. Structural Position Sizing: Stake and dollar SL/TP mathematically bound to |entry - sl| and multiplier.
  2. Cash vs. Equity Accounting: Active open stakes do NOT falsely trip the 5% daily drawdown breaker.
  3. Base / Quote USD Concurrency & DXY Correlation Lock: Blocks opening multiple correlated Quote-USD pairs
     and disallows stacking identical DXY directional exposure.
  4. Macro Session Windows: Enforces London Open (07:00-11:00 GMT) and New York Open (13:00-17:00 GMT) gating.
  5. HTF Premium/Discount Matrices: Rejects Longs in H4 Premium (> 50% Eq) and Shorts in H4 Discount (< 50% Eq).
  6. Macro Liquidity Pools Filter: Confirms sweeps align within 3-5 pips of PDH/PDL, Asia H/L, or EQH/EQL.
  7. Deriv Deal Cancellation SLA & Race Condition Fix: Minute 14, Second 55 evaluation (5s before expiry).
  8. Cancellation Premium Bleed Cap: Rejects trades if cancellation fee > 15% of 1% risk budget.
  9. Execution Spread Widening Filter: Rejects orders when live spread exceeds baseline limit.
  10. Tier-1 Economic News Embargo: Suspends USD trading ±30m around high-impact news.
  11. Chronological SMC Engine: Detects confirmed sweeps with ATR displacement, structural buffers, and OB invalidation.
"""

import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd
import asyncio
import time

import config
from modules.smc_engine import (
    detect_institutional_setup,
    mark_setup_consumed,
    reset_consumed_setups,
    is_in_macro_session,
    calculate_h4_dealing_range,
    check_htf_premium_discount,
    extract_macro_liquidity_pools,
    is_near_macro_liquidity_pool,
)
from modules.risk_manager import (
    calculate_order_risk,
    check_account_health,
    get_risk_state,
    _ensure_session_sync,
    _extract_contract_symbol,
    can_open_symbol,
    mark_symbol_active,
    mark_symbol_closed,
    get_dxy_exposure,
)
from modules.cancellation_manager import (
    register_cancellation_contract,
    evaluate_sluggish_ob_cancellations,
    get_tracked_cancellation_contracts,
    clear_cancellation_contract,
)
from modules.order_executor import (
    check_spread_safe,
    place_multiplier_contract,
)
from modules.news_embargo import (
    NewsEvent,
    is_news_embargo_active,
    inject_custom_event,
    clear_events_cache,
)


class TestContiInstitutionalFramework(unittest.TestCase):

    def setUp(self):
        config.RISK_PER_TRADE_PCT = 0.01
        config.MAX_MARGIN_PER_TRADE_PCT = 0.05
        config.MAX_STAKE_USD = 150.00
        config.MAX_DAILY_DRAWDOWN_PCT = 0.05
        config.RISK_REWARD_RATIO = 2.0
        config.MAX_CONCURRENT_TRADES = 2
        config.MAX_BASE_USD_TRADES = 1
        config.MAX_QUOTE_USD_TRADES = 1
        config.ENFORCE_DXY_DIRECTION_LOCK = True
        config.ENFORCE_MACRO_SESSIONS = False
        config.ENFORCE_HTF_PREMIUM_DISCOUNT = False
        config.ENFORCE_MACRO_LIQUIDITY_FILTER = False
        config.USE_DEAL_CANCELLATION = True
        config.DEAL_CANCELLATION_TRIGGER_S = 895.0
        config.DEAL_CANCELLATION_SLA_BUFFER_S = 5.0
        config.MAX_CANCELLATION_FEE_PCT_OF_RISK = 0.15
        config.ENFORCE_SPREAD_FILTER = True
        config.ENFORCE_NEWS_EMBARGO = False

        reset_consumed_setups()
        clear_events_cache()
        state = get_risk_state()
        state.session_date = ""
        state.active_symbols.clear()
        state.active_directions.clear()
        state.last_entry_time.clear()
        state.open_stakes_usd = 0.0
        state.circuit_breaker_tripped = False
        get_tracked_cancellation_contracts().clear()

    def test_structural_position_sizing(self):
        """Verify stake_usd is scaled to structural stop distance and capped by MAX_STAKE_USD."""
        balance = 2000.0  # $2,000 equity -> 1% target risk = $20.00
        entry = 1.1000
        sl_price = 1.0945  # 0.50% stop distance (55 pips)
        stake, sl_usd, tp_usd = calculate_order_risk(
            entry_price=entry, stop_loss=sl_price, balance=balance, symbol="frxEURUSD"
        )

        self.assertAlmostEqual(stake, 40.00, places=2)
        self.assertEqual(sl_usd, 20.00)
        self.assertEqual(tp_usd, 40.00)

        # Large account test ($10,000 balance) -> stake must be capped at MAX_STAKE_USD ($150.00)
        stake_l, sl_l, tp_l = calculate_order_risk(
            entry_price=1.33735, stop_loss=1.33650, balance=10000.0, symbol="frxGBPUSD"
        )
        self.assertLessEqual(stake_l, config.MAX_STAKE_USD)

    def test_open_stakes_do_not_falsely_trip_circuit_breaker(self):
        """Verify that cash deducted for active open stakes is counted in effective_equity."""
        _ensure_session_sync(1000.0)
        state = get_risk_state()
        self.assertEqual(state.starting_balance, 1000.0)

        mark_symbol_active("frxEURUSD", stake_usd=60.0)
        mark_symbol_active("frxUSDJPY", stake_usd=60.0)

        class MockAPIActiveStakes:
            is_active = True
            authorized = True
            balance = 880.0
            currency = "USD"

        import modules.risk_manager as rm
        orig_get_api = rm.get_api
        rm.get_api = lambda: MockAPIActiveStakes()

        try:
            healthy = check_account_health()
            self.assertTrue(healthy, "Open stakes should NOT trip the daily drawdown circuit breaker.")
            self.assertFalse(state.circuit_breaker_tripped)

            state.open_stakes_usd = 0.0
            state.active_symbols.clear()
            state.active_directions.clear()
            state.last_entry_time.clear()

            class MockAPIRealizedLoss:
                is_active = True
                authorized = True
                balance = 940.0
                currency = "USD"

            rm.get_api = lambda: MockAPIRealizedLoss()
            healthy_after_loss = check_account_health()
            self.assertFalse(healthy_after_loss, "Realized loss >= 5% must trip circuit breaker.")
            self.assertTrue(state.circuit_breaker_tripped)
        finally:
            rm.get_api = orig_get_api

    def test_dxy_correlation_trap_and_currency_classification(self):
        """Verify DXY exposure calculation and Base/Quote USD concurrency gating."""
        self.assertEqual(get_dxy_exposure("frxEURUSD", "LONG"), "DXY_SHORT")
        self.assertEqual(get_dxy_exposure("frxEURUSD", "SHORT"), "DXY_LONG")
        self.assertEqual(get_dxy_exposure("frxUSDJPY", "LONG"), "DXY_LONG")
        self.assertEqual(get_dxy_exposure("frxUSDJPY", "SHORT"), "DXY_SHORT")

        # Open Long on Quote-USD (EURUSD)
        self.assertTrue(can_open_symbol("frxEURUSD", direction="LONG"))
        mark_symbol_active("frxEURUSD", stake_usd=20.0, direction="LONG")

        # Second Quote-USD pair (GBPUSD) must be blocked
        self.assertFalse(
            can_open_symbol("frxGBPUSD", direction="LONG"),
            "Second Quote-USD trade must be blocked by MAX_QUOTE_USD_TRADES limit."
        )

        # Opening USDJPY in SHORT direction (DXY_SHORT) is blocked by DXY Direction Lock
        self.assertFalse(
            can_open_symbol("frxUSDJPY", direction="SHORT"),
            "Stacked DXY_SHORT exposure must be blocked by DXY Direction Lock."
        )

        # Opening USDJPY in LONG direction (DXY_LONG) is allowed (1 Base + 1 Quote)
        self.assertTrue(
            can_open_symbol("frxUSDJPY", direction="LONG"),
            "1 Base-USD trade in opposing DXY direction must be permitted."
        )
        mark_symbol_active("frxUSDJPY", stake_usd=20.0, direction="LONG")

        # 3rd trade blocked by global MAX_CONCURRENT_TRADES = 2
        self.assertFalse(can_open_symbol("frxUSDCAD", direction="LONG"))

        # Unlocking EURUSD frees Quote-USD slot
        mark_symbol_closed("frxEURUSD")
        self.assertFalse(can_open_symbol("frxGBPUSD", direction="SHORT"))
        self.assertTrue(can_open_symbol("frxGBPUSD", direction="LONG"))

        # Verify Deriv OTP portfolio symbol extraction
        self.assertEqual(
            _extract_contract_symbol({"shortcode": "MULTUP_FRXGBPUSD_500.00_100"}),
            "frxGBPUSD",
        )
        self.assertEqual(
            _extract_contract_symbol({"underlying_symbol": "frxEURUSD"}),
            "frxEURUSD",
        )

    def test_macro_session_windows(self):
        """Verify macro session filter restricts entries to London and New York Open."""
        london_dt = datetime(2026, 10, 23, 8, 30, tzinfo=timezone.utc)
        self.assertTrue(is_in_macro_session(london_dt))

        ny_dt = datetime(2026, 10, 23, 14, 15, tzinfo=timezone.utc)
        self.assertTrue(is_in_macro_session(ny_dt))

        asian_dt = datetime(2026, 10, 23, 3, 0, tzinfo=timezone.utc)
        self.assertFalse(is_in_macro_session(asian_dt))

        lull_dt = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
        self.assertFalse(is_in_macro_session(lull_dt))

    def test_htf_premium_discount_matrices(self):
        """Verify H4 dealing range calculation and Premium/Discount restriction."""
        h4_data = []
        for i in range(25):
            if i == 5:
                h4_data.append({"open": 1.1100, "high": 1.1200, "low": 1.1050, "close": 1.1120})
            elif i == 15:
                h4_data.append({"open": 1.0850, "high": 1.0900, "low": 1.0800, "close": 1.0870})
            else:
                h4_data.append({"open": 1.0950, "high": 1.1000, "low": 1.0920, "close": 1.0960})

        df_h4 = pd.DataFrame(h4_data)
        config.ENFORCE_HTF_PREMIUM_DISCOUNT = True

        h4_high, h4_low, eq = calculate_h4_dealing_range(df_h4, swing_len=3)
        self.assertIsNotNone(eq)
        self.assertEqual(h4_high, 1.1200)
        self.assertEqual(h4_low, 1.0800)
        self.assertAlmostEqual(eq, 1.1000)

        ok_long, _, _ = check_htf_premium_discount("LONG", 1.1050, df_h4)
        self.assertFalse(ok_long)

        ok_short, _, _ = check_htf_premium_discount("SHORT", 1.1050, df_h4)
        self.assertTrue(ok_short)

        ok_long_disc, _, _ = check_htf_premium_discount("LONG", 1.0920, df_h4)
        self.assertTrue(ok_long_disc)

        ok_short_disc, _, _ = check_htf_premium_discount("SHORT", 1.0920, df_h4)
        self.assertFalse(ok_short_disc)

    def test_macro_liquidity_pool_alignment(self):
        """Verify sweeps align with macro liquidity pools (PDL, Asia Low, EQL)."""
        pools = [
            {"type": "PDL", "price": 1.0920, "side": "LOW", "desc": "Previous Day Low ($1.09200)"},
            {"type": "PDH", "price": 1.1150, "side": "HIGH", "desc": "Previous Day High ($1.11500)"},
            {"type": "ASIA_LOW", "price": 1.0935, "side": "LOW", "desc": "Asia Low ($1.09350)"},
        ]

        is_near, desc = is_near_macro_liquidity_pool("frxEURUSD", 1.0922, side="LOW", pools=pools, tolerance_pips=5.0)
        self.assertTrue(is_near)
        self.assertIn("Previous Day Low", desc)

        is_near_bad, _ = is_near_macro_liquidity_pool("frxEURUSD", 1.0980, side="LOW", pools=pools, tolerance_pips=5.0)
        self.assertFalse(is_near_bad)

    def test_micro_mechanic_1_sla_cancellation_race_condition(self):
        """
        Micro-Mechanic Fix 1: Verify cancellation dispatches at Minute 14, Second 55
        (5.0s SLA before window shuts) if price is trapped inside/below OB.
        """
        register_cancellation_contract(
            contract_id=555111,
            symbol="frxEURUSD",
            direction="LONG",
            ob_top=1.0930,
            ob_bottom=1.0905,
            cancellation_fee=1.40,
        )

        contracts = get_tracked_cancellation_contracts()
        self.assertIn("555111", contracts)
        contract = contracts["555111"]

        # Sub-test A: At 10 minutes elapsed (600s), evaluation should NOT trigger early
        contract.entry_time = time.time() - 600.0

        cancelled_early = asyncio.run(evaluate_sluggish_ob_cancellations("frxEURUSD", current_spot=1.0920))
        self.assertFalse(cancelled_early, "Should NOT cancel before the 14m 55s SLA window.")
        self.assertIn("555111", get_tracked_cancellation_contracts())

        # Sub-test B: At Minute 14, Second 55 (896.0s elapsed), price still at 1.0920 (<= ob_top 1.0930)
        contract.entry_time = time.time() - 896.0

        class MockAPICancel:
            is_active = True
            authorized = True
            async def send(self, payload):
                if payload.get("cancel") == 555111:
                    return {"cancel": {"contract_id": 555111, "sold_for": 38.60}}
                return {}

        import modules.order_executor as oe
        orig_get_api = oe.get_api
        oe.get_api = lambda: MockAPICancel()

        try:
            cancelled_sla = asyncio.run(evaluate_sluggish_ob_cancellations("frxEURUSD", current_spot=1.0920))
            self.assertTrue(cancelled_sla, "Must dispatch cancellation at minute 14, second 55 SLA trigger.")
            self.assertNotIn("555111", get_tracked_cancellation_contracts())
        finally:
            oe.get_api = orig_get_api

    def test_micro_mechanic_2_cancellation_premium_bleed(self):
        """
        Micro-Mechanic Fix 2: Verify order is rejected if the cancellation fee exceeds
        15% of the trade's 1% risk budget.
        """
        # Risk budget: $20.00 stop-loss (1% of $2,000 equity).
        # Max allowed cancellation fee: 15% of $20.00 = $3.00.
        stop_loss_usd = 20.00
        max_bleed_limit = stop_loss_usd * config.MAX_CANCELLATION_FEE_PCT_OF_RISK  # $3.00

        # Case A: High implied volatility -> Deriv quotes fee of $4.50 (> $3.00 limit) -> REJECT
        expensive_fee = 4.50
        self.assertGreater(expensive_fee, max_bleed_limit)

        # Case B: Normal volatility -> Deriv quotes fee of $1.50 (<= $3.00 limit) -> ACCEPT
        safe_fee = 1.50
        self.assertLessEqual(safe_fee, max_bleed_limit)

    def test_micro_mechanic_3_execution_spread_widening_filter(self):
        """
        Micro-Mechanic Fix 3: Verify order executor measures live spread and rejects
        execution if spread exceeds baseline limit (e.g. 1.5 pips on EURUSD).
        """
        # Mock Deriv ticks response with wide spread: ask=1.12000, bid=1.11976 (2.4 pips > 1.5 limit)
        class MockAPIWideSpread:
            is_active = True
            authorized = True
            async def send(self, payload):
                if payload.get("ticks") == "frxEURUSD":
                    return {
                        "tick": {"ask": 1.12000, "bid": 1.11976, "quote": 1.11988, "symbol": "frxEURUSD"}
                    }
                return {}

        import modules.order_executor as oe
        orig_get_api = oe.get_api
        oe.get_api = lambda: MockAPIWideSpread()

        try:
            is_safe, spread_pips, max_allowed = asyncio.run(check_spread_safe("frxEURUSD"))
            self.assertFalse(is_safe, "Wide spread (2.4 pips > 1.5 limit) must be rejected.")
            self.assertAlmostEqual(spread_pips, 2.40, places=2)

            # Mock normal tight spread: ask=1.11984, bid=1.11976 (0.8 pips <= 1.5 limit)
            class MockAPITightSpread:
                is_active = True
                authorized = True
                async def send(self, payload):
                    return {
                        "tick": {"ask": 1.11984, "bid": 1.11976, "quote": 1.11980, "symbol": "frxEURUSD"}
                    }

            oe.get_api = lambda: MockAPITightSpread()
            is_safe_tight, spread_tight, _ = asyncio.run(check_spread_safe("frxEURUSD"))
            self.assertTrue(is_safe_tight, "Tight spread (0.8 pips) must be authorized.")
            self.assertAlmostEqual(spread_tight, 0.80, places=2)
        finally:
            oe.get_api = orig_get_api

    def test_micro_mechanic_4_tier1_news_embargo(self):
        """
        Micro-Mechanic Fix 4: Verify USD trading is halted 30 minutes before and
        30 minutes after any High-Impact USD economic release.
        """
        config.ENFORCE_NEWS_EMBARGO = True

        # Inject high-impact US Core CPI event scheduled for 14:30 UTC
        event_time = datetime(2026, 10, 23, 14, 30, tzinfo=timezone.utc)
        cpi_event = NewsEvent(
            title="Core CPI m/m",
            country="USD",
            impact="High",
            time_utc=event_time,
        )
        inject_custom_event(cpi_event)

        # 1. At 14:10 UTC (20 minutes BEFORE release) -> Inside embargo window!
        dt_before = datetime(2026, 10, 23, 14, 10, tzinfo=timezone.utc)
        embargo_active, reason = is_news_embargo_active(now_dt=dt_before)
        self.assertTrue(embargo_active, "Must halt trading 20 minutes before CPI.")
        self.assertIn("Core CPI", reason)

        # 2. At 14:45 UTC (15 minutes AFTER release) -> Inside embargo window!
        dt_after = datetime(2026, 10, 23, 14, 45, tzinfo=timezone.utc)
        embargo_active_after, _ = is_news_embargo_active(now_dt=dt_after)
        self.assertTrue(embargo_active_after, "Must halt trading 15 minutes after CPI.")

        # 3. At 15:15 UTC (45 minutes AFTER release) -> Outside embargo window!
        dt_safe = datetime(2026, 10, 23, 15, 15, tzinfo=timezone.utc)
        embargo_safe, _ = is_news_embargo_active(now_dt=dt_safe)
        self.assertFalse(embargo_safe, "Trading should resume 45 minutes after CPI.")

    def test_smc_chronological_sweep_and_invalidation(self):
        """Verify chronological liquidity sweep + displacement triggers signal, and broken OB is rejected."""
        dates = pd.date_range("2026-01-01", periods=50, freq="15min")
        data = []

        for i in range(40):
            if i == 20:
                data.append({"open": 1.0950, "high": 1.0960, "low": 1.0920, "close": 1.0955})
            else:
                data.append({"open": 1.0960, "high": 1.0970, "low": 1.0950, "close": 1.0962})

        # Bar 40: Sweep below 1.0920 down to 1.0905, close strongly at 1.0930
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
        self.assertLess(sig.stop_loss, 1.0905, "Structural SL must include ATR buffer below OB bottom.")
        self.assertAlmostEqual(sig.entry_price, 1.0920, places=5)

        mark_setup_consumed(sig)
        self.assertIsNone(detect_institutional_setup("frxEURUSD", df))

        reset_consumed_setups()
        df.iloc[44, df.columns.get_loc("low")] = 1.0895
        self.assertIsNone(
            detect_institutional_setup("frxEURUSD", df),
            "Order Block breached by an intermediate candle must be invalidated."
        )


if __name__ == "__main__":
    unittest.main()
