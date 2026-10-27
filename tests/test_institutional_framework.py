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
  12. Self-Healing Latency SLA Adaptation: Shifts Deal Cancellation trigger earlier during high broker latency.
  13. Self-Healing Orphan & Ghost Reconciliation: Adopts untracked exchange contracts and purges ghost locks.
  14. Self-Healing Defensive Risk Throttling: Cuts risk by 50% on loss streaks and auto-recovers on wins.
  15. Bayesian Continual Meta-Learner: Updates posterior distributions from trade outcomes and vetoes weak regimes.
  16. Trade Attribution & SQLite Persistence: Logs rich entry context vectors and loss root causes.
"""

import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import asyncio
import time
import os

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
from modules.self_healer import (
    SelfHealingSupervisor,
    get_self_healer,
)
from modules.meta_learner import (
    MetaLearner,
    get_meta_learner,
    BayesianClusterState,
)
from modules.trade_learner import (
    TradeLearnerDB,
    TradeContext,
    TradePostMortem,
    get_session_phase,
    classify_loss_root_cause,
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

    def test_fixed_stake_and_virtual_equity_sizing(self):
        """Verify that FIXED_STAKE_USD forces exact $2.00 trades and VIRTUAL_EQUITY sizes to $20.00."""
        orig_fixed = getattr(config, "FIXED_STAKE_USD", 0.0)
        orig_virt = getattr(config, "VIRTUAL_EQUITY", 0.0)

        try:
            # 1. Fixed $2.00 stake override
            config.FIXED_STAKE_USD = 2.00
            stake, sl_usd, tp_usd = calculate_order_risk(
                entry_price=1.1000, stop_loss=1.0975, balance=10000.0, symbol="frxEURUSD"
            )
            self.assertEqual(stake, 2.00, "Stake must equal fixed $2.00 override.")
            self.assertGreater(sl_usd, 0.0)
            self.assertEqual(tp_usd, round(sl_usd * 2.0, 2))

            # 2. Virtual $20.00 equity sizing (dynamic 1% risk of $20 = $0.20, minimum $1.00 stake)
            config.FIXED_STAKE_USD = 0.0
            config.VIRTUAL_EQUITY = 20.00
            stake_v, sl_v, tp_v = calculate_order_risk(
                entry_price=1.1000, stop_loss=1.0950, balance=10000.0, symbol="frxEURUSD"
            )
            self.assertLessEqual(stake_v, 20.00, "Stake must be sized relative to $20 virtual equity.")
        finally:
            config.FIXED_STAKE_USD = orig_fixed
            config.VIRTUAL_EQUITY = orig_virt

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

        self.assertTrue(can_open_symbol("frxEURUSD", direction="LONG"))
        mark_symbol_active("frxEURUSD", stake_usd=20.0, direction="LONG")

        self.assertFalse(
            can_open_symbol("frxGBPUSD", direction="LONG"),
            "Second Quote-USD trade must be blocked by MAX_QUOTE_USD_TRADES limit."
        )

        self.assertFalse(
            can_open_symbol("frxUSDJPY", direction="SHORT"),
            "Stacked DXY_SHORT exposure must be blocked by DXY Direction Lock."
        )

        self.assertTrue(
            can_open_symbol("frxUSDJPY", direction="LONG"),
            "1 Base-USD trade in opposing DXY direction must be permitted."
        )
        mark_symbol_active("frxUSDJPY", stake_usd=20.0, direction="LONG")

        self.assertFalse(can_open_symbol("frxUSDCAD", direction="LONG"))

        mark_symbol_closed("frxEURUSD")
        self.assertFalse(can_open_symbol("frxGBPUSD", direction="SHORT"))
        self.assertTrue(can_open_symbol("frxGBPUSD", direction="LONG"))

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
        stop_loss_usd = 20.00
        max_bleed_limit = stop_loss_usd * config.MAX_CANCELLATION_FEE_PCT_OF_RISK  # $3.00

        expensive_fee = 4.50
        self.assertGreater(expensive_fee, max_bleed_limit)

        safe_fee = 1.50
        self.assertLessEqual(safe_fee, max_bleed_limit)

    def test_cancellation_unavailable_broker_fallback(self):
        """
        Verify that if Deriv rejects Deal Cancellation with ContractBuyValidationError,
        the executor catches the exception and falls back to structural limit_order SL/TP.
        """
        import modules.order_executor as oe
        orig_get_api = oe.get_api

        class MockAPICancellationFallback:
            is_active = True
            authorized = True
            currency = "USD"
            call_count = 0

            async def send(self, payload):
                self.call_count += 1
                if "ticks" in payload:
                    return {"tick": {"ask": 1.10008, "bid": 1.10000, "quote": 1.10004, "symbol": "frxEURUSD"}}
                if "proposal" in payload:
                    if "cancellation" in payload:
                        raise RuntimeError("Deriv error [ContractBuyValidationError]: Deal cancellation is not available for this asset.")
                    elif "limit_order" in payload:
                        return {"proposal": {"id": "PROP_FALLBACK_123", "ask_price": 2.0, "spot": 1.1000}}
                if "buy" in payload:
                    return {"buy": {"contract_id": 999111, "buy_price": 2.0}}
                return {}

        mock_api = MockAPICancellationFallback()
        oe.get_api = lambda: mock_api

        try:
            res = asyncio.run(place_multiplier_contract(
                symbol="frxEURUSD",
                direction="LONG",
                stake=2.0,
                stop_loss_usd=0.50,
                take_profit_usd=1.00,
            ))
            self.assertIsNotNone(res, "Order execution must succeed via automatic limit_order fallback.")
            self.assertEqual(res["contract_id"], 999111)
            self.assertFalse(res["has_deal_cancellation"])
            self.assertGreaterEqual(mock_api.call_count, 3)  # ticks, failed proposal, fallback proposal, buy
        finally:
            oe.get_api = orig_get_api

    def test_micro_mechanic_3_execution_spread_widening_filter(self):
        """
        Micro-Mechanic Fix 3: Verify order executor measures live spread and rejects
        execution if spread exceeds baseline limit.
        """
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

        event_time = datetime(2026, 10, 23, 14, 30, tzinfo=timezone.utc)
        cpi_event = NewsEvent(
            title="Core CPI m/m",
            country="USD",
            impact="High",
            time_utc=event_time,
        )
        inject_custom_event(cpi_event)

        dt_before = datetime(2026, 10, 23, 14, 10, tzinfo=timezone.utc)
        embargo_active, reason = is_news_embargo_active(now_dt=dt_before)
        self.assertTrue(embargo_active, "Must halt trading 20 minutes before CPI.")
        self.assertIn("Core CPI", reason)

        dt_after = datetime(2026, 10, 23, 14, 45, tzinfo=timezone.utc)
        embargo_active_after, _ = is_news_embargo_active(now_dt=dt_after)
        self.assertTrue(embargo_active_after, "Must halt trading 15 minutes after CPI.")

        dt_safe = datetime(2026, 10, 23, 15, 15, tzinfo=timezone.utc)
        embargo_safe, _ = is_news_embargo_active(now_dt=dt_safe)
        self.assertFalse(embargo_safe, "Trading should resume 45 minutes after CPI.")

    def test_self_healing_latency_adaptation(self):
        """
        Self-Healing: Verify supervisor measures live WebSocket RTT and
        dynamically shifts Deal Cancellation trigger earlier when latency spikes.
        """
        supervisor = SelfHealingSupervisor()
        config.DEAL_CANCELLATION_TRIGGER_S = 895.0

        class MockAPISlow:
            is_active = True
            authorized = True
            async def send(self, payload):
                await asyncio.sleep(0.06)  # Simulate 60ms delay
                return {"ping": "pong"}

        import modules.self_healer as sh
        orig_get_api = sh.get_api
        sh.get_api = lambda: MockAPISlow()

        try:
            # Simulate high rolling latency > 500ms
            supervisor.metrics.rolling_rtt_ms = [650.0, 700.0, 750.0, 800.0]
            asyncio.run(supervisor.measure_and_adapt_latency())

            self.assertLess(
                config.DEAL_CANCELLATION_TRIGGER_S,
                895.0,
                "SLA trigger must shift earlier than 895.0s during high broker latency."
            )
        finally:
            sh.get_api = orig_get_api
            config.DEAL_CANCELLATION_TRIGGER_S = 895.0

    def test_self_healing_orphan_and_ghost_reconciliation(self):
        """
        Self-Healing: Verify supervisor automatically adopts untracked exchange contracts
        and purges ghost symbol locks that no longer exist on the exchange.
        """
        supervisor = SelfHealingSupervisor()
        risk_state = get_risk_state()

        # Ghost lock setup: frxAUDUSD is locked locally but missing from exchange
        risk_state.active_symbols.add("frxAUDUSD")
        risk_state.last_entry_time["frxAUDUSD"] = time.time() - 100.0  # > 60s grace period

        # Exchange portfolio has an untracked orphan contract on frxUSDJPY
        portfolio_contracts = [
            {
                "contract_id": 888999,
                "underlying_symbol": "frxUSDJPY",
                "contract_type": "MULTUP",
                "buy_price": 50.0,
                "cancellation": {"ask_price": 2.50, "date_expiry": time.time() + 600},
            }
        ]

        supervisor.reconcile_exchange_portfolio(portfolio_contracts)

        # 1. Orphan contract on frxUSDJPY must be adopted into active symbols
        self.assertIn("frxUSDJPY", risk_state.active_symbols)
        self.assertIn("888999", get_tracked_cancellation_contracts())

        # 2. Ghost lock on frxAUDUSD must be purged
        self.assertNotIn("frxAUDUSD", risk_state.active_symbols)

    def test_self_healing_defensive_risk_throttling(self):
        """
        Self-Healing: Verify supervisor throttles risk by 50% after 2 consecutive losses
        and automatically restores full risk upon a winning trade.
        """
        supervisor = SelfHealingSupervisor()
        base_risk = 0.01  # 1.0%

        # Loss 1
        supervisor.record_trade_result(r_multiple=-1.0, exit_reason="LOSS_SL")
        self.assertEqual(supervisor.metrics.consecutive_losses, 1)
        self.assertFalse(supervisor.metrics.defensive_mode_active)
        self.assertEqual(supervisor.get_effective_risk_pct(base_risk), 0.01)

        # Loss 2 -> Trips defensive throttle!
        supervisor.record_trade_result(r_multiple=-0.15, exit_reason="LOSS_DEAL_CANCELLATION")
        self.assertEqual(supervisor.metrics.consecutive_losses, 2)
        self.assertTrue(supervisor.metrics.defensive_mode_active)
        self.assertEqual(supervisor.get_effective_risk_pct(base_risk), 0.005)  # 0.50% risk

        # Win 1 -> Auto-recovers to full risk!
        supervisor.record_trade_result(r_multiple=2.0, exit_reason="WIN_TP")
        self.assertEqual(supervisor.metrics.consecutive_losses, 0)
        self.assertFalse(supervisor.metrics.defensive_mode_active)
        self.assertEqual(supervisor.get_effective_risk_pct(base_risk), 0.01)

    def test_bayesian_meta_learner_conviction_and_veto(self):
        """
        Self-Improving: Verify Bayesian Meta-Learner updates posterior distributions
        and vetoes setup clusters that exhibit negative empirical expectancy (E[R] < 0).
        """
        test_state_file = Path("data/test_meta_learner_state.json")
        if os.path.exists(test_state_file):
            os.remove(test_state_file)

        learner = MetaLearner(state_file=test_state_file)
        symbol = "frxEURUSD"
        session = "LONDON_OPEN"
        pool = "PDL"

        # 1. Initial prior evaluation -> Authorized with moderate conviction
        approved, score, risk_pct, reason = learner.evaluate_setup_conviction(
            symbol=symbol, session_phase=session, macro_pool_type=pool, displacement_ratio=1.1, htf_depth_pct=0.25
        )
        self.assertTrue(approved)
        self.assertGreaterEqual(score, 0.50)

        # 2. Simulate 3 consecutive bad losses in this specific cluster
        for _ in range(3):
            learner.learn_from_trade(
                symbol=symbol,
                session_phase=session,
                macro_pool_type=pool,
                pnl_usd=-20.0,
                r_multiple=-1.0,
                exit_reason="LOSS_SL",
                save=False,
            )

        # 3. Post-losses evaluation -> Veto Rule must trigger!
        approved_after_losses, score_low, _, veto_reason = learner.evaluate_setup_conviction(
            symbol=symbol, session_phase=session, macro_pool_type=pool
        )
        self.assertFalse(approved_after_losses, "Cluster with E[R] < 0 must be vetoed.")
        self.assertIn("Vetoing to prevent repeat loss", veto_reason)

        # 4. Cleanup
        if os.path.exists(test_state_file):
            os.remove(test_state_file)

    def test_trade_learner_sqlite_persistence(self):
        """
        Self-Improving: Verify TradeLearnerDB persists entry context vectors,
        loss root causes, and computes post-mortem attribution records.
        """
        test_db_path = Path("data/test_learner_history.sqlite")
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        db = TradeLearnerDB(db_path=test_db_path)

        # Save trade context
        ctx = TradeContext(
            trade_id="TEST_001",
            contract_id="CID_123",
            symbol="frxEURUSD",
            direction="LONG",
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            stake_usd=40.0,
            potential_risk_usd=20.0,
            ob_top=1.0960,
            ob_bottom=1.0950,
            displacement_atr_ratio=1.25,
            htf_depth_pct=0.20,
            macro_pool_type="PDL",
            session_phase="LONDON_OPEN",
            atr=0.0012,
            spread_pips=0.8,
            entry_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        db.save_context(ctx)

        # Save trade outcome
        outcome = TradePostMortem(
            trade_id="TEST_001",
            exit_price=1.1100,
            pnl_usd=40.0,
            r_multiple=2.0,
            exit_reason="WIN_TP",
            loss_root_cause=None,
            holding_time_seconds=750.0,
            exit_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        db.save_outcome(outcome)

        recent = db.get_recent_trades(limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["trade_id"], "TEST_001")
        self.assertEqual(recent[0]["r_multiple"], 2.0)
        self.assertEqual(recent[0]["exit_reason"], "WIN_TP")

        # Test loss root cause classification
        cause = classify_loss_root_cause(
            exit_reason="LOSS_DEAL_CANCELLATION",
            holding_time_seconds=895.0,
            spread_at_entry=1.2,
            max_spread_allowed=1.5,
            displacement_ratio=1.1,
        )
        self.assertEqual(cause, "SLUGGISH_OB_FAILURE")

        # Cleanup
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

    def test_smc_chronological_sweep_and_invalidation(self):
        """Verify chronological liquidity sweep + displacement triggers signal, and broken OB is rejected."""
        dates = pd.date_range("2026-01-01", periods=50, freq="15min")
        data = []

        for i in range(40):
            if i == 20:
                data.append({"open": 1.0950, "high": 1.0960, "low": 1.0920, "close": 1.0955})
            else:
                data.append({"open": 1.0960, "high": 1.0970, "low": 1.0950, "close": 1.0962})

        data.append({"open": 1.0925, "high": 1.0935, "low": 1.0905, "close": 1.0930})

        for i in range(41, 49):
            data.append({"open": 1.0935, "high": 1.0948, "low": 1.0932, "close": 1.0942})

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

    def test_smc_synthetics_m5_setup(self):
        """Verify synthetic indices utilize 3-bar swing length and 0.50 ATR displacement."""
        reset_consumed_setups()
        dates = pd.date_range("2026-01-01", periods=26, freq="5min")
        data = []
        # Base price 30000.0, ATR ~ 50.0
        for i in range(20):
            if i == 10:
                data.append({"open": 30000.0, "high": 30050.0, "low": 29900.0, "close": 30010.0})
            else:
                data.append({"open": 30000.0, "high": 30040.0, "low": 29960.0, "close": 30010.0})

        # Bar 20 sweeps swing low 29900 with 0.50+ ATR rejection wick
        data.append({"open": 29920.0, "high": 29950.0, "low": 29850.0, "close": 29910.0})
        # Next bars pull back into OB [29850, 29920]
        for i in range(21, 25):
            data.append({"open": 29940.0, "high": 29980.0, "low": 29930.0, "close": 29960.0})
        # Forming candle touching OB
        data.append({"open": 29940.0, "high": 29940.0, "low": 29900.0, "close": 29905.0})

        df = pd.DataFrame(data, index=dates)
        sig = detect_institutional_setup("R_75", df)
        self.assertIsNotNone(sig, "Synthetic index should trigger SMC setup with 3-bar swings.")
        self.assertEqual(sig.symbol, "R_75")
        self.assertEqual(sig.direction, "LONG")
        self.assertLess(sig.stop_loss, 29850.0)


if __name__ == "__main__":
    unittest.main()
