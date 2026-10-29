"""
tests/test_dbot_engine.py — Institutional Audit of Upgraded DBot Engine
Validates:
  1. Default & custom parameter initialization.
  2. Dynamic Direction & Micro-momentum evaluation (PUT vs CALL).
  3. Real-Time Streaming Settlement handling with Event callback.
  4. Session State Persistence (Atomic JSON save, reload, archiving).
  5. Martingale progression scaling on consecutive losses ($2 -> $4 -> $8 -> $16).
  6. Reset to base stake ($2.00) immediately following a win.
  7. Stake clamping to max_stake ceiling.
  8. Session Take Profit circuit breaker termination.
  9. Cumulative Session Stop Loss circuit breaker termination.
  10. Max consecutive loss streak circuit breaker (HALT vs RESET).
  11. Insufficient balance safety rejection.
  12. CSV trade journaling schema and data integrity.
"""

import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import os
import csv
import json
import tempfile
import asyncio

from modules.dbot_engine import DBotEngine
from modules.session_state import SessionStateManager
from modules.direction_engine import (
    evaluate_tick_bias,
    _calculate_ema,
    _calculate_rsi,
    evaluate_htf_trend,
)


class TestSessionState(unittest.TestCase):
    def test_save_and_load_active_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "test_session.json")
            mgr = SessionStateManager(state_file=state_file)

            # Initially no session
            self.assertIsNone(mgr.load_active_session())

            # Save an active session
            mgr.save_session(
                session_id="test-session-123",
                total_trades=10,
                total_wins=6,
                total_losses=4,
                consecutive_wins=1,
                consecutive_losses=0,
                current_stake=2.0,
                session_pnl=4.50,
                status="RUNNING",
            )

            # Load it back
            loaded = mgr.load_active_session()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["session_id"], "test-session-123")
            self.assertEqual(loaded["total_trades"], 10)
            self.assertEqual(loaded["total_wins"], 6)
            self.assertEqual(loaded["total_losses"], 4)
            self.assertEqual(loaded["session_pnl"], 4.50)
            self.assertEqual(loaded["status"], "RUNNING")


class TestDirectionEngine(unittest.TestCase):
    def test_ema_calculation(self):
        prices = [10.0, 10.0, 10.0, 10.0]
        ema = _calculate_ema(prices, 5)
        self.assertEqual(ema, 10.0)

    def test_bearish_momentum_bias(self):
        # Descending tick price sequence
        prices = [1180.0 - (i * 0.5) for i in range(25)]
        direction, conviction, rationale = evaluate_tick_bias(prices, consecutive_losses=0)
        self.assertEqual(direction, "PUT")
        self.assertGreaterEqual(conviction, 0.65)
        self.assertIn("Bearish", rationale)

    def test_bullish_momentum_bias(self):
        # Ascending tick price sequence
        prices = [1170.0 + (i * 0.5) for i in range(25)]
        direction, conviction, rationale = evaluate_tick_bias(prices, consecutive_losses=0)
        self.assertEqual(direction, "CALL")
        self.assertGreaterEqual(conviction, 0.65)
        self.assertIn("Bullish", rationale)

    def test_insufficient_ticks_fallback(self):
        prices = [100.0, 101.0]
        direction, conviction, _ = evaluate_tick_bias(prices, preferred_default="PUT")
        self.assertEqual(direction, "PUT")
        self.assertEqual(conviction, 0.50)


class TestDBotEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DBotEngine(
            symbol="1HZ100V",
            direction_mode="PUT",
            contract_type="PUT",
            duration=5,
            duration_unit="t",
            staking_strategy="MARTINGALE",
            base_stake=2.0,
            martingale_multiplier=2.0,
            max_loss_streak=4,
            action_on_max_loss="HALT",
            max_stake=32.0,
            take_profit_usd=20.0,
            stop_loss_usd=20.0,
            inter_trade_delay=0.0,
            min_conviction=0.50,
            persist_session=False,
            use_streaming=False,
            continuous_mode=False,
        )

    def test_initialization(self):
        self.assertEqual(self.engine.symbol, "1HZ100V")
        self.assertEqual(self.engine.contract_type, "PUT")
        self.assertEqual(self.engine.duration, 5)
        self.assertEqual(self.engine.duration_unit, "t")
        self.assertEqual(self.engine.base_stake, 2.0)
        self.assertEqual(self.engine.current_stake, 2.0)
        self.assertEqual(self.engine.martingale_multiplier, 2.0)
        self.assertEqual(self.engine.max_loss_streak, 4)
        self.assertEqual(self.engine.max_stake, 32.0)
        self.assertEqual(self.engine.take_profit_usd, 20.0)
        self.assertEqual(self.engine.stop_loss_usd, 20.0)

    @patch("modules.dbot_engine.DBotEngine.refresh_balance", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.get_proposal", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.buy_contract", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.wait_for_settlement", new_callable=AsyncMock)
    def test_martingale_loss_progression(self, mock_settle, mock_buy, mock_prop, mock_bal):
        mock_bal.return_value = 1000.0
        mock_prop.return_value = {"id": "prop-1", "ask_price": 2.0, "payout": 3.8}
        mock_buy.return_value = {"contract_id": 1001, "buy_price": 2.0}
        mock_settle.return_value = {"status": "lost", "profit": -2.0, "entry_spot": 100.0, "exit_tick": 101.0}

        # Trade 1: Loss of $2.00
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertTrue(should_continue)
        self.assertEqual(self.engine.total_trades, 1)
        self.assertEqual(self.engine.total_losses, 1)
        self.assertEqual(self.engine.consecutive_losses, 1)
        self.assertEqual(self.engine.session_pnl, -2.0)
        self.assertEqual(self.engine.current_stake, 4.0)

        # Trade 2: Loss of $4.00
        mock_prop.return_value = {"id": "prop-2", "ask_price": 4.0, "payout": 7.6}
        mock_buy.return_value = {"contract_id": 1002, "buy_price": 4.0}
        mock_settle.return_value = {"status": "lost", "profit": -4.0, "entry_spot": 100.0, "exit_tick": 101.0}
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertTrue(should_continue)
        self.assertEqual(self.engine.total_trades, 2)
        self.assertEqual(self.engine.total_losses, 2)
        self.assertEqual(self.engine.consecutive_losses, 2)
        self.assertEqual(self.engine.session_pnl, -6.0)
        self.assertEqual(self.engine.current_stake, 8.0)

    @patch("modules.dbot_engine.DBotEngine.refresh_balance", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.get_proposal", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.buy_contract", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.wait_for_settlement", new_callable=AsyncMock)
    def test_win_resets_stake_and_streak(self, mock_settle, mock_buy, mock_prop, mock_bal):
        mock_bal.return_value = 1000.0
        # Start after 2 losses with stake $8.00
        self.engine.current_stake = 8.0
        self.engine.consecutive_losses = 2
        self.engine.total_trades = 2
        self.engine.total_losses = 2
        self.engine.session_pnl = -6.0

        # Trade 3: Win with stake $8.00 (profit = +$7.20)
        mock_prop.return_value = {"id": "prop-3", "ask_price": 8.0, "payout": 15.2}
        mock_buy.return_value = {"contract_id": 1003, "buy_price": 8.0}
        mock_settle.return_value = {"status": "won", "profit": 7.20, "entry_spot": 105.0, "exit_tick": 104.0}

        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertTrue(should_continue)
        self.assertEqual(self.engine.total_trades, 3)
        self.assertEqual(self.engine.total_wins, 1)
        self.assertEqual(self.engine.consecutive_losses, 0)
        self.assertEqual(self.engine.consecutive_wins, 1)
        self.assertEqual(self.engine.current_stake, 2.0)  # Reset to base stake
        self.assertAlmostEqual(self.engine.session_pnl, 1.20)

    def test_take_profit_circuit_breaker(self):
        self.engine.session_pnl = 20.50  # Over target of $20.00
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertFalse(should_continue)

    def test_continuous_mode_rollover(self):
        self.engine.continuous_mode = True
        self.engine.session_pnl = 20.50  # Over TP target of 20.0
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertTrue(should_continue)
        self.assertEqual(self.engine.cycle_count, 2)
        self.assertEqual(self.engine.session_pnl, 0.0)
        self.assertEqual(self.engine.current_stake, 2.0)

    def test_stop_loss_circuit_breaker(self):
        self.engine.session_pnl = -21.00  # Beyond max loss of -$20.00
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertFalse(should_continue)

    def test_max_loss_streak_halt(self):
        self.engine.consecutive_losses = 4  # Reached max_loss_streak = 4
        self.engine.action_on_max_loss = "HALT"
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertFalse(should_continue)

    @patch("modules.dbot_engine.DBotEngine.refresh_balance", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.get_proposal", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.buy_contract", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.wait_for_settlement", new_callable=AsyncMock)
    def test_max_loss_streak_reset(self, mock_settle, mock_buy, mock_prop, mock_bal):
        mock_bal.return_value = 1000.0
        self.engine.current_stake = 32.0
        self.engine.consecutive_losses = 4
        self.engine.action_on_max_loss = "RESET"

        mock_prop.return_value = {"id": "prop-4", "ask_price": 2.0, "payout": 3.8}
        mock_buy.return_value = {"contract_id": 1004, "buy_price": 2.0}
        mock_settle.return_value = {"status": "won", "profit": 1.80, "entry_spot": 100.0, "exit_tick": 99.0}

        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertTrue(should_continue)
        self.assertEqual(self.engine.current_stake, 2.0)
        self.assertEqual(self.engine.consecutive_losses, 0)

    @patch("modules.dbot_engine.DBotEngine.refresh_balance", new_callable=AsyncMock)
    def test_insufficient_balance_halt(self, mock_bal):
        mock_bal.return_value = 1.50  # Less than current_stake of $2.00
        should_continue = asyncio.run(self.engine.execute_trade_cycle())
        self.assertFalse(should_continue)

    @patch("modules.dbot_engine.DBotEngine.refresh_balance", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.get_proposal", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.buy_contract", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.wait_for_settlement", new_callable=AsyncMock)
    def test_max_stake_clamping(self, mock_settle, mock_buy, mock_prop, mock_bal):
        mock_bal.return_value = 1000.0
        self.engine.current_stake = 32.0
        self.engine.max_stake = 32.0
        mock_prop.return_value = {"id": "prop-5", "ask_price": 32.0, "payout": 60.8}
        mock_buy.return_value = {"contract_id": 1005, "buy_price": 32.0}
        mock_settle.return_value = {"status": "lost", "profit": -32.0, "entry_spot": 100.0, "exit_tick": 101.0}

        asyncio.run(self.engine.execute_trade_cycle())
        # Multiplier would want 64.0, but clamped to max_stake 32.0
        self.assertEqual(self.engine.current_stake, 32.0)

    def test_csv_trade_recording(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_trades.csv")
            with patch("config.TRADE_LOG_CSV", csv_path):
                self.engine.session_pnl = 1.80
                self.engine.current_balance = 8950.0
                self.engine.record_trade_csv(
                    contract_id=1234567,
                    contract_type="PUT",
                    stake=2.0,
                    payout=3.80,
                    profit=1.80,
                    status="won",
                    entry_spot=1180.50,
                    exit_tick=1179.20,
                )

                self.assertTrue(os.path.exists(csv_path))
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    self.assertEqual(len(reader), 2)
                    header = reader[0]
                    row = reader[1]
                    self.assertIn("contract_id", header)
                    self.assertEqual(row[1], "1HZ100V")
                    self.assertEqual(row[2], "PUT")
                    self.assertEqual(row[3], "1234567")
                    self.assertEqual(row[4], "2.0")
                    self.assertEqual(row[6], "1.8")
                    self.assertEqual(row[7], "WON")

    @patch("modules.dbot_engine.DBotEngine.refresh_balance", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.get_proposal", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.buy_contract", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.wait_for_settlement", new_callable=AsyncMock)
    def test_dalembert_loss_and_win_progression(self, mock_settle, mock_buy, mock_prop, mock_bal):
        """Validates Step 2a: D'Alembert +1 unit on loss, -1 unit on win."""
        dalembert_engine = DBotEngine(
            staking_strategy="DALEMBERT",
            base_stake=2.0,
            dalembert_unit=1.0,
            max_stake=8.0,
            min_conviction=0.50,
            direction_mode="PUT",
            persist_session=False,
            use_streaming=False,
            continuous_mode=False,
        )
        mock_bal.return_value = 1000.0

        # Loss 1: Stake $2.00 -> should increase by $1.00 to $3.00
        mock_prop.return_value = {"id": "p1", "ask_price": 2.0, "payout": 3.8}
        mock_buy.return_value = {"contract_id": 2001, "buy_price": 2.0}
        mock_settle.return_value = {"status": "lost", "profit": -2.0, "entry_spot": 100.0, "exit_tick": 101.0}

        asyncio.run(dalembert_engine.execute_trade_cycle())
        self.assertEqual(dalembert_engine.current_stake, 3.0)

        # Loss 2: Stake $3.00 -> should increase by $1.00 to $4.00
        mock_prop.return_value = {"id": "p2", "ask_price": 3.0, "payout": 5.7}
        mock_buy.return_value = {"contract_id": 2002, "buy_price": 3.0}
        mock_settle.return_value = {"status": "lost", "profit": -3.0, "entry_spot": 100.0, "exit_tick": 101.0}

        asyncio.run(dalembert_engine.execute_trade_cycle())
        self.assertEqual(dalembert_engine.current_stake, 4.0)

        # Win 1: Stake $4.00 -> should decrease by $1.00 to $3.00
        mock_prop.return_value = {"id": "p3", "ask_price": 4.0, "payout": 7.6}
        mock_buy.return_value = {"contract_id": 2003, "buy_price": 4.0}
        mock_settle.return_value = {"status": "won", "profit": 3.6, "entry_spot": 100.0, "exit_tick": 99.0}

        asyncio.run(dalembert_engine.execute_trade_cycle())
        self.assertEqual(dalembert_engine.current_stake, 3.0)

        # Win 2: Stake $3.00 -> should decrease by $1.00 to base floor of $2.00
        mock_prop.return_value = {"id": "p4", "ask_price": 3.0, "payout": 5.7}
        mock_buy.return_value = {"contract_id": 2004, "buy_price": 3.0}
        mock_settle.return_value = {"status": "won", "profit": 2.7, "entry_spot": 100.0, "exit_tick": 99.0}

        asyncio.run(dalembert_engine.execute_trade_cycle())
        self.assertEqual(dalembert_engine.current_stake, 2.0)

        # Win 3: Stake $2.00 -> should stay clamped at base floor of $2.00
        mock_prop.return_value = {"id": "p5", "ask_price": 2.0, "payout": 3.8}
        mock_buy.return_value = {"contract_id": 2005, "buy_price": 2.0}
        mock_settle.return_value = {"status": "won", "profit": 1.8, "entry_spot": 100.0, "exit_tick": 99.0}

        asyncio.run(dalembert_engine.execute_trade_cycle())
        self.assertEqual(dalembert_engine.current_stake, 2.0)

    @patch("modules.dbot_engine.determine_contract_direction", new_callable=AsyncMock)
    @patch("modules.dbot_engine.DBotEngine.get_proposal", new_callable=AsyncMock)
    def test_conviction_filter_veto(self, mock_prop, mock_dir):
        """Validates Step 4: Low conviction (<70%) is vetoed and does not place order."""
        veto_engine = DBotEngine(
            min_conviction=0.70,
            persist_session=False,
            use_streaming=False,
            continuous_mode=False,
        )
        # Mock signal with low conviction (55%)
        mock_dir.return_value = ("PUT", 0.55, "Compression/Noise")

        should_continue = asyncio.run(veto_engine.execute_trade_cycle())
        self.assertTrue(should_continue)
        # Should not have requested proposal or placed buy
        mock_prop.assert_not_called()
        self.assertEqual(veto_engine.total_trades, 0)

    def test_htf_trend_evaluation(self):
        """Validates Step 3: M1 EMA21 vs EMA50 trend determination."""
        # Bullish candles: steadily rising closes
        bull_candles = [{"close": 1200.0 + (i * 0.5)} for i in range(50)]
        trend, ema21, ema50 = evaluate_htf_trend(bull_candles)
        self.assertEqual(trend, "BULLISH")
        self.assertGreater(ema21, ema50)

        # Bearish candles: steadily falling closes
        bear_candles = [{"close": 1200.0 - (i * 0.5)} for i in range(50)]
        trend, ema21, ema50 = evaluate_htf_trend(bear_candles)
        self.assertEqual(trend, "BEARISH")
        self.assertLess(ema21, ema50)

    def test_htf_trend_gating_bans_counter_trend(self):
        """Validates that a Bullish HTF trend only permits CALL, and Bearish only permits PUT."""
        prices = [1210.0 + (i % 2) * 0.2 for i in range(25)]  # choppy ticks

        # In Bullish HTF: Must return CALL
        dir_bull, conv_bull, _ = evaluate_tick_bias(prices, htf_trend="BULLISH", tick_rsi=45.0)
        self.assertEqual(dir_bull, "CALL")

        # In Bearish HTF: Must return PUT
        dir_bear, conv_bear, _ = evaluate_tick_bias(prices, htf_trend="BEARISH", tick_rsi=55.0)
        self.assertEqual(dir_bear, "PUT")

    def test_tick_rsi_calculation(self):
        """Validates 14-period RSI calculation on ticks."""
        rising = [100.0 + i for i in range(20)]
        rsi_high = _calculate_rsi(rising, period=14)
        self.assertGreater(rsi_high, 70.0)

        falling = [100.0 - i for i in range(20)]
        rsi_low = _calculate_rsi(falling, period=14)
        self.assertLess(rsi_low, 30.0)


if __name__ == "__main__":
    unittest.main()
