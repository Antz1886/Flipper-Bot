"""
tests/test_dbot_engine.py — Quantitative & Behavioral Audit of DBot Execution Engine
Validates:
  1. Default & custom parameter initialization matching Inbuilt Deriv Bot.
  2. Martingale progression scaling on consecutive losses ($2 -> $4 -> $8 -> $16).
  3. Reset to base stake ($2.00) immediately following a win.
  4. Stake clamping to max_stake ceiling.
  5. Session Take Profit circuit breaker termination.
  6. Cumulative Session Stop Loss circuit breaker termination.
  7. Max consecutive loss streak circuit breaker (HALT vs RESET).
  8. Account balance sufficiency check prior to ordering.
  9. CSV trade journaling schema and data integrity.
"""

import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import os
import csv
import tempfile
import asyncio

from modules.dbot_engine import DBotEngine


class TestDBotEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DBotEngine(
            symbol="1HZ100V",
            contract_type="PUT",
            duration=5,
            duration_unit="t",
            base_stake=2.0,
            martingale_multiplier=2.0,
            max_loss_streak=4,
            action_on_max_loss="HALT",
            max_stake=32.0,
            take_profit_usd=20.0,
            stop_loss_usd=20.0,
            inter_trade_delay=0.0,
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


if __name__ == "__main__":
    unittest.main()
