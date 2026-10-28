"""
modules/dbot_engine.py — Inbuilt Deriv Bot (DBot) Execution Engine
Replicates the native Deriv Bot (DBot) strategy with institutional safety controls:
  - Asset: Volatility 100 (1s) Index (1HZ100V) or user-configured synthetic
  - Contract Type: 5-tick PUT (Fall) / CALL (Rise)
  - Staking: Martingale progression ($2.00 -> $4.00 -> $8.00 -> $16.00 ...)
  - Reset to Base Stake on every Win
  - Circuit Breakers: Session Take Profit, Cumulative Stop Loss, Max Loss Streak Ceiling
"""

import asyncio
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone

import config
from modules import connector

log = logging.getLogger("dbot_engine")


class DBotEngine:
    def __init__(
        self,
        symbol: str = getattr(config, "DBOT_SYMBOL", "1HZ100V"),
        contract_type: str = getattr(config, "DBOT_CONTRACT_TYPE", "PUT"),
        duration: int = getattr(config, "DBOT_DURATION", 5),
        duration_unit: str = getattr(config, "DBOT_DURATION_UNIT", "t"),
        base_stake: float = getattr(config, "DBOT_BASE_STAKE", 2.0),
        martingale_multiplier: float = getattr(config, "DBOT_MARTINGALE_MULTIPLIER", 2.0),
        max_loss_streak: int = getattr(config, "DBOT_MAX_LOSS_STREAK", 5),
        action_on_max_loss: str = getattr(config, "DBOT_ACTION_ON_MAX_LOSS", "HALT"),
        max_stake: float = getattr(config, "DBOT_MAX_STAKE", 64.0),
        take_profit_usd: float = getattr(config, "DBOT_TAKE_PROFIT_USD", 50.0),
        stop_loss_usd: float = getattr(config, "DBOT_STOP_LOSS_USD", 50.0),
        inter_trade_delay: float = getattr(config, "DBOT_INTER_TRADE_DELAY_S", 1.0),
    ):
        self.symbol = symbol
        self.contract_type = contract_type.upper()
        self.duration = duration
        self.duration_unit = duration_unit
        self.base_stake = float(base_stake)
        self.martingale_multiplier = float(martingale_multiplier)
        self.max_loss_streak = int(max_loss_streak)
        self.action_on_max_loss = action_on_max_loss.upper()
        self.max_stake = float(max_stake)
        self.take_profit_usd = float(take_profit_usd)
        self.stop_loss_usd = float(stop_loss_usd)
        self.inter_trade_delay = float(inter_trade_delay)

        self.current_stake: float = self.base_stake
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.session_pnl: float = 0.0

        self.start_balance: float | None = None
        self.current_balance: float | None = None
        self.is_running: bool = False

    def log_header(self):
        """Print institutional launch banner for Deriv Bot Mode."""
        log.info("=" * 80)
        log.info("   DERIV INBUILT BOT (DBOT) REPLICA ENGINE — MARTINGALE QUANT")
        log.info("=" * 80)
        log.info(f"   Asset Symbol        : {self.symbol}")
        log.info(f"   Contract Type       : {self.contract_type} ({'Fall' if self.contract_type == 'PUT' else 'Rise'})")
        log.info(f"   Duration            : {self.duration} {self.duration_unit} (ticks)")
        log.info(f"   Base Stake          : ${self.base_stake:.2f} USD")
        log.info(f"   Martingale Ratio    : {self.martingale_multiplier:.1f}x (on loss)")
        log.info(f"   Max Stake Cap       : ${self.max_stake:.2f} USD")
        log.info(f"   Max Loss Streak     : {self.max_loss_streak} (Action: {self.action_on_max_loss})")
        log.info(f"   Session Take Profit : +${self.take_profit_usd:.2f} USD")
        log.info(f"   Session Stop Loss   : -${self.stop_loss_usd:.2f} USD")
        log.info(f"   Inter-Trade Delay   : {self.inter_trade_delay}s")
        log.info(f"   Account Mode        : {'DEMO' if getattr(config, 'DEMO_MODE', True) else 'REAL'}")
        log.info("=" * 80)

    async def _ensure_connection(self):
        """Verify API is connected and authorized; reconnect if necessary."""
        api = connector.get_api()
        if not api or not api.is_active or not api.authorized:
            log.warning("Deriv API session disconnected or unauthorized. Re-establishing connection...")
            success = await connector.reconnect()
            if not success:
                raise ConnectionError("Failed to reconnect to Deriv WebSocket API.")
            api = connector.get_api()
        return api

    async def refresh_balance(self) -> float:
        """Fetch current balance from Deriv."""
        api = await self._ensure_connection()
        try:
            resp = await api.send({"balance": 1})
            bal = float(resp.get("balance", {}).get("balance", 0.0))
            self.current_balance = bal
            if self.start_balance is None:
                self.start_balance = bal
            return bal
        except Exception as e:
            log.warning(f"Failed to fetch balance: {e}")
            return self.current_balance or 0.0

    async def get_proposal(self, stake: float) -> dict | None:
        """Request price proposal for binary options contract."""
        api = await self._ensure_connection()
        symbol_key = "underlying_symbol" if getattr(api, "using_otp_ws", False) else "symbol"
        payload = {
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": self.contract_type,
            "currency": api.currency or "USD",
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            symbol_key: self.symbol,
        }

        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                resp = await api.send(payload)
                prop = resp.get("proposal")
                if prop and "id" in prop:
                    return prop
                error = resp.get("error")
                if error:
                    log.error(f"Proposal error: {error.get('message', error)}")
                    return None
            except Exception as e:
                log.warning(f"Proposal request attempt {attempt} failed: {e}")
                await asyncio.sleep(config.RETRY_DELAY_S)
        return None

    async def buy_contract(self, proposal_id: str, ask_price: float) -> dict | None:
        """Purchase contract via proposal id."""
        api = await self._ensure_connection()
        payload = {
            "buy": proposal_id,
            "price": ask_price,
        }
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                resp = await api.send(payload)
                buy_data = resp.get("buy")
                if buy_data and "contract_id" in buy_data:
                    return buy_data
                error = resp.get("error")
                if error:
                    log.error(f"Buy execution error: {error.get('message', error)}")
                    return None
            except Exception as e:
                log.warning(f"Buy request attempt {attempt} failed: {e}")
                await asyncio.sleep(config.RETRY_DELAY_S)
        return None

    async def wait_for_settlement(self, contract_id: int, timeout_s: float = 30.0) -> dict | None:
        """Poll proposal_open_contract until contract settles (is_sold == 1)."""
        api = await self._ensure_connection()
        start_t = time.time()
        while time.time() - start_t < timeout_s:
            try:
                resp = await api.send({"proposal_open_contract": 1, "contract_id": contract_id})
                poc = resp.get("proposal_open_contract")
                if poc:
                    is_sold = bool(poc.get("is_sold"))
                    if is_sold:
                        return poc
            except Exception as e:
                log.warning(f"Error querying contract settlement: {e}")
            await asyncio.sleep(0.8)

        log.warning(f"Contract {contract_id} settlement timed out after {timeout_s}s.")
        return None

    def record_trade_csv(
        self,
        contract_id: int,
        stake: float,
        payout: float,
        profit: float,
        status: str,
        entry_spot: str | float,
        exit_tick: str | float,
    ):
        """Append trade details to trades.csv."""
        try:
            os.makedirs("logs", exist_ok=True)
            file_exists = os.path.exists(config.TRADE_LOG_CSV)
            with open(config.TRADE_LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "symbol", "contract_type", "contract_id",
                        "stake", "payout", "profit", "status", "entry_spot",
                        "exit_tick", "session_pnl", "balance",
                    ])
                writer.writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    self.symbol,
                    self.contract_type,
                    contract_id,
                    round(stake, 2),
                    round(payout, 2),
                    round(profit, 2),
                    status.upper(),
                    entry_spot,
                    exit_tick,
                    round(self.session_pnl, 2),
                    round(self.current_balance or 0.0, 2),
                ])
        except Exception as e:
            log.error(f"Failed to record trade in CSV: {e}")

    async def execute_trade_cycle(self) -> bool:
        """
        Execute a single DBot trade cycle:
          1. Check circuit breakers (TP, SL, Max Loss Streak).
          2. Sizing clamp to max_stake.
          3. Get proposal.
          4. Execute buy.
          5. Await settlement.
          6. Apply Martingale progression / reset.
        Returns:
          True if trading should continue, False if halted by circuit breaker.
        """
        # Circuit Breaker 1: Take Profit Target Hit
        if self.session_pnl >= self.take_profit_usd:
            log.info(
                f"🎯 [TAKE PROFIT REACHED] Session target of +${self.take_profit_usd:.2f} achieved! "
                f"Current Session P&L: +${self.session_pnl:.2f}. Halting bot."
            )
            return False

        # Circuit Breaker 2: Stop Loss Limit Hit
        if self.session_pnl <= -self.stop_loss_usd:
            log.error(
                f"🛑 [STOP LOSS TRIGGERED] Maximum session drawdown of -${self.stop_loss_usd:.2f} reached! "
                f"Current Session P&L: ${self.session_pnl:.2f}. Halting bot to preserve capital."
            )
            return False

        # Circuit Breaker 3: Max Loss Streak Reached
        if self.consecutive_losses >= self.max_loss_streak:
            if self.action_on_max_loss == "HALT":
                log.error(
                    f"⚠️ [CIRCUIT BREAKER] Max consecutive loss streak of {self.max_loss_streak} reached! "
                    f"Halting bot to prevent Martingale drawdown bleed."
                )
                return False
            else:
                log.warning(
                    f"⚠️ Max consecutive loss streak of {self.max_loss_streak} reached. "
                    f"Resetting stake to base stake (${self.base_stake:.2f})."
                )
                self.current_stake = self.base_stake
                self.consecutive_losses = 0

        # Enforce max stake ceiling
        stake_to_use = min(round(self.current_stake, 2), self.max_stake)

        # Check account balance sufficiency
        current_bal = await self.refresh_balance()
        if current_bal < stake_to_use:
            log.error(
                f"🛑 Insufficient account balance (${current_bal:.2f}) for next stake (${stake_to_use:.2f}). "
                f"Trading halted."
            )
            return False

        self.total_trades += 1
        trade_num = self.total_trades

        log.info(
            f"─── Trade #{trade_num} | Sizing: ${stake_to_use:.2f} on {self.symbol} "
            f"({self.duration}t {self.contract_type}) | Losses: {self.consecutive_losses} ───"
        )

        # 1. Obtain Proposal
        proposal = await self.get_proposal(stake_to_use)
        if not proposal:
            log.error("Unable to obtain valid Deriv price proposal. Waiting before retry...")
            await asyncio.sleep(3.0)
            return True

        prop_id = proposal["id"]
        ask_price = float(proposal["ask_price"])
        expected_payout = float(proposal.get("payout", 0.0))

        # 2. Buy Contract
        buy_data = await self.buy_contract(prop_id, ask_price)
        if not buy_data:
            log.error("Failed to execute buy order. Waiting before retry...")
            await asyncio.sleep(3.0)
            return True

        contract_id = int(buy_data["contract_id"])
        log.info(
            f"Contract #{contract_id} OPEN | Stake: ${ask_price:.2f} | "
            f"Expected Payout: ${expected_payout:.2f} | Awaiting {self.duration} ticks..."
        )

        # 3. Wait for Settlement
        poc = await self.wait_for_settlement(contract_id)
        if not poc:
            log.warning(f"Could not confirm settlement for contract #{contract_id}. Fetching updated balance...")
            await self.refresh_balance()
            return True

        # 4. Parse Outcome
        status = str(poc.get("status", "lost")).lower()
        profit = float(poc.get("profit", -ask_price))
        payout = float(poc.get("payout", 0.0))
        entry_spot = poc.get("entry_spot", "N/A")
        exit_tick = poc.get("exit_tick", "N/A")

        self.session_pnl += profit
        await self.refresh_balance()

        win_rate = (self.total_wins / self.total_trades) * 100 if self.total_trades > 0 else 0.0

        if status == "won" or profit > 0:
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.current_stake = self.base_stake

            log.info(
                f"✅ [WIN] #{trade_num} Contract #{contract_id} | Entry: {entry_spot} -> Exit: {exit_tick} | "
                f"Profit: +${profit:.2f} | Session PnL: ${self.session_pnl:+.2f} "
                f"(Wins: {self.total_wins}/{self.total_trades}, WR: {win_rate:.1f}%) | "
                f"Reset Stake to ${self.current_stake:.2f}"
            )
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0

            next_stake = round(stake_to_use * self.martingale_multiplier, 2)
            self.current_stake = min(next_stake, self.max_stake)

            log.info(
                f"❌ [LOSS] #{trade_num} Contract #{contract_id} | Entry: {entry_spot} -> Exit: {exit_tick} | "
                f"Loss: -${abs(profit):.2f} | Session PnL: ${self.session_pnl:+.2f} "
                f"(Losses: {self.total_losses}/{self.total_trades}, WR: {win_rate:.1f}%) | "
                f"Next Stake (x{self.martingale_multiplier:.1f}): ${self.current_stake:.2f} (Streak: {self.consecutive_losses})"
            )

        self.record_trade_csv(
            contract_id=contract_id,
            stake=ask_price,
            payout=payout,
            profit=profit,
            status=status,
            entry_spot=entry_spot,
            exit_tick=exit_tick,
        )

        return True

    async def run(self):
        """Main DBot trading loop."""
        self.is_running = True
        self.log_header()

        api = await self._ensure_connection()
        start_bal = await self.refresh_balance()
        log.info(f"Authorized on Account {api.loginid} | Starting Balance: ${start_bal:.2f} {api.currency}")

        try:
            while self.is_running:
                should_continue = await self.execute_trade_cycle()
                if not should_continue:
                    break
                if self.inter_trade_delay > 0:
                    await asyncio.sleep(self.inter_trade_delay)

        except asyncio.CancelledError:
            log.info("DBot trading loop cancelled.")
        except Exception as e:
            log.error(f"Unexpected error in DBot execution loop: {e}", exc_info=True)
        finally:
            self.is_running = False
            final_bal = await self.refresh_balance()
            win_rate = (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0.0
            log.info("=" * 80)
            log.info("   DBOT SESSION FINISHED")
            log.info("=" * 80)
            log.info(f"   Total Trades Taken  : {self.total_trades}")
            log.info(f"   Wins / Losses       : {self.total_wins} W / {self.total_losses} L (Win Rate: {win_rate:.1f}%)")
            log.info(f"   Net Session Profit  : ${self.session_pnl:+.2f} USD")
            log.info(f"   Final Balance       : ${final_bal:.2f} USD")
            log.info("=" * 80)

    def stop(self):
        """Halt trading loop."""
        self.is_running = False
