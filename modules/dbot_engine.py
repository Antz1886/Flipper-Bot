"""
modules/dbot_engine.py — Institutional Deriv Bot (DBot) Execution Engine
Quant Upgrade:
  - Dynamic Direction Edge: Real-time tick momentum & micro-trend evaluation (PUT vs CALL)
  - Real-Time WebSocket Streaming: Event-driven settlement tracking (zero polling lag)
  - Atomic Session Persistence: Resilient state across reboots (P&L, streaks, Martingale tier)
  - Institutional Circuit Breakers: Take profit, stop loss, max loss streak, balance pre-check
"""

import asyncio
import csv
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import config
from modules import connector
from modules.direction_engine import determine_contract_direction
from modules.session_state import SessionStateManager

log = logging.getLogger("dbot_engine")


class DBotEngine:
    def __init__(
        self,
        symbol: str = getattr(config, "DBOT_SYMBOL", "1HZ100V"),
        direction_mode: str = getattr(config, "DBOT_DIRECTION_MODE", "DYNAMIC"),
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
        persist_session: bool = getattr(config, "DBOT_PERSIST_SESSION", True),
        use_streaming: bool = getattr(config, "DBOT_USE_STREAMING", True),
    ):
        self.symbol = symbol
        self.direction_mode = direction_mode.upper()
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
        self.persist_session = persist_session
        self.use_streaming = use_streaming

        self.session_id: str = uuid.uuid4().hex[:12]
        self.current_stake: float = self.base_stake
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.total_trades: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.session_pnl: float = 0.0
        self.session_status: str = "RUNNING"

        self.start_balance: float | None = None
        self.current_balance: float | None = None
        self.is_running: bool = False

        self.state_mgr = SessionStateManager()
        if self.persist_session:
            self._restore_persisted_state()

    def _restore_persisted_state(self):
        """Restore active session metrics from disk if available."""
        saved = self.state_mgr.load_active_session()
        if saved:
            self.session_id = saved.get("session_id", self.session_id)
            self.total_trades = saved.get("total_trades", 0)
            self.total_wins = saved.get("total_wins", 0)
            self.total_losses = saved.get("total_losses", 0)
            self.consecutive_wins = saved.get("consecutive_wins", 0)
            self.consecutive_losses = saved.get("consecutive_losses", 0)
            self.current_stake = float(saved.get("current_stake", self.base_stake))
            self.session_pnl = float(saved.get("session_pnl", 0.0))
            self.session_status = saved.get("status", "RUNNING")

    def _save_current_state(self, status: str = "RUNNING"):
        """Save active state to disk."""
        self.session_status = status
        if self.persist_session:
            self.state_mgr.save_session(
                session_id=self.session_id,
                total_trades=self.total_trades,
                total_wins=self.total_wins,
                total_losses=self.total_losses,
                consecutive_wins=self.consecutive_wins,
                consecutive_losses=self.consecutive_losses,
                current_stake=self.current_stake,
                session_pnl=self.session_pnl,
                status=status,
            )

    def log_header(self):
        """Print institutional launch banner for Deriv Bot Mode."""
        log.info("=" * 80)
        log.info("   INSTITUTIONAL DERIV BOT (DBOT) ENGINE — QUANT EDITION")
        log.info("=" * 80)
        log.info(f"   Asset Symbol        : {self.symbol}")
        log.info(f"   Direction Mode      : {self.direction_mode} (Dynamic Tick Momentum)")
        log.info(f"   Default Contract    : {self.contract_type} ({'Fall' if self.contract_type == 'PUT' else 'Rise'})")
        log.info(f"   Duration            : {self.duration} {self.duration_unit} (ticks)")
        log.info(f"   Base Stake          : ${self.base_stake:.2f} USD")
        log.info(f"   Martingale Ratio    : {self.martingale_multiplier:.1f}x (on loss)")
        log.info(f"   Max Stake Cap       : ${self.max_stake:.2f} USD")
        log.info(f"   Max Loss Streak     : {self.max_loss_streak} (Action: {self.action_on_max_loss})")
        log.info(f"   Session Take Profit : +${self.take_profit_usd:.2f} USD")
        log.info(f"   Session Stop Loss   : -${self.stop_loss_usd:.2f} USD")
        log.info(f"   Streaming Execution : {'Real-Time WebSocket' if self.use_streaming else 'Polling (0.8s)'}")
        log.info(f"   Session Persistence : {'Active (data/dbot_session.json)' if self.persist_session else 'Disabled'}")
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

    async def get_proposal(self, stake: float, contract_type: str) -> dict | None:
        """Request price proposal for binary options contract."""
        api = await self._ensure_connection()
        symbol_key = "underlying_symbol" if getattr(api, "using_otp_ws", False) else "symbol"
        payload = {
            "proposal": 1,
            "amount": round(stake, 2),
            "basis": "stake",
            "contract_type": contract_type,
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

    async def wait_for_settlement(self, contract_id: int, timeout_s: float = 25.0) -> dict | None:
        """
        Wait for contract settlement.
        Uses Real-Time WebSocket Streaming subscription for zero-lag notification,
        with seamless fallback to polling if subscription stream drops.
        """
        api = await self._ensure_connection()

        if self.use_streaming:
            settled_data: dict = {}
            event = asyncio.Event()

            async def on_contract_update(poc: dict):
                try:
                    cid = int(poc.get("contract_id", 0))
                    if cid == contract_id:
                        if poc.get("is_sold") or poc.get("status") in ("won", "lost"):
                            settled_data.update(poc)
                            event.set()
                except Exception as ex:
                    log.debug(f"Error handling streaming contract update: {ex}")

            api.add_contract_handler(on_contract_update)
            sub_id = None
            try:
                # Subscribe to contract stream
                sub_resp = await api.send({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                    "subscribe": 1,
                })
                sub_id = sub_resp.get("subscription", {}).get("id")

                # Check if it was already settled in initial snapshot
                init_poc = sub_resp.get("proposal_open_contract", {})
                if init_poc.get("is_sold") or init_poc.get("status") in ("won", "lost"):
                    return init_poc

                await asyncio.wait_for(event.wait(), timeout=timeout_s)
                if settled_data:
                    return settled_data
            except asyncio.TimeoutError:
                log.warning(f"Streaming settlement timed out for #{contract_id}. Falling back to polling query.")
            except Exception as e:
                log.warning(f"Streaming subscription encountered error: {e}. Falling back to polling.")
            finally:
                api.remove_contract_handler(on_contract_update)
                if sub_id:
                    try:
                        await api.send({"forget": sub_id})
                    except Exception:
                        pass

        # Fallback Polling Loop
        start_t = time.time()
        while time.time() - start_t < timeout_s:
            try:
                resp = await api.send({"proposal_open_contract": 1, "contract_id": contract_id})
                poc = resp.get("proposal_open_contract")
                if poc and (poc.get("is_sold") or poc.get("status") in ("won", "lost")):
                    return poc
            except Exception as e:
                log.warning(f"Polling settlement error: {e}")
            await asyncio.sleep(0.8)

        log.warning(f"Contract {contract_id} settlement timed out after {timeout_s}s.")
        return None

    def record_trade_csv(
        self,
        contract_id: int,
        contract_type: str,
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
                    contract_type,
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
        Execute an institutional DBot trade cycle:
          1. Circuit breaker checks (TP, SL, Max Loss Streak).
          2. Dynamic direction & micro-momentum evaluation (PUT vs CALL).
          3. Sizing clamp to max_stake ceiling.
          4. Get proposal & buy contract.
          5. Real-time streaming settlement.
          6. Martingale scaling or win reset.
          7. Atomic session persistence.
        """
        # Circuit Breaker 1: Take Profit Target Hit
        if self.session_pnl >= self.take_profit_usd:
            log.info(
                f"🎯 [TAKE PROFIT REACHED] Session target of +${self.take_profit_usd:.2f} achieved! "
                f"Current Session P&L: +${self.session_pnl:.2f}. Halting bot."
            )
            self._save_current_state(status="TP_HIT")
            return False

        # Circuit Breaker 2: Stop Loss Limit Hit
        if self.session_pnl <= -self.stop_loss_usd:
            log.error(
                f"🛑 [STOP LOSS TRIGGERED] Maximum session drawdown of -${self.stop_loss_usd:.2f} reached! "
                f"Current Session P&L: ${self.session_pnl:.2f}. Halting bot to preserve capital."
            )
            self._save_current_state(status="SL_HIT")
            return False

        # Circuit Breaker 3: Max Loss Streak Reached
        if self.consecutive_losses >= self.max_loss_streak:
            if self.action_on_max_loss == "HALT":
                log.error(
                    f"⚠️ [CIRCUIT BREAKER] Max consecutive loss streak of {self.max_loss_streak} reached! "
                    f"Halting bot to prevent Martingale drawdown bleed."
                )
                self._save_current_state(status="MAX_STREAK_HALT")
                return False
            else:
                log.warning(
                    f"⚠️ Max consecutive loss streak of {self.max_loss_streak} reached. "
                    f"Resetting stake to base stake (${self.base_stake:.2f})."
                )
                self.current_stake = self.base_stake
                self.consecutive_losses = 0
                self._save_current_state(status="RUNNING")

        # Dynamic Direction Determination
        chosen_dir, conviction, rationale = await determine_contract_direction(
            symbol=self.symbol,
            direction_mode=self.direction_mode,
            preferred_default=self.contract_type,
            consecutive_losses=self.consecutive_losses,
        )

        # Enforce max stake ceiling
        stake_to_use = min(round(self.current_stake, 2), self.max_stake)

        # Pre-execution balance sufficiency check
        current_bal = await self.refresh_balance()
        if current_bal < stake_to_use:
            log.error(
                f"🛑 Insufficient account balance (${current_bal:.2f}) for next stake (${stake_to_use:.2f}). "
                f"Trading halted."
            )
            self._save_current_state(status="INSUFFICIENT_FUNDS")
            return False

        self.total_trades += 1
        trade_num = self.total_trades

        log.info(
            f"─── Trade #{trade_num} | Sizing: ${stake_to_use:.2f} | "
            f"Direction: {chosen_dir} ({conviction*100:.0f}% conv) | Streak: {self.consecutive_losses}L ───"
        )
        log.info(f"    Edge Rationale: {rationale}")

        # 1. Obtain Proposal
        proposal = await self.get_proposal(stake_to_use, contract_type=chosen_dir)
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
            f"Contract #{contract_id} OPEN | {chosen_dir} | Stake: ${ask_price:.2f} | "
            f"Expected Payout: ${expected_payout:.2f} | Streaming settlement..."
        )

        # 3. Real-Time Streaming Settlement
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

        self._save_current_state(status="RUNNING")

        self.record_trade_csv(
            contract_id=contract_id,
            contract_type=chosen_dir,
            stake=ask_price,
            payout=payout,
            profit=profit,
            status=status,
            entry_spot=entry_spot,
            exit_tick=exit_tick,
        )

        return True

    async def run(self):
        """Main DBot trading loop with crash-resilient persistence."""
        self.is_running = True
        self.log_header()

        api = await self._ensure_connection()
        start_bal = await self.refresh_balance()
        log.info(f"Authorized on Account {api.loginid} | Starting Balance: ${start_bal:.2f} {api.currency}")

        if self.session_status in ("TP_HIT", "SL_HIT", "MAX_STREAK_HALT"):
            log.warning(
                f"⚠️ Today's session was already finalized with status: '{self.session_status}' "
                f"(PnL: ${self.session_pnl:+.2f}). Halting execution to prevent over-trading."
            )
            return

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
            log.info("   DBOT SESSION SUMMARY")
            log.info("=" * 80)
            log.info(f"   Total Trades Taken  : {self.total_trades}")
            log.info(f"   Wins / Losses       : {self.total_wins} W / {self.total_losses} L (Win Rate: {win_rate:.1f}%)")
            log.info(f"   Net Session Profit  : ${self.session_pnl:+.2f} USD")
            log.info(f"   Final Balance       : ${final_bal:.2f} USD")
            log.info(f"   Final Status        : {self.session_status}")
            log.info("=" * 80)

    def stop(self):
        """Halt trading loop."""
        self.is_running = False
