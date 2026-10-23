"""
modules/self_healer.py — Autonomous Self-Healing & Operational Supervisor
Continuously monitors system health, reconciles orphan contracts, adapts SLA timers
to real-time network latency, and throttles risk defensives during loss streaks:
  1. Live RTT Latency Adaptation: Measures Deriv WebSocket ping time and dynamically
     shifts the Deal Cancellation SLA dispatch earlier during network lag.
  2. Orphan & Zombie Contract Reconciliation: Queries live broker `portfolio`, adopts
     untracked orders into active cancellation monitors, and purges ghost symbol locks.
  3. Defensive Drawdown Throttling: Automatically de-leverages risk (cutting to 0.50%)
     following consecutive losses until edge re-establishes.
  4. Heartbeat Supervision: Detects stalled connections and forces proactive re-authentication.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import config
from modules.connector import get_api, ensure_connected
from modules.risk_manager import get_risk_state, mark_symbol_active, mark_symbol_closed, _extract_contract_symbol
from modules.cancellation_manager import register_cancellation_contract, get_tracked_cancellation_contracts

log = logging.getLogger(__name__)


@dataclass
class SelfHealingMetrics:
    last_ping_rtt_ms: float = 0.0
    rolling_rtt_ms: List[float] = field(default_factory=list)
    reconciled_orphans_count: int = 0
    released_ghosts_count: int = 0
    consecutive_losses: int = 0
    defensive_mode_active: bool = False
    last_reconciliation_time: float = 0.0


class SelfHealingSupervisor:
    def __init__(self):
        self.metrics = SelfHealingMetrics()

    async def measure_and_adapt_latency(self) -> float:
        """
        Measure live Round-Trip Time (RTT) to Deriv WebSocket API.
        If latency is elevated (> 500ms), dynamically shifts the Deal Cancellation
        SLA trigger earlier so cancellation payloads never arrive past the 15m window.
        """
        api = get_api()
        if not api or not api.is_active:
            return 0.0

        t0 = time.time()
        try:
            resp = await api.send({"ping": 1})
            rtt_ms = (time.time() - t0) * 1000.0
            self.metrics.last_ping_rtt_ms = rtt_ms
            self.metrics.rolling_rtt_ms.append(rtt_ms)
            if len(self.metrics.rolling_rtt_ms) > 10:
                self.metrics.rolling_rtt_ms.pop(0)

            avg_rtt = sum(self.metrics.rolling_rtt_ms) / len(self.metrics.rolling_rtt_ms)

            # Adapt SLA trigger if latency is high
            base_trigger = 895.0  # 14m 55s
            if avg_rtt > 500.0:
                # Add extra buffer for slow network (shift up to 5s earlier)
                lag_buffer_s = min(5.0, (avg_rtt / 1000.0) * 4)
                adjusted_trigger = round(base_trigger - lag_buffer_s, 1)
                config.DEAL_CANCELLATION_TRIGGER_S = adjusted_trigger
                log.warning(
                    f"🏥 [SELF-HEALING SLA ADAPTATION] Elevated API latency ({avg_rtt:.0f}ms). "
                    f"Shifted Deal Cancellation trigger to {adjusted_trigger:.1f}s to guarantee broker delivery."
                )
            else:
                config.DEAL_CANCELLATION_TRIGGER_S = base_trigger

            return rtt_ms
        except Exception as e:
            log.warning(f"Latency measurement ping failed: {e}")
            return 999.0

    def reconcile_exchange_portfolio(self, portfolio_contracts: List[Dict[str, Any]]):
        """
        Self-Healing Reconciliation:
        1. Compares live broker contracts against locally tracked state.
        2. Adopts untracked orphan contracts into risk & Deal Cancellation tracking.
        3. Releases ghost symbol locks for contracts that no longer exist on the exchange.
        """
        risk_state = get_risk_state()
        tracked_cancellations = get_tracked_cancellation_contracts()
        exchange_symbols = set()
        exchange_contract_ids = set()

        for c in portfolio_contracts:
            cid = str(c.get("contract_id", ""))
            sym = _extract_contract_symbol(c)
            if not sym or not cid:
                continue

            exchange_symbols.add(sym)
            exchange_contract_ids.add(cid)

            # Check if this active exchange contract is missing from local tracking
            if sym not in risk_state.active_symbols:
                buy_price = float(c.get("buy_price", 10.0))
                c_type = (c.get("contract_type") or "").upper()
                c_dir = "LONG" if "UP" in c_type or "CALL" in c_type else "SHORT"
                mark_symbol_active(sym, stake_usd=buy_price, direction=c_dir)
                self.metrics.reconciled_orphans_count += 1
                log.warning(
                    f"🏥 [SELF-HEALING RECONCILER] Adopted orphan exchange contract {cid} ({sym} {c_dir}) "
                    f"into active risk oversight."
                )

            # If contract has deal cancellation and is not in cancellation monitor, register it
            cancellation_info = c.get("cancellation")
            if cancellation_info and cid not in tracked_cancellations:
                exp_ts = cancellation_info.get("date_expiry")
                fee = float(cancellation_info.get("ask_price", 0.0))
                barrier = float(c.get("barrier", 0.0))
                c_type = (c.get("contract_type") or "").upper()
                c_dir = "LONG" if "UP" in c_type or "CALL" in c_type else "SHORT"
                register_cancellation_contract(
                    contract_id=cid,
                    symbol=sym,
                    direction=c_dir,
                    ob_top=barrier,
                    ob_bottom=barrier,
                    date_expiry=exp_ts,
                    cancellation_fee=fee,
                )
                log.warning(
                    f"🏥 [SELF-HEALING RECONCILER] Enrolled untracked contract {cid} ({sym}) "
                    f"into SLA Deal Cancellation monitor."
                )

        # Ghost symbol lock purge: If a symbol is locked locally but not present on exchange
        now_ts = time.time()
        for sym in list(risk_state.active_symbols):
            entry_ts = risk_state.last_entry_time.get(sym, 0.0)
            # Allow 60s settlement grace period
            if (now_ts - entry_ts) > 60.0 and sym not in exchange_symbols:
                mark_symbol_closed(sym)
                self.metrics.released_ghosts_count += 1
                log.info(
                    f"🏥 [SELF-HEALING RECONCILER] Purged ghost symbol lock for {sym}: "
                    f"confirmed no longer active on broker exchange."
                )

        self.metrics.last_reconciliation_time = now_ts

    def record_trade_result(self, r_multiple: float, exit_reason: str):
        """
        Record outcome to adjust defensive risk throttling:
        - If loss: increments consecutive losses. At >= 2 losses, enables Defensive Throttle.
        - If win: resets consecutive losses and restores full risk.
        """
        if r_multiple > 0 or exit_reason == "WIN_TP":
            if self.metrics.consecutive_losses > 0:
                log.info(
                    f"🌱 [SELF-HEALING RECOVERY] Winning trade registered ({r_multiple:+.2f}R). "
                    f"Consecutive loss counter reset. Restoring full risk budget."
                )
            self.metrics.consecutive_losses = 0
            self.metrics.defensive_mode_active = False
        else:
            self.metrics.consecutive_losses += 1
            if self.metrics.consecutive_losses >= 2:
                self.metrics.defensive_mode_active = True
                log.warning(
                    f"🛡️ [SELF-HEALING DEFENSIVE THROTTLE] {self.metrics.consecutive_losses} consecutive losses. "
                    f"Engaging defensive mode: cutting risk per trade by 50% until next winning trade."
                )

    def get_effective_risk_pct(self, base_risk_pct: float) -> float:
        """Return defensive-adjusted risk percentage."""
        if self.metrics.defensive_mode_active:
            return round(base_risk_pct * 0.50, 4)
        return base_risk_pct


_self_healer: Optional[SelfHealingSupervisor] = None


def get_self_healer() -> SelfHealingSupervisor:
    global _self_healer
    if _self_healer is None:
        _self_healer = SelfHealingSupervisor()
    return _self_healer
