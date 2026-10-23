"""
modules/trade_learner.py — Post-Trade Attribution & Continual Learning Database
Captures rich structural entry contexts, tracks trade lifecycles, classifies loss root causes,
and stores immutable post-mortems in SQLite for online Bayesian learning.
"""

import os
import sqlite3
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

import config

log = logging.getLogger(__name__)

DB_PATH = Path("data/learner_history.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_contexts (
    trade_id TEXT PRIMARY KEY,
    contract_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    stake_usd REAL NOT NULL,
    potential_risk_usd REAL NOT NULL,
    ob_top REAL NOT NULL,
    ob_bottom REAL NOT NULL,
    displacement_atr_ratio REAL NOT NULL,
    htf_depth_pct REAL NOT NULL,
    macro_pool_type TEXT NOT NULL,
    session_phase TEXT NOT NULL,
    atr REAL NOT NULL,
    spread_pips REAL NOT NULL,
    entry_time_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_outcomes (
    trade_id TEXT PRIMARY KEY,
    exit_price REAL NOT NULL,
    pnl_usd REAL NOT NULL,
    r_multiple REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    loss_root_cause TEXT,
    holding_time_seconds REAL NOT NULL,
    exit_time_utc TEXT NOT NULL,
    FOREIGN KEY(trade_id) REFERENCES trade_contexts(trade_id)
);
"""


@dataclass
class TradeContext:
    trade_id: str
    contract_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    stake_usd: float
    potential_risk_usd: float
    ob_top: float
    ob_bottom: float
    displacement_atr_ratio: float
    htf_depth_pct: float
    macro_pool_type: str
    session_phase: str
    atr: float
    spread_pips: float
    entry_time_utc: str


@dataclass
class TradePostMortem:
    trade_id: str
    exit_price: float
    pnl_usd: float
    r_multiple: float
    exit_reason: str  # "WIN_TP", "LOSS_SL", "LOSS_DEAL_CANCELLATION", "MANUAL_CLOSE"
    loss_root_cause: Optional[str]
    holding_time_seconds: float
    exit_time_utc: str


class TradeLearnerDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        os.makedirs(self.db_path.parent, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        log.debug(f"Trade Learner DB initialized at {self.db_path}")

    def save_context(self, ctx: TradeContext):
        sql = """
        INSERT OR REPLACE INTO trade_contexts (
            trade_id, contract_id, symbol, direction, entry_price, stop_loss,
            take_profit, stake_usd, potential_risk_usd, ob_top, ob_bottom,
            displacement_atr_ratio, htf_depth_pct, macro_pool_type,
            session_phase, atr, spread_pips, entry_time_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        vals = (
            ctx.trade_id, ctx.contract_id, ctx.symbol, ctx.direction,
            ctx.entry_price, ctx.stop_loss, ctx.take_profit, ctx.stake_usd,
            ctx.potential_risk_usd, ctx.ob_top, ctx.ob_bottom,
            ctx.displacement_atr_ratio, ctx.htf_depth_pct, ctx.macro_pool_type,
            ctx.session_phase, ctx.atr, ctx.spread_pips, ctx.entry_time_utc
        )
        with self._get_connection() as conn:
            conn.execute(sql, vals)
            conn.commit()
        log.debug(f"Saved entry context for trade {ctx.trade_id} ({ctx.symbol})")

    def save_outcome(self, outcome: TradePostMortem):
        sql = """
        INSERT OR REPLACE INTO trade_outcomes (
            trade_id, exit_price, pnl_usd, r_multiple, exit_reason,
            loss_root_cause, holding_time_seconds, exit_time_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        vals = (
            outcome.trade_id, outcome.exit_price, outcome.pnl_usd,
            outcome.r_multiple, outcome.exit_reason, outcome.loss_root_cause,
            outcome.holding_time_seconds, outcome.exit_time_utc
        )
        with self._get_connection() as conn:
            conn.execute(sql, vals)
            conn.commit()
        log.info(
            f"🧠 [POST-TRADE ATTRIBUTION] Trade {outcome.trade_id} settled: "
            f"{outcome.exit_reason} ({outcome.r_multiple:+.2f}R, ${outcome.pnl_usd:+.2f}) | "
            f"Cause: {outcome.loss_root_cause or 'N/A'}"
        )

    def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        sql = """
        SELECT c.*, o.exit_price, o.pnl_usd, o.r_multiple, o.exit_reason,
               o.loss_root_cause, o.holding_time_seconds, o.exit_time_utc
        FROM trade_contexts c
        INNER JOIN trade_outcomes o ON c.trade_id = o.trade_id
        ORDER BY o.exit_time_utc DESC
        LIMIT ?
        """
        with self._get_connection() as conn:
            cur = conn.execute(sql, (limit,))
            return [dict(row) for row in cur.fetchall()]


# Singleton database instance
_learner_db: Optional[TradeLearnerDB] = None


def get_learner_db() -> TradeLearnerDB:
    global _learner_db
    if _learner_db is None:
        _learner_db = TradeLearnerDB()
    return _learner_db


def classify_loss_root_cause(
    exit_reason: str,
    holding_time_seconds: float,
    spread_at_entry: float,
    max_spread_allowed: float,
    displacement_ratio: float,
) -> str:
    """Classify the root cause of a losing or cancelled trade."""
    if exit_reason == "LOSS_DEAL_CANCELLATION":
        return "SLUGGISH_OB_FAILURE"
    if spread_at_entry >= max_spread_allowed * 0.90:
        return "SPREAD_SLIPPAGE_DRAG"
    if displacement_ratio < 0.95:
        return "WEAK_IMPULSE_DISPLACEMENT"
    if holding_time_seconds < 180:
        return "AGGRESSIVE_COUNTER_IMPULSE"
    return "ORGANIC_STOP_OUT"


def get_session_phase(dt: Optional[datetime] = None) -> str:
    """Categorize UTC time into specific liquidity session phases."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    hour = dt.hour

    if 7 <= hour < 9:
        return "LONDON_OPEN"
    elif 9 <= hour < 11:
        return "LONDON_LATE"
    elif 13 <= hour < 15:
        return "NY_OPEN"
    elif 15 <= hour < 17:
        return "NY_LATE"
    return "OFF_SESSION"
