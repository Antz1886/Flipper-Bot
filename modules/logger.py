"""
modules/logger.py — Structured Logging & Trade Journal
Writes all bot activity to a rotating log file (bot.log) and
appends every trade event to a CSV journal (trades.csv).

v2.0 — Added trade outcome logging (WIN/LOSS/PnL tracking).
"""

import csv
import logging
import logging.handlers
import os
from datetime import datetime, timezone
from typing import Any

from config import LOG_FILE, TRADE_CSV

_CSV_FIELDS = [
    "timestamp", "symbol", "direction", "signal_type",
    "entry_price", "ob_top", "ob_bottom", "fvg_top", "fvg_bottom",
    "confidence", "stake_usd", "stop_loss_usd", "take_profit_usd",
    "contract_id", "balance_before", "result", "pnl_usd", "balance_after", "notes",
]


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


def _ensure_header():
    if not os.path.exists(TRADE_CSV):
        with open(TRADE_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()


def log_trade(
    *,
    signal,
    stake:              float,
    stop_loss_amount:   float,
    take_profit_amount: float,
    balance_before:     float,
    order_result:       dict | None,
    result:             str,
    notes:              str = "",
) -> None:
    """Append one trade row to trades.csv."""
    _ensure_header()

    buy_resp    = (order_result or {}).get("buy", {})
    contract_id = buy_resp.get("contract_id", "")

    row: dict[str, Any] = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "symbol":         signal.symbol,
        "direction":      signal.direction,
        "signal_type":    signal.signal_type,
        "entry_price":    signal.entry_price,
        "ob_top":         signal.ob_top,
        "ob_bottom":      signal.ob_bottom,
        "fvg_top":        signal.fvg_top,
        "fvg_bottom":     signal.fvg_bottom,
        "confidence":     round(signal.confidence, 2),
        "stake_usd":      stake,
        "stop_loss_usd":  stop_loss_amount,
        "take_profit_usd": take_profit_amount,
        "contract_id":    contract_id,
        "balance_before": round(balance_before, 4),
        "result":         result,
        "pnl_usd":        "",
        "balance_after":  "",
        "notes":          notes,
    }

    with open(TRADE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


def log_trade_outcome(
    *,
    symbol:         str,
    contract_id:    str = "",
    pnl_usd:        float = 0.0,
    balance_after:  float = 0.0,
    result:         str = "CLOSED",
    notes:          str = "",
) -> None:
    """
    Append a trade outcome (close/win/loss) row to trades.csv.
    Called when a contract is detected as closed.
    """
    _ensure_header()
    row = {field: "" for field in _CSV_FIELDS}
    row["timestamp"]     = datetime.now(timezone.utc).isoformat()
    row["symbol"]        = symbol
    row["contract_id"]   = contract_id
    row["result"]        = result
    row["pnl_usd"]       = round(pnl_usd, 2) if pnl_usd else ""
    row["balance_after"] = round(balance_after, 2) if balance_after else ""
    row["notes"]         = notes

    with open(TRADE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)


def log_session_event(event: str, notes: str = "") -> None:
    """Append a session-level event (BOT_START, BOT_STOP, etc.) to the CSV."""
    _ensure_header()
    row = {field: "" for field in _CSV_FIELDS}
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["result"]    = event
    row["notes"]     = notes
    with open(TRADE_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)
