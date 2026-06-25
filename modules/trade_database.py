# modules/trade_database.py
"""SQLite trade database wrapper.
Provides schema creation, insert, query, and analytics helpers.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "trades.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stake REAL NOT NULL,
    pnl REAL,
    regime TEXT,
    adx REAL,
    adx_slope REAL,
    atr REAL,
    atr_percentile REAL,
    ema_distance REAL,
    rsi REAL,
    score INTEGER,
    score_bucket TEXT,
    duration_seconds INTEGER,
    mfe REAL,
    mae REAL,
    recovery_factor REAL
);
"""

class TradeDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()
        log.info(f"Trade database initialised at {self.db_path}")

    def _create_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def insert_trade(self, trade: Dict[str, Any]):
        """Insert a trade dictionary matching the schema fields.
        Missing optional fields are allowed and will be stored as NULL.
        """
        columns = ", ".join(trade.keys())
        placeholders = ", ".join(["?" for _ in trade])
        sql = f"INSERT INTO trades ({columns}) VALUES ({placeholders})"
        self.conn.execute(sql, tuple(trade.values()))
        self.conn.commit()
        log.debug(f"Inserted trade id={self.conn.lastrowid}")
        return self.conn.lastrowid

    def query_trades(self, where: str = "", params: Tuple = ()) -> List[sqlite3.Row]:
        sql = "SELECT * FROM trades"
        if where:
            sql += f" WHERE {where}"
        cur = self.conn.execute(sql, params)
        return cur.fetchall()

    def close(self):
        self.conn.close()
        log.info("Trade database connection closed")
