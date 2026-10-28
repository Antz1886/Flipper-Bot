"""
modules/session_state.py — Atomic Session State Persistence
Maintains resilient session state across container crashes, VM restarts, and reconnections:
  - Tracks real-time session P&L, current Martingale tier, loss/win streaks, and total trade metrics.
  - Automatically restores active trading state on container reboot within the same UTC day.
  - Enforces session finalization when Take Profit or Stop Loss circuit breakers trigger.
"""

import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

log = logging.getLogger("session_state")

STATE_FILE = "data/dbot_session.json"


class SessionStateManager:
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    def load_active_session(self) -> dict | None:
        """
        Loads persisted session from disk if it exists, belongs to current UTC day,
        and is still active or halted.
        """
        if not os.path.exists(self.state_file):
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            saved_date = data.get("date_utc")

            if saved_date != today_utc:
                log.info(f"Persisted session is from previous date ({saved_date}). Starting fresh for {today_utc}.")
                self.archive_session(data)
                return None

            log.info(
                f"🔄 [SESSION RESUMED] Restored state from disk (ID: {data.get('session_id', 'N/A')[:8]}): "
                f"Trades: {data.get('total_trades', 0)} | W/L: {data.get('total_wins', 0)}/{data.get('total_losses', 0)} | "
                f"Streak: {data.get('consecutive_losses', 0)}L | Next Stake: ${data.get('current_stake', 2.0):.2f} | "
                f"Session P&L: ${data.get('session_pnl', 0.0):+.2f} | Status: {data.get('status', 'RUNNING')}"
            )
            return data
        except Exception as e:
            log.warning(f"Could not load session state: {e}. Starting fresh session.")
            return None

    def save_session(
        self,
        session_id: str,
        total_trades: int,
        total_wins: int,
        total_losses: int,
        consecutive_wins: int,
        consecutive_losses: int,
        current_stake: float,
        session_pnl: float,
        status: str = "RUNNING",
    ):
        """Atomically persist session state to prevent corruption on crash."""
        now_utc = datetime.now(timezone.utc)
        payload = {
            "session_id": session_id,
            "date_utc": now_utc.strftime("%Y-%m-%d"),
            "updated_at": now_utc.isoformat(),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "current_stake": round(current_stake, 2),
            "session_pnl": round(session_pnl, 2),
            "status": status,
        }

        try:
            dir_name = os.path.dirname(self.state_file) or "."
            os.makedirs(dir_name, exist_ok=True)
            # Write to temporary file first, then atomic rename
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
                json.dump(payload, tf, indent=2)
                temp_name = tf.name

            os.replace(temp_name, self.state_file)
        except Exception as e:
            log.error(f"Failed to persist session state: {e}")

    def archive_session(self, data: dict):
        """Archives previous day's session."""
        try:
            archive_dir = os.path.join(os.path.dirname(self.state_file), "archive")
            os.makedirs(archive_dir, exist_ok=True)
            sid = data.get("session_id", uuid.uuid4().hex[:8])
            date = data.get("date_utc", "unknown")
            archive_file = os.path.join(archive_dir, f"session_{date}_{sid[:8]}.json")
            with open(archive_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
        except Exception as e:
            log.warning(f"Failed to archive session: {e}")

    def archive_active_file(self):
        """Archives current active session file to archive directory."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.archive_session(data)
        except Exception as e:
            log.warning(f"Failed to archive active state file: {e}")

    def clear(self):
        """Clear active session file."""
        try:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
        except Exception as e:
            log.warning(f"Failed to clear session state file: {e}")
