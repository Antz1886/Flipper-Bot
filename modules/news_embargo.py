"""
modules/news_embargo.py — Tier-1 Macroeconomic Data News Embargo Engine
Fetches the institutional economic calendar from ForexFactory JSON feed and enforces
a strict "No-Trade" state across USD pairs 30 minutes before and 30 minutes after
any High-Impact USD event (e.g., US CPI, NFP, FOMC Interest Rate Decisions, PPI, GDP).
"""

import os
import json
import time
import logging
import urllib.request
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional

import config

log = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    title: str
    country: str
    impact: str
    time_utc: datetime

    @property
    def is_tier_1_usd(self) -> bool:
        if self.country.upper() != "USD":
            return False
        if self.impact.upper() == "HIGH":
            return True
        # Hardcoded Tier-1 keywords if broker or feed classifies as medium
        tier1_keywords = ("CPI", "NON-FARM", "PAYROLL", "FOMC", "RATE DECISION", "FED FUNDS", "GDP")
        title_upper = self.title.upper()
        return any(k in title_upper for k in tier1_keywords)


_cached_events: list[NewsEvent] = []
_last_sync_timestamp: float = 0.0


def parse_event_dict(item: dict) -> Optional[NewsEvent]:
    """Parse a single ForexFactory event dict into a timezone-aware NewsEvent."""
    try:
        title = item.get("title", "")
        country = item.get("country", "")
        impact = item.get("impact", "")
        date_str = item.get("date", "")
        if not date_str:
            return None

        # Format: 2026-10-18T10:15:00-04:00 or ISO format
        dt = datetime.fromisoformat(date_str).astimezone(timezone.utc)
        return NewsEvent(
            title=title,
            country=country,
            impact=impact,
            time_utc=dt,
        )
    except Exception as e:
        log.debug(f"Failed to parse calendar event {item}: {e}")
        return None


def fetch_economic_calendar() -> list[NewsEvent]:
    """Fetch live economic calendar events from ForexFactory JSON feed."""
    url = getattr(config, "ECONOMIC_CALENDAR_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
    cache_path = getattr(config, "CALENDAR_CACHE_FILE", "logs/calendar_cache.json")
    events = []

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FlipperBot/2.0 QuantNewsMonitor",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        for item in data:
            evt = parse_event_dict(item)
            if evt and evt.is_tier_1_usd:
                events.append(evt)

        # Cache valid events
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        log.info(f"📅 Economic Calendar synchronized: {len(events)} High-Impact USD events tracked.")
        return events

    except Exception as e:
        log.warning(f"Failed to fetch live economic calendar ({e}). Checking local disk cache...")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    evt = parse_event_dict(item)
                    if evt and evt.is_tier_1_usd:
                        events.append(evt)
                log.info(f"Loaded {len(events)} Tier-1 USD events from local calendar cache.")
                return events
            except Exception as cache_err:
                log.error(f"Failed to read calendar cache: {cache_err}")

        return events


def sync_calendar_if_needed(force: bool = False):
    """Synchronize the calendar hourly or upon initialization."""
    global _cached_events, _last_sync_timestamp
    now_ts = time.time()
    sync_interval = getattr(config, "CALENDAR_SYNC_INTERVAL_S", 3600)

    if force or not _cached_events or (now_ts - _last_sync_timestamp) > sync_interval:
        fetched = fetch_economic_calendar()
        if fetched or not _cached_events:
            _cached_events = fetched
            _last_sync_timestamp = now_ts


def is_news_embargo_active(
    now_dt: Optional[datetime] = None,
    events: Optional[list[NewsEvent]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Check if current time is within 30 minutes before or 30 minutes after
    any High-Impact USD news release.
    Returns (is_active, embargo_reason_str).
    """
    if not getattr(config, "ENFORCE_NEWS_EMBARGO", True):
        return False, None

    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    else:
        now_dt = now_dt.astimezone(timezone.utc)

    event_list = events if events is not None else _cached_events
    if not event_list:
        sync_calendar_if_needed()
        event_list = _cached_events

    before_min = getattr(config, "NEWS_EMBARGO_MINUTES_BEFORE", 30)
    after_min = getattr(config, "NEWS_EMBARGO_MINUTES_AFTER", 30)

    for evt in event_list:
        diff_s = (now_dt - evt.time_utc).total_seconds()
        diff_min = diff_s / 60.0

        # Within [-before_min, +after_min] window
        if -before_min <= diff_min <= after_min:
            phase = f"T{-int(abs(diff_min))}m BEFORE" if diff_min < 0 else f"T{+int(diff_min)}m AFTER"
            reason = (
                f"🚨 [TIER-1 DATA EMBARGO] {evt.title} ({evt.country}) at "
                f"{evt.time_utc.strftime('%H:%M UTC')} [{phase} release]. All USD entries locked."
            )
            return True, reason

    return False, None


def inject_custom_event(event: NewsEvent):
    """Helper for testing: inject custom news event into active cache."""
    global _cached_events
    _cached_events.append(event)


def clear_events_cache():
    """Helper for testing: clear active cache."""
    global _cached_events
    _cached_events.clear()
