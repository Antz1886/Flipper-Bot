"""
modules/connector.py — Deriv WebSocket API Connection Manager
Maintains a single persistent WebSocket connection to the Deriv API.
Routes responses via req_id-keyed Futures and dispatches tick/balance
subscription updates to registered callbacks.

v2.0 — Added connection_healthy flag, duplicate handler prevention,
       and improved disconnect resilience.
"""

import asyncio
import json
import logging
from typing import Callable

import websockets
import websockets.exceptions
try:
    from websockets import State
except ImportError:
    from websockets.protocol import State

from config import DERIV_WS_URL, DERIV_API_TOKEN, MAX_RETRIES, RETRY_DELAY_S

log = logging.getLogger(__name__)


class DerivAPI:
    """Persistent Deriv WebSocket connection with async request/response routing."""

    def __init__(self):
        self._ws = None
        self._req_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_handlers: list[Callable] = []
        self._recv_task: asyncio.Task | None = None
        self.balance: float = 0.0
        self.currency: str = "USD"
        self.authorized: bool = False
        self.is_active: bool = False
        self.connection_healthy: bool = False  # True only when WS is confirmed alive

    async def _cleanup(self):
        """Cancel receiver task, close websocket, and reject pending futures."""
        self.authorized = False
        self.is_active = False
        self.connection_healthy = False
        
        # 1. Cancel receiver task
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None
            
        # 2. Close websocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            
        # 3. Reject pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Connection cleaned up"))
        self._pending.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def _recv_loop(self):
        """Background task: reads all incoming WebSocket messages and routes them."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    log.warning(f"Non-JSON message received: {raw[:80]}")
                    continue

                msg_type = msg.get("msg_type")
                req_id   = msg.get("req_id")
                error    = msg.get("error")

                # ── Resolve pending request ───────────────────────────────────
                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        if error:
                            fut.set_exception(
                                RuntimeError(f"Deriv error [{error.get('code')}]: {error.get('message')}")
                            )
                        else:
                            fut.set_result(msg)

                # ── Tick subscription updates (no req_id after first response) ─
                elif msg_type == "tick":
                    for cb in self._tick_handlers:
                        # Dispatch as task to avoid deadlocking if handler calls api.send()
                        asyncio.create_task(cb(msg["tick"]))

                # ── Balance subscription updates ──────────────────────────────
                elif msg_type == "balance":
                    bal = msg.get("balance", {})
                    self.balance  = float(bal.get("balance", self.balance))
                    self.currency = bal.get("currency", self.currency)
                    log.debug(f"Balance update: {self.currency} {self.balance:.4f}")

        except websockets.exceptions.ConnectionClosed as e:
            log.warning(f"WebSocket connection closed: {e}")
        except Exception as e:
            log.error(f"recv_loop error: {e}", exc_info=True)
        finally:
            self.is_active = False
            self.connection_healthy = False
            # Reject all pending futures on disconnect
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket disconnected"))
            self._pending.clear()

    # ── Public API ────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Open WebSocket, start receiver, authorize. Returns True on success."""
        for attempt in range(1, MAX_RETRIES + 1):
            await self._cleanup()
            try:
                self._ws = await websockets.connect(
                    DERIV_WS_URL,
                    ping_interval=None,  # Disable library keepalive to prevent 1011 errors
                    ping_timeout=None,
                    close_timeout=10,
                    open_timeout=10,
                )
                self.is_active = True
                self._recv_task = asyncio.create_task(self._recv_loop())
                log.info(f"🌐 WebSocket connected to Deriv API (attempt {attempt})")

                # Authorize
                resp = await self.send({"authorize": DERIV_API_TOKEN})
                info = resp.get("authorize", {})
                self.balance  = float(info.get("balance", 0))
                self.currency = info.get("currency", "USD")
                self.authorized = True
                self.connection_healthy = True
                log.info(
                    f"Authorized | Account: {info.get('email','?')} | "
                    f"Balance: {self.currency} {self.balance:.2f}"
                )

                # Subscribe to balance updates
                await self.send({"balance": 1, "subscribe": 1})
                return True

            except Exception as e:
                log.warning(f"Connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
                await self._cleanup()
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S * attempt)

        log.critical("Failed to connect to Deriv API after all retries.")
        return False

    async def send(self, payload: dict) -> dict:
        """Send a request and await its response. Thread-safe via req_id."""
        if not self.is_active or not self._ws or self._ws.state != State.OPEN:
            raise ConnectionError("WebSocket is not connected")
            
        rid = self._next_id()
        payload["req_id"] = rid
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
            # Wrap write operation in a timeout to prevent hanging on dead/half-open sockets
            await asyncio.wait_for(self._ws.send(json.dumps(payload)), timeout=10.0)
            return await asyncio.wait_for(fut, timeout=30.0)
        except Exception:
            if rid in self._pending:
                self._pending.pop(rid)
            raise

    async def subscribe_ticks(self, symbol: str, callback: Callable) -> str:
        """Subscribe to live ticks for a symbol. Returns subscription id."""
        # Prevent duplicate handler registration
        if callback not in self._tick_handlers:
            self._tick_handlers.append(callback)
        
        try:
            resp = await self.send({"ticks": symbol, "subscribe": 1})
            sub_id = resp.get("subscription", {}).get("id", "")
            log.info(f"Subscribed to ticks for {symbol} | sub_id: {sub_id}")
            return sub_id
        except Exception as e:
            log.error(f"Subscription failed for {symbol}: {e}")
            raise

    async def unsubscribe(self, sub_id: str):
        """Cancel a subscription by its id."""
        await self.send({"forget": sub_id})
        log.info(f"Unsubscribed: {sub_id}")

    async def is_connected(self) -> bool:
        """Ping the server to check connection health."""
        if not self.is_active or not self._ws or self._ws.state != State.OPEN:
            self.connection_healthy = False
            return False
        rid = self._next_id()
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
            # Wrap ping send and response in a short timeout
            await asyncio.wait_for(self._ws.send(json.dumps({"ping": 1, "req_id": rid})), timeout=3.0)
            await asyncio.wait_for(fut, timeout=3.0)
            self.connection_healthy = True
            return True
        except Exception:
            if rid in self._pending:
                self._pending.pop(rid)
            self.connection_healthy = False
            return False

    async def disconnect(self):
        """Gracefully close the WebSocket connection."""
        await self._cleanup()
        log.info("Deriv WebSocket disconnected.")


# ── Module-level singleton ────────────────────────────────────────────────────
_api: DerivAPI | None = None


async def initialize() -> bool:
    global _api
    _api = DerivAPI()
    return await _api.connect()


def get_api() -> DerivAPI:
    return _api


async def shutdown():
    if _api:
        await _api.disconnect()
