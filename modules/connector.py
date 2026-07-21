"""
modules/connector.py — Deriv WebSocket API Connection Manager
Maintains a persistent WebSocket connection to Deriv API.
Routes responses via req_id-keyed Futures and dispatches tick/balance updates.
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

import config
from config import DERIV_WS_URL, DERIV_API_TOKEN, MAX_RETRIES, RETRY_DELAY_S

log = logging.getLogger(__name__)


class DerivAPI:
    """Persistent Deriv WebSocket connection with async request/response routing."""

    def __init__(self):
        self._ws = None
        self._req_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_handlers: list[Callable] = []
        self._transaction_handlers: list[Callable] = []
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self.balance: float = 0.0
        self.currency: str = "USD"
        self.loginid: str = ""
        self.authorized: bool = False
        self.is_active: bool = False
        self.connection_healthy: bool = False

    async def _cleanup(self):
        """Cancel receiver task, heartbeat task, close websocket, and reject pending futures."""
        self.authorized = False
        self.is_active = False
        self.connection_healthy = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None

        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):
                pass
            self._recv_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Connection cleaned up"))
        self._pending.clear()

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

                if req_id and req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        if error:
                            fut.set_exception(
                                RuntimeError(f"Deriv error [{error.get('code')}]: {error.get('message')}")
                            )
                        else:
                            fut.set_result(msg)

                elif msg_type == "tick":
                    for cb in self._tick_handlers:
                        asyncio.create_task(cb(msg["tick"]))

                elif msg_type == "balance":
                    bal = msg.get("balance", {})
                    self.balance  = float(bal.get("balance", self.balance))
                    self.currency = bal.get("currency", self.currency)
                    log.debug(f"Balance update: {self.currency} {self.balance:.2f}")

                elif msg_type == "transaction":
                    tx = msg.get("transaction", {})
                    for cb in self._transaction_handlers:
                        asyncio.create_task(cb(tx))

        except websockets.exceptions.ConnectionClosed as e:
            log.warning(f"WebSocket connection closed: {e}")
        except Exception as e:
            log.error(f"recv_loop error: {e}", exc_info=True)
        finally:
            self.is_active = False
            self.connection_healthy = False
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("WebSocket disconnected"))
            self._pending.clear()

    async def _heartbeat_loop(self):
        """Send a ping frame to Deriv API every 25 seconds to keep connection alive."""
        log.info("Starting WebSocket heartbeat loop (ping every 25s)")
        try:
            while self.is_active:
                await asyncio.sleep(25.0)
                healthy = await self.is_connected()
                if not healthy:
                    log.warning("Heartbeat health check failed! WebSocket connection is dead.")
                    if self._ws:
                        await self._ws.close()
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Error in heartbeat loop: {e}")

    async def connect(self) -> bool:
        """Connect to Deriv WebSocket and authorize with API Token. Returns True on success."""
        ws_url = config.DERIV_WS_URL
        api_token = config.DERIV_API_TOKEN

        for attempt in range(1, MAX_RETRIES + 1):
            await self._cleanup()
            try:
                log.info(f"Connecting to Deriv WebSocket... (attempt {attempt}/{MAX_RETRIES})")
                self._ws = await websockets.connect(
                    ws_url,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                    open_timeout=10,
                )
                self.is_active = True
                self._recv_task = asyncio.create_task(self._recv_loop())

                if api_token and api_token != "your_api_token_here":
                    log.info("Authorizing with Deriv API Token...")
                    auth_resp = await self.send({"authorize": api_token})
                    auth_data = auth_resp.get("authorize", {})
                    self.balance = float(auth_data.get("balance", 0.0))
                    self.currency = auth_data.get("currency", "USD")
                    self.loginid = auth_data.get("loginid", "")
                    self.authorized = True
                    log.info(f"✅ Authorized | Account: {self.loginid} | Balance: {self.currency} {self.balance:.2f}")

                    # Subscribe to balance updates
                    await self.send({"balance": 1, "subscribe": 1})

                else:
                    log.warning("No DERIV_API_TOKEN set. Running in unauthenticated mode (read-only feeds).")

                self.connection_healthy = True
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                return True

            except Exception as e:
                log.warning(f"Connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
                await self._cleanup()
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY_S * attempt)

        log.error("Failed to connect to Deriv API after retries.")
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
            await asyncio.wait_for(self._ws.send(json.dumps(payload)), timeout=10.0)
            return await asyncio.wait_for(fut, timeout=30.0)
        except Exception:
            if rid in self._pending:
                self._pending.pop(rid)
            raise

    async def is_connected(self) -> bool:
        """Ping the server to check connection health."""
        if not self.is_active or not self._ws or self._ws.state != State.OPEN:
            self.connection_healthy = False
            return False
        rid = self._next_id()
        fut = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        try:
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


# Singleton instance
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
