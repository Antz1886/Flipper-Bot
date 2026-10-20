"""
modules/connector.py — Deriv WebSocket API Connection Manager
Supports both:
  1. Deriv v1 REST + OTP WebSocket authentication (using DERIV_APP_ID='33z3Jc7fSBXsnKQzT1tXm' + 'pat_...' token)
  2. Deriv v3 WebSocket direct token authentication (15-char token) & Read-Only Market Scanner fallback
Maintains a persistent WebSocket connection with automatic reconnection.
"""

import asyncio
import json
import logging
import urllib.request
import urllib.error
from typing import Callable

import websockets
import websockets.exceptions
try:
    from websockets import State
except ImportError:
    from websockets.protocol import State

import config
from config import MAX_RETRIES, RETRY_DELAY_S

log = logging.getLogger(__name__)


class DerivAPI:
    """Persistent Deriv WebSocket connection with async request/response routing and auto-reconnect."""

    def __init__(self):
        self._ws = None
        self._req_id: int = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_handlers: list[Callable] = []
        self._transaction_handlers: list[Callable] = []
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._balance_event: asyncio.Event | None = None
        self.balance: float | None = None
        self.currency: str = "USD"
        self.loginid: str = ""
        self.authorized: bool = False
        self.using_otp_ws: bool = False
        self.is_active: bool = False
        self.connection_healthy: bool = False

    async def _fetch_rest(self, path: str, api_token: str, app_id: str, method: str = "GET", payload: dict | None = None) -> dict:
        """Perform an asynchronous REST API call to Deriv (api.derivws.com)."""
        url = f"{config.DERIV_REST_URL}{path}"

        def blocking_req():
            req = urllib.request.Request(url, method=method)
            req.add_header("Authorization", f"Bearer {api_token}")
            req.add_header("Deriv-App-ID", str(app_id))
            req.add_header("User-Agent", "Mozilla/5.0")
            if payload is not None:
                req.data = json.dumps(payload).encode("utf-8")
                req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        return await asyncio.to_thread(blocking_req)

    async def _cleanup(self):
        """Cancel receiver task, heartbeat task, close websocket, and reject pending futures."""
        self.authorized = False
        self.using_otp_ws = False
        self.is_active = False
        self.connection_healthy = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None

        if self._balance_event and not self._balance_event.is_set():
            self._balance_event.set()
        self._balance_event = None

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
                    self.balance  = float(bal.get("balance", self.balance if self.balance is not None else 0.0))
                    self.currency = bal.get("currency", self.currency)
                    if self._balance_event and not self._balance_event.is_set():
                        self._balance_event.set()
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
                    log.warning("Heartbeat health check failed! Marking connection inactive for auto-reconnect.")
                    self.is_active = False
                    self.connection_healthy = False
                    if self._ws:
                        try:
                            await self._ws.close()
                        except Exception:
                            pass
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Error in heartbeat loop: {e}")

    async def ensure_connected(self) -> bool:
        """Ensure the WebSocket connection is active, automatically reconnecting if dropped."""
        if self.is_active and self._ws and self._ws.state == State.OPEN:
            return True
        log.info("🔄 Deriv WebSocket disconnected — initiating automatic reconnection...")
        return await self.connect()

    async def _connect_via_rest_otp(self, api_token: str, app_id: str) -> bool:
        """
        Authenticate via Deriv v1 REST API (/trading/v1/options/accounts -> /otp)
        using the registered 21-char App ID (33z3Jc7fSBXsnKQzT1tXm) and PAT token (pat_...).
        """
        log.info(f"Authenticating PAT token via Deriv REST API (App ID: {app_id})...")
        accounts_resp = await self._fetch_rest("/trading/v1/options/accounts", api_token=api_token, app_id=app_id)
        accounts = accounts_resp.get("data", [])
        if not accounts:
            raise ValueError("No trading accounts returned from Deriv REST API.")

        target_type = "demo" if config.DEMO_MODE else "real"
        selected_account = None
        for acc in accounts:
            if acc.get("account_type") == target_type and acc.get("status") == "active":
                selected_account = acc
                break
        if not selected_account:
            for acc in accounts:
                if acc.get("account_type") == target_type:
                    selected_account = acc
                    break
        if not selected_account:
            selected_account = accounts[0]

        account_id = selected_account.get("account_id")
        log.info(f"Selected {target_type} account: {account_id}")

        otp_resp = await self._fetch_rest(
            f"/trading/v1/options/accounts/{account_id}/otp",
            api_token=api_token,
            app_id=app_id,
            method="POST",
        )
        otp_url = otp_resp.get("data", {}).get("url")
        if not otp_url:
            raise ValueError("Failed to retrieve pre-authenticated WebSocket OTP URL from Deriv.")

        log.info("Connecting to pre-authenticated Deriv OTP WebSocket...")
        self._ws = await websockets.connect(
            otp_url,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=10,
            open_timeout=10,
        )
        self.is_active = True
        self.using_otp_ws = True
        self._recv_task = asyncio.create_task(self._recv_loop())

        self.balance = float(selected_account.get("balance", 0.0))
        self.currency = selected_account.get("currency", "USD")
        self.loginid = str(account_id)
        self.authorized = True
        self.connection_healthy = True

        log.info(
            f"✅ Authorized via Deriv OTP | Account: {self.loginid} | "
            f"Balance: {self.currency} {self.balance:.2f}"
        )

        try:
            await self.send({"balance": 1, "subscribe": 1})
        except Exception as e:
            log.debug(f"Balance subscription note: {e}")

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        return True

    async def connect(self) -> bool:
        """Connect to Deriv WebSocket and authorize with API Token. Returns True on success."""
        public_app_id = getattr(config, "DERIV_PUBLIC_APP_ID", "1089")
        ws_urls = [
            f"wss://red.derivws.com/websockets/v3?app_id={public_app_id}",
            f"wss://red.derivws.com/websockets/v3?app_id={public_app_id}",
            f"wss://ws.derivws.com/websockets/v3?app_id={public_app_id}",
        ]
        api_token = (config.DERIV_API_TOKEN or "").strip()
        app_id = (config.DERIV_APP_ID or "33z3Jc7fSBXsnKQzT1tXm").strip()

        # Check if user accidentally pasted the 21-char App ID into DERIV_API_TOKEN
        if api_token == "33z3Jc7fSBXsnKQzT1tXm":
            log.warning(
                "ℹ️ '33z3Jc7fSBXsnKQzT1tXm' is your Deriv Application ID (DERIV_APP_ID), not the 'pat_...' API token. "
                "Using it as DERIV_APP_ID and starting in Read-Only Market Scanner Mode until DERIV_API_TOKEN=pat_... is set."
            )
            app_id = api_token
            api_token = ""

        for attempt in range(1, MAX_RETRIES + 1):
            await self._cleanup()
            try:
                # 1. If a Deriv v1 PAT token (pat_...) is configured, try REST + OTP first
                if api_token.startswith("pat_"):
                    try:
                        return await self._connect_via_rest_otp(api_token=api_token, app_id=app_id)
                    except urllib.error.HTTPError as http_err:
                        err_body = http_err.read().decode("utf-8", errors="ignore")[:160]
                        log.error(
                            f"⚠️ Deriv REST PAT authentication failed (HTTP {http_err.code}: {err_body}). "
                            "Falling back to Read-Only Market Scanner Mode on red.derivws.com..."
                        )
                    except Exception as otp_err:
                        log.error(
                            f"⚠️ Deriv REST OTP error ({otp_err}). "
                            "Falling back to Read-Only Market Scanner Mode on red.derivws.com..."
                        )

                # 2. Connect to Deriv v3 WebSocket gateway
                ws_url = ws_urls[(attempt - 1) % len(ws_urls)]
                log.info(f"Connecting to Deriv WebSocket ({ws_url})... (attempt {attempt}/{MAX_RETRIES})")
                self._ws = await websockets.connect(
                    ws_url,
                    origin="https://app.deriv.com",
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=10,
                    open_timeout=10,
                )
                self.is_active = True
                self.using_otp_ws = False
                self._recv_task = asyncio.create_task(self._recv_loop())

                if api_token and api_token != "your_api_token_here" and not api_token.startswith("pat_"):
                    log.info("Authorizing with Deriv API Token...")
                    try:
                        auth_resp = await self.send({"authorize": api_token})
                        auth_data = auth_resp.get("authorize", {})
                        self.authorized = True
                        self.loginid = auth_data.get("loginid", "")
                        self.currency = auth_data.get("currency", "USD")
                        self.balance = None
                        if auth_data.get("balance") is not None:
                            self.balance = float(auth_data.get("balance", 0.0))
                        log.info(
                            f"✅ Authorized | Account: {self.loginid} | Balance: "
                            f"{self.currency} {self.balance if self.balance is not None else 'N/A'}"
                        )

                        self._balance_event = asyncio.Event()
                        await self.send({"balance": 1, "subscribe": 1})
                        try:
                            await asyncio.wait_for(self._balance_event.wait(), timeout=10.0)
                            log.debug("Received initial balance update from Deriv.")
                        except asyncio.TimeoutError:
                            log.warning("Balance subscription did not return an update within 10s; using cached balance if available.")

                        if self.balance is None:
                            self.balance = 0.0
                    except RuntimeError as auth_err:
                        log.error(
                            f"⚠️ API Token rejected by Deriv WebSocket ({auth_err}). "
                            "Falling back to unauthenticated Read-Only Market Scanner Mode so container stays alive."
                        )
                        self.authorized = False
                        self.balance = 0.0
                elif not api_token or api_token == "your_api_token_here":
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
            await asyncio.wait_for(self._ws.send(json.dumps({"ping": 1, "req_id": rid})), timeout=10.0)
            await asyncio.wait_for(fut, timeout=10.0)
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


async def ensure_connected() -> bool:
    global _api
    if _api is None:
        _api = DerivAPI()
    return await _api.ensure_connected()


def get_api() -> DerivAPI:
    return _api


async def shutdown():
    if _api:
        await _api.disconnect()
