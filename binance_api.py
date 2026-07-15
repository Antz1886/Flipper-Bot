import aiohttp
import time
import hmac
import hashlib
import urllib.parse
import logging
import config

log = logging.getLogger(__name__)

class BinanceFuturesClient:
    def __init__(self):
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.base_url = config.BASE_URL
        self.headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        self.time_offset = 0

    async def sync_time(self):
        """Synchronize time offset with Binance server to prevent timestamp errors."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/fapi/v1/time"
                start = int(time.time() * 1000)
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        server_time = int(data.get("serverTime", 0))
                        end = int(time.time() * 1000)
                        latency = (end - start) // 2
                        local_time = end - latency
                        self.time_offset = server_time - local_time
                        log.info(f"⏰ Clock synchronized with Binance Futures. Offset: {self.time_offset}ms")
        except Exception as e:
            log.error(f"Failed to sync time with Binance: {e}")

    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, path: str, params: dict = None, signed: bool = False) -> dict:
        if params is None:
            params = {}

        url = f"{self.base_url}{path}"

        if signed:
            params["timestamp"] = int(time.time() * 1000) + self.time_offset
            query_string = urllib.parse.urlencode(params)
            params["signature"] = self._generate_signature(query_string)

        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url, params=params, headers=self.headers) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            log.error(f"Binance GET {path} Error: Status {resp.status} | {data}")
                            return None
                        return data
                elif method.upper() == "POST":
                    async with session.post(url, params=params, headers=self.headers) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            log.error(f"Binance POST {path} Error: Status {resp.status} | {data}")
                            return None
                        return data
                elif method.upper() == "DELETE":
                    async with session.delete(url, params=params, headers=self.headers) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            log.error(f"Binance DELETE {path} Error: Status {resp.status} | {data}")
                            return None
                        return data
            except Exception as e:
                log.error(f"Connection error requesting {path}: {e}")
                return None

    async def get_ping(self) -> bool:
        """Test connectivity to the REST API."""
        res = await self._request("GET", "/fapi/v1/ping")
        return res is not None

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> list:
        """Fetch historical klines (candlesticks)."""
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        return await self._request("GET", "/fapi/v1/klines", params=params)

    async def get_balance(self) -> float:
        """Fetch USDT wallet balance."""
        res = await self._request("GET", "/fapi/v2/balance", signed=True)
        if res:
            for asset in res:
                if asset.get("asset") == "USDT":
                    return float(asset.get("balance", 0))
        return 0.0

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        params = {
            "symbol": symbol.upper(),
            "leverage": leverage
        }
        res = await self._request("POST", "/fapi/v1/leverage", params=params, signed=True)
        if res and int(res.get("leverage", 0)) == leverage:
            log.info(f"✅ Leverage successfully set to {leverage}x for {symbol}.")
            return True
        return False

    async def get_position_info(self, symbol: str) -> dict:
        """Get active position details for a symbol."""
        params = {"symbol": symbol.upper()}
        res = await self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        if res and isinstance(res, list):
            for pos in res:
                if pos.get("symbol") == symbol.upper():
                    return {
                        "size": float(pos.get("positionAmt", 0)),
                        "entry_price": float(pos.get("entryPrice", 0)),
                        "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                        "leverage": int(pos.get("leverage", 0))
                    }
        return {"size": 0.0, "entry_price": 0.0, "unrealized_pnl": 0.0, "leverage": 0}

    async def get_open_orders(self, symbol: str) -> list:
        """Get all open orders for a symbol."""
        params = {"symbol": symbol.upper()}
        res = await self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
        return res if isinstance(res, list) else []

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancel all open orders for a symbol."""
        params = {"symbol": symbol.upper()}
        res = await self._request("DELETE", "/fapi/v1/allOpenOrders", params=params, signed=True)
        if res and "code" not in res:
            log.info(f"✅ Cancelled all open orders for {symbol}.")
            return True
        return False

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None, stop_price: float = None, reduce_only: bool = False) -> dict:
        """Place an order (Market, Limit, Stop Market)."""
        params = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": round(quantity, config.QTY_PRECISION)
        }

        if price is not None:
            params["price"] = str(round(price, 2))
            params["timeInForce"] = "GTC"
        if stop_price is not None:
            params["stopPrice"] = str(round(stop_price, 2))
        if reduce_only:
            params["reduceOnly"] = "true"

        return await self._request("POST", "/fapi/v1/order", params=params, signed=True)
