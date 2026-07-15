import asyncio
import os
import sys
import logging
import time
import csv
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from binance_api import BinanceFuturesClient

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    ]
)
log = logging.getLogger("main")
sys.stdout.reconfigure(encoding="utf-8")

def calculate_ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return 0.0
    k = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema

def calculate_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        return 100.0
        
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    rs = avg_gain / max(1e-10, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))

def log_trade(symbol, side, entry_price, size, stop_loss, take_profit, result="OPEN", pnl=0.0):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(config.TRADE_LOG_CSV)
    with open(config.TRADE_LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "side", "entry_price", "size", "stop_loss", "take_profit", "result", "pnl"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol, side, entry_price, size, stop_loss, take_profit, result, pnl
        ])

async def run_scan_cycle(client: BinanceFuturesClient):
    symbol = config.SYMBOL
    log.info(f"─── Starting Scan Cycle for {symbol} ───")

    balance = await client.get_balance()
    log.info(f"Wallet Balance: {balance:.2f} USDT")

    pos = await client.get_position_info(symbol)
    pos_size = pos.get("size", 0.0)
    entry_price = pos.get("entry_price", 0.0)
    
    if abs(pos_size) > 0.0:
        log.info(f"📊 Active Position: {pos_size} BTC @ ${entry_price:.2f} | Unrealized PnL: ${pos.get('unrealized_pnl'):.2f}")
        return True

    open_orders = await client.get_open_orders(symbol)
    if open_orders:
        log.info(f"🧹 Found {len(open_orders)} open orders without an active position. Cleaning up...")
        await client.cancel_all_orders(symbol)

    h1_klines = await client.get_klines(symbol, "1h", limit=250)
    if not h1_klines:
        log.warning("Failed to fetch 1H candlesticks.")
        return True
    h1_closes = [float(k[4]) for k in h1_klines]
    
    m15_klines = await client.get_klines(symbol, "15m", limit=100)
    if not m15_klines:
        log.warning("Failed to fetch 15M candlesticks.")
        return True
    m15_closes = [float(k[4]) for k in m15_klines]
    
    current_price = m15_closes[-1]
    
    h1_trend_ema = calculate_ema(h1_closes, config.EMA_TREND_PERIOD)
    m15_fast_ema = calculate_ema(m15_closes, config.EMA_FAST_PERIOD)
    m15_slow_ema = calculate_ema(m15_closes, config.EMA_SLOW_PERIOD)
    m15_rsi = calculate_rsi(m15_closes, config.RSI_PERIOD)
    
    log.info(f"Price: ${current_price:.2f} | H1 Trend EMA: ${h1_trend_ema:.2f}")
    log.info(f"M15 Fast EMA: ${m15_fast_ema:.2f} | M15 Slow EMA: ${m15_slow_ema:.2f} | RSI: {m15_rsi:.2f}")

    is_bullish_trend = current_price > h1_trend_ema
    is_bearish_trend = current_price < h1_trend_ema

    raw_qty = config.TRADE_STAKE_USDT / current_price
    quantity = round(raw_qty, 3)
    if quantity == 0:
        log.warning("Calculated quantity is 0. Increase TRADE_STAKE_USDT.")
        return True

    if is_bullish_trend:
        ema_aligned = m15_fast_ema > m15_slow_ema
        price_in_pullback = current_price <= m15_slow_ema
        rsi_oversold = m15_rsi < config.RSI_BUY_THRESHOLD
        
        log.info(f"Long Scan: EMA aligned: {ema_aligned} | Pullback: {price_in_pullback} | RSI oversold: {rsi_oversold}")
        
        if ema_aligned and price_in_pullback and rsi_oversold:
            log.info("🚀 [LONG SIGNAL] Triggering buy order...")
            buy_res = await client.place_order(symbol, "BUY", "MARKET", quantity)
            if not buy_res or "orderId" not in buy_res:
                log.error("Failed to execute market BUY order.")
                return True
                
            entry_spot = float(buy_res.get("avgPrice", current_price) or current_price)
            log.info(f"✅ Market BUY filled @ ${entry_spot:.2f} | Qty: {quantity}")
            
            sl_price = entry_spot * (1.0 - config.STOP_LOSS_PCT / 100.0)
            sl_res = await client.place_order(symbol, "SELL", "STOP_MARKET", quantity, stop_price=sl_price, reduce_only=True)
            
            tp_price = entry_spot * (1.0 + config.TAKE_PROFIT_PCT / 100.0)
            tp_res = await client.place_order(symbol, "SELL", "LIMIT", quantity, price=tp_price, reduce_only=True)
            
            if sl_res and tp_res:
                log.info(f"✅ Bracket Orders Placed: SL @ ${sl_price:.2f} | TP @ ${tp_price:.2f}")
                log_trade(symbol, "LONG", entry_spot, quantity, sl_price, tp_price)
            else:
                log.error("Failed to place bracket orders. Cleaning position.")
                await client.place_order(symbol, "SELL", "MARKET", quantity, reduce_only=True)

    elif is_bearish_trend:
        ema_aligned = m15_fast_ema < m15_slow_ema
        price_in_pullback = current_price >= m15_slow_ema
        rsi_overbought = m15_rsi > config.RSI_SELL_THRESHOLD
        
        log.info(f"Short Scan: EMA aligned: {ema_aligned} | Pullback: {price_in_pullback} | RSI overbought: {rsi_overbought}")
        
        if ema_aligned and price_in_pullback and rsi_overbought:
            log.info("📉 [SHORT SIGNAL] Triggering sell order...")
            sell_res = await client.place_order(symbol, "SELL", "MARKET", quantity)
            if not sell_res or "orderId" not in sell_res:
                log.error("Failed to execute market SELL order.")
                return True
                
            entry_spot = float(sell_res.get("avgPrice", current_price) or current_price)
            log.info(f"✅ Market SELL filled @ ${entry_spot:.2f} | Qty: {quantity}")
            
            sl_price = entry_spot * (1.0 + config.STOP_LOSS_PCT / 100.0)
            sl_res = await client.place_order(symbol, "BUY", "STOP_MARKET", quantity, stop_price=sl_price, reduce_only=True)
            
            tp_price = entry_spot * (1.0 - config.TAKE_PROFIT_PCT / 100.0)
            tp_res = await client.place_order(symbol, "BUY", "LIMIT", quantity, price=tp_price, reduce_only=True)
            
            if sl_res and tp_res:
                log.info(f"✅ Bracket Orders Placed: SL @ ${sl_price:.2f} | TP @ ${tp_price:.2f}")
                log_trade(symbol, "SHORT", entry_spot, quantity, sl_price, tp_price)
            else:
                log.error("Failed to place bracket orders. Cleaning position.")
                await client.place_order(symbol, "BUY", "MARKET", quantity, reduce_only=True)
                
    return True

async def main():
    log.info("=" * 60)
    log.info("       BINANCE FUTURES TRADING BOT (TESTNET)")
    log.info("=" * 60)
    log.info(f"  Symbol    : {config.SYMBOL}")
    log.info(f"  Leverage  : {config.LEVERAGE}x")
    log.info(f"  Stake Size: {config.TRADE_STAKE_USDT} USDT")
    log.info(f"  SL / TP   : {config.STOP_LOSS_PCT}% / {config.TAKE_PROFIT_PCT}%")
    log.info(f"  Testnet   : {config.USE_TESTNET}")
    log.info("=" * 60)

    client = BinanceFuturesClient()

    if not client.api_key or not client.api_secret:
        log.critical("❌ ERROR: API Credentials missing! Please configure BINANCE_API_KEY and BINANCE_API_SECRET in your .env file.")
        sys.exit(1)

    log.info("Testing connectivity to Binance Futures...")
    if not await client.get_ping():
        log.critical("❌ Cannot connect to Binance Futures API. Check connection or endpoints.")
        sys.exit(1)
    log.info("✅ Connectivity test passed.")

    log.info(f"Synchronizing leverage settings on exchange...")
    await client.set_leverage(config.SYMBOL, config.LEVERAGE)

    if "--dry-run" in sys.argv:
        log.info("🧪 [DRY RUN] Verification completed successfully. Exiting.")
        return

    log.info("🚀 Starting main scanning loop...")
    while True:
        try:
            should_continue = await run_scan_cycle(client)
            if not should_continue:
                break
        except Exception as e:
            log.error(f"Error in scan cycle: {e}", exc_info=True)
            
        await asyncio.sleep(config.SCAN_INTERVAL_S)

if __name__ == "__main__":
    asyncio.run(main())
