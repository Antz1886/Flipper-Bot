import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SYMBOLS, PRIMARY_TF_S, HTF_TF_S
from modules.strategy_manager import TrendStrategy
from modules.trend_engine import TrendEngine
from modules.volatility_filter import VolatilityFilter
from modules.signal_validator import SignalValidator
from modules.risk_manager import can_open_position, calculate_stake
from modules.trade_scoring import TradeScoring
from modules.backtester import Backtester

# Mock risk manager with can_open_position mapping
class MockRiskManager:
    def can_open_position(self, signal):
        return True

def run_backtest():
    print("Initializing Flipper-Bot v3.0 Backtest Engine...")
    
    # 1. Check/Prepare data directory and files
    os.makedirs("data", exist_ok=True)
    
    # Create synthetic OHLC data for testing if not present
    for symbol in SYMBOLS:
        csv_path = f"data/{symbol}.csv"
        if not os.path.exists(csv_path):
            print(f"Generating synthetic OHLC data for {symbol} at {csv_path}...")
            np.random.seed(42)
            n_bars = 1000
            close_prices = 1000.0 + np.cumsum(np.random.normal(0, 5, n_bars))
            high_prices = close_prices + np.random.uniform(0, 10, n_bars)
            low_prices = close_prices - np.random.uniform(0, 10, n_bars)
            open_prices = np.roll(close_prices, 1)
            open_prices[0] = close_prices[0]
            
            df = pd.DataFrame({
                "timestamp": pd.date_range(start="2026-01-01", periods=n_bars, freq="5min"),
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.randint(100, 1000, n_bars),
                "adx": np.random.uniform(15, 45, n_bars),
                "rsi": np.random.uniform(25, 75, n_bars),
                "ema_fast": pd.Series(close_prices).ewm(span=20).mean(),
                "ema_slow": pd.Series(close_prices).ewm(span=50).mean()
            })
            df.to_csv(csv_path, index=False)
            print(f"Generated {csv_path} successfully.")

    # 2. Setup modules
    # In order to run the TrendStrategy, we need configurations
    config_dict = {
        'FAST_EMA_PERIOD': 20,
        'SLOW_EMA_PERIOD': 50,
        'HTF_EMA_PERIOD': 200,
        'MIN_ADX_TREND': 20.0,
        'RSI_BUY_THRESHOLD': 45,
        'RSI_SELL_THRESHOLD': 55,
        'ATR_PERIOD': 14,
        'ATR_MEDIAN_WINDOW': 20,
        'TRADE_SCORE_BUCKETS': {
            'high': (90, 100),
            'mid': (80, 90),
            'low': (70, 80)
        },
        'SYMBOLS': SYMBOLS,
        'N_BARS': 200,
        'HOLDING_PERIOD_BARS': 10,
        'MULTIPLIER': 10
    }
    
    components = {
        "trend_engine": TrendEngine(config_dict),
        "volatility_filter": VolatilityFilter(config_dict),
        "signal_validator": SignalValidator(config_dict),
        "risk_manager": MockRiskManager(),
        "trade_scoring": TradeScoring(config_dict),
    }
    
    strategy = TrendStrategy(config_dict, components)
    components["strategy_manager"] = strategy
    
    # 3. Instantiate Backtester
    backtester = Backtester(config_dict, components)
    
    # 4. Run Backtest
    print("Running Backtest...")
    summary = backtester.run()
    
    print("\n================ BACKTEST RESULTS ================")
    print(f"Total PnL     : ${summary['total_pnl']:.2f}")
    print(f"Win Rate      : {summary['win_rate']*100:.2f}%")
    print(f"Profit Factor : {summary['profit_factor']:.3f}")
    print(f"Max Drawdown  : {summary['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio  : {summary['sharpe']:.3f}")
    print(f"Total Trades  : {summary['num_trades']}")
    print("==================================================")

if __name__ == "__main__":
    run_backtest()
