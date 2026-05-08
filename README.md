# Flipper Bot
### $10 → $100 | Smart Money Concepts + AI Predictive Filtering + Kelly Compounding

A fully autonomous Linux-native trading bot designed for Deriv Synthetic Indices (specifically **Volatility 75**). It uses Smart Money Concepts (FVG + Order Blocks) for signal generation, an **XGBoost/Random Forest AI model** for high-probability trade filtering, and dynamic **Full Kelly position sizing** for aggressive account growth.

---

## Architecture

```
small_account_bot/
├── main.py                  # Async orchestrator (entry point)
├── config.py                # Global configuration & risk parameters
├── ml_data_collector.py     # Extensive historical data fetcher & backtester
├── ml_trainer.py            # AI model training pipeline
├── models/
│   └── rf_setup_scorer.pkl  # Trained Random Forest model
├── ml_data/
│   └── historical_setups.csv# Generated dataset for training
├── .env                     # Deriv API credentials
├── requirements.txt         # Dependencies (includes scikit-learn & pandas_ta)
└── modules/
    ├── connector.py         # Deriv WebSocket API manager
    ├── data_feed.py         # OHLC & Tick data streams
    ├── smc_engine.py        # SMC Logic + AI Inference Engine
    ├── risk_manager.py      # Kelly Criterion & Stake calculator
    ├── order_executor.py    # Deriv contract execution
    └── logger.py            # CSV trade journal & logging
```

---

## New Feature: AI Phase 1 (Predictive Setup Scoring)
The bot now includes a machine learning confirmation layer. 
- **The Filter:** Every SMC signal is evaluated by a Random Forest model trained on 100,000+ historical candles.
- **Features:** The AI analyzes current Volatility (ATR), Momentum (RSI), and Trend Context (EMA 50) before approving a trade.
- **The Veto:** Even if a "textbook" SMC setup is found, the AI will veto the trade if the predicted win probability is < 60%, significantly reducing drawdown during choppy market regimes.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
Create a `.env` file from the template:
```bash
cp .env.template .env
```
Set your Deriv API Token (Trade + Read scope) and App ID:
```
DERIV_API_TOKEN=your_token_here
DERIV_APP_ID=your_app_id
```

### 3. Training the AI
To update the AI model with the latest market data:
1. Run `python ml_data_collector.py` to fetch fresh historical data.
2. Run `python ml_trainer.py` to retrain the model and save the new `.pkl` file.

---

## Run

```bash
python main.py
```

The bot will:
1. Establish a persistent WebSocket connection to Deriv.
2. Scan M5/M15 timeframes for unmitigated FVG + Order Block confluences.
3. Pass signals through the AI filter.
4. Execute trades using Multipliers with automated Kelly stake sizing.

---

## Key Parameters (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `SYMBOL` | `R_75` | Volatility 75 Index |
| `KELLY_FRACTION` | `0.22` | Full Kelly (22% of balance) |
| `RISK_REWARD_RATIO` | `2.0` | Minimum 1:2 R:R |
| `MULTIPLIER` | `50` | Deriv Multiplier setting |
| `SCAN_INTERVAL_S` | `30` | Seconds between scan cycles |

---

## Risk Disclosure
This bot uses **Full Kelly sizing** on a micro-account ($10). This is extremely aggressive. While the AI filter increases win rates, synthetic indices are highly volatile. **Use at your own risk.**
