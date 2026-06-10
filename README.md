# Flipper Bot
### $10 → $100 | Smart Money Concepts + AI Predictive Filtering + Growth Mode Compounding

A fully autonomous Linux-native trading bot designed for Deriv Synthetic Indices (specifically **Volatility 75** and **Volatility 100**). It uses Smart Money Concepts (FVG + Order Blocks) for signal generation, a **Random Forest AI model** for high-probability trade filtering, and a dual-mode risk management system for account compounding and drawdown survival.

---

## Growth Mode & Risk Management Architecture

The bot has been transitioned into **Growth Mode** to compound the account balance up to $20.00 (and eventually to the $100.00 target). This is governed by three core pillars:

### 1. Threshold Compounding
To protect the balance when the account is low, the bot runs a threshold gate:
* **Account Flip Mode (`balance < $5.00`):** The stake is hardcoded at exactly **`$1.00`** to survive variance and drawdowns.
* **Growth Mode (`balance >= $5.00`):** The bot unlocks dynamic position sizing, setting the stake to **`20%` of the account balance** (`balance * 0.20`).

### 2. Fixed 1:3 Asymmetric Risk-to-Reward
To ensure that winning trades rapidly recover losing streaks, the bot enforces an asymmetric payout:
* **Stop Loss (SL):** Calculated dynamically based on the SMC Order Block invalidation level.
* **Take Profit (TP):** Forced to be exactly **`3.0 * absolute_sl_amount`** (1:3 Risk-to-Reward). A single win recovers three typical losses.

### 3. Upgraded AI Confidence Filter
* Every setup is passed to a Random Forest model trained on candle structure, momentum (RSI), volatility (ATR), and trend context.
* To protect the compounding phase, the minimum win probability required to approve a trade is set to **`45%`** (vetoing any setup where `win_prob < 45%`).

---

## Directory Layout

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
    ├── risk_manager.py      # Sizing & asymmetric R:R calculator
    ├── order_executor.py    # Deriv contract execution
    └── logger.py            # CSV trade journal & logging
```

---

## Setup & Execution

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
```env
DERIV_API_TOKEN=your_token_here
DERIV_APP_ID=your_app_id
```

### 3. Running the Bot
To start the bot in the background:
```bash
nohup python3 main.py > bot_stdout.log 2>&1 &
```

---

## Key Parameters (`config.py`)

| Parameter | Current Value | Description |
|---|---|---|
| `SYMBOLS` | `["R_75", "R_100"]` | Tradable synthetic indices |
| `AI_VETO_THRESHOLD` | `0.45` | Minimum win probability required (45%) |
| `KELLY_FRACTION` | `0.08` | Fractional Kelly (8% risk per trade fallback) |
| `MAX_STAKE_PCT_OF_BALANCE` | `0.50` | Maximum allocation allowed across open trades |
| `ZONE_TTL_S` | `14400` | Zone Time-To-Live (4 hours before expiry) |
| `SCAN_INTERVAL_S` | `30` | Seconds between scan cycles |

---

## Risk Disclosure
This bot uses aggressive position sizing and is designed for micro-accounts. While the AI filter and dynamic stop-loss preserve capital during drawdown, synthetic indices are highly leveraged and volatile. **Use at your own risk.**
