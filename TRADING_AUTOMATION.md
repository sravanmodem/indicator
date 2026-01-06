# High-Quality Signal Filtering & Automated Trading System

## Overview

This system implements institutional-grade signal quality filtering and automated order placement targeting **₹25,000 daily profit** with robust risk management.

## Key Features

### 1. Signal Quality Scoring System

Every signal is scored 0-100 based on multi-factor analysis:

**Scoring Breakdown (100 points total):**
- **Trend Strength (30 points):** SuperTrend alignment, EMA alignment, ADX strength
- **Momentum Confirmation (20 points):** RSI position, MACD alignment, Stochastic confirmation
- **Volume Analysis (15 points):** Above-average volume, volume trend
- **Greeks Quality (20 points):** Delta range, Theta acceptability, Liquidity (bid-ask spread)
- **PCR Analysis (10 points):** Market sentiment alignment
- **Risk/Reward (5 points):** R:R ratio quality

**Quality Threshold:**
- **High Quality:** Score ≥ 70/100 ⭐
- **Needs Review:** Score < 70/100 ⚠️

Only high-quality signals (≥70) are recommended for trading.

### 2. Automated Order Placement

**Features:**
- Market and limit order support
- Bracket orders (Entry + SL + Target)
- Order modification and cancellation
- Position tracking
- Error handling with retry logic

**Integration:**
- Zerodha Kite Connect API
- NFO options trading
- MIS product (Intraday)

### 3. Risk Management System

**Capital Requirements:**
- Recommended: ₹5,00,000 for ₹25,000 daily target
- Risk per trade: 2% max (₹10,000)
- Daily loss limit: 3% max (₹15,000)
- Max concurrent positions: 5

**Position Sizing:**
- Automatic calculation based on risk parameters
- Capital allocation limit: 20% per position
- Minimum R:R requirement: 1.5:1
- Greeks-based validation

**Daily P&L Tracking:**
- Real-time profit/loss monitoring
- Auto-exit when ₹25k target reached ✅
- Auto-stop when 3% loss limit hit 🛑
- Win rate, profit factor, largest win/loss tracking

### 4. Trading Automation Modes

**Three Operating Modes:**

1. **MANUAL Mode** (Default)
   - Shows high-quality signals only
   - No automatic order placement
   - User manually places trades

2. **SEMI-AUTO Mode**
   - Shows high-quality signals
   - Requires user confirmation before placing orders
   - Pending signals queue

3. **FULL-AUTO Mode** (Target: 25k daily)
   - Fully automated signal scanning
   - Automatic order placement on high-quality signals
   - Automatic position sizing
   - Automatic exit on target/loss limits

## System Architecture

### Core Services

**1. Signal Quality Analyzer** (`app/services/signal_quality.py`)
- Multi-factor scoring algorithm
- Quality threshold enforcement
- Factor breakdown and warnings

**2. Order Manager** (`app/services/order_manager.py`)
- Kite Connect integration
- Order lifecycle management
- Bracket order support

**3. Risk Manager** (`app/services/risk_manager.py`)
- Position sizing calculator
- Daily limit enforcement
- P&L tracking
- Capital requirement analysis

**4. Trading Automation** (`app/services/trading_automation.py`)
- Index scanning (NIFTY, BANKNIFTY, SENSEX)
- Signal-to-order pipeline
- Mode switching (Manual/Semi/Full)
- Execution tracking

## Usage

### Basic Setup

```python
from app.services.trading_automation import (
    TradingAutomation,
    TradingConfig,
    AutomationMode
)
from app.services.signal_engine import TradingStyle

# Configure trading automation
config = TradingConfig(
    mode=AutomationMode.MANUAL,  # Start with manual
    capital=500000,  # 5 lakh
    trading_style=TradingStyle.INTRADAY,
    indices_to_trade=["NIFTY", "BANKNIFTY"],
    auto_exit_on_target=True,
    auto_stop_on_loss=True,
)

# Initialize automation
automation = TradingAutomation(config)
```

### Scanning for High-Quality Signals

```python
# Scan all configured indices
executions = await automation.scan_and_execute()

for execution in executions:
    if execution.quality_score.is_high_quality:
        print(f"HIGH QUALITY: {execution.signal.direction}")
        print(f"Score: {execution.quality_score.total_score}/100")
        print(f"Factors: {execution.quality_score.factors}")
```

### Manual Trading Workflow

```python
# 1. Get signal with quality score
signal = signal_engine.analyze(df, option_chain)
quality_score = quality_analyzer.analyze_quality(signal)

# 2. Check if high quality
if quality_score.is_high_quality:
    print(f"✅ HIGH QUALITY ({quality_score.total_score}/100)")

    # 3. Validate risk constraints
    can_trade, reason = risk_manager.can_take_trade(signal)

    if can_trade:
        # 4. Calculate position size
        position = risk_manager.calculate_position_size(
            signal=signal,
            entry_price=signal.recommended_option.ltp,
            stop_loss=signal.recommended_option.expected_at_stop
        )

        print(f"Quantity: {position.quantity}")
        print(f"Risk: ₹{position.risk_amount:,.0f}")
        print(f"Potential: ₹{position.potential_profit:,.0f}")

        # 5. Place order (manual confirmation)
        order_result = order_manager.place_signal_order(
            signal=signal,
            quantity=position.quantity,
            with_bracket=True
        )
```

### Automated Trading (Full-Auto Mode)

```python
# Switch to full automation
automation.config.mode = AutomationMode.FULL_AUTO

# Run continuous scanning (in background)
while True:
    # Check if daily target reached
    summary = automation.get_daily_summary()

    if summary["target_reached"]:
        print("🎯 Daily target reached!")
        break

    if summary["max_loss_hit"]:
        print("🛑 Loss limit hit - Stopped trading")
        break

    # Scan and execute
    await automation.scan_and_execute()

    # Wait before next scan
    await asyncio.sleep(60)  # Scan every minute
```

### Daily Summary

```python
summary = automation.get_daily_summary()

print(f"Date: {summary['date']}")
print(f"Total Trades: {summary['total_trades']}")
print(f"Win Rate: {summary['win_rate']}")
print(f"Net P&L: ₹{summary['net_pnl']:,.0f}")
print(f"Progress: {summary['progress_pct']:.1f}% of ₹25k target")
print(f"Active Positions: {summary['active_positions']}")
```

## Dashboard Integration

The system is fully integrated into the web dashboard:

### Visual Indicators

1. **Signal Card Quality Badge**
   - Shows quality score prominently
   - Green badge for high quality (≥70)
   - Amber badge for needs review (<70)
   - Breakdown: Trend, Momentum, Greeks scores

2. **Recommended Option Quality Section**
   - ⭐ HIGH QUALITY badge for scores ≥70
   - Quality factors list with checkmarks
   - Score breakdown visualization
   - Real-time quality assessment

3. **Color Coding**
   - Green: High-quality signals, profitable targets
   - Red: Low-quality signals, stop losses
   - Amber: Medium quality, needs review
   - White/Gray: Neutral indicators

## Risk Management Rules

### Per-Trade Limits
- ✅ Max risk: 2% of capital (₹10,000 on ₹5L)
- ✅ Min R:R ratio: 1.5:1
- ✅ Max capital per position: 20%
- ✅ Delta range: 0.30-0.75 (avoid deep OTM/ITM)
- ✅ Min OI: 10,000 contracts
- ✅ Min premium: ₹5

### Daily Limits
- ✅ Profit target: ₹25,000 (auto-exit all)
- ✅ Max loss: 3% of capital (₹15,000 - auto-stop)
- ✅ Max positions: 5 concurrent
- ✅ Win rate target: >50%

### Signal Quality Gates
- ✅ Confidence: ≥60%
- ✅ Quality score: ≥70/100
- ✅ Trend alignment: Strong
- ✅ Momentum confirmation: Present
- ✅ Greeks profile: Acceptable

## Expected Performance

### Conservative Estimates
- **Capital:** ₹5,00,000
- **Daily Target:** ₹25,000 (5% of capital)
- **Risk per trade:** ₹10,000 (2%)
- **Win rate:** 50-60%
- **Avg R:R:** 2:1
- **Trades per day:** 3-5

### Example Day
```
Trade 1: +₹12,000 (Win)
Trade 2: -₹4,000 (Loss)
Trade 3: +₹15,000 (Win)
Trade 4: +₹8,000 (Win) - Target reached
---
Total: +₹31,000 ✅ (Target exceeded, auto-stopped)
Win rate: 75% (3/4)
Largest win: ₹15,000
```

## Safety Features

1. **Multiple Quality Checks**
   - Signal must pass quality threshold
   - Risk manager must approve
   - Position sizing must succeed
   - Order validation before placement

2. **Automatic Safeguards**
   - Daily loss limit enforcement
   - Profit target auto-exit
   - Position limit enforcement
   - Capital allocation limits

3. **Error Handling**
   - Order placement retry logic
   - Connection failure recovery
   - Market closed detection
   - Invalid data rejection

## Configuration Files

**Risk Parameters** (`app/services/risk_manager.py`):
```python
params = RiskParameters(
    total_capital=500000,
    daily_profit_target=25000,
    max_risk_per_trade_pct=2.0,
    max_daily_loss_pct=3.0,
    max_positions=5,
    min_risk_reward=1.5,
)
```

**Quality Thresholds** (`app/services/signal_quality.py`):
```python
HIGH_QUALITY_THRESHOLD = 70  # 70/100 minimum score
```

## Monitoring & Logging

All trading activity is logged:
- Signal generation with quality scores
- Order placement attempts and results
- Position entries and exits
- P&L updates
- Daily summaries

**Log Example:**
```
INFO: NIFTY Signal: CE | Confidence: 78.3% | Quality Score: 74.5/100
INFO: ✅ NIFTY: HIGH QUALITY SIGNAL DETECTED!
INFO: Quality Factors: Strong trend alignment, Strong momentum confirmation
INFO: NIFTY Position: 50 contracts | Risk: ₹8,500 | Potential: ₹17,000 | R:R: 2.00
INFO: 🚀 NIFTY: Placing order...
INFO: ✅ NIFTY: Order placed successfully - 240106000123456
```

## Testing

Before enabling FULL_AUTO mode:

1. **Test in MANUAL mode** - Verify signal quality scoring
2. **Test in SEMI_AUTO mode** - Verify order placement with confirmation
3. **Paper trade** - Test with small quantities
4. **Monitor carefully** - Watch first few days closely
5. **Adjust parameters** - Fine-tune based on results

## Support

For issues or questions about the trading automation system, refer to:
- Code documentation in respective service files
- Log files in `logs/app.log`
- Dashboard real-time monitoring

---

**⚠️ IMPORTANT DISCLAIMER:**

This is a fully automated trading system. While it includes robust risk management, trading involves inherent risks. Never invest more than you can afford to lose. Past performance does not guarantee future results. Always test thoroughly before deploying with real capital.

**Start with MANUAL mode and small capital to understand the system before scaling up.**
