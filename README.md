# NIFTY Options Indicator

Institutional-grade CE/PE trading signal system for NIFTY, Bank NIFTY, and SENSEX options trading. Built with FastAPI and real-time Zerodha Kite integration.

## Features

- **Real-time Trading Signals**: Generate CE (Call) and PE (Put) signals with confidence scores
- **Multiple Indices**: Support for NIFTY, Bank NIFTY, and SENSEX
- **Trading Styles**: Scalping, Intraday, and Swing trading modes
- **Technical Indicators**:
  - Trend: SuperTrend, EMA Crossover, VWAP, ADX
  - Momentum: RSI, MACD, Stochastic
  - Volatility: Bollinger Bands, ATR, VIX Analysis
  - Volume: OBV
- **Options Analytics**:
  - Put-Call Ratio (PCR)
  - Max Pain Analysis
  - Open Interest Analysis
  - Greeks Calculation (Delta, Gamma, Theta, Vega)
- **Support/Resistance Levels**: Pivot Points, CPR, Camarilla
- **Live WebSocket Feed**: Real-time price updates via Zerodha Kite
- **Responsive Dashboard**: Mobile-optimized UI with HTMX

## Tech Stack

- **Backend**: FastAPI, Python 3.12+
- **Data Processing**: Pandas, NumPy
- **Technical Analysis**: TA-Lib (ta), SciPy
- **Frontend**: Jinja2 Templates, HTMX, Alpine.js, TailwindCSS
- **Broker Integration**: Zerodha Kite Connect API
- **Database**: SQLite (aiosqlite)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd indicator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your Zerodha credentials:
```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
APP_SECRET_KEY=your_secret_key
DEBUG=false
```

## Usage

Start the application:
```bash
python run.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access the dashboard at `http://localhost:8000`

## Project Structure

```
indicator/
├── app/
│   ├── api/              # API routes
│   │   ├── auth.py       # Authentication endpoints
│   │   ├── htmx.py       # HTMX partial endpoints
│   │   ├── market.py     # Market data endpoints
│   │   ├── signals.py    # Signal endpoints
│   │   └── paper_trading.py  # Paper trading API
│   ├── core/             # Core configuration
│   │   ├── config.py     # Settings and constants
│   │   ├── logging.py    # Logging setup
│   │   └── security.py   # Security utilities
│   ├── indicators/       # Technical indicators
│   │   ├── trend.py      # SuperTrend, EMA, VWAP, ADX
│   │   ├── momentum.py   # RSI, MACD, Stochastic
│   │   ├── volatility.py # Bollinger Bands, ATR
│   │   ├── volume.py     # OBV
│   │   ├── options.py    # PCR, Max Pain, OI Analysis
│   │   ├── pivots.py     # Pivot Points, CPR, Camarilla
│   │   └── greeks.py     # Options Greeks
│   ├── services/         # Business logic
│   │   ├── data_fetcher.py      # Market data fetching
│   │   ├── signal_engine.py     # Signal generation
│   │   ├── paper_trading.py     # Paper trading service
│   │   ├── websocket_manager.py # WebSocket handling
│   │   └── zerodha_auth.py      # Zerodha authentication
│   ├── templates/        # Jinja2 templates
│   │   ├── paper_trading.html           # Paper trading dashboard
│   │   ├── paper_strategy_5_percent.html   # 5% Daily strategy
│   │   ├── paper_strategy_15_minute.html   # 15-Minute strategy
│   │   ├── paper_strategy_expiry_day.html  # Expiry Day strategy
│   │   └── partials/
│   │       └── paper_order_history.html    # Order history partial
│   └── main.py           # FastAPI app entry
├── data/                 # Data storage
│   └── paper_trading/    # Paper trading data files
│       ├── positions_*.json    # Strategy positions
│       └── orders_*.json       # Strategy order history
├── logs/                 # Application logs
├── requirements.txt
├── run.py               # Application runner
└── start.py             # Production starter
```

## Signal Generation

The signal engine combines multiple indicators to generate trading signals:

1. **Trend Analysis** (40% weight)
   - SuperTrend direction
   - EMA crossover system (9/21/50)
   - VWAP position
   - ADX trend strength

2. **Momentum Analysis** (30% weight)
   - RSI levels and divergences
   - MACD crossovers
   - Stochastic signals

3. **Options Flow** (20% weight)
   - Put-Call Ratio sentiment
   - Max Pain magnet effect
   - OI-based support/resistance

4. **Risk Management** (10% weight)
   - ATR-based stop loss
   - VIX regime analysis
   - Pivot level proximity

## Configuration

Key settings in `app/core/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_timeframe` | 5minute | Chart timeframe |
| `max_positions` | 5 | Maximum concurrent positions |
| `risk_per_trade` | 0.01 | Risk per trade (1%) |

Indicator parameters and signal thresholds can be customized in the same file.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Trading dashboard |
| `/health` | GET | Health check |
| `/auth/login` | GET | Zerodha login redirect |
| `/zerodha/callback` | GET | OAuth callback |
| `/api/signals/{index}` | GET | Get current signal |
| `/api/market/quote/{index}` | GET | Get market quote |

## Paper Trading System

The application includes a comprehensive paper trading system for testing strategies without real money.

### Features

- **Capital Management**: ₹1,00,000 virtual capital
- **Multiple Strategies**: Each strategy operates independently
- **Broker Charges**: Realistic charge calculation (brokerage, STT, GST, etc.)
- **Order History**: Track all trades with P&L

### Trading Strategies

| Strategy | Trading Window | Description |
|----------|---------------|-------------|
| **5% Daily** | 9:15 AM - 3:30 PM | Target 5% daily return |
| **15-Minute** | 9:15 AM - 3:30 PM | Quick 15-minute trades |
| **Expiry Day** | 1:00 PM - 3:00 PM | Expiry day only trades |

### Smart Entry Price Logic

The system uses **Swing Trading Analysis** for optimal entry:

```
1. Analyze last 30 minutes of market data
2. Calculate swing high/low from recent candles
3. Determine position in swing range

For CE (Call) signals:
- Don't enter if price is falling
- Wait for reversal confirmation
- Best entry near swing low when rising

For PE (Put) signals:
- Don't enter if price is rising
- Wait for reversal confirmation
- Best entry near swing high when falling
```

### Stop Loss & Target Calculation

- **Delta-based SL**: 12-20% based on option delta
- **Risk/Reward**: 1:2.5 ratio for targets
- **ATM Options**: 15% SL
- **OTM Options**: 18-20% SL (higher risk)
- **ITM Options**: 12% SL (lower risk)

### Trailing Stop Loss

All exits happen through stop loss only:

```
1. At 50% of target reached → Move SL to breakeven (entry price)
2. Every 10% profit above 50% → Trail SL by 10%
3. When target achieved → Set SL at target price and wait

Example:
- Entry: ₹100, Target: ₹125 (25% profit)
- At ₹112.50 (50% of target): SL moves to ₹100 (breakeven)
- At ₹115 (60%): SL moves to ₹110
- At ₹120 (80%): SL moves to ₹115
- At ₹125 (target): SL set at ₹125, continue if trending
```

### API Endpoints - Paper Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/paper/trading` | GET | Main paper trading dashboard |
| `/paper/strategy/5-percent-daily` | GET | 5% Daily strategy page |
| `/paper/strategy/15-minute` | GET | 15-Minute strategy page |
| `/paper/strategy/expiry-day` | GET | Expiry Day strategy page |
| `/api/paper/execute` | POST | Execute signal as paper trade |
| `/api/paper/positions` | GET | Get current positions |
| `/api/paper/orders` | GET | Get order history |
| `/api/paper/metrics` | GET | Get performance metrics |

### Broker Charges Calculation

Realistic charge calculation per trade:

| Charge Type | Rate |
|-------------|------|
| Brokerage | ₹20 per order |
| STT | 0.0625% on sell |
| Exchange Txn | 0.053% |
| SEBI | 0.0001% |
| Stamp Duty | 0.003% on buy |
| GST | 18% on (brokerage + txn + SEBI) |

## Configuration

Key settings in `app/core/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_timeframe` | 5minute | Chart timeframe |
| `max_positions` | 5 | Maximum concurrent positions |
| `risk_per_trade` | 0.01 | Risk per trade (1%) |

Indicator parameters and signal thresholds can be customized in the same file.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Trading dashboard |
| `/health` | GET | Health check |
| `/auth/login` | GET | Zerodha login redirect |
| `/zerodha/callback` | GET | OAuth callback |
| `/api/signals/{index}` | GET | Get current signal |
| `/api/market/quote/{index}` | GET | Get market quote |

## License

MIT License
