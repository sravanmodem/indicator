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
│   │   └── signals.py    # Signal endpoints
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
│   │   ├── websocket_manager.py # WebSocket handling
│   │   └── zerodha_auth.py      # Zerodha authentication
│   ├── templates/        # Jinja2 templates
│   └── main.py           # FastAPI app entry
├── data/                 # Data storage
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

## License

MIT License
