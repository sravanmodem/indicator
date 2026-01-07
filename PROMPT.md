# NIFTY Options Indicator - Project Generation Prompt

Use this prompt to regenerate the entire project from scratch.

---

## Project Overview

Create an institutional-grade options trading signal system for Indian indices (NIFTY, Bank NIFTY, SENSEX) using Python FastAPI with real-time Zerodha Kite integration. The system should generate CE (Call) and PE (Put) trading signals based on multiple technical indicators.

## Tech Stack

- **Backend**: FastAPI (Python 3.12+)
- **Data Processing**: Pandas, NumPy
- **Technical Analysis**: ta library, SciPy
- **Frontend**: Jinja2 Templates, HTMX, Alpine.js, TailwindCSS (CDN)
- **Broker Integration**: Zerodha Kite Connect API
- **Database**: SQLite with aiosqlite (async)
- **Logging**: Loguru
- **Configuration**: Pydantic Settings with .env support

## Project Structure

```
indicator/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app with lifespan
│   ├── api/
│   │   ├── __init__.py
│   │   ├── auth.py               # Zerodha OAuth endpoints
│   │   ├── htmx.py               # HTMX partial endpoints
│   │   ├── market.py             # Market data API
│   │   └── signals.py            # Signal API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Pydantic Settings + constants
│   │   ├── logging.py            # Loguru setup
│   │   └── security.py           # Token encryption/storage
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── trend.py              # SuperTrend, EMA, VWAP, ADX
│   │   ├── momentum.py           # RSI, MACD, Stochastic, CCI
│   │   ├── volatility.py         # Bollinger Bands, ATR
│   │   ├── volume.py             # OBV
│   │   ├── options.py            # PCR, Max Pain, OI, GEX, VIX
│   │   ├── pivots.py             # Pivot Points, CPR, Camarilla
│   │   └── greeks.py             # Options Greeks (Delta, Gamma, Theta, Vega)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── zerodha_auth.py       # Authentication service
│   │   ├── data_fetcher.py       # Historical data & option chain
│   │   ├── websocket_manager.py  # Real-time WebSocket feed
│   │   └── signal_engine.py      # Signal generation engine
│   ├── templates/
│   │   ├── base.html             # Base template with nav
│   │   ├── index.html            # Home/login page
│   │   ├── dashboard.html        # Main trading dashboard
│   │   └── partials/
│   │       ├── auth_status.html
│   │       ├── signal_card.html
│   │       ├── indicator_panel.html
│   │       ├── option_chain_table.html
│   │       ├── market_overview.html
│   │       ├── recommended_option.html
│   │       ├── ws_status.html
│   │       ├── live_ticks.html
│   │       └── toast.html
│   └── static/                   # Static assets
├── data/                         # Token storage, DB
├── logs/                         # Log files
├── requirements.txt
├── run.py                        # Dev runner
├── start.py                      # Production starter
└── .env                          # Environment variables
```

## Core Features to Implement

### 1. Configuration (app/core/config.py)

```python
# Settings class with:
- kite_api_key, kite_api_secret (Zerodha credentials)
- app_secret_key (for token encryption)
- app_host (default: "0.0.0.0"), app_port (default: 8000)
- debug mode
- database_url (SQLite)
- log_level

# Constants:
- NIFTY_LOT_SIZE = 25
- BANKNIFTY_LOT_SIZE = 15
- SENSEX_LOT_SIZE = 10
- Index tokens: NIFTY_INDEX_TOKEN = 256265, BANKNIFTY_INDEX_TOKEN = 260105, SENSEX_INDEX_TOKEN = 265

# Indicator parameters:
INDICATOR_PARAMS = {
    "supertrend": {"period": 10, "multiplier": 3.0},
    "ema_fast": 9, "ema_slow": 21, "ema_trend": 50,
    "rsi_period": 14,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "adx_period": 14,
    "bollinger_period": 20, "bollinger_std": 2.0,
    "atr_period": 14,
}

# Signal thresholds:
SIGNAL_THRESHOLDS = {
    "adx_trend": 25, "adx_strong": 40,
    "rsi_oversold": 30, "rsi_overbought": 70,
    "pcr_oversold": 1.3, "pcr_overbought": 0.7,
    "vix_low": 12, "vix_high": 25,
}
```

### 2. Technical Indicators (app/indicators/)

#### trend.py
- **SuperTrend**: ATR-based trend indicator (returns direction, bands)
- **EMA Crossover System**: Fast(9)/Slow(21)/Trend(50) with crossover detection
- **VWAP**: Volume Weighted Average Price with standard deviation bands
- **ADX**: Average Directional Index with +DI/-DI and trend strength classification

#### momentum.py
- **RSI**: With oversold/overbought zones and divergence detection
- **MACD**: With histogram, crossovers, and zero-line crosses
- **Stochastic**: %K/%D with crossover signals in zones
- **CCI**: With breakout detection

#### volatility.py
- **Bollinger Bands**: With squeeze detection and %B
- **ATR**: With volatility state (low/normal/high/extreme)
- ATR-based stop loss and target calculators

#### options.py
- **PCR (Put-Call Ratio)**: With sentiment classification
- **Max Pain**: Calculate strike where buyers lose most
- **OI Analysis**: Long buildup, short covering detection, call/put walls
- **GEX (Gamma Exposure)**: Market maker hedging behavior
- **VIX Analysis**: Volatility regime classification

#### pivots.py
- **Standard Pivot Points**: P, R1-R3, S1-S3
- **CPR (Central Pivot Range)**: TC, BC, Pivot
- **Camarilla**: H1-H4, L1-L4 with breakout zones

#### greeks.py
- **Black-Scholes Greeks**: Delta, Gamma, Theta, Vega calculation
- **Expected price calculation**: At target, at stop loss, tomorrow (theta decay)
- **Implied Volatility estimation**: From ATM option premiums

### 3. Signal Engine (app/services/signal_engine.py)

```python
# Enums:
class SignalType(Enum):
    STRONG_CE, CE, WEAK_CE, NEUTRAL, WEAK_PE, PE, STRONG_PE

class TradingStyle(Enum):
    SCALPING, INTRADAY, SWING

# Dataclasses:
@dataclass
class IndicatorSignal:
    name, value, signal (ce/pe/neutral), strength (0-1), reason

@dataclass
class RecommendedOption:
    symbol, strike, option_type, ltp, oi, volume, bid, ask, reason
    # Greeks: delta, gamma, theta, vega
    # Expected prices: expected_at_target, expected_at_stop, price_tomorrow

@dataclass
class TradeSignal:
    timestamp, instrument, signal_type, direction, confidence
    entry_price, stop_loss, target_1, target_2
    indicators: list[IndicatorSignal]
    supporting_factors, warning_factors
    risk_reward, trading_style
    recommended_option: RecommendedOption | None

# Signal Engine analyze() method:
1. Calculate all technical indicators
2. Score each indicator for CE/PE bias
3. Aggregate scores with weights
4. Determine signal type based on confidence difference
5. Calculate ATR-based stops and targets
6. Find best option to buy based on trading style:
   - Scalping: ATM or 1 ITM
   - Intraday: ATM or 1 OTM
   - Swing: 1-2 OTM
7. Calculate Greeks and expected prices for recommended option
```

### 4. Zerodha Integration

#### zerodha_auth.py
- OAuth flow: login URL generation -> callback with request_token -> exchange for access_token
- Encrypted token storage (using cryptography library)
- Session restoration on startup
- Logout and token invalidation

#### data_fetcher.py
- Fetch instruments list (NFO, BFO exchanges)
- Get options for NIFTY/BANKNIFTY/SENSEX by expiry and strike range
- Fetch historical OHLCV data
- Build option chain with OI, volume, bid/ask from quotes
- Calculate PCR from chain data

#### websocket_manager.py
- Connect to Kite WebSocket for real-time ticks
- Subscribe to index tokens
- Store latest tick data
- Connection status tracking

### 5. API Endpoints

#### auth.py
```
GET /auth/login -> Redirect to Zerodha login
GET /zerodha/callback -> Handle OAuth callback
GET /auth/logout -> Clear session
GET /auth/status -> Return auth status JSON
```

#### htmx.py (Server-side rendered partials)
```
GET /htmx/auth-status -> Auth status partial
GET /htmx/signal-card/{index}?style={style} -> Signal card with recommendation
GET /htmx/indicator-panel/{index} -> Technical indicator values
GET /htmx/option-chain-table/{index} -> Option chain table
GET /htmx/market-overview -> NIFTY/BANKNIFTY/SENSEX prices
GET /htmx/ws-status -> WebSocket connection status
GET /htmx/recommended-option/{index}?style={style} -> Option recommendation with Greeks
GET /htmx/live-ticks -> Live tick data table
```

#### market.py
```
GET /api/market/quote/{index} -> Current quote
GET /api/market/historical/{index} -> Historical data
GET /api/market/option-chain/{index} -> Full option chain
```

### 6. Frontend Templates

#### base.html
- Dark theme with TailwindCSS
- Glass morphism cards
- Custom colors: ce (green), pe (red), surface (slate)
- Navigation with logo, market overview, auth status
- HTMX + Alpine.js integration

#### dashboard.html
- Index tabs (NIFTY, Bank NIFTY, SENSEX)
- Trading style selector (Scalping, Intraday, Swing)
- Signal card with CE/PE recommendation and confidence
- Recommended option card with Greeks display
- Technical indicator panel
- Option chain table
- Auto-refresh via HTMX polling

#### Signal Card Design
- Large CE/PE badge with color (green/red)
- Confidence percentage
- Entry, Stop Loss, Target 1, Target 2 prices
- Risk:Reward ratio
- Supporting factors (bullish signals)
- Warning factors (caution signals)
- Individual indicator signals list

#### Recommended Option Card
- Option symbol and strike
- Premium (LTP) with bid/ask spread
- Greeks display: Delta (with %), Gamma, Theta (with daily decay), Vega
- Expected prices: At target, At stop loss, Tomorrow
- OI and Volume metrics

### 7. Trading Logic

#### Signal Weighting
- Trend indicators: 40% (SuperTrend, EMA, VWAP, ADX)
- Momentum indicators: 30% (RSI, MACD, Stochastic)
- Options flow: 20% (PCR, Max Pain, OI)
- Risk factors: 10% (VIX, Pivots)

#### Signal Strength
- confidence_diff > 40: STRONG signal
- confidence_diff > 20: Normal signal
- confidence_diff > 10: WEAK signal
- confidence_diff < 10: NEUTRAL

#### Stop Loss & Targets
- Based on ATR with multiplier
- Scalping: 1.5x ATR stop
- Intraday: 2.0x ATR stop
- Target 1: 1.5x stop distance
- Target 2: 2.5x stop distance

### 8. Option Selection Criteria

```python
# For recommended option:
1. Find ATM strike from spot price
2. Select target strikes based on trading style
3. Filter by:
   - OI >= 10,000 (liquidity)
   - LTP >= 5 (avoid illiquid)
   - Reasonable bid-ask spread
4. Score based on:
   - Spread score (30%)
   - OI score (30%)
   - Volume score (20%)
   - ATM bonus (20%)
5. Calculate Greeks using Black-Scholes
6. Calculate expected prices using Delta
```

### 9. Greeks Calculation

```python
# Black-Scholes Greeks:
d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
d2 = d1 - σ√T

Delta (CE) = N(d1)
Delta (PE) = N(d1) - 1
Gamma = N'(d1) / (Sσ√T)
Theta (CE) = -Sσ N'(d1)/(2√T) - rK e^(-rT) N(d2)
Vega = S√T N'(d1)

# Expected price at target:
expected = current_premium + (delta * (target - spot))

# Tomorrow's price (theta decay):
price_tomorrow = current_premium + theta
```

## Environment Variables (.env)

```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
APP_SECRET_KEY=your_secret_key_for_encryption
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
LOG_LEVEL=INFO
```

## Requirements.txt

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
kiteconnect>=5.0.1
websockets>=12.0
pandas>=2.2.0
numpy>=1.26.0
ta>=0.11.0
scipy>=1.12.0
python-dotenv>=1.0.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
jinja2>=3.1.3
aiohttp>=3.9.0
httpx>=0.26.0
aiosqlite>=0.19.0
cryptography>=42.0.0
loguru>=0.7.2
orjson>=3.9.0
```

## Key Implementation Notes

1. **All indicators return dataclasses** with computed values and signals
2. **Use async/await** for all Zerodha API calls (wrap sync calls with asyncio.to_thread)
3. **Singleton pattern** for services (auth, data_fetcher, signal_engine, ws_manager)
4. **HTMX partials** for real-time UI updates without full page reload
5. **Alpine.js** for client-side state (active tab, trading style)
6. **Token encryption** using Fernet (cryptography library)
7. **Loguru** for structured logging with file rotation
8. **Error handling** with graceful fallbacks (show "Market closed" when data unavailable)

## UI/UX Requirements

1. **Dark theme** optimized for trading (reduce eye strain)
2. **Mobile responsive** - dashboard works on all screen sizes
3. **Color coding**: Green for CE/bullish, Red for PE/bearish, Slate for neutral
4. **Real-time updates**: Auto-refresh signal card every second during market hours
5. **Clear visual hierarchy**: Signal direction is the most prominent element
6. **Professional typography**: Inter for UI, JetBrains Mono for numbers/prices

## Testing the Application

1. Start with `python run.py`
2. Navigate to http://localhost:8000
3. Click "Login with Zerodha"
4. After callback, redirected to dashboard
5. Select index (NIFTY/BANKNIFTY/SENSEX)
6. Select trading style (Scalping/Intraday/Swing)
7. View generated signal with recommended option

---

This prompt should provide enough context to regenerate the entire project with all features intact.
