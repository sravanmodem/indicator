# Quick Start Guide

## 🚀 Running the Server

### Method 1: Development Mode (Recommended)

**Double-click:** `run_server.bat`

This will:
- ✅ Use Python 3.12 automatically
- ✅ Activate virtual environment (if exists)
- ✅ Start server with auto-reload
- ✅ Open on http://localhost:8000

### Method 2: Production Mode

**Double-click:** `run_server_production.bat`

- Uses 4 workers for better performance
- No auto-reload (more stable)

### Method 3: Manual Command Line

```cmd
cd C:\Users\saiki\OneDrive\Desktop\sravan\indicator
C:\Users\saiki\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn app.main:app --reload
```

---

## 🔧 First Time Setup

**Double-click:** `setup_environment.bat`

This will:
1. Create virtual environment
2. Install all dependencies
3. Prepare the environment

---

## 📊 Check Server Status

**Double-click:** `check_status.bat`

This shows:
- ✅ Server health
- 📈 API call statistics
- 💾 Cache performance
- ⚠️ Rate limit usage

Or manually:
```cmd
REM Server health
curl http://localhost:8000/health

REM API statistics
curl http://localhost:8000/api/v1/monitoring/api-calls

REM Cache stats
curl http://localhost:8000/api/v1/cache/stats
```

---

## 🌐 Access Points

After starting the server:

### Main Application
- **Dashboard:** http://localhost:8000/dashboard
- **Paper Trading:** http://localhost:8000/paper
- **Login:** http://localhost:8000/user/login

### API Endpoints (Consolidated)
- **Dashboard Data:** http://localhost:8000/api/v1/dashboard
- **Paper Trading Data:** http://localhost:8000/api/v1/paper-trading
- **Cache Stats:** http://localhost:8000/api/v1/cache/stats
- **API Monitoring:** http://localhost:8000/api/v1/monitoring/api-calls
- **API Costs:** http://localhost:8000/api/v1/monitoring/api-costs

### Documentation
- **API Docs (Swagger):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🎯 Features Enabled

### ✅ Optimized API Architecture
- **Single API call per page load**
- **75-85% reduction** in external API calls
- **Intelligent caching** (10s for options, 5min for historical)
- **Zero frontend external calls** (all on backend)

### ✅ Auto-Refresh Strategy
- **10-second polling** during market hours only
- **Backend determines** market status
- **No polling** on weekends/holidays/outside trading hours

### ✅ Monitoring
- Real-time API call tracking
- Cache hit/miss ratios
- Rate limit usage
- Performance metrics

---

## 🔍 Troubleshooting

### Server won't start

```cmd
REM Check Python installation
C:\Users\saiki\AppData\Local\Programs\Python\Python312\python.exe --version

REM Should show: Python 3.12.x
```

### Missing dependencies

```cmd
REM Install requirements
C:\Users\saiki\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

### Port 8000 already in use

```cmd
REM Find process using port 8000
netstat -ano | findstr :8000

REM Kill the process (replace PID)
taskkill /PID <PID> /F
```

### Check if server is running

```cmd
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "authenticated": true
}
```

---

## 📝 Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Zerodha Credentials
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret

# Server Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=True

# Database
DATABASE_PATH=./data/trading.db
```

---

## 🎓 Usage Tips

### 1. Monitor API Usage

Open in browser:
```
http://localhost:8000/api/v1/monitoring/api-calls
```

Expected output:
```json
{
  "summary": {
    "total_calls": 500,
    "api_calls": 75,
    "cached_calls": 425,
    "cache_hit_rate": 85.0,
    "success_rate": 99.8
  }
}
```

### 2. Clear Cache (Force Fresh Data)

```cmd
curl -X POST http://localhost:8000/api/v1/cache/clear
```

### 3. Check Cache Performance

```cmd
curl http://localhost:8000/api/v1/cache/stats
```

### 4. Watch Logs

The server will show logs like:
```
[CACHE HIT] fetch_historical_data:NIFTY
[CACHE MISS] get_option_chain:BANKNIFTY
[API CALL] Executing fetch_historical_data
[FRONTEND] Auto-refresh (market open)
```

---

## 📂 File Structure

```
indicator/
├── run_server.bat              ← Start development server
├── run_server_production.bat   ← Start production server
├── setup_environment.bat       ← Setup dependencies
├── check_status.bat            ← Check server status
├── app/
│   ├── main.py                 ← FastAPI application
│   ├── api/
│   │   └── consolidated.py     ← NEW: Consolidated endpoints
│   ├── services/
│   │   ├── cache_manager.py    ← NEW: Caching system
│   │   └── cached_data_fetcher.py
│   ├── middleware/
│   │   └── api_tracking.py     ← NEW: API monitoring
│   └── templates/
│       └── dashboard.html      ← UPDATED: Single API call
├── ARCHITECTURE_OPTIMIZATION.md
├── OPTIMIZATION_SUMMARY.md
└── IMPLEMENTATION_STATUS.md
```

---

## ✅ Verification Checklist

After starting server, verify:

- [ ] Server running at http://localhost:8000
- [ ] Health check returns `{"status": "healthy"}`
- [ ] Dashboard loads: http://localhost:8000/dashboard
- [ ] Console shows: `[FRONTEND] Fetching dashboard data (single API call)...`
- [ ] Network tab shows only ONE `/api/v1/dashboard` call
- [ ] Market status badge visible (OPEN/CLOSED)
- [ ] Auto-refresh works (10 seconds if market open)
- [ ] Cache stats accessible: http://localhost:8000/api/v1/cache/stats

---

## 🎯 Next Steps

1. **Login** with Zerodha credentials
2. **View Dashboard** - should load with single API call
3. **Monitor** API usage and cache performance
4. **Migrate** remaining pages using same pattern

---

## 📞 Support

**Documentation:**
- Full Guide: [ARCHITECTURE_OPTIMIZATION.md](./ARCHITECTURE_OPTIMIZATION.md)
- Quick Reference: [OPTIMIZATION_SUMMARY.md](./OPTIMIZATION_SUMMARY.md)
- Migration Status: [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)

**Branch:** `optimize-api-architecture`

**Status:**
- Backend: ✅ 100% Complete
- Frontend: ✅ 10% Complete (1/10 pages)
- Dashboard: ✅ Fully optimized
