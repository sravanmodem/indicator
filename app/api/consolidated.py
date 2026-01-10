"""
Consolidated API Endpoints
Single endpoint per page - all external API calls happen in backend only
Zero frontend calls to Zerodha or market APIs

Frontend refresh: Every 10 seconds during market hours (backend checks market status)
"""

from typing import Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app.services.cached_data_fetcher import CachedDataFetcher, get_cached_data_fetcher
from app.services.signal_engine import SignalEngine, get_signal_engine
from app.services.websocket_manager import get_ws_manager
from app.services.paper_trading import get_paper_trading_service
from app.services.zerodha_auth import get_auth_service
from app.services.cache_manager import get_cache_manager
from app.core.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["consolidated"])


class PageDataAggregator:
    """Aggregates all data needed for a page in single backend call"""

    def __init__(
        self,
        data_fetcher: CachedDataFetcher,
        signal_engine: SignalEngine,
    ):
        self.data_fetcher = data_fetcher
        self.signal_engine = signal_engine

    async def get_dashboard_data(
        self,
        indices: List[str] = ["NIFTY", "BANKNIFTY", "SENSEX"],
        style: str = "intraday"
    ) -> Dict:
        """
        Single endpoint for dashboard page
        Aggregates ALL data needed - no frontend calls to external APIs

        Returns:
        - Market status
        - All indices data (signals, indicators, option chains)
        - WebSocket status
        - Header cards data
        """
        logger.info(f"[CONSOLIDATED API] Dashboard data requested for {indices}")

        # Initialize result structure
        result = {
            "market_status": None,
            "market_overview": None,
            "websocket_status": None,
            "indices": {},
            "timestamp": None
        }

        try:
            # 1. Get market status (internal check, no external API)
            result["market_status"] = await self._get_market_status()

            # 2. Get WebSocket status (internal state, no external API)
            ws_manager = get_ws_manager()
            result["websocket_status"] = {
                "connected": ws_manager.is_connected,
                "subscribed_count": len(ws_manager._subscriptions) if hasattr(ws_manager, '_subscriptions') else 0
            }

            # 3. Aggregate data for all indices in parallel
            # This makes external API calls but batched on backend
            import asyncio
            tasks = []
            for index_name in indices:
                task = self._get_index_complete_data(index_name, style)
                tasks.append(task)

            indices_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Package results
            for index_name, index_data in zip(indices, indices_results):
                if isinstance(index_data, Exception):
                    logger.error(f"Error fetching {index_name}: {index_data}")
                    result["indices"][index_name] = {"error": str(index_data)}
                else:
                    result["indices"][index_name] = index_data

            # 5. Get market overview (combines all index data)
            result["market_overview"] = await self._get_market_overview(indices)

            # 6. Timestamp
            from datetime import datetime
            result["timestamp"] = datetime.now().isoformat()

            logger.info(f"[CONSOLIDATED API] Dashboard data compiled successfully")
            return result

        except Exception as e:
            logger.error(f"[CONSOLIDATED API] Error in dashboard data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_index_complete_data(self, index_name: str, style: str) -> Dict:
        """
        Get complete data for single index
        All external API calls happen here on backend
        """
        logger.info(f"[BACKEND] Fetching complete data for {index_name}")

        try:
            # Backend call 1: Historical data (Zerodha API)
            historical_data = await self.data_fetcher.fetch_historical_data(index_name)

            # Backend call 2: Option chain (Zerodha API)
            option_chain = await self.data_fetcher.get_option_chain(index_name)

            # Backend processing: Analyze signals (no external API)
            signal_result = await self.signal_engine.analyze(
                index=index_name,
                timeframe="5minute",
                style=style
            )

            # Backend processing: Calculate option chain stats (no external API)
            option_stats = self._calculate_option_stats(option_chain) if option_chain else None

            # Backend processing: Get recommended option (no external API)
            recommended_option = self._get_recommended_option(
                option_chain,
                signal_result
            ) if option_chain else None

            # Package everything
            return {
                "signal": {
                    "action": signal_result.action,
                    "confidence": signal_result.confidence,
                    "option_type": signal_result.option_type,
                    "entry_price": signal_result.entry_price,
                    "stop_loss": signal_result.stop_loss,
                    "target": signal_result.target,
                    "reasoning": signal_result.reasoning,
                },
                "indicators": {
                    "trend": signal_result.indicators.get("trend", {}),
                    "momentum": signal_result.indicators.get("momentum", {}),
                    "volatility": signal_result.indicators.get("volatility", {}),
                    "pivot": signal_result.indicators.get("pivot", {}),
                },
                "option_chain_summary": option_stats,
                "recommended_option": recommended_option,
                "last_price": historical_data[-1]["close"] if historical_data else None,
                "historical_summary": {
                    "open": historical_data[0]["open"] if historical_data else None,
                    "high": max([c["high"] for c in historical_data]) if historical_data else None,
                    "low": min([c["low"] for c in historical_data]) if historical_data else None,
                    "close": historical_data[-1]["close"] if historical_data else None,
                } if historical_data else None
            }

        except Exception as e:
            logger.error(f"[BACKEND] Error fetching {index_name} data: {e}")
            raise

    async def _get_market_status(self) -> Dict:
        """Get market status - internal calculation, no external API"""
        from datetime import datetime, time
        import pytz

        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        current_time = now.time()

        # Market hours: 9:15 AM to 3:30 PM IST
        market_open = time(9, 15)
        market_close = time(15, 30)

        is_open = (
            now.weekday() < 5 and  # Monday to Friday
            market_open <= current_time <= market_close
        )

        return {
            "is_open": is_open,
            "current_time": now.isoformat(),
            "message": "Market is open" if is_open else "Market is closed"
        }

    async def _get_market_overview(self, indices: List[str]) -> Dict:
        """Get overview of all indices - uses WebSocket ticks (no external API)"""
        ws_manager = get_ws_manager()

        overview = {}
        for index_name in indices:
            # Get from WebSocket ticks (no external API call)
            ticks = ws_manager._latest_ticks if hasattr(ws_manager, '_latest_ticks') else {}

            # Find tick for this index
            index_tick = None
            for token, tick_data in ticks.items():
                symbol = tick_data.instrument_token if hasattr(tick_data, 'instrument_token') else None
                if symbol and str(symbol).startswith(index_name):
                    index_tick = tick_data
                    break

            if index_tick and hasattr(index_tick, 'last_price'):
                overview[index_name] = {
                    "ltp": index_tick.last_price,
                    "change": index_tick.change if hasattr(index_tick, 'change') else None,
                    "change_percent": index_tick.change_percent if hasattr(index_tick, 'change_percent') else None,
                }
            else:
                overview[index_name] = {"ltp": None, "change": None, "change_percent": None}

        return overview

    def _calculate_option_stats(self, option_chain: List[Dict]) -> Dict:
        """Calculate option chain statistics - pure calculation, no external API"""
        if not option_chain:
            return None

        total_ce_oi = sum([opt.get("oi", 0) for opt in option_chain if opt.get("option_type") == "CE"])
        total_pe_oi = sum([opt.get("oi", 0) for opt in option_chain if opt.get("option_type") == "PE"])

        pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0

        # Find ATM strike
        if option_chain:
            mid_strike = option_chain[len(option_chain) // 2].get("strike", 0)
        else:
            mid_strike = 0

        return {
            "pcr": pcr,
            "total_ce_oi": total_ce_oi,
            "total_pe_oi": total_pe_oi,
            "atm_strike": mid_strike,
            "total_options": len(option_chain)
        }

    def _get_recommended_option(self, option_chain: List[Dict], signal_result) -> Optional[Dict]:
        """Get recommended option to trade - pure logic, no external API"""
        if not option_chain or not signal_result:
            return None

        # Filter by option type from signal
        option_type = signal_result.option_type
        filtered = [opt for opt in option_chain if opt.get("option_type") == option_type]

        if not filtered:
            return None

        # Find best option (e.g., highest OI, ATM or ITM)
        # This is simplified logic
        best_option = max(filtered, key=lambda x: x.get("oi", 0))

        return {
            "symbol": best_option.get("symbol"),
            "strike": best_option.get("strike"),
            "option_type": best_option.get("option_type"),
            "ltp": best_option.get("ltp"),
            "oi": best_option.get("oi"),
            "iv": best_option.get("iv"),
        }

    async def get_paper_trading_page_data(self) -> Dict:
        """
        Single endpoint for paper trading page
        All data fetched from backend only
        """
        logger.info("[CONSOLIDATED API] Paper trading page data requested")

        paper_service = get_paper_trading_service()

        try:
            # Get all paper trading data in one go
            stats = await paper_service.get_stats()
            positions = await paper_service.get_positions()
            orders = await paper_service.get_orders()
            trading_config = await paper_service.get_trading_config()

            # Get current signal for active index
            current_index = trading_config.get("index", "NIFTY")
            signal_data = await self._get_index_complete_data(
                current_index,
                trading_config.get("style", "intraday")
            )

            return {
                "stats": stats,
                "positions": positions,
                "orders": orders,
                "trading_config": trading_config,
                "current_signal": signal_data,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[CONSOLIDATED API] Error in paper trading data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_option_chain_page_data(self, index: str) -> Dict:
        """
        Single endpoint for option chain page
        All data fetched from backend only
        """
        logger.info(f"[CONSOLIDATED API] Option chain page data requested for {index}")

        try:
            # Backend call: Get option chain (Zerodha API)
            option_chain = await self.data_fetcher.get_option_chain(index)

            # Backend processing: Calculate all statistics
            stats = self._calculate_option_stats(option_chain)

            # Backend processing: Max pain calculation
            max_pain = self._calculate_max_pain(option_chain) if option_chain else None

            # Backend processing: OI analysis
            oi_analysis = self._analyze_oi(option_chain) if option_chain else None

            return {
                "option_chain": option_chain,
                "statistics": stats,
                "max_pain": max_pain,
                "oi_analysis": oi_analysis,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[CONSOLIDATED API] Error in option chain data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _calculate_max_pain(self, option_chain: List[Dict]) -> Optional[float]:
        """Calculate max pain - pure calculation, no external API"""
        # Simplified max pain calculation
        # In production, this would be more sophisticated
        if not option_chain:
            return None

        strikes = list(set([opt.get("strike") for opt in option_chain]))

        # Calculate pain at each strike
        pain_values = {}
        for strike in strikes:
            pain = 0
            for opt in option_chain:
                opt_strike = opt.get("strike", 0)
                oi = opt.get("oi", 0)
                opt_type = opt.get("option_type")

                if opt_type == "CE" and strike > opt_strike:
                    pain += (strike - opt_strike) * oi
                elif opt_type == "PE" and strike < opt_strike:
                    pain += (opt_strike - strike) * oi

            pain_values[strike] = pain

        # Return strike with minimum pain
        if pain_values:
            max_pain_strike = min(pain_values, key=pain_values.get)
            return max_pain_strike

        return None

    def _analyze_oi(self, option_chain: List[Dict]) -> Optional[Dict]:
        """Analyze open interest - pure calculation, no external API"""
        if not option_chain:
            return None

        # Find strikes with highest OI
        ce_options = [opt for opt in option_chain if opt.get("option_type") == "CE"]
        pe_options = [opt for opt in option_chain if opt.get("option_type") == "PE"]

        max_ce_oi = max(ce_options, key=lambda x: x.get("oi", 0)) if ce_options else None
        max_pe_oi = max(pe_options, key=lambda x: x.get("oi", 0)) if pe_options else None

        return {
            "max_ce_strike": max_ce_oi.get("strike") if max_ce_oi else None,
            "max_ce_oi": max_ce_oi.get("oi") if max_ce_oi else None,
            "max_pe_strike": max_pe_oi.get("strike") if max_pe_oi else None,
            "max_pe_oi": max_pe_oi.get("oi") if max_pe_oi else None,
        }


# Dependency injection
def get_aggregator() -> PageDataAggregator:
    """Get page data aggregator instance"""
    return PageDataAggregator(
        data_fetcher=get_cached_data_fetcher(),  # Use cached fetcher
        signal_engine=get_signal_engine()
    )


# API Endpoints
@router.get("/dashboard")
async def dashboard_data(
    indices: str = "NIFTY,BANKNIFTY,SENSEX",
    style: str = "intraday",
    aggregator: PageDataAggregator = Depends(get_aggregator)
):
    """
    Consolidated dashboard endpoint
    Returns ALL data needed for dashboard page
    Frontend makes ONLY this one call
    """
    indices_list = [idx.strip().upper() for idx in indices.split(",")]
    return await aggregator.get_dashboard_data(indices_list, style)


@router.get("/paper-trading")
async def paper_trading_data(
    aggregator: PageDataAggregator = Depends(get_aggregator)
):
    """
    Consolidated paper trading endpoint
    Returns ALL data needed for paper trading page
    Frontend makes ONLY this one call
    """
    return await aggregator.get_paper_trading_page_data()


@router.get("/option-chain/{index}")
async def option_chain_data(
    index: str,
    aggregator: PageDataAggregator = Depends(get_aggregator)
):
    """
    Consolidated option chain endpoint
    Returns ALL data needed for option chain page
    Frontend makes ONLY this one call
    """
    return await aggregator.option_chain_data(index.upper())


@router.get("/history")
async def history_data():
    """
    Consolidated history endpoint
    Returns ALL data needed for history page
    Frontend makes ONLY this one call
    """
    from app.services.signal_history_service import get_history_service

    history_service = get_history_service()
    stats = history_service.get_statistics(days=30)
    recent_signals = history_service.get_recent_signals(limit=50)

    return {
        "stats": stats,
        "recent_signals": recent_signals,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/settings")
async def settings_data():
    """
    Consolidated settings endpoint
    Returns ALL data needed for settings page
    Frontend makes ONLY this one call
    """
    from app.services.signal_history_service import get_history_service
    from app.services.auto_trader import get_auto_trader

    history_service = get_history_service()
    auto_trader = get_auto_trader()

    return {
        "signal_settings": history_service.get_settings(),
        "auto_trade_settings": auto_trader.get_settings(),
        "auth_status": get_auth_service().get_auth_status(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/cache/stats")
async def cache_stats():
    """
    Get cache statistics
    Shows cache hit/miss ratio and current entries
    """
    cache = get_cache_manager()
    return cache.get_stats()


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cache
    Forces fresh data fetch on next request
    """
    cache = get_cache_manager()
    await cache.clear()
    return {"message": "Cache cleared successfully"}


@router.get("/monitoring/api-calls")
async def api_call_stats(hours: int = 24):
    """
    Get API call statistics
    Shows external API usage, cache hit rates, and performance metrics
    """
    from app.middleware import get_api_tracker

    tracker = get_api_tracker()
    return tracker.get_stats(hours=hours)


@router.get("/monitoring/api-costs")
async def api_cost_estimate():
    """
    Get API cost estimate and rate limit usage
    Helps identify if approaching Zerodha rate limits
    """
    from app.middleware import get_api_tracker

    tracker = get_api_tracker()
    return tracker.get_cost_estimate()


@router.post("/monitoring/reset")
async def reset_monitoring():
    """
    Reset all monitoring statistics
    """
    from app.middleware import get_api_tracker

    tracker = get_api_tracker()
    tracker.reset()
    return {"message": "Monitoring statistics reset successfully"}
