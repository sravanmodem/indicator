"""
Cached Data Fetcher Wrapper
Wraps DataFetcher with intelligent caching to minimize Zerodha API calls
All external API calls go through this layer
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from app.services.data_fetcher import DataFetcher, get_data_fetcher, is_api_allowed
from app.services.cache_manager import get_cache_manager, get_deduplicator, cached


class CachedDataFetcher:
    """
    Wrapper around DataFetcher that adds intelligent caching
    Prevents redundant Zerodha API calls
    """

    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.cache = get_cache_manager()
        self.deduplicator = get_deduplicator()

    @cached("historical_data")
    async def fetch_historical_data(
        self,
        index_name: str,
        timeframe: str = "5minute",
        days: int = 5
    ) -> List[Dict]:
        """
        Fetch historical data with caching (5 minute TTL)
        EXTERNAL API CALL: Zerodha kite.historical()
        """
        logger.info(f"[CACHED FETCHER] Historical data for {index_name}")

        # Check if API calls are allowed
        is_allowed, reason = is_api_allowed()
        if not is_allowed:
            logger.warning(f"[API BLOCKED] {reason}")
            # Return cached data even if expired, or empty
            cache_key = f"fetch_historical_data:historical_data:{index_name}:{timeframe}:{days}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                logger.info(f"[USING EXPIRED CACHE] Market closed - using last known data")
                return cached_data
            return []

        # Make actual API call
        return await self.data_fetcher.fetch_historical_data(index_name, timeframe, days)

    @cached("option_chain")
    async def get_option_chain(
        self,
        index_name: str,
        expiry_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get option chain with caching (10 second TTL)
        EXTERNAL API CALL: Zerodha kite.instruments()
        """
        logger.info(f"[CACHED FETCHER] Option chain for {index_name}")

        # Check if API calls are allowed
        is_allowed, reason = is_api_allowed()
        if not is_allowed:
            logger.warning(f"[API BLOCKED] {reason}")
            # Return cached data even if expired, or empty
            cache_key = f"get_option_chain:option_chain:{index_name}:{expiry_date}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                logger.info(f"[USING EXPIRED CACHE] Market closed - using last known option chain")
                return cached_data
            return []

        # Make actual API call
        return await self.data_fetcher.get_option_chain(index_name, expiry_date)

    @cached("quote")
    async def get_quote(self, symbol: str) -> Dict:
        """
        Get quote with caching (5 second TTL)
        EXTERNAL API CALL: Zerodha kite.quote()
        """
        logger.info(f"[CACHED FETCHER] Quote for {symbol}")

        # Check if API calls are allowed
        is_allowed, reason = is_api_allowed()
        if not is_allowed:
            logger.warning(f"[API BLOCKED] {reason}")
            return {}

        # Make actual API call
        return await self.data_fetcher.get_quote(symbol)

    @cached("instruments")
    async def get_instruments(self, exchange: str = "NFO") -> List[Dict]:
        """
        Get instruments with caching (1 hour TTL)
        EXTERNAL API CALL: Zerodha kite.instruments()
        """
        logger.info(f"[CACHED FETCHER] Instruments for {exchange}")

        # Check if API calls are allowed
        is_allowed, reason = is_api_allowed()
        if not is_allowed:
            # Instruments rarely change, use cached data
            cache_key = f"get_instruments:instruments:{exchange}"
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return cached_data
            return []

        # Make actual API call
        return await self.data_fetcher.get_instruments(exchange)

    async def get_available_expiries(self, index_name: str) -> List[str]:
        """
        Get available expiries (uses cached option chain)
        NO EXTERNAL API CALL - uses cached data
        """
        logger.info(f"[CACHED FETCHER] Available expiries for {index_name}")

        # This uses the cached option chain internally
        return await self.data_fetcher.get_available_expiries(index_name)

    async def get_lot_size(self, index_name: str) -> int:
        """
        Get lot size for index
        NO EXTERNAL API CALL - internal lookup
        """
        return await self.data_fetcher.get_lot_size(index_name)


# Global cached fetcher instance
_cached_fetcher: Optional[CachedDataFetcher] = None


def get_cached_data_fetcher() -> CachedDataFetcher:
    """Get global cached data fetcher instance"""
    global _cached_fetcher
    if _cached_fetcher is None:
        _cached_fetcher = CachedDataFetcher(get_data_fetcher())
    return _cached_fetcher
