"""
Intelligent Caching Layer
Reduces redundant Zerodha API calls by caching with TTL
"""

import time
from typing import Any, Dict, Optional, Callable
from functools import wraps
import asyncio
from loguru import logger


class CacheManager:
    """
    Multi-tier caching system with TTL strategies
    Prevents redundant external API calls
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

        # Cache TTL configuration (in seconds)
        self.TTL_CONFIG = {
            # Real-time data (from WebSocket, no external API cost)
            "websocket_ticks": 0,  # Live updates, no caching needed

            # Fast-changing data (external API calls)
            "option_chain": 10,  # 10 seconds - options change frequently
            "market_status": 30,  # 30 seconds - market hours don't change often

            # Medium-changing data (external API calls)
            "signal_analysis": 120,  # 2 minutes - signals recalculated
            "quote": 5,  # 5 seconds - price quotes

            # Slow-changing data (external API calls)
            "historical_data": 300,  # 5 minutes - OHLC candles
            "instruments": 3600,  # 1 hour - instrument list rarely changes
        }

    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate unique cache key"""
        key_parts = [prefix]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return ":".join(key_parts)

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False

        timestamp = cache_entry.get("timestamp", 0)
        ttl = cache_entry.get("ttl", 0)
        current_time = time.time()

        is_valid = (current_time - timestamp) < ttl
        return is_valid

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_entry = self._cache.get(key)

        if cache_entry and self._is_cache_valid(cache_entry):
            logger.debug(f"[CACHE HIT] {key}")
            return cache_entry.get("value")

        logger.debug(f"[CACHE MISS] {key}")
        return None

    async def set(self, key: str, value: Any, ttl: int):
        """Set value in cache with TTL"""
        self._cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl
        }
        logger.debug(f"[CACHE SET] {key} (TTL: {ttl}s)")

    async def delete(self, key: str):
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"[CACHE DELETE] {key}")

    async def clear(self):
        """Clear all cache"""
        self._cache.clear()
        logger.info("[CACHE CLEAR] All cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self._cache)
        valid_entries = sum(1 for entry in self._cache.values() if self._is_cache_valid(entry))
        expired_entries = total_entries - valid_entries

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_keys": list(self._cache.keys())
        }

    async def get_lock(self, key: str) -> asyncio.Lock:
        """Get or create lock for key (prevents duplicate concurrent requests)"""
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]


# Global cache instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(cache_type: str):
    """
    Decorator to cache function results
    Prevents redundant external API calls

    Usage:
        @cached("historical_data")
        async def fetch_historical_data(index: str):
            # This will be cached for 5 minutes
            return await zerodha_api.get_historical(index)
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = get_cache_manager()

            # Generate cache key
            cache_key = cache._get_cache_key(
                f"{func.__name__}:{cache_type}",
                *args,
                **kwargs
            )

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.info(f"[CACHE] Using cached {cache_type} for {func.__name__}")
                return cached_value

            # Get lock to prevent duplicate concurrent requests
            lock = await cache.get_lock(cache_key)

            async with lock:
                # Double-check cache after acquiring lock
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    logger.info(f"[CACHE] Using cached {cache_type} (after lock)")
                    return cached_value

                # Cache miss - call actual function
                logger.info(f"[API CALL] Executing {func.__name__} for {cache_type}")
                result = await func(*args, **kwargs)

                # Store in cache
                ttl = cache.TTL_CONFIG.get(cache_type, 60)
                await cache.set(cache_key, result, ttl)

                return result

        return wrapper

    return decorator


# Request deduplication for identical in-flight requests
class RequestDeduplicator:
    """
    Coalesces identical concurrent requests
    If same endpoint is called multiple times simultaneously,
    only one actual request is made
    """

    def __init__(self):
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def deduplicate(self, key: str, func: Callable, *args, **kwargs):
        """Execute function or wait for existing execution"""

        # Check if request already in progress
        if key in self._pending_requests:
            logger.info(f"[DEDUPE] Waiting for existing request: {key}")
            return await self._pending_requests[key]

        # Create new request
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            # Double-check after acquiring lock
            if key in self._pending_requests:
                logger.info(f"[DEDUPE] Waiting for existing request (after lock): {key}")
                return await self._pending_requests[key]

            # Create future for this request
            future = asyncio.create_task(func(*args, **kwargs))
            self._pending_requests[key] = future

            try:
                logger.info(f"[DEDUPE] Executing new request: {key}")
                result = await future
                return result
            finally:
                # Clean up
                if key in self._pending_requests:
                    del self._pending_requests[key]


# Global deduplicator instance
_deduplicator: Optional[RequestDeduplicator] = None


def get_deduplicator() -> RequestDeduplicator:
    """Get global request deduplicator"""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = RequestDeduplicator()
    return _deduplicator
