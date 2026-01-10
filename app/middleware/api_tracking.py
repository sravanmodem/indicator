"""
API Call Tracking Middleware
Monitors and logs all external API calls to Zerodha
Tracks usage, costs, and helps identify optimization opportunities
"""

import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class APICallTracker:
    """Tracks external API calls and generates usage statistics"""

    def __init__(self):
        self._calls: List[Dict] = []
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._call_times: Dict[str, List[float]] = defaultdict(list)
        self._start_time = datetime.now()

    def log_call(
        self,
        endpoint: str,
        method: str = "GET",
        duration: float = 0,
        success: bool = True,
        cached: bool = False
    ):
        """Log an API call"""
        call_info = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "duration_ms": round(duration * 1000, 2),
            "success": success,
            "cached": cached
        }

        self._calls.append(call_info)
        self._call_counts[endpoint] += 1
        self._call_times[endpoint].append(duration)

        # Log to console
        cache_status = " [CACHED]" if cached else " [API CALL]"
        logger.info(
            f"{cache_status} {method} {endpoint} - "
            f"{call_info['duration_ms']}ms - "
            f"{'SUCCESS' if success else 'FAILED'}"
        )

    def get_stats(self, hours: Optional[int] = None) -> Dict:
        """Get API call statistics"""
        calls = self._calls

        # Filter by time window if specified
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            calls = [
                c for c in calls
                if datetime.fromisoformat(c["timestamp"]) >= cutoff
            ]

        total_calls = len(calls)
        successful_calls = sum(1 for c in calls if c["success"])
        cached_calls = sum(1 for c in calls if c["cached"])
        api_calls = total_calls - cached_calls

        # Calculate average duration per endpoint
        avg_durations = {}
        for endpoint, times in self._call_times.items():
            if times:
                avg_durations[endpoint] = round(sum(times) / len(times) * 1000, 2)

        # Top endpoints by call count
        top_endpoints = sorted(
            self._call_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "summary": {
                "total_calls": total_calls,
                "api_calls": api_calls,
                "cached_calls": cached_calls,
                "cache_hit_rate": round((cached_calls / total_calls * 100), 2) if total_calls > 0 else 0,
                "success_rate": round((successful_calls / total_calls * 100), 2) if total_calls > 0 else 0,
                "uptime_hours": round((datetime.now() - self._start_time).total_seconds() / 3600, 2)
            },
            "top_endpoints": [
                {"endpoint": endpoint, "count": count}
                for endpoint, count in top_endpoints
            ],
            "average_durations_ms": avg_durations,
            "recent_calls": calls[-20:]  # Last 20 calls
        }

    def get_cost_estimate(self) -> Dict:
        """
        Estimate API costs based on Zerodha rate limits
        Note: Zerodha doesn't charge per-call, but has rate limits
        This helps understand potential rate limit issues
        """
        api_calls = sum(
            1 for c in self._calls
            if not c.get("cached", False)
        )

        # Zerodha typical rate limits (approximate)
        RATE_LIMIT_PER_SECOND = 10
        RATE_LIMIT_PER_MINUTE = 180

        uptime_seconds = (datetime.now() - self._start_time).total_seconds()
        calls_per_second = api_calls / uptime_seconds if uptime_seconds > 0 else 0
        calls_per_minute = calls_per_second * 60

        return {
            "total_api_calls": api_calls,
            "calls_per_second": round(calls_per_second, 2),
            "calls_per_minute": round(calls_per_minute, 2),
            "rate_limit_usage": {
                "per_second": f"{round(calls_per_second / RATE_LIMIT_PER_SECOND * 100, 2)}%",
                "per_minute": f"{round(calls_per_minute / RATE_LIMIT_PER_MINUTE * 100, 2)}%"
            },
            "status": "SAFE" if calls_per_second < RATE_LIMIT_PER_SECOND * 0.5 else "WARNING"
        }

    def reset(self):
        """Reset all tracking data"""
        self._calls.clear()
        self._call_counts.clear()
        self._call_times.clear()
        self._start_time = datetime.now()
        logger.info("[API TRACKER] Reset all tracking data")


# Global tracker instance
_tracker: Optional[APICallTracker] = None


def get_api_tracker() -> APICallTracker:
    """Get global API tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = APICallTracker()
    return _tracker


class APITrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track API endpoints called
    Logs timing and success/failure of each request
    """

    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()

        # Process request
        response: Response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Track if this is an API endpoint
        path = request.url.path
        if path.startswith("/api/"):
            tracker = get_api_tracker()

            # Check if response came from cache (look for cache headers)
            cached = response.headers.get("X-Cache-Status") == "HIT"

            tracker.log_call(
                endpoint=path,
                method=request.method,
                duration=duration,
                success=response.status_code < 400,
                cached=cached
            )

        return response


# Wrapper to mark cached responses
def mark_cached_response(response: Response) -> Response:
    """Add cache header to response"""
    response.headers["X-Cache-Status"] = "HIT"
    return response


def mark_api_response(response: Response) -> Response:
    """Add API call header to response"""
    response.headers["X-Cache-Status"] = "MISS"
    return response
