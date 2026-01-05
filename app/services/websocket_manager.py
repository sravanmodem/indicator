"""
WebSocket Manager for Real-Time Market Data
Low-latency streaming from Zerodha Kite Ticker
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from kiteconnect import KiteTicker
from loguru import logger

from app.core.config import (
    BANKNIFTY_INDEX_TOKEN,
    NIFTY_INDEX_TOKEN,
    get_settings,
)
from app.services.zerodha_auth import get_auth_service


class TickMode(Enum):
    """Tick subscription modes."""
    LTP = "ltp"        # Last traded price only
    QUOTE = "quote"    # LTP + OHLC + Volume
    FULL = "full"      # Complete market depth


@dataclass
class TickData:
    """Structured tick data."""
    instrument_token: int
    timestamp: datetime
    last_price: float
    volume: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    oi: int = 0
    oi_change: int = 0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0
    depth: dict = field(default_factory=dict)

    @classmethod
    def from_kite_tick(cls, tick: dict) -> "TickData":
        """Create TickData from Kite ticker response."""
        return cls(
            instrument_token=tick.get("instrument_token", 0),
            timestamp=tick.get("exchange_timestamp") or datetime.now(),
            last_price=tick.get("last_price", 0.0),
            volume=tick.get("volume_traded", 0),
            open=tick.get("ohlc", {}).get("open", 0.0),
            high=tick.get("ohlc", {}).get("high", 0.0),
            low=tick.get("ohlc", {}).get("low", 0.0),
            close=tick.get("ohlc", {}).get("close", 0.0),
            change=tick.get("change", 0.0),
            oi=tick.get("oi", 0),
            oi_change=tick.get("oi_day_high", 0) - tick.get("oi_day_low", 0),
            bid_price=tick.get("depth", {}).get("buy", [{}])[0].get("price", 0.0),
            ask_price=tick.get("depth", {}).get("sell", [{}])[0].get("price", 0.0),
            bid_qty=tick.get("depth", {}).get("buy", [{}])[0].get("quantity", 0),
            ask_qty=tick.get("depth", {}).get("sell", [{}])[0].get("quantity", 0),
            depth=tick.get("depth", {}),
        )


class WebSocketManager:
    """
    Manages WebSocket connection to Zerodha Kite Ticker.

    Features:
    - Auto-reconnection with exponential backoff
    - Subscription management
    - Callback-based event handling
    - Thread-safe tick storage
    """

    def __init__(self):
        self.settings = get_settings()
        self.auth_service = get_auth_service()
        self._ticker: KiteTicker | None = None
        self._is_connected = False
        self._subscriptions: dict[int, TickMode] = {}
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._latest_ticks: dict[int, TickData] = {}
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._is_connected

    def get_latest_tick(self, instrument_token: int) -> TickData | None:
        """Get latest tick for an instrument."""
        return self._latest_ticks.get(instrument_token)

    def get_all_ticks(self) -> dict[int, TickData]:
        """Get all latest ticks."""
        return self._latest_ticks.copy()

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for events.

        Events:
        - on_tick: Called on each tick
        - on_connect: Called when connected
        - on_disconnect: Called when disconnected
        - on_error: Called on error
        """
        self._callbacks[event].append(callback)
        logger.debug(f"Registered callback for event: {event}")

    def unregister_callback(self, event: str, callback: Callable) -> None:
        """Remove callback for event."""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        """Emit event to all registered callbacks."""
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    if self._loop:
                        asyncio.run_coroutine_threadsafe(
                            callback(*args, **kwargs), self._loop
                        )
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connection successful
        """
        if not self.auth_service.is_authenticated:
            logger.error("Cannot connect WebSocket: Not authenticated")
            return False

        try:
            self._loop = asyncio.get_event_loop()

            access_token = self.auth_service.kite.access_token
            self._ticker = KiteTicker(
                api_key=self.settings.kite_api_key,
                access_token=access_token,
            )

            # Register callbacks
            self._ticker.on_ticks = self._on_ticks
            self._ticker.on_connect = self._on_connect
            self._ticker.on_close = self._on_close
            self._ticker.on_error = self._on_error
            self._ticker.on_reconnect = self._on_reconnect
            self._ticker.on_noreconnect = self._on_noreconnect

            # Start connection in thread
            self._ticker.connect(threaded=True)

            # Wait for connection
            for _ in range(50):  # 5 second timeout
                if self._is_connected:
                    logger.info("WebSocket connected successfully")
                    return True
                await asyncio.sleep(0.1)

            logger.warning("WebSocket connection timeout")
            return False

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ticker:
            try:
                self._ticker.close()
            except Exception:
                pass
            self._ticker = None
        self._is_connected = False
        self._subscriptions.clear()
        logger.info("WebSocket disconnected")

    async def subscribe(
        self,
        instrument_tokens: list[int],
        mode: TickMode = TickMode.FULL,
    ) -> bool:
        """
        Subscribe to instruments.

        Args:
            instrument_tokens: List of instrument tokens
            mode: Subscription mode (LTP, QUOTE, FULL)

        Returns:
            True if subscription successful
        """
        if not self._is_connected or not self._ticker:
            logger.error("Cannot subscribe: WebSocket not connected")
            return False

        try:
            self._ticker.subscribe(instrument_tokens)

            mode_map = {
                TickMode.LTP: self._ticker.MODE_LTP,
                TickMode.QUOTE: self._ticker.MODE_QUOTE,
                TickMode.FULL: self._ticker.MODE_FULL,
            }
            self._ticker.set_mode(mode_map[mode], instrument_tokens)

            for token in instrument_tokens:
                self._subscriptions[token] = mode

            logger.info(f"Subscribed to {len(instrument_tokens)} instruments in {mode.value} mode")
            return True

        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return False

    async def unsubscribe(self, instrument_tokens: list[int]) -> bool:
        """Unsubscribe from instruments."""
        if not self._is_connected or not self._ticker:
            return False

        try:
            self._ticker.unsubscribe(instrument_tokens)
            for token in instrument_tokens:
                self._subscriptions.pop(token, None)
                self._latest_ticks.pop(token, None)
            logger.info(f"Unsubscribed from {len(instrument_tokens)} instruments")
            return True
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
            return False

    async def subscribe_nifty_chain(self, expiry_date: str, strikes: list[int]) -> bool:
        """
        Subscribe to NIFTY option chain for given expiry and strikes.

        Args:
            expiry_date: Expiry in format "YYMDD" (e.g., "24125" for Dec 5, 2024)
            strikes: List of strike prices
        """
        # This would need instrument lookup - placeholder for now
        # In production, fetch instrument list and map strikes to tokens
        tokens = [NIFTY_INDEX_TOKEN]  # Start with index
        logger.info(f"Subscribing to NIFTY chain: {len(strikes)} strikes")
        return await self.subscribe(tokens, TickMode.FULL)

    # Kite Ticker Callbacks
    def _on_ticks(self, ws, ticks: list[dict]) -> None:
        """Handle incoming ticks."""
        for tick in ticks:
            tick_data = TickData.from_kite_tick(tick)
            self._latest_ticks[tick_data.instrument_token] = tick_data
        self._emit("on_tick", ticks)

    def _on_connect(self, ws, response) -> None:
        """Handle connection established."""
        self._is_connected = True
        self._reconnect_attempts = 0
        logger.info("Kite Ticker connected")

        # Resubscribe to previous subscriptions
        if self._subscriptions:
            for token, mode in self._subscriptions.items():
                ws.subscribe([token])
                mode_map = {
                    TickMode.LTP: ws.MODE_LTP,
                    TickMode.QUOTE: ws.MODE_QUOTE,
                    TickMode.FULL: ws.MODE_FULL,
                }
                ws.set_mode(mode_map[mode], [token])

        self._emit("on_connect", response)

    def _on_close(self, ws, code, reason) -> None:
        """Handle connection closed."""
        self._is_connected = False
        logger.warning(f"Kite Ticker closed: {code} - {reason}")
        self._emit("on_disconnect", code, reason)

    def _on_error(self, ws, code, reason) -> None:
        """Handle connection error."""
        logger.error(f"Kite Ticker error: {code} - {reason}")
        self._emit("on_error", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        """Handle reconnection attempt."""
        self._reconnect_attempts = attempts_count
        logger.info(f"Reconnecting... Attempt {attempts_count}")

    def _on_noreconnect(self, ws) -> None:
        """Handle reconnection failure."""
        logger.error("Max reconnection attempts reached")
        self._is_connected = False

    def get_connection_status(self) -> dict[str, Any]:
        """Get WebSocket connection status."""
        return {
            "is_connected": self._is_connected,
            "subscriptions_count": len(self._subscriptions),
            "subscribed_tokens": list(self._subscriptions.keys()),
            "reconnect_attempts": self._reconnect_attempts,
            "ticks_received": len(self._latest_ticks),
        }


# Singleton instance
_ws_manager: WebSocketManager | None = None


def get_ws_manager() -> WebSocketManager:
    """Get or create WebSocket manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
