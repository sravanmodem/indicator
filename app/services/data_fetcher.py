"""
Historical Data Fetcher
Retrieves OHLCV data and option chain from Zerodha
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from kiteconnect import KiteConnect
from loguru import logger

from app.core.config import (
    BANKNIFTY_INDEX_TOKEN,
    NIFTY_INDEX_TOKEN,
    SENSEX_INDEX_TOKEN,
    TIMEFRAME_MAP,
    get_settings,
)
from app.services.zerodha_auth import get_auth_service


class DataFetcher:
    """
    Fetches historical data, option chain, and market info from Zerodha.
    """

    def __init__(self):
        self.settings = get_settings()
        self.auth_service = get_auth_service()
        self._instruments_cache: dict[str, pd.DataFrame] = {}
        self._instruments_last_updated: datetime | None = None
        # Option chain cache with short TTL (10 seconds)
        self._option_chain_cache: dict[str, dict] = {}
        self._option_chain_last_updated: dict[str, datetime] = {}

    @property
    def kite(self) -> KiteConnect:
        """Get authenticated Kite instance."""
        return self.auth_service.kite

    async def fetch_instruments(self, exchange: str = "NFO") -> pd.DataFrame:
        """
        Fetch instruments list from Zerodha.

        Args:
            exchange: Exchange (NSE, NFO, BSE, BFO)

        Returns:
            DataFrame with instrument details
        """
        cache_key = exchange
        now = datetime.now()

        # Use cache if less than 1 hour old
        if (
            cache_key in self._instruments_cache
            and self._instruments_last_updated
            and (now - self._instruments_last_updated).seconds < 3600
        ):
            return self._instruments_cache[cache_key]

        try:
            instruments = await asyncio.to_thread(self.kite.instruments, exchange)
            df = pd.DataFrame(instruments)
            self._instruments_cache[cache_key] = df
            self._instruments_last_updated = now
            logger.info(f"Fetched {len(df)} instruments from {exchange}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch instruments: {e}")
            return pd.DataFrame()

    async def get_index_lot_size(self, index: str = "NIFTY") -> int:
        """
        Get the actual lot size for an index from Kite instruments data.

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX

        Returns:
            Lot size as integer
        """
        try:
            exchange = "BFO" if index == "SENSEX" else "NFO"
            instruments = await self.fetch_instruments(exchange)
            if instruments.empty:
                # Return default values if fetch fails
                defaults = {"NIFTY": 25, "BANKNIFTY": 15, "SENSEX": 10}
                return defaults.get(index, 25)

            # Filter for the index options
            index_opts = instruments[
                (instruments["name"] == index)
                & (instruments["instrument_type"].isin(["CE", "PE"]))
            ]

            if index_opts.empty:
                defaults = {"NIFTY": 25, "BANKNIFTY": 15, "SENSEX": 10}
                return defaults.get(index, 25)

            # Get lot size from first matching instrument
            lot_size = index_opts.iloc[0]["lot_size"]
            logger.info(f"{index} lot size from Kite: {lot_size}")
            return int(lot_size)

        except Exception as e:
            logger.error(f"Failed to fetch lot size for {index}: {e}")
            defaults = {"NIFTY": 25, "BANKNIFTY": 15, "SENSEX": 10}
            return defaults.get(index, 25)

    async def get_nifty_options(
        self,
        expiry_date: datetime | None = None,
        strike_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """
        Get NIFTY options for specific expiry and strike range.

        Args:
            expiry_date: Option expiry date (None = current week)
            strike_range: (min_strike, max_strike) tuple

        Returns:
            DataFrame with CE and PE options
        """
        instruments = await self.fetch_instruments("NFO")
        if instruments.empty:
            return pd.DataFrame()

        # Filter NIFTY options
        nifty_opts = instruments[
            (instruments["name"] == "NIFTY")
            & (instruments["instrument_type"].isin(["CE", "PE"]))
        ].copy()

        if expiry_date:
            nifty_opts = nifty_opts[
                nifty_opts["expiry"] == expiry_date.strftime("%Y-%m-%d")
            ]
        else:
            # Get current weekly expiry
            today = datetime.now().date()
            # Convert expiry to date if it's not already
            nifty_opts["expiry"] = pd.to_datetime(nifty_opts["expiry"]).dt.date
            upcoming = nifty_opts[nifty_opts["expiry"] >= today]
            if not upcoming.empty:
                next_expiry = upcoming["expiry"].min()
                nifty_opts = nifty_opts[nifty_opts["expiry"] == next_expiry]

        if strike_range:
            min_strike, max_strike = strike_range
            nifty_opts = nifty_opts[
                (nifty_opts["strike"] >= min_strike)
                & (nifty_opts["strike"] <= max_strike)
            ]

        return nifty_opts.sort_values(["strike", "instrument_type"])

    async def get_banknifty_options(
        self,
        expiry_date: datetime | None = None,
        strike_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Get BANKNIFTY options similar to NIFTY."""
        instruments = await self.fetch_instruments("NFO")
        if instruments.empty:
            return pd.DataFrame()

        banknifty_opts = instruments[
            (instruments["name"] == "BANKNIFTY")
            & (instruments["instrument_type"].isin(["CE", "PE"]))
        ].copy()

        if expiry_date:
            banknifty_opts = banknifty_opts[
                banknifty_opts["expiry"] == expiry_date.strftime("%Y-%m-%d")
            ]
        else:
            today = datetime.now().date()
            # Convert expiry to date if it's not already
            banknifty_opts["expiry"] = pd.to_datetime(banknifty_opts["expiry"]).dt.date
            upcoming = banknifty_opts[banknifty_opts["expiry"] >= today]
            if not upcoming.empty:
                next_expiry = upcoming["expiry"].min()
                banknifty_opts = banknifty_opts[banknifty_opts["expiry"] == next_expiry]

        if strike_range:
            min_strike, max_strike = strike_range
            banknifty_opts = banknifty_opts[
                (banknifty_opts["strike"] >= min_strike)
                & (banknifty_opts["strike"] <= max_strike)
            ]

        return banknifty_opts.sort_values(["strike", "instrument_type"])

    async def get_sensex_options(
        self,
        expiry_date: datetime | None = None,
        strike_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Get SENSEX options from BFO exchange."""
        instruments = await self.fetch_instruments("BFO")
        if instruments.empty:
            return pd.DataFrame()

        sensex_opts = instruments[
            (instruments["name"] == "SENSEX")
            & (instruments["instrument_type"].isin(["CE", "PE"]))
        ].copy()

        if expiry_date:
            sensex_opts = sensex_opts[
                sensex_opts["expiry"] == expiry_date.strftime("%Y-%m-%d")
            ]
        else:
            today = datetime.now().date()
            # Convert expiry to date if it's not already
            sensex_opts["expiry"] = pd.to_datetime(sensex_opts["expiry"]).dt.date
            upcoming = sensex_opts[sensex_opts["expiry"] >= today]
            if not upcoming.empty:
                next_expiry = upcoming["expiry"].min()
                sensex_opts = sensex_opts[sensex_opts["expiry"] == next_expiry]

        if strike_range:
            min_strike, max_strike = strike_range
            sensex_opts = sensex_opts[
                (sensex_opts["strike"] >= min_strike)
                & (sensex_opts["strike"] <= max_strike)
            ]

        return sensex_opts.sort_values(["strike", "instrument_type"])

    async def get_next_expiry(self, index: str = "NIFTY") -> dict[str, Any]:
        """
        Get the next expiry date for an index from Kite instruments.

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX

        Returns:
            Dictionary with expiry details
        """
        try:
            today = datetime.now().date()
            current_time = datetime.now()

            # Determine exchange and index name
            if index.upper() == "SENSEX":
                exchange = "BFO"
                name = "SENSEX"
            else:
                exchange = "NFO"
                name = index.upper()

            instruments = await self.fetch_instruments(exchange)
            if instruments.empty:
                return {"error": "Could not fetch instruments"}

            # Filter options for the index
            options = instruments[
                (instruments["name"] == name)
                & (instruments["instrument_type"].isin(["CE", "PE"]))
            ].copy()

            if options.empty:
                return {"error": f"No options found for {index}"}

            # Convert expiry to date
            options["expiry"] = pd.to_datetime(options["expiry"]).dt.date

            # Get all unique expiry dates
            all_expiries = sorted(options["expiry"].unique())

            # Filter to future expiries (including today if market still open)
            market_close_hour = 15
            market_close_minute = 30

            if current_time.hour >= market_close_hour and current_time.minute >= market_close_minute:
                # Market closed, exclude today
                future_expiries = [e for e in all_expiries if e > today]
            else:
                # Market open or today is before market close
                future_expiries = [e for e in all_expiries if e >= today]

            if not future_expiries:
                return {"error": "No future expiries found"}

            next_expiry = future_expiries[0]
            days_to_expiry = (next_expiry - today).days
            is_expiry_day = (next_expiry == today)

            # Get all expiry dates for display (next 4 expiries)
            upcoming_expiries = future_expiries[:4]

            return {
                "index": index.upper(),
                "expiry_date": next_expiry,
                "days_to_expiry": days_to_expiry,
                "is_expiry_day": is_expiry_day,
                "expiry_weekday": next_expiry.strftime("%a"),  # Mon, Tue, etc.
                "all_expiries": upcoming_expiries,
            }

        except Exception as e:
            logger.error(f"Failed to get next expiry for {index}: {e}")
            return {"error": str(e)}

    async def get_all_indices_expiry(self) -> dict[str, Any]:
        """
        Get next expiry for all indices.

        Returns:
            Dictionary with expiry info for NIFTY, BANKNIFTY, SENSEX
        """
        results = {}
        for index in ["NIFTY", "BANKNIFTY", "SENSEX"]:
            expiry = await self.get_next_expiry(index)
            results[index] = expiry
        return results

    async def fetch_historical_data(
        self,
        instrument_token: int,
        timeframe: str = "5minute",
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        days: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            instrument_token: Zerodha instrument token
            timeframe: Candle interval (minute, 3minute, 5minute, etc.)
            from_date: Start date
            to_date: End date
            days: Number of days if from_date not specified

        Returns:
            DataFrame with OHLCV data
        """
        try:
            to_date = to_date or datetime.now()
            from_date = from_date or (to_date - timedelta(days=days))

            # Map timeframe
            interval = TIMEFRAME_MAP.get(timeframe, timeframe)

            data = await asyncio.to_thread(
                self.kite.historical_data,
                instrument_token,
                from_date,
                to_date,
                interval,
                continuous=False,
                oi=True,
            )

            df = pd.DataFrame(data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df = df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "oi": "OI",
                })

            logger.debug(f"Fetched {len(df)} candles for token {instrument_token}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    async def fetch_quote(self, instruments: list[str]) -> dict[str, Any]:
        """
        Fetch current quote for instruments.

        Args:
            instruments: List of instrument identifiers (e.g., ["NSE:NIFTY 50"])

        Returns:
            Quote data dictionary
        """
        try:
            return await asyncio.to_thread(self.kite.quote, instruments)
        except Exception as e:
            logger.error(f"Failed to fetch quote: {e}")
            return {}

    async def fetch_ohlc(self, instruments: list[str]) -> dict[str, Any]:
        """Fetch OHLC for instruments."""
        try:
            return await asyncio.to_thread(self.kite.ohlc, instruments)
        except Exception as e:
            logger.error(f"Failed to fetch OHLC: {e}")
            return {}

    async def fetch_ltp(self, instruments: list[str]) -> dict[str, Any]:
        """Fetch Last Traded Price."""
        try:
            return await asyncio.to_thread(self.kite.ltp, instruments)
        except Exception as e:
            logger.error(f"Failed to fetch LTP: {e}")
            return {}

    async def get_option_chain(
        self,
        index: str = "NIFTY",
        expiry_date: datetime | None = None,
        strike_count: int = 10,
    ) -> dict[str, Any]:
        """
        Build option chain with OI and Greeks data.

        Args:
            index: NIFTY, BANKNIFTY, or SENSEX
            expiry_date: Expiry date (None = current week)
            strike_count: Number of strikes around ATM

        Returns:
            Option chain data
        """
        # Check cache first (10 second TTL)
        cache_key = f"{index}_{strike_count}"
        now = datetime.now()
        if (
            cache_key in self._option_chain_cache
            and cache_key in self._option_chain_last_updated
            and (now - self._option_chain_last_updated[cache_key]).seconds < 10
        ):
            logger.debug(f"Using cached option chain for {index}")
            return self._option_chain_cache[cache_key]

        try:
            # Get current spot price
            if index == "NIFTY":
                spot_key = "NSE:NIFTY 50"
            elif index == "BANKNIFTY":
                spot_key = "NSE:NIFTY BANK"
            elif index == "SENSEX":
                spot_key = "BSE:SENSEX"
            else:
                spot_key = f"NSE:{index}"
            quote = await self.fetch_quote([spot_key])
            spot_price = quote.get(spot_key, {}).get("last_price", 0)

            if not spot_price:
                return {"error": "Could not fetch spot price"}

            # Round to nearest strike
            if index == "NIFTY":
                strike_interval = 50
            elif index == "SENSEX":
                strike_interval = 100
            else:
                strike_interval = 100
            atm_strike = round(spot_price / strike_interval) * strike_interval

            # Get options
            if index == "NIFTY":
                options = await self.get_nifty_options(
                    expiry_date=expiry_date,
                    strike_range=(
                        atm_strike - (strike_count * strike_interval),
                        atm_strike + (strike_count * strike_interval),
                    ),
                )
            elif index == "SENSEX":
                options = await self.get_sensex_options(
                    expiry_date=expiry_date,
                    strike_range=(
                        atm_strike - (strike_count * strike_interval),
                        atm_strike + (strike_count * strike_interval),
                    ),
                )
            else:
                options = await self.get_banknifty_options(
                    expiry_date=expiry_date,
                    strike_range=(
                        atm_strike - (strike_count * strike_interval),
                        atm_strike + (strike_count * strike_interval),
                    ),
                )

            if options.empty:
                return {"error": "No options found"}

            # Determine exchange for options
            exchange = "BFO" if index == "SENSEX" else "NFO"

            # Build instrument strings for quote
            instrument_strings = [
                f"{exchange}:{row['tradingsymbol']}" for _, row in options.iterrows()
            ]

            # Fetch quotes for all options
            quotes = await self.fetch_quote(instrument_strings)

            # Build chain
            chain = []
            strikes = sorted(options["strike"].unique())

            for strike in strikes:
                ce_row = options[
                    (options["strike"] == strike) & (options["instrument_type"] == "CE")
                ]
                pe_row = options[
                    (options["strike"] == strike) & (options["instrument_type"] == "PE")
                ]

                ce_data = {}
                pe_data = {}

                if not ce_row.empty:
                    ce_symbol = f"{exchange}:{ce_row.iloc[0]['tradingsymbol']}"
                    ce_quote = quotes.get(ce_symbol, {})
                    ce_data = {
                        "symbol": ce_row.iloc[0]["tradingsymbol"],
                        "token": ce_row.iloc[0]["instrument_token"],
                        "ltp": ce_quote.get("last_price", 0),
                        "oi": ce_quote.get("oi", 0),
                        "volume": ce_quote.get("volume", 0),
                        "bid": ce_quote.get("depth", {}).get("buy", [{}])[0].get("price", 0),
                        "ask": ce_quote.get("depth", {}).get("sell", [{}])[0].get("price", 0),
                        "change": ce_quote.get("net_change", 0),
                        "change_pct": ce_quote.get("change", 0),
                    }

                if not pe_row.empty:
                    pe_symbol = f"{exchange}:{pe_row.iloc[0]['tradingsymbol']}"
                    pe_quote = quotes.get(pe_symbol, {})
                    pe_data = {
                        "symbol": pe_row.iloc[0]["tradingsymbol"],
                        "token": pe_row.iloc[0]["instrument_token"],
                        "ltp": pe_quote.get("last_price", 0),
                        "oi": pe_quote.get("oi", 0),
                        "volume": pe_quote.get("volume", 0),
                        "bid": pe_quote.get("depth", {}).get("buy", [{}])[0].get("price", 0),
                        "ask": pe_quote.get("depth", {}).get("sell", [{}])[0].get("price", 0),
                        "change": pe_quote.get("net_change", 0),
                        "change_pct": pe_quote.get("change", 0),
                    }

                chain.append({
                    "strike": strike,
                    "is_atm": strike == atm_strike,
                    "ce": ce_data,
                    "pe": pe_data,
                })

            # Calculate totals
            total_ce_oi = sum(c["ce"].get("oi", 0) for c in chain)
            total_pe_oi = sum(c["pe"].get("oi", 0) for c in chain)
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0

            result = {
                "index": index,
                "spot_price": spot_price,
                "atm_strike": atm_strike,
                "expiry": options.iloc[0]["expiry"] if not options.empty else None,
                "chain": chain,
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "pcr": round(pcr, 3),
                "timestamp": datetime.now().isoformat(),
            }

            # Cache the result
            self._option_chain_cache[cache_key] = result
            self._option_chain_last_updated[cache_key] = datetime.now()

            return result

        except Exception as e:
            logger.error(f"Failed to build option chain: {e}")
            return {"error": str(e)}


# Singleton instance
_data_fetcher: DataFetcher | None = None


def get_data_fetcher() -> DataFetcher:
    """Get or create data fetcher instance."""
    global _data_fetcher
    if _data_fetcher is None:
        _data_fetcher = DataFetcher()
    return _data_fetcher
