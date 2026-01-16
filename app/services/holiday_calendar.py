"""
Holiday Calendar Service
Fetches and caches NSE market holidays dynamically
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Set
import httpx
from loguru import logger


class HolidayCalendar:
    """
    Service to fetch and cache NSE market holidays.

    Fetches holidays from NSE website and caches them for 24 hours.
    Falls back to previous year's holidays if fetch fails.
    """

    def __init__(self):
        self._holidays: Set[date] = set()
        self._last_fetch: datetime | None = None
        self._cache_duration = timedelta(hours=24)
        self._fetch_lock = asyncio.Lock()

    async def get_holidays(self) -> Set[date]:
        """
        Get market holidays for current and next year.

        Returns:
            Set of holiday dates
        """
        # Check if cache is still valid
        if (
            self._last_fetch
            and datetime.now() - self._last_fetch < self._cache_duration
            and self._holidays
        ):
            return self._holidays

        # Fetch new holidays
        async with self._fetch_lock:
            # Double-check after acquiring lock
            if (
                self._last_fetch
                and datetime.now() - self._last_fetch < self._cache_duration
                and self._holidays
            ):
                return self._holidays

            await self._fetch_holidays()
            return self._holidays

    async def _fetch_holidays(self):
        """Fetch holidays from NSE API."""
        try:
            current_year = datetime.now().year
            next_year = current_year + 1

            holidays = set()

            # Try to fetch from NSE holiday calendar API
            # NSE provides holiday calendar on their website
            # Format: https://www.nseindia.com/api/holiday-master?type=trading

            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                }

                try:
                    response = await client.get(
                        'https://www.nseindia.com/api/holiday-master?type=trading',
                        headers=headers,
                        follow_redirects=True
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Parse holiday data
                        # NSE API returns: {"CM": [...], "FO": [...], ...}
                        # We need FO (Futures & Options) holidays
                        fo_holidays = data.get('FO', [])

                        for holiday in fo_holidays:
                            holiday_date_str = holiday.get('tradingDate')
                            if holiday_date_str:
                                # Parse date (format: "DD-MMM-YYYY" like "26-Jan-2024")
                                try:
                                    holiday_date = datetime.strptime(holiday_date_str, '%d-%b-%Y').date()
                                    # Only add if in current or next year
                                    if holiday_date.year in [current_year, next_year]:
                                        holidays.add(holiday_date)
                                except ValueError:
                                    logger.warning(f"Could not parse holiday date: {holiday_date_str}")

                        if holidays:
                            self._holidays = holidays
                            self._last_fetch = datetime.now()
                            logger.info(f"Fetched {len(holidays)} market holidays from NSE API")
                            return

                except Exception as e:
                    logger.warning(f"Failed to fetch from NSE API: {e}")

            # Fallback: Use hardcoded holidays for current year if fetch failed
            if not holidays:
                logger.warning("Using fallback holiday list")
                self._holidays = self._get_fallback_holidays()
                self._last_fetch = datetime.now()

        except Exception as e:
            logger.error(f"Error fetching holidays: {e}")
            # Use fallback if we don't have any holidays
            if not self._holidays:
                self._holidays = self._get_fallback_holidays()
                self._last_fetch = datetime.now()

    def _get_fallback_holidays(self) -> Set[date]:
        """
        Fallback holiday list based on typical NSE holidays.

        Includes:
        - Republic Day (Jan 26)
        - Mahashivratri (Feb/Mar - varies)
        - Holi (Mar - varies)
        - Good Friday (Mar/Apr - varies)
        - Ram Navami (Apr - varies)
        - Mahavir Jayanti (Apr - varies)
        - Ambedkar Jayanti (Apr 14)
        - Id-ul-Fitr (varies)
        - Maharashtra Day (May 1)
        - Buddha Purnima (May - varies)
        - Id-ul-Adha (varies)
        - Muharram (varies)
        - Independence Day (Aug 15)
        - Janmashtami (Aug/Sep - varies)
        - Ganesh Chaturthi (Sep - varies)
        - Gandhi Jayanti (Oct 2)
        - Dussehra (Sep/Oct - varies)
        - Diwali (Oct/Nov - varies - multiple days)
        - Guru Nanak Jayanti (Nov - varies)
        - Christmas (Dec 25)

        Returns:
            Set of date objects for known holidays
        """
        # 2024 holidays
        holidays_2024 = {
            date(2024, 1, 26),   # Republic Day
            date(2024, 3, 8),    # Mahashivratri
            date(2024, 3, 25),   # Holi
            date(2024, 3, 29),   # Good Friday
            date(2024, 4, 11),   # Id-ul-Fitr
            date(2024, 4, 14),   # Ambedkar Jayanti / Mahavir Jayanti
            date(2024, 4, 17),   # Ram Navami
            date(2024, 4, 21),   # Mahavir Jayanti
            date(2024, 5, 1),    # Maharashtra Day
            date(2024, 5, 23),   # Buddha Purnima
            date(2024, 6, 17),   # Id-ul-Adha (Bakri Id)
            date(2024, 7, 17),   # Muharram
            date(2024, 8, 15),   # Independence Day
            date(2024, 9, 16),   # Milad-un-Nabi
            date(2024, 10, 2),   # Gandhi Jayanti
            date(2024, 10, 12),  # Dussehra
            date(2024, 11, 1),   # Diwali Laxmi Pujan
            date(2024, 11, 15),  # Guru Nanak Jayanti
            date(2024, 12, 25),  # Christmas
        }

        # 2025 holidays (estimated based on lunar calendar)
        holidays_2025 = {
            date(2025, 1, 26),   # Republic Day
            date(2025, 2, 26),   # Mahashivratri
            date(2025, 3, 14),   # Holi
            date(2025, 3, 31),   # Id-ul-Fitr
            date(2025, 4, 14),   # Ambedkar Jayanti / Mahavir Jayanti
            date(2025, 4, 18),   # Good Friday
            date(2025, 5, 1),    # Maharashtra Day
            date(2025, 5, 12),   # Buddha Purnima
            date(2025, 6, 7),    # Id-ul-Adha
            date(2025, 8, 15),   # Independence Day
            date(2025, 8, 27),   # Janmashtami
            date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
            date(2025, 10, 20),  # Diwali Laxmi Pujan
            date(2025, 10, 21),  # Diwali Balipratipada
            date(2025, 11, 5),   # Guru Nanak Jayanti
            date(2025, 12, 25),  # Christmas
        }

        # 2026 holidays (estimated)
        holidays_2026 = {
            date(2026, 1, 26),   # Republic Day
            date(2026, 3, 3),    # Holi
            date(2026, 3, 20),   # Id-ul-Fitr
            date(2026, 4, 3),    # Good Friday
            date(2026, 4, 6),    # Ram Navami
            date(2026, 4, 14),   # Ambedkar Jayanti / Mahavir Jayanti
            date(2026, 5, 1),    # Maharashtra Day
            date(2026, 5, 27),   # Id-ul-Adha
            date(2026, 8, 15),   # Independence Day
            date(2026, 9, 16),   # Ganesh Chaturthi
            date(2026, 10, 2),   # Gandhi Jayanti
            date(2026, 10, 22),  # Dussehra
            date(2026, 11, 8),   # Diwali Laxmi Pujan
            date(2026, 11, 9),   # Diwali Balipratipada
            date(2026, 11, 25),  # Guru Nanak Jayanti
            date(2026, 12, 25),  # Christmas
        }

        return holidays_2024 | holidays_2025 | holidays_2026

    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if a given date is a trading day.

        Args:
            check_date: Date to check

        Returns:
            True if it's a trading day, False if weekend or holiday
        """
        # Weekend check
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Holiday check
        return check_date not in self._holidays


# Global singleton
_holiday_calendar: HolidayCalendar | None = None


def get_holiday_calendar() -> HolidayCalendar:
    """Get the global holiday calendar instance."""
    global _holiday_calendar
    if _holiday_calendar is None:
        _holiday_calendar = HolidayCalendar()
    return _holiday_calendar


async def get_market_holidays() -> Set[date]:
    """
    Convenience function to get market holidays.

    Returns:
        Set of holiday dates
    """
    calendar = get_holiday_calendar()
    return await calendar.get_holidays()
