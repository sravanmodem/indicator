"""
Signal History Service
Tracks and stores signal history for analysis
"""

from datetime import datetime, timedelta
from typing import Any

from loguru import logger


class SignalHistoryService:
    """
    Service for tracking signal history.

    Stores signals in memory for now (can be extended to database).
    """

    def __init__(self):
        self._signals: list[dict] = []

    def add_signal(self, signal_data: dict) -> None:
        """Add a signal to history."""
        signal_data["recorded_at"] = datetime.now()
        self._signals.append(signal_data)

        # Keep only last 1000 signals
        if len(self._signals) > 1000:
            self._signals = self._signals[-1000:]

    def get_signals(
        self,
        days: int = 30,
        index: str | None = None,
        signal_type: str | None = None,
        min_quality: float | None = None,
    ) -> list[dict]:
        """
        Get signals from history with optional filters.

        Args:
            days: Number of days to look back
            index: Filter by index (NIFTY, BANKNIFTY, SENSEX)
            signal_type: Filter by type (CE, PE)
            min_quality: Minimum quality score

        Returns:
            List of matching signals
        """
        cutoff = datetime.now() - timedelta(days=days)

        results = []
        for signal in self._signals:
            recorded = signal.get("recorded_at", datetime.now())
            if recorded < cutoff:
                continue

            if index and signal.get("index") != index:
                continue

            if signal_type and signal.get("signal_type") != signal_type:
                continue

            if min_quality and signal.get("quality_score", 0) < min_quality:
                continue

            results.append(signal)

        return sorted(results, key=lambda x: x.get("recorded_at", datetime.min), reverse=True)

    def get_statistics(self, days: int = 30) -> dict[str, Any]:
        """
        Get statistics for signals over a period.

        Args:
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        signals = self.get_signals(days=days)

        if not signals:
            return {
                "total_signals": 0,
                "win_rate": 0,
                "avg_quality_score": 0,
                "target_hit": 0,
                "stop_loss_hit": 0,
                "pending": 0,
                "total_pnl": 0,
                "ce_signals": 0,
                "pe_signals": 0,
                "high_quality_count": 0,
            }

        total = len(signals)
        target_hit = sum(1 for s in signals if s.get("outcome") == "target_hit")
        sl_hit = sum(1 for s in signals if s.get("outcome") == "stop_loss_hit")
        pending = sum(1 for s in signals if s.get("outcome") in (None, "pending", "open"))
        closed = target_hit + sl_hit

        win_rate = (target_hit / closed * 100) if closed > 0 else 0

        quality_scores = [s.get("quality_score", 0) for s in signals if s.get("quality_score")]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        total_pnl = sum(s.get("pnl", 0) for s in signals)

        ce_signals = sum(1 for s in signals if s.get("signal_type") in ("CE", "ce", "STRONG_CE"))
        pe_signals = sum(1 for s in signals if s.get("signal_type") in ("PE", "pe", "STRONG_PE"))

        high_quality = sum(1 for s in signals if s.get("quality_score", 0) >= 70)

        return {
            "total_signals": total,
            "win_rate": round(win_rate, 1),
            "avg_quality_score": round(avg_quality, 1),
            "target_hit": target_hit,
            "stop_loss_hit": sl_hit,
            "pending": pending,
            "total_pnl": round(total_pnl, 2),
            "ce_signals": ce_signals,
            "pe_signals": pe_signals,
            "high_quality_count": high_quality,
        }

    def update_outcome(self, signal_id: str, outcome: str, pnl: float = 0) -> bool:
        """
        Update the outcome of a signal.

        Args:
            signal_id: Signal identifier
            outcome: 'target_hit', 'stop_loss_hit', 'manual_exit'
            pnl: Profit/Loss amount

        Returns:
            True if updated, False if not found
        """
        for signal in self._signals:
            if signal.get("id") == signal_id:
                signal["outcome"] = outcome
                signal["pnl"] = pnl
                signal["closed_at"] = datetime.now()
                return True
        return False

    def clear_history(self) -> None:
        """Clear all signal history."""
        self._signals.clear()

    def get_settings(self) -> dict[str, Any] | None:
        """
        Get trading settings.

        Returns default settings for now.
        """
        return {
            "total_capital": 250000.0,  # 2.5L per index
            "max_daily_loss_pct": 20.0,
            "max_trade_loss_pct": 10.0,
            "fund_utilization_pct": 100.0,
        }

    def get_recent_signals(self, limit: int = 10, days: int = 7) -> list[dict]:
        """
        Get recent signals.

        Args:
            limit: Maximum number of signals to return
            days: Number of days to look back

        Returns:
            List of recent signals
        """
        signals = self.get_signals(days=days)
        return signals[:limit]


# Singleton instance
_history_service: SignalHistoryService | None = None


def get_history_service() -> SignalHistoryService:
    """Get or create the history service singleton."""
    global _history_service
    if _history_service is None:
        _history_service = SignalHistoryService()
    return _history_service
