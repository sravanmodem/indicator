"""
Signal Quality Analyzer
Evaluates the quality of trade signals based on multiple factors
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from app.services.signal_engine import TradeSignal


@dataclass
class QualityScore:
    """Quality score breakdown for a signal."""

    # Component scores (max values)
    trend: float = 0.0  # Max 30
    momentum: float = 0.0  # Max 20
    volume: float = 0.0  # Max 15
    greeks: float = 0.0  # Max 20
    pcr: float = 0.0  # Max 10
    rr: float = 0.0  # Max 5 (risk/reward)

    # Total and meta
    total_score: float = 0.0  # Max 100
    is_high_quality: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Alias for total_score for template compatibility."""
        return self.total_score


class SignalQualityAnalyzer:
    """
    Analyzes signal quality based on multiple factors.

    Quality Components:
    - Trend Strength (30 pts): SuperTrend alignment, EMA alignment, ADX strength
    - Momentum (20 pts): RSI position, MACD confirmation, Stochastic alignment
    - Volume (15 pts): Volume confirmation, OBV trend
    - Greeks Quality (20 pts): Delta, Theta, IV levels
    - PCR Alignment (10 pts): Put-Call ratio supporting direction
    - Risk/Reward (5 pts): R:R ratio quality
    """

    def __init__(self):
        self.min_high_quality_score = 70

    def analyze_quality(self, signal: "TradeSignal") -> QualityScore:
        """
        Analyze the quality of a trade signal.

        Args:
            signal: TradeSignal object to analyze

        Returns:
            QualityScore with component breakdown
        """
        score = QualityScore()
        warnings = []

        try:
            # Analyze trend strength (30 pts)
            score.trend = self._analyze_trend(signal, warnings)

            # Analyze momentum (20 pts)
            score.momentum = self._analyze_momentum(signal, warnings)

            # Analyze volume (15 pts)
            score.volume = self._analyze_volume(signal, warnings)

            # Analyze greeks quality (20 pts)
            score.greeks = self._analyze_greeks(signal, warnings)

            # Analyze PCR alignment (10 pts)
            score.pcr = self._analyze_pcr(signal, warnings)

            # Analyze risk/reward (5 pts)
            score.rr = self._analyze_risk_reward(signal, warnings)

            # Calculate total
            score.total_score = (
                score.trend
                + score.momentum
                + score.volume
                + score.greeks
                + score.pcr
                + score.rr
            )

            # Determine if high quality
            score.is_high_quality = score.total_score >= self.min_high_quality_score

            # Add warning factors from signal
            score.warnings = warnings + signal.warning_factors

        except Exception as e:
            logger.error(f"Error analyzing signal quality: {e}")
            score.warnings.append(f"Analysis error: {str(e)[:50]}")

        return score

    def _analyze_trend(self, signal: "TradeSignal", warnings: list) -> float:
        """Analyze trend strength indicators (max 30 pts)."""
        points = 0.0

        for ind in signal.indicators:
            name = ind.name.lower()

            # SuperTrend (10 pts)
            if "supertrend" in name:
                if ind.signal == signal.direction.lower():
                    points += 10 * ind.strength
                elif ind.signal == "neutral":
                    points += 3
                else:
                    warnings.append("SuperTrend against signal direction")

            # EMA alignment (10 pts)
            elif "ema" in name:
                if ind.signal == signal.direction.lower():
                    points += 10 * ind.strength
                elif ind.signal == "neutral":
                    points += 3

            # ADX strength (10 pts)
            elif "adx" in name:
                if isinstance(ind.value, dict):
                    adx_value = ind.value.get("adx", 0)
                else:
                    adx_value = ind.value if ind.value else 0

                if adx_value >= 40:
                    points += 10
                elif adx_value >= 30:
                    points += 7
                elif adx_value >= 25:
                    points += 5
                elif adx_value >= 20:
                    points += 3
                else:
                    warnings.append(f"Weak trend strength (ADX: {adx_value:.1f})")
                    points += 1

        return min(points, 30)

    def _analyze_momentum(self, signal: "TradeSignal", warnings: list) -> float:
        """Analyze momentum indicators (max 20 pts)."""
        points = 0.0

        for ind in signal.indicators:
            name = ind.name.lower()

            # RSI (8 pts)
            if "rsi" in name:
                if ind.signal == signal.direction.lower():
                    points += 8 * ind.strength
                elif ind.signal == "neutral":
                    points += 4
                else:
                    warnings.append("RSI divergence from signal")

            # MACD (7 pts)
            elif "macd" in name:
                if ind.signal == signal.direction.lower():
                    points += 7 * ind.strength
                elif ind.signal == "neutral":
                    points += 3

            # Stochastic (5 pts)
            elif "stoch" in name:
                if ind.signal == signal.direction.lower():
                    points += 5 * ind.strength
                elif ind.signal == "neutral":
                    points += 2

        return min(points, 20)

    def _analyze_volume(self, signal: "TradeSignal", warnings: list) -> float:
        """Analyze volume indicators (max 15 pts)."""
        points = 0.0

        for ind in signal.indicators:
            name = ind.name.lower()

            # OBV trend (8 pts)
            if "obv" in name:
                if ind.signal == signal.direction.lower():
                    points += 8 * ind.strength
                elif ind.signal == "neutral":
                    points += 3
                else:
                    warnings.append("Volume not confirming move")

            # Volume analysis (7 pts)
            elif "volume" in name:
                if ind.signal == signal.direction.lower():
                    points += 7 * ind.strength
                elif ind.signal == "neutral":
                    points += 3

        # If no volume indicators, give base points
        if points == 0:
            points = 7  # Default moderate volume assumption

        return min(points, 15)

    def _analyze_greeks(self, signal: "TradeSignal", warnings: list) -> float:
        """Analyze option greeks quality (max 20 pts)."""
        points = 0.0

        if not signal.recommended_option:
            return 10  # Default if no option recommended

        opt = signal.recommended_option

        # Delta quality (8 pts) - Prefer ATM options (0.4-0.6 delta)
        if opt.delta is not None:
            abs_delta = abs(opt.delta)
            if 0.4 <= abs_delta <= 0.6:
                points += 8  # ATM - best for directional
            elif 0.3 <= abs_delta < 0.4 or 0.6 < abs_delta <= 0.7:
                points += 6  # Near ATM
            elif 0.2 <= abs_delta < 0.3 or 0.7 < abs_delta <= 0.8:
                points += 4  # Slightly ITM/OTM
            else:
                points += 2
                warnings.append(f"Delta ({abs_delta:.2f}) not optimal")

        # Theta impact (6 pts) - Lower theta decay is better for buyers
        if opt.theta is not None:
            # Theta is negative for long options
            abs_theta = abs(opt.theta)
            if abs_theta < 5:
                points += 6  # Minimal decay
            elif abs_theta < 10:
                points += 4
            elif abs_theta < 20:
                points += 2
            else:
                warnings.append(f"High theta decay: {abs_theta:.1f}")
                points += 1

        # Vega/IV quality (6 pts) - Moderate IV is ideal
        if opt.vega is not None:
            if opt.vega > 0.1:
                points += 6  # Good vega exposure
            elif opt.vega > 0.05:
                points += 4
            else:
                points += 2

        # If no greeks available, give base points
        if points == 0:
            points = 10

        return min(points, 20)

    def _analyze_pcr(self, signal: "TradeSignal", warnings: list) -> float:
        """Analyze put-call ratio alignment (max 10 pts)."""
        points = 0.0

        for ind in signal.indicators:
            name = ind.name.lower()

            if "pcr" in name or "put" in name and "call" in name:
                if ind.signal == signal.direction.lower():
                    points += 10 * ind.strength
                elif ind.signal == "neutral":
                    points += 5
                else:
                    warnings.append("PCR not supporting signal direction")
                    points += 2
                break

        # If no PCR indicator, give neutral points
        if points == 0:
            points = 5

        return min(points, 10)

    def _analyze_risk_reward(self, signal: "TradeSignal", warnings: list) -> float:
        """Analyze risk/reward ratio (max 5 pts)."""
        rr = signal.risk_reward

        if rr >= 3.0:
            return 5  # Excellent R:R
        elif rr >= 2.5:
            return 4
        elif rr >= 2.0:
            return 3
        elif rr >= 1.5:
            return 2
        elif rr >= 1.0:
            warnings.append(f"Low risk/reward ratio: {rr:.1f}")
            return 1
        else:
            warnings.append(f"Poor risk/reward ratio: {rr:.1f}")
            return 0


# Singleton instance
_quality_analyzer: SignalQualityAnalyzer | None = None


def get_quality_analyzer() -> SignalQualityAnalyzer:
    """Get or create the quality analyzer singleton."""
    global _quality_analyzer
    if _quality_analyzer is None:
        _quality_analyzer = SignalQualityAnalyzer()
    return _quality_analyzer
