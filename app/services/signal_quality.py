"""
Signal Quality Scoring System
Filters and scores signals based on multi-factor quality analysis
Only high-quality signals (score > 70) are shown and traded
"""

from dataclasses import dataclass
from loguru import logger

from app.services.signal_engine import TradeSignal, SignalType


@dataclass
class QualityScore:
    """Signal quality score breakdown."""
    total_score: float  # 0-100
    trend_score: float  # 0-30
    momentum_score: float  # 0-20
    volume_score: float  # 0-15
    greeks_score: float  # 0-20
    pcr_score: float  # 0-10
    risk_reward_score: float  # 0-5

    is_high_quality: bool
    factors: list[str]
    warnings: list[str]


class SignalQualityAnalyzer:
    """
    Analyzes signal quality using multi-factor scoring.

    Scoring breakdown (total 100 points):
    1. Trend Strength (30 points):
       - SuperTrend alignment (10)
       - EMA alignment (10)
       - ADX strength (10)

    2. Momentum Confirmation (20 points):
       - RSI position (7)
       - MACD alignment (7)
       - Stochastic confirmation (6)

    3. Volume Analysis (15 points):
       - Above average volume (10)
       - Volume trend (5)

    4. Greeks Quality (20 points):
       - Delta in good range (10)
       - Theta acceptable (5)
       - Liquidity (bid-ask spread) (5)

    5. PCR Analysis (10 points):
       - Market sentiment alignment (10)

    6. Risk/Reward (5 points):
       - R:R ratio quality (5)
    """

    HIGH_QUALITY_THRESHOLD = 70  # Minimum score for high quality

    def analyze_quality(self, signal: TradeSignal) -> QualityScore:
        """
        Analyze signal quality and return score breakdown.

        Args:
            signal: TradeSignal to analyze

        Returns:
            QualityScore with detailed breakdown
        """
        trend_score = self._score_trend(signal)
        momentum_score = self._score_momentum(signal)
        volume_score = self._score_volume(signal)
        greeks_score = self._score_greeks(signal)
        pcr_score = self._score_pcr(signal)
        rr_score = self._score_risk_reward(signal)

        total = trend_score + momentum_score + volume_score + greeks_score + pcr_score + rr_score

        factors = []
        warnings = []

        # Collect quality factors
        if trend_score >= 24:  # 80% of 30
            factors.append("Strong trend alignment")
        elif trend_score < 18:  # 60% of 30
            warnings.append("Weak trend alignment")

        if momentum_score >= 16:  # 80% of 20
            factors.append("Strong momentum confirmation")
        elif momentum_score < 12:  # 60% of 20
            warnings.append("Weak momentum")

        if volume_score >= 12:  # 80% of 15
            factors.append("High volume support")
        elif volume_score < 9:  # 60% of 15
            warnings.append("Low volume")

        if greeks_score >= 16:  # 80% of 20
            factors.append("Excellent Greeks profile")
        elif greeks_score < 12:  # 60% of 20
            warnings.append("Poor Greeks quality")

        if pcr_score >= 8:  # 80% of 10
            factors.append("PCR sentiment aligned")

        if rr_score >= 4:  # 80% of 5
            factors.append(f"Excellent R:R ({signal.risk_reward:.2f})")
        elif rr_score < 3:  # 60% of 5
            warnings.append(f"Poor R:R ({signal.risk_reward:.2f})")

        # Additional quality factors from signal
        if signal.confidence > 70:
            factors.append(f"High confidence ({signal.confidence:.1f}%)")

        if signal.signal_type in [SignalType.STRONG_CE, SignalType.STRONG_PE]:
            factors.append("Strong directional signal")

        is_high_quality = total >= self.HIGH_QUALITY_THRESHOLD

        return QualityScore(
            total_score=round(total, 1),
            trend_score=round(trend_score, 1),
            momentum_score=round(momentum_score, 1),
            volume_score=round(volume_score, 1),
            greeks_score=round(greeks_score, 1),
            pcr_score=round(pcr_score, 1),
            risk_reward_score=round(rr_score, 1),
            is_high_quality=is_high_quality,
            factors=factors,
            warnings=warnings,
        )

    def _score_trend(self, signal: TradeSignal) -> float:
        """Score trend strength (max 30 points)."""
        score = 0.0

        # Get indicator signals
        supertrend = self._get_indicator(signal, "SuperTrend")
        ema = self._get_indicator(signal, "EMA Crossover")
        adx = self._get_indicator(signal, "ADX")

        # SuperTrend alignment (10 points)
        if supertrend:
            if supertrend.signal == signal.direction.lower():
                score += 10 * supertrend.strength

        # EMA alignment (10 points)
        if ema:
            if ema.signal == signal.direction.lower():
                score += 10 * ema.strength

        # ADX strength (10 points)
        if adx:
            adx_value = adx.value
            if adx_value >= 40:  # Very strong trend
                score += 10
            elif adx_value >= 30:  # Strong trend
                score += 8
            elif adx_value >= 25:  # Decent trend
                score += 6
            elif adx_value >= 20:  # Weak trend
                score += 3

        return min(score, 30)

    def _score_momentum(self, signal: TradeSignal) -> float:
        """Score momentum confirmation (max 20 points)."""
        score = 0.0

        rsi = self._get_indicator(signal, "RSI")
        macd = self._get_indicator(signal, "MACD")
        stoch = self._get_indicator(signal, "Stochastic")

        # RSI position (7 points)
        if rsi:
            rsi_value = rsi.value
            if signal.direction == "CE":
                # For CE, RSI should be strong but not overbought
                if 55 <= rsi_value <= 75:
                    score += 7
                elif 50 <= rsi_value < 55 or 75 < rsi_value <= 80:
                    score += 5
                elif 45 <= rsi_value < 50:
                    score += 3
            else:  # PE
                # For PE, RSI should be weak but not oversold
                if 25 <= rsi_value <= 45:
                    score += 7
                elif 20 <= rsi_value < 25 or 45 < rsi_value <= 50:
                    score += 5
                elif 50 < rsi_value <= 55:
                    score += 3

        # MACD alignment (7 points)
        if macd:
            if macd.signal == signal.direction.lower():
                score += 7 * macd.strength

        # Stochastic confirmation (6 points)
        if stoch:
            if stoch.signal == signal.direction.lower():
                score += 6 * stoch.strength

        return min(score, 20)

    def _score_volume(self, signal: TradeSignal) -> float:
        """Score volume quality (max 15 points)."""
        score = 0.0

        # Check for volume-related supporting factors
        volume_factors = [
            f for f in signal.supporting_factors
            if "volume" in f.lower() or "oi" in f.lower()
        ]

        # Award points based on volume mentions
        if len(volume_factors) >= 2:
            score += 15
        elif len(volume_factors) == 1:
            score += 10
        else:
            # Base volume score if no specific mentions
            score += 7

        return min(score, 15)

    def _score_greeks(self, signal: TradeSignal) -> float:
        """Score Greeks quality (max 20 points)."""
        score = 0.0

        if not signal.recommended_option:
            return 0

        option = signal.recommended_option

        # Delta quality (10 points)
        if option.delta is not None:
            delta_abs = abs(option.delta)

            # For ATM/near-ATM options (best for trading)
            if 0.40 <= delta_abs <= 0.60:  # ATM range
                score += 10
            elif 0.35 <= delta_abs < 0.40 or 0.60 < delta_abs <= 0.65:  # Near ATM
                score += 8
            elif 0.30 <= delta_abs < 0.35 or 0.65 < delta_abs <= 0.75:  # Slightly OTM/ITM
                score += 6
            elif 0.20 <= delta_abs < 0.30 or 0.75 < delta_abs <= 0.85:  # More OTM/ITM
                score += 4
            else:  # Too far OTM or deep ITM
                score += 2

        # Theta quality (5 points)
        # Lower absolute theta is better (less time decay)
        if option.theta is not None:
            theta_abs = abs(option.theta)
            if theta_abs <= 0.03:
                score += 5
            elif theta_abs <= 0.05:
                score += 4
            elif theta_abs <= 0.08:
                score += 3
            elif theta_abs <= 0.12:
                score += 2
            else:
                score += 1

        # Liquidity (bid-ask spread) (5 points)
        if option.bid > 0 and option.ask > 0:
            spread_pct = (option.ask - option.bid) / option.ltp
            if spread_pct <= 0.02:  # 2% or less
                score += 5
            elif spread_pct <= 0.05:  # 5% or less
                score += 4
            elif spread_pct <= 0.10:  # 10% or less
                score += 3
            else:
                score += 1

        # High OI bonus
        if option.oi >= 100000:
            score += 2

        return min(score, 20)

    def _score_pcr(self, signal: TradeSignal) -> float:
        """Score PCR sentiment alignment (max 10 points)."""
        score = 0.0

        pcr = self._get_indicator(signal, "PCR")
        if not pcr:
            return 5  # Neutral if no PCR data

        pcr_value = pcr.value

        # CE signals work best in extreme fear (high PCR > 1.3)
        # PE signals work best in extreme greed (low PCR < 0.7)
        if signal.direction == "CE":
            if pcr_value >= 1.3:  # Extreme fear
                score += 10
            elif pcr_value >= 1.1:  # Fear
                score += 7
            elif pcr_value >= 0.9:  # Neutral
                score += 5
            else:  # Greed
                score += 3
        else:  # PE
            if pcr_value <= 0.7:  # Extreme greed
                score += 10
            elif pcr_value <= 0.9:  # Greed
                score += 7
            elif pcr_value <= 1.1:  # Neutral
                score += 5
            else:  # Fear
                score += 3

        return min(score, 10)

    def _score_risk_reward(self, signal: TradeSignal) -> float:
        """Score risk/reward ratio (max 5 points)."""
        rr = signal.risk_reward

        if rr >= 3.0:
            return 5
        elif rr >= 2.5:
            return 4.5
        elif rr >= 2.0:
            return 4
        elif rr >= 1.5:
            return 3
        elif rr >= 1.2:
            return 2
        else:
            return 1

    def _get_indicator(self, signal: TradeSignal, name: str):
        """Get indicator by name from signal."""
        for ind in signal.indicators:
            if ind.name == name:
                return ind
        return None


# Singleton instance
_quality_analyzer: SignalQualityAnalyzer | None = None


def get_quality_analyzer() -> SignalQualityAnalyzer:
    """Get or create quality analyzer instance."""
    global _quality_analyzer
    if _quality_analyzer is None:
        _quality_analyzer = SignalQualityAnalyzer()
    return _quality_analyzer
