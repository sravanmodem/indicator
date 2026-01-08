"""
AI Trading Service
Claude Haiku integration for intelligent trade decisions

Features:
- Entry signal analysis with 15-min OHLCV + indicators + Greeks
- Exit signal generation (waits for AI decision)
- Expiry day ultra-pro strategy support
- Auto-execute on AI approval
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from loguru import logger

from app.core.config import get_settings


@dataclass
class AITradeDecision:
    """AI decision result for entry or exit."""
    action: Literal["BUY", "SKIP", "EXIT", "HOLD"]
    confidence: float  # 0-100
    reasoning: str
    adjusted_target: Optional[float] = None
    adjusted_stop_loss: Optional[float] = None
    risk_assessment: str = ""
    market_sentiment: str = ""


@dataclass
class SignalContext:
    """Complete context for AI analysis."""
    # Signal info
    signal_type: str  # CE or PE
    confidence: float
    entry_price: float
    stop_loss: float
    target: float
    index: str  # NIFTY, BANKNIFTY, SENSEX

    # 15-min of 1-min OHLCV data
    ohlcv_data: list[dict] = field(default_factory=list)

    # All indicators
    supertrend: dict = field(default_factory=dict)
    rsi: float = 50.0
    macd: dict = field(default_factory=dict)
    ema: dict = field(default_factory=dict)
    adx: dict = field(default_factory=dict)
    vwap: float = 0.0
    bollinger: dict = field(default_factory=dict)
    stochastic: dict = field(default_factory=dict)

    # Option Greeks
    delta: float = 0.5
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    # Expiry info
    hours_to_expiry: float = 168.0  # Default 1 week
    is_expiry_day: bool = False
    atm_premium: float = 0.0

    # Current position info (for exit analysis)
    current_price: float = 0.0
    pnl_percent: float = 0.0
    position_duration_minutes: int = 0
    max_price_reached: float = 0.0
    min_price_reached: float = 0.0


class AITradingService:
    """
    AI-powered trading decision service using Claude Haiku.

    Responsibilities:
    - Analyze entry signals with full market context
    - Generate exit signals based on indicators + Greeks
    - Special handling for expiry day (1 PM - 3 PM)
    - Risk assessment and position sizing recommendations
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[Any] = None
        self._enabled = False
        self._model = "claude-3-haiku-20240307"
        self._initialized = False

    def initialize(self, api_key: str = None, model: str = None, enabled: bool = True):
        """Initialize AI service with API key."""
        try:
            import anthropic

            api_key = api_key or self.settings.anthropic_api_key
            if not api_key:
                logger.warning("No Anthropic API key provided - AI trading disabled")
                self._enabled = False
                return

            self._client = anthropic.Anthropic(api_key=api_key)
            self._model = model or self.settings.ai_model or self._model
            self._enabled = enabled
            self._initialized = True
            logger.info(f"AI Trading Service initialized: model={self._model}, enabled={enabled}")
        except ImportError:
            logger.error("anthropic package not installed - run: pip install anthropic")
            self._enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize AI service: {e}")
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None

    @property
    def model_name(self) -> str:
        return self._model

    async def analyze_entry_signal(self, context: SignalContext) -> AITradeDecision:
        """
        Analyze entry signal with Claude.

        Sends:
        - Signal details (CE/PE, entry, SL, target)
        - Last 15 minutes of 1-min OHLCV data
        - All technical indicators
        - Option Greeks (Delta, Gamma, Theta, Vega)
        - Expiry information

        Returns BUY/SKIP decision with optional SL/Target adjustments.
        """
        if not self.is_enabled:
            logger.info("AI disabled - auto-approving signal")
            return AITradeDecision(
                action="BUY",
                confidence=context.confidence,
                reasoning="AI disabled - signal auto-approved",
            )

        prompt = self._build_entry_prompt(context)

        try:
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self._model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_entry_response(response.content[0].text, context)

        except Exception as e:
            logger.error(f"AI entry analysis error: {e}")
            return AITradeDecision(
                action="BUY",
                confidence=context.confidence * 0.8,
                reasoning=f"AI error fallback - proceeding with signal: {str(e)[:100]}",
            )

    async def analyze_exit_signal(self, context: SignalContext) -> AITradeDecision:
        """
        Analyze exit decision with Claude.

        Called periodically while position is open.
        Returns EXIT/HOLD decision based on:
        - Current price and P&L
        - Updated indicators
        - Greeks (especially Theta decay)
        - Time in position
        - Expiry proximity

        Only exits when AI says EXIT (or hard stop at -15%).
        """
        if not self.is_enabled:
            # Fallback rules when AI is disabled
            if context.pnl_percent <= -15:
                return AITradeDecision(
                    action="EXIT",
                    confidence=100,
                    reasoning="Hard stop triggered: -15% loss limit",
                )
            return AITradeDecision(
                action="HOLD",
                confidence=50,
                reasoning="AI disabled - holding position",
            )

        prompt = self._build_exit_prompt(context)

        try:
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self._model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )

            decision = self._parse_exit_response(response.content[0].text)

            # Hard stop override
            if context.pnl_percent <= -15 and decision.action == "HOLD":
                return AITradeDecision(
                    action="EXIT",
                    confidence=100,
                    reasoning=f"Hard stop override: -15% loss limit (AI said HOLD but P&L is {context.pnl_percent:.1f}%)",
                )

            return decision

        except Exception as e:
            logger.error(f"AI exit analysis error: {e}")
            # On error, hold position unless hard stop
            if context.pnl_percent <= -15:
                return AITradeDecision(
                    action="EXIT",
                    confidence=100,
                    reasoning=f"AI error + hard stop: {str(e)[:50]}",
                )
            return AITradeDecision(
                action="HOLD",
                confidence=30,
                reasoning=f"AI error - holding position: {str(e)[:50]}",
            )

    def _build_entry_prompt(self, ctx: SignalContext) -> str:
        """Build structured prompt for entry analysis."""
        ohlcv_summary = self._format_ohlcv(ctx.ohlcv_data)
        expiry_warning = ""

        if ctx.is_expiry_day:
            expiry_warning = f"""
## EXPIRY DAY WARNING
- Hours to expiry: {ctx.hours_to_expiry:.1f}
- Gamma exposure: {"HIGH - VOLATILE!" if ctx.gamma > 0.01 else "MODERATE" if ctx.gamma > 0.005 else "LOW"}
- Theta decay accelerated
- RECOMMENDATION: {"SCALP ONLY (quick in/out)" if ctx.hours_to_expiry < 3 else "Trade with tight stops"}
"""

        return f"""You are an expert Indian options trading AI with 15+ years experience. Analyze this signal and decide BUY or SKIP.

## SIGNAL DETAILS
- Index: {ctx.index}
- Direction: {ctx.signal_type} ({"CALL - Bullish" if ctx.signal_type == "CE" else "PUT - Bearish"})
- Entry Price: Rs.{ctx.entry_price:.2f}
- Stop Loss: Rs.{ctx.stop_loss:.2f} ({((ctx.entry_price - ctx.stop_loss) / ctx.entry_price * 100):.1f}% risk)
- Target: Rs.{ctx.target:.2f} ({((ctx.target - ctx.entry_price) / ctx.entry_price * 100):.1f}% reward)
- Signal Confidence: {ctx.confidence:.0f}%
- Risk:Reward = 1:{((ctx.target - ctx.entry_price) / (ctx.entry_price - ctx.stop_loss)):.1f}
{expiry_warning}
## LAST 15 MINUTES (1-min candles)
{ohlcv_summary}

## TECHNICAL INDICATORS
- SuperTrend: Direction={ctx.supertrend.get('direction', 'N/A')}, Value={ctx.supertrend.get('value', 0):.2f}
- RSI(14): {ctx.rsi:.1f} {"(OVERBOUGHT)" if ctx.rsi > 70 else "(OVERSOLD)" if ctx.rsi < 30 else ""}
- MACD: Line={ctx.macd.get('macd', 0):.2f}, Signal={ctx.macd.get('signal', 0):.2f}, Histogram={ctx.macd.get('histogram', 0):.2f}
- EMA: Fast(9)={ctx.ema.get('fast', 0):.2f}, Slow(21)={ctx.ema.get('slow', 0):.2f}, Trend(50)={ctx.ema.get('trend', 0):.2f}
- ADX: {ctx.adx.get('adx', 0):.1f} (Trend: {"STRONG" if ctx.adx.get('adx', 0) > 25 else "WEAK"})
- VWAP: {ctx.vwap:.2f}
- Stochastic: K={ctx.stochastic.get('k', 50):.1f}, D={ctx.stochastic.get('d', 50):.1f}

## OPTION GREEKS
- Delta: {ctx.delta:.3f} ({ctx.delta * 100:.0f}% directional exposure)
- Gamma: {ctx.gamma:.5f} ({"HIGH RISK" if ctx.gamma > 0.01 else "Normal"})
- Theta: Rs.{ctx.theta:.2f}/day (daily time decay)
- Vega: Rs.{ctx.vega:.2f}/1% IV change

## EXPIRY INFO
- Hours to Expiry: {ctx.hours_to_expiry:.1f}
- Is Expiry Day: {"YES - HIGH VOLATILITY EXPECTED" if ctx.is_expiry_day else "No"}
- ATM Premium: Rs.{ctx.atm_premium:.2f}

## YOUR TASK
1. Analyze if this is a GOOD entry based on:
   - Indicator alignment (trend + momentum)
   - Greeks risk (high gamma = volatile)
   - Risk/Reward ratio
   - Time to expiry (theta impact)
2. If signal looks weak or risky, SKIP it
3. Optionally adjust target/SL if you see better levels

RESPOND IN STRICT JSON FORMAT ONLY:
{{"action": "BUY" or "SKIP", "confidence": 0-100, "reasoning": "2-3 sentence explanation", "adjusted_target": null or number, "adjusted_stop_loss": null or number}}
"""

    def _build_exit_prompt(self, ctx: SignalContext) -> str:
        """Build prompt for exit analysis."""
        ohlcv_summary = self._format_ohlcv(ctx.ohlcv_data[-5:])  # Last 5 candles

        return f"""You are an expert Indian options trading AI. Decide if we should EXIT or HOLD this position.

## OPEN POSITION
- Index: {ctx.index}
- Type: {ctx.signal_type} ({"CALL" if ctx.signal_type == "CE" else "PUT"})
- Entry: Rs.{ctx.entry_price:.2f}
- Current: Rs.{ctx.current_price:.2f}
- P&L: {ctx.pnl_percent:+.1f}%
- Stop Loss: Rs.{ctx.stop_loss:.2f}
- Target: Rs.{ctx.target:.2f}
- Time in Position: {ctx.position_duration_minutes} minutes
- Max Price Reached: Rs.{ctx.max_price_reached:.2f} ({((ctx.max_price_reached - ctx.entry_price) / ctx.entry_price * 100):+.1f}%)
- Min Price Reached: Rs.{ctx.min_price_reached:.2f} ({((ctx.min_price_reached - ctx.entry_price) / ctx.entry_price * 100):+.1f}%)

## RECENT PRICE ACTION (Last 5 min)
{ohlcv_summary}

## CURRENT INDICATORS
- SuperTrend: {ctx.supertrend.get('direction', 'N/A')}
- RSI: {ctx.rsi:.1f}
- MACD Histogram: {ctx.macd.get('histogram', 0):.2f}

## GREEKS & TIME
- Delta: {ctx.delta:.3f}
- Theta Decay: Rs.{abs(ctx.theta):.2f}/day
- Hours to Expiry: {ctx.hours_to_expiry:.1f}
- Is Expiry Day: {"YES" if ctx.is_expiry_day else "No"}

## DECISION FACTORS
1. If in profit: Is momentum fading? Book profits?
2. If in loss: Is trend reversing? Cut losses?
3. Time decay: Is theta eating into position?
4. Expiry risk: Close before expiry if < 30 mins remaining

RESPOND IN STRICT JSON FORMAT ONLY:
{{"action": "EXIT" or "HOLD", "confidence": 0-100, "reasoning": "1-2 sentence explanation"}}
"""

    def _format_ohlcv(self, ohlcv: list[dict]) -> str:
        """Format OHLCV data for prompt."""
        if not ohlcv:
            return "No data available"

        lines = []
        for i, candle in enumerate(ohlcv):
            time_label = f"T-{len(ohlcv) - i}m"
            o = candle.get('Open', candle.get('open', 0))
            h = candle.get('High', candle.get('high', 0))
            l = candle.get('Low', candle.get('low', 0))
            c = candle.get('Close', candle.get('close', 0))
            v = candle.get('Volume', candle.get('volume', 0))

            change = ((c - o) / o * 100) if o > 0 else 0
            direction = "UP" if c > o else "DOWN" if c < o else "FLAT"

            lines.append(f"{time_label}: O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} [{direction} {change:+.2f}%]")

        return "\n".join(lines)

    def _parse_entry_response(self, response: str, ctx: SignalContext) -> AITradeDecision:
        """Parse Claude's entry response."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return AITradeDecision(
                    action=data.get("action", "BUY").upper(),
                    confidence=float(data.get("confidence", ctx.confidence)),
                    reasoning=data.get("reasoning", "No reason provided"),
                    adjusted_target=data.get("adjusted_target"),
                    adjusted_stop_loss=data.get("adjusted_stop_loss"),
                )
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in AI response: {e}")
        except Exception as e:
            logger.warning(f"Failed to parse AI entry response: {e}")

        # Fallback - check for keywords
        response_upper = response.upper()
        if "SKIP" in response_upper or "AVOID" in response_upper or "DON'T" in response_upper:
            return AITradeDecision(
                action="SKIP",
                confidence=60,
                reasoning=f"AI recommended skip (parsed from text): {response[:200]}",
            )

        return AITradeDecision(
            action="BUY",
            confidence=ctx.confidence,
            reasoning="Could not parse AI response - proceeding with signal",
        )

    def _parse_exit_response(self, response: str) -> AITradeDecision:
        """Parse Claude's exit response."""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return AITradeDecision(
                    action=data.get("action", "HOLD").upper(),
                    confidence=float(data.get("confidence", 50)),
                    reasoning=data.get("reasoning", ""),
                )
        except Exception as e:
            logger.warning(f"Failed to parse AI exit response: {e}")

        # Fallback - check for keywords
        response_upper = response.upper()
        if "EXIT" in response_upper or "CLOSE" in response_upper or "SELL" in response_upper:
            return AITradeDecision(
                action="EXIT",
                confidence=60,
                reasoning=f"AI recommended exit (parsed from text): {response[:150]}",
            )

        return AITradeDecision(
            action="HOLD",
            confidence=50,
            reasoning="Parse error - defaulting to HOLD",
        )


# Singleton instance
_ai_service: Optional[AITradingService] = None


def get_ai_trading_service() -> AITradingService:
    """Get or create singleton AI trading service."""
    global _ai_service
    if _ai_service is None:
        _ai_service = AITradingService()
        # Auto-initialize from settings if API key is configured
        settings = get_settings()
        if settings.anthropic_api_key:
            _ai_service.initialize(
                api_key=settings.anthropic_api_key,
                model=settings.ai_model,
                enabled=settings.ai_enabled,
            )
    return _ai_service
