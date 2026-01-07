"""
AI Trading Service
Supports both Claude (Anthropic) and ChatGPT (OpenAI) for intelligent trade decisions
Automatically uses whichever API key is configured
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from app.core.config import get_settings


@dataclass
class AITradeDecision:
    """AI-generated trade decision."""
    action: str  # "ENTER", "EXIT", "HOLD", "SKIP"
    confidence: float  # 0-100
    reasoning: str
    entry_price: float | None = None
    stop_loss: float | None = None
    target: float | None = None
    risk_reward: float | None = None
    position_size_recommendation: str | None = None
    market_sentiment: str | None = None  # "BULLISH", "BEARISH", "NEUTRAL"
    key_factors: list[str] | None = None
    warnings: list[str] | None = None
    timestamp: datetime | None = None
    ai_provider: str | None = None  # "Claude" or "ChatGPT"


class AIProviderBase(ABC):
    """Base class for AI providers."""

    @abstractmethod
    async def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from AI."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name."""
        pass


class ClaudeProvider(AIProviderBase):
    """Anthropic Claude AI provider."""

    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Claude AI initialized with model: {model}")

    async def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @property
    def provider_name(self) -> str:
        return "Claude"


class OpenAIProvider(AIProviderBase):
    """OpenAI ChatGPT provider."""

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"OpenAI ChatGPT initialized with model: {model}")

    async def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    @property
    def provider_name(self) -> str:
        return "ChatGPT"


class AITrader:
    """AI-powered trading assistant supporting Claude and ChatGPT."""

    def __init__(self):
        settings = get_settings()
        self.provider: AIProviderBase | None = None

        # Try Claude first, then OpenAI
        if settings.anthropic_api_key:
            try:
                self.provider = ClaudeProvider(
                    api_key=settings.anthropic_api_key,
                    model=settings.claude_model,
                )
            except Exception as e:
                logger.error(f"Failed to initialize Claude: {e}")

        if not self.provider and settings.openai_api_key:
            try:
                self.provider = OpenAIProvider(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                )
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")

        if not self.provider:
            logger.warning("No AI provider configured - provide ANTHROPIC_API_KEY or OPENAI_API_KEY")

    @property
    def is_configured(self) -> bool:
        """Check if AI is configured."""
        return self.provider is not None

    @property
    def provider_name(self) -> str:
        """Get active provider name."""
        return self.provider.provider_name if self.provider else "None"

    def _create_market_context(
        self,
        signal_data: dict,
        indicators: list[dict],
        option_data: dict | None = None,
        position_data: dict | None = None,
    ) -> str:
        """Create detailed market context for AI."""
        context = f"""
## Current Market Data
- **Index**: {signal_data.get('index', 'NIFTY')}
- **Spot Price**: ₹{signal_data.get('spot_price', 0):,.2f}
- **Signal Direction**: {signal_data.get('direction', 'N/A')}
- **Signal Confidence**: {signal_data.get('confidence', 0)}%
- **Signal Type**: {signal_data.get('signal_type', 'N/A')}
- **Time**: {datetime.now().strftime('%H:%M:%S')}

## Technical Indicators
"""
        for ind in indicators:
            context += f"- **{ind.get('name', 'Unknown')}**: Signal={ind.get('signal', 'N/A')}, Strength={ind.get('strength', 'N/A')}, Reason={ind.get('reason', '')}\n"

        if option_data:
            context += f"""
## Recommended Option
- **Strike**: {option_data.get('strike', 'N/A')}
- **Type**: {option_data.get('option_type', 'N/A')}
- **LTP**: ₹{option_data.get('ltp', 0):.2f}
- **Delta**: {option_data.get('delta', 'N/A')}
- **Theta**: {option_data.get('theta', 'N/A')}
- **Open Interest**: {option_data.get('oi', 0):,}
- **Volume**: {option_data.get('volume', 0):,}
- **Bid/Ask**: ₹{option_data.get('bid', 0):.2f} / ₹{option_data.get('ask', 0):.2f}
"""

        if position_data:
            context += f"""
## Current Position
- **Symbol**: {position_data.get('symbol', 'N/A')}
- **Entry Price**: ₹{position_data.get('entry_price', 0):.2f}
- **Current Price**: ₹{position_data.get('current_price', 0):.2f}
- **P&L**: ₹{position_data.get('pnl', 0):.2f} ({position_data.get('pnl_percent', 0):.1f}%)
- **Max Price**: ₹{position_data.get('max_price', 0):.2f}
- **Current SL**: ₹{position_data.get('stop_loss', 0):.2f}
- **Target**: ₹{position_data.get('target', 0):.2f}
"""

        return context

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from AI response."""
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        return json.loads(response_text.strip())

    async def analyze_entry(
        self,
        signal_data: dict,
        indicators: list[dict],
        option_data: dict | None = None,
    ) -> AITradeDecision:
        """
        Analyze whether to enter a trade based on signal and market data.
        """
        if not self.provider:
            return AITradeDecision(
                action="SKIP",
                confidence=0,
                reasoning="AI not configured - provide ANTHROPIC_API_KEY or OPENAI_API_KEY in .env",
                timestamp=datetime.now(),
                ai_provider="None",
            )

        context = self._create_market_context(signal_data, indicators, option_data)

        prompt = f"""You are an expert options trader analyzing Indian market (NIFTY/BANKNIFTY/SENSEX) for intraday trading.

{context}

## Trading Rules
1. Only trade options with premium between ₹45-55
2. Stop loss: 10% of premium (trailing after profit)
3. Trail stop loss 5 points below highest price reached
4. Prefer high probability setups (multiple indicators aligned)
5. Avoid trading during high volatility without clear direction

## Your Task
Analyze this trading opportunity and decide:
1. Should we ENTER this trade or SKIP?
2. What's your confidence level (0-100)?
3. Key reasons for your decision
4. Risk assessment and warnings

Respond ONLY in JSON format (no other text):
{{
    "action": "ENTER" or "SKIP",
    "confidence": 0-100,
    "reasoning": "Your detailed analysis in 2-3 sentences",
    "market_sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "entry_price": recommended entry price or null,
    "stop_loss": recommended stop loss or null,
    "target": recommended target or null,
    "risk_reward": calculated risk:reward ratio or null,
    "position_size_recommendation": "FULL" or "HALF" or "QUARTER",
    "key_factors": ["factor1", "factor2"],
    "warnings": ["warning1", "warning2"]
}}"""

        try:
            response_text = await self.provider.generate_response(prompt)
            result = self._parse_json_response(response_text)

            return AITradeDecision(
                action=result.get("action", "SKIP"),
                confidence=result.get("confidence", 0),
                reasoning=result.get("reasoning", ""),
                entry_price=result.get("entry_price"),
                stop_loss=result.get("stop_loss"),
                target=result.get("target"),
                risk_reward=result.get("risk_reward"),
                position_size_recommendation=result.get("position_size_recommendation"),
                market_sentiment=result.get("market_sentiment"),
                key_factors=result.get("key_factors", []),
                warnings=result.get("warnings", []),
                timestamp=datetime.now(),
                ai_provider=self.provider_name,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return AITradeDecision(
                action="SKIP",
                confidence=0,
                reasoning=f"Failed to parse AI response",
                timestamp=datetime.now(),
                ai_provider=self.provider_name,
            )
        except Exception as e:
            logger.error(f"AI entry analysis error: {e}")
            return AITradeDecision(
                action="SKIP",
                confidence=0,
                reasoning=f"AI error: {str(e)}",
                timestamp=datetime.now(),
                ai_provider=self.provider_name,
            )

    async def analyze_exit(
        self,
        position_data: dict,
        signal_data: dict,
        indicators: list[dict],
    ) -> AITradeDecision:
        """
        Analyze whether to exit an existing position.
        """
        if not self.provider:
            return AITradeDecision(
                action="HOLD",
                confidence=0,
                reasoning="AI not configured",
                timestamp=datetime.now(),
                ai_provider="None",
            )

        context = self._create_market_context(
            signal_data, indicators, position_data=position_data
        )

        prompt = f"""You are an expert options trader managing an active position in Indian market.

{context}

## Exit Rules
1. Trailing stop loss: 5 points below highest price
2. Initial stop loss: 10% of entry premium
3. Exit on trend reversal signals (SuperTrend flip, MACD crossover)
4. Book profits on momentum exhaustion (RSI extreme)
5. Market closes at 3:30 PM - square off by 3:20 PM

## Your Task
Analyze this position and decide:
1. Should we EXIT now, or HOLD the position?
2. What's your confidence level (0-100)?
3. Key reasons for your decision

Respond ONLY in JSON format (no other text):
{{
    "action": "EXIT" or "HOLD",
    "confidence": 0-100,
    "reasoning": "Your detailed analysis in 2-3 sentences",
    "market_sentiment": "BULLISH" or "BEARISH" or "NEUTRAL",
    "stop_loss": updated stop loss recommendation,
    "target": updated target recommendation,
    "key_factors": ["factor1", "factor2"],
    "warnings": ["warning1", "warning2"]
}}"""

        try:
            response_text = await self.provider.generate_response(prompt)
            result = self._parse_json_response(response_text)

            return AITradeDecision(
                action=result.get("action", "HOLD"),
                confidence=result.get("confidence", 0),
                reasoning=result.get("reasoning", ""),
                stop_loss=result.get("stop_loss"),
                target=result.get("target"),
                market_sentiment=result.get("market_sentiment"),
                key_factors=result.get("key_factors", []),
                warnings=result.get("warnings", []),
                timestamp=datetime.now(),
                ai_provider=self.provider_name,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            return AITradeDecision(
                action="HOLD",
                confidence=0,
                reasoning="Failed to parse AI response",
                timestamp=datetime.now(),
                ai_provider=self.provider_name,
            )
        except Exception as e:
            logger.error(f"AI exit analysis error: {e}")
            return AITradeDecision(
                action="HOLD",
                confidence=0,
                reasoning=f"AI error: {str(e)}",
                timestamp=datetime.now(),
                ai_provider=self.provider_name,
            )


# Singleton instance
_ai_trader: AITrader | None = None


def get_ai_trader() -> AITrader:
    """Get or create AI trader instance."""
    global _ai_trader
    if _ai_trader is None:
        _ai_trader = AITrader()
    return _ai_trader
