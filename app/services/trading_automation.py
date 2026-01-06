"""
Trading Automation Service
Integrates signal generation, quality filtering, risk management, and order execution
Fully automated trading system targeting 25k daily profit
"""

from datetime import datetime, date
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from app.services.signal_engine import get_signal_engine, TradeSignal, TradingStyle
from app.services.signal_quality import get_quality_analyzer, QualityScore
from app.services.risk_manager import get_risk_manager, create_default_risk_params, PositionSize
from app.services.order_manager import get_order_manager, OrderType, OrderResult
from app.services.data_fetcher import get_data_fetcher
from app.core.config import NIFTY_INDEX_TOKEN, BANKNIFTY_INDEX_TOKEN, SENSEX_INDEX_TOKEN


class AutomationMode(Enum):
    """Trading automation mode."""
    MANUAL = "manual"  # Show signals only, no auto trading
    SEMI_AUTO = "semi_auto"  # Show signals, require confirmation
    FULL_AUTO = "full_auto"  # Fully automated trading


@dataclass
class TradingConfig:
    """Trading automation configuration."""
    mode: AutomationMode = AutomationMode.MANUAL
    capital: float = 500000  # 5 lakh capital for 25k target
    trading_style: TradingStyle = TradingStyle.INTRADAY
    indices_to_trade: list[str] = None  # ["NIFTY", "BANKNIFTY", "SENSEX"]
    auto_exit_on_target: bool = True  # Auto exit all when target hit
    auto_stop_on_loss: bool = True  # Auto stop when loss limit hit

    def __post_init__(self):
        if self.indices_to_trade is None:
            self.indices_to_trade = ["NIFTY", "BANKNIFTY"]


@dataclass
class TradeExecution:
    """Trade execution result."""
    signal: TradeSignal
    quality_score: QualityScore
    position_size: PositionSize | None
    order_result: OrderResult | None
    executed: bool
    reason: str
    timestamp: datetime


class TradingAutomation:
    """
    Fully automated trading system.

    Workflow:
    1. Generate signals from market data
    2. Score signal quality (only >70 score pass)
    3. Check risk management constraints
    4. Calculate position size
    5. Place orders (if automation enabled)
    6. Track P&L and daily targets
    7. Auto-exit on profit target or loss limit
    """

    def __init__(self, config: TradingConfig):
        """Initialize trading automation."""
        self.config = config

        # Initialize services
        self.signal_engine = get_signal_engine(config.trading_style)
        self.quality_analyzer = get_quality_analyzer()

        risk_params = create_default_risk_params(capital=config.capital)
        self.risk_manager = get_risk_manager(params=risk_params)

        self.order_manager = get_order_manager()
        self.data_fetcher = get_data_fetcher()

        # Tracking
        self.executed_trades: list[TradeExecution] = []
        self.pending_signals: list[tuple[TradeSignal, QualityScore]] = []

        logger.info(f"Trading automation initialized: Mode={config.mode.value}, Capital=₹{config.capital:,.0f}")

    async def scan_and_execute(self) -> list[TradeExecution]:
        """
        Scan all configured indices and execute high-quality signals.

        Returns:
            List of trade executions
        """
        executions = []

        for index in self.config.indices_to_trade:
            try:
                execution = await self._process_index(index)
                if execution:
                    executions.append(execution)
            except Exception as e:
                logger.error(f"Error processing {index}: {e}")

        return executions

    async def _process_index(self, index: str) -> TradeExecution | None:
        """Process a single index and execute if signal is high quality."""
        try:
            # Step 1: Fetch market data
            logger.info(f"Scanning {index}...")

            token = self._get_token(index)
            df = await self.data_fetcher.fetch_historical_data(
                instrument_token=token,
                timeframe="5minute",
                days=3,
            )

            if df.empty:
                logger.warning(f"{index}: No data available")
                return None

            # Fetch option chain
            chain_data = await self.data_fetcher.get_option_chain(index=index)
            option_chain = chain_data.get("chain", []) if "error" not in chain_data else None

            # Step 2: Generate signal
            signal = self.signal_engine.analyze(df=df, option_chain=option_chain)

            if not signal:
                logger.info(f"{index}: No signal generated")
                return None

            # Step 3: Score signal quality
            quality_score = self.quality_analyzer.analyze_quality(signal)

            logger.info(
                f"{index} Signal: {signal.direction} | "
                f"Confidence: {signal.confidence:.1f}% | "
                f"Quality Score: {quality_score.total_score:.1f}/100"
            )

            # Only proceed with high-quality signals
            if not quality_score.is_high_quality:
                logger.info(
                    f"{index}: Quality score {quality_score.total_score:.1f} below threshold "
                    f"({self.quality_analyzer.HIGH_QUALITY_THRESHOLD})"
                )
                return TradeExecution(
                    signal=signal,
                    quality_score=quality_score,
                    position_size=None,
                    order_result=None,
                    executed=False,
                    reason=f"Quality score {quality_score.total_score:.1f} below threshold",
                    timestamp=datetime.now(),
                )

            logger.info(f"✅ {index}: HIGH QUALITY SIGNAL DETECTED!")
            logger.info(f"   Quality Factors: {', '.join(quality_score.factors)}")
            if quality_score.warnings:
                logger.warning(f"   Warnings: {', '.join(quality_score.warnings)}")

            # Step 4: Check risk constraints
            can_trade, reason = self.risk_manager.can_take_trade(signal)

            if not can_trade:
                logger.warning(f"{index}: Trade blocked - {reason}")
                return TradeExecution(
                    signal=signal,
                    quality_score=quality_score,
                    position_size=None,
                    order_result=None,
                    executed=False,
                    reason=reason,
                    timestamp=datetime.now(),
                )

            # Step 5: Calculate position size
            if not signal.recommended_option:
                logger.warning(f"{index}: No recommended option")
                return TradeExecution(
                    signal=signal,
                    quality_score=quality_score,
                    position_size=None,
                    order_result=None,
                    executed=False,
                    reason="No recommended option",
                    timestamp=datetime.now(),
                )

            position_size = self.risk_manager.calculate_position_size(
                signal=signal,
                entry_price=signal.recommended_option.ltp,
                stop_loss=signal.recommended_option.expected_at_stop or signal.stop_loss,
            )

            if not position_size:
                logger.warning(f"{index}: Position sizing failed")
                return TradeExecution(
                    signal=signal,
                    quality_score=quality_score,
                    position_size=None,
                    order_result=None,
                    executed=False,
                    reason="Position sizing failed",
                    timestamp=datetime.now(),
                )

            logger.info(
                f"{index} Position: {position_size.quantity} contracts | "
                f"Risk: ₹{position_size.risk_amount:,.0f} | "
                f"Potential: ₹{position_size.potential_profit:,.0f} | "
                f"R:R: {position_size.risk_reward_ratio:.2f}"
            )

            # Step 6: Execute trade (if automation enabled)
            order_result = None

            if self.config.mode == AutomationMode.FULL_AUTO:
                logger.info(f"🚀 {index}: Placing order...")

                order_result = self.order_manager.place_signal_order(
                    signal=signal,
                    quantity=position_size.quantity,
                    order_type=OrderType.MARKET,
                    with_bracket=True,  # Place SL and target orders
                )

                if order_result.success:
                    logger.info(f"✅ {index}: Order placed successfully - {order_result.order_id}")

                    # Track position
                    self.risk_manager.add_active_position({
                        "order_id": order_result.order_id,
                        "index": index,
                        "signal": signal,
                        "position_size": position_size,
                    })

                    execution = TradeExecution(
                        signal=signal,
                        quality_score=quality_score,
                        position_size=position_size,
                        order_result=order_result,
                        executed=True,
                        reason="Order placed successfully",
                        timestamp=datetime.now(),
                    )

                    self.executed_trades.append(execution)
                    return execution

                else:
                    logger.error(f"❌ {index}: Order failed - {order_result.message}")
                    return TradeExecution(
                        signal=signal,
                        quality_score=quality_score,
                        position_size=position_size,
                        order_result=order_result,
                        executed=False,
                        reason=f"Order failed: {order_result.message}",
                        timestamp=datetime.now(),
                    )

            elif self.config.mode == AutomationMode.SEMI_AUTO:
                # Store for user confirmation
                self.pending_signals.append((signal, quality_score))
                logger.info(f"{index}: Signal added to pending - Awaiting confirmation")

                return TradeExecution(
                    signal=signal,
                    quality_score=quality_score,
                    position_size=position_size,
                    order_result=None,
                    executed=False,
                    reason="Awaiting user confirmation (semi-auto mode)",
                    timestamp=datetime.now(),
                )

            else:  # MANUAL mode
                logger.info(f"{index}: High-quality signal available (manual mode)")

                return TradeExecution(
                    signal=signal,
                    quality_score=quality_score,
                    position_size=position_size,
                    order_result=None,
                    executed=False,
                    reason="Manual mode - signal available",
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error processing {index}: {e}")
            return None

    def get_daily_summary(self) -> dict:
        """Get daily trading summary with P&L."""
        stats = self.risk_manager.get_daily_summary()

        return {
            "date": stats.date.isoformat(),
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "win_rate": f"{stats.win_rate:.1f}%",
            "gross_profit": stats.gross_profit,
            "gross_loss": stats.gross_loss,
            "net_pnl": stats.net_pnl,
            "largest_win": stats.largest_win,
            "largest_loss": stats.largest_loss,
            "profit_factor": f"{stats.profit_factor:.2f}",
            "active_positions": stats.active_positions,
            "target_reached": stats.target_reached,
            "max_loss_hit": stats.max_loss_hit,
            "target_amount": self.risk_manager.params.daily_profit_target,
            "progress_pct": (stats.net_pnl / self.risk_manager.params.daily_profit_target) * 100,
        }

    def get_pending_signals(self) -> list[tuple[TradeSignal, QualityScore]]:
        """Get pending signals awaiting confirmation."""
        return self.pending_signals

    def confirm_pending_signal(self, signal_index: int) -> OrderResult | None:
        """
        Confirm and execute a pending signal.

        Args:
            signal_index: Index in pending_signals list

        Returns:
            OrderResult if executed, None otherwise
        """
        if signal_index >= len(self.pending_signals):
            logger.error(f"Invalid signal index: {signal_index}")
            return None

        signal, quality_score = self.pending_signals[signal_index]

        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            signal=signal,
            entry_price=signal.recommended_option.ltp,
            stop_loss=signal.recommended_option.expected_at_stop or signal.stop_loss,
        )

        if not position_size:
            logger.error("Position sizing failed")
            return None

        # Place order
        order_result = self.order_manager.place_signal_order(
            signal=signal,
            quantity=position_size.quantity,
            order_type=OrderType.MARKET,
            with_bracket=True,
        )

        if order_result.success:
            # Remove from pending
            self.pending_signals.pop(signal_index)

            # Track position
            self.risk_manager.add_active_position({
                "order_id": order_result.order_id,
                "signal": signal,
                "position_size": position_size,
            })

        return order_result

    def _get_token(self, index: str) -> int:
        """Get instrument token for index."""
        if index == "NIFTY":
            return NIFTY_INDEX_TOKEN
        elif index == "SENSEX":
            return SENSEX_INDEX_TOKEN
        else:
            return BANKNIFTY_INDEX_TOKEN


# Singleton instance
_trading_automation: TradingAutomation | None = None


def get_trading_automation(config: TradingConfig | None = None) -> TradingAutomation:
    """Get or create trading automation instance."""
    global _trading_automation
    if _trading_automation is None:
        if config is None:
            config = TradingConfig()
        _trading_automation = TradingAutomation(config)
    return _trading_automation
