"""
Email Service
Optional email notifications for daily/monthly/yearly P&L reports
"""

import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import asyncio
from threading import Thread

from loguru import logger


@dataclass
class EmailSettings:
    """Email configuration settings."""
    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    sender_email: str = ""
    recipient_email: str = ""
    use_tls: bool = True
    # Schedule settings
    send_daily_report: bool = True
    send_monthly_report: bool = True
    send_yearly_report: bool = True
    daily_report_time: str = "18:00"  # 6 PM after market close


@dataclass
class PnLReport:
    """P&L Report data."""
    report_type: str  # daily, monthly, yearly
    period_start: date
    period_end: date
    starting_capital: float
    ending_capital: float
    total_pnl: float
    total_pnl_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    max_drawdown_percent: float
    best_trade: float
    worst_trade: float
    average_trade: float
    trade_details: List[Dict[str, Any]]


class EmailService:
    """Handles email notifications for P&L reports."""

    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.settings_file = self.data_dir / "email_settings.json"
        self.reports_file = self.data_dir / "pnl_reports.json"

        self._settings: EmailSettings = EmailSettings()
        self._reports: List[Dict[str, Any]] = []
        self._last_daily_report: Optional[date] = None
        self._last_monthly_report: Optional[date] = None
        self._last_yearly_report: Optional[date] = None

        self._load_settings()
        self._load_reports()

    def _load_settings(self):
        """Load email settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                    self._settings = EmailSettings(**data.get('settings', {}))
                    self._last_daily_report = (
                        datetime.fromisoformat(data['last_daily_report']).date()
                        if data.get('last_daily_report') else None
                    )
                    self._last_monthly_report = (
                        datetime.fromisoformat(data['last_monthly_report']).date()
                        if data.get('last_monthly_report') else None
                    )
                    self._last_yearly_report = (
                        datetime.fromisoformat(data['last_yearly_report']).date()
                        if data.get('last_yearly_report') else None
                    )
                logger.info("Email settings loaded")
            except Exception as e:
                logger.error(f"Error loading email settings: {e}")
        else:
            self._save_settings()

    def _save_settings(self):
        """Save email settings to file."""
        try:
            data = {
                'settings': asdict(self._settings),
                'last_daily_report': self._last_daily_report.isoformat() if self._last_daily_report else None,
                'last_monthly_report': self._last_monthly_report.isoformat() if self._last_monthly_report else None,
                'last_yearly_report': self._last_yearly_report.isoformat() if self._last_yearly_report else None,
            }
            with open(self.settings_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving email settings: {e}")

    def _load_reports(self):
        """Load saved reports from file."""
        if self.reports_file.exists():
            try:
                with open(self.reports_file, 'r') as f:
                    self._reports = json.load(f)
            except Exception as e:
                logger.error(f"Error loading reports: {e}")
                self._reports = []

    def _save_reports(self):
        """Save reports to file."""
        try:
            # Keep only last 365 days of reports
            cutoff = (datetime.now() - timedelta(days=365)).isoformat()
            self._reports = [r for r in self._reports if r.get('created_at', '') >= cutoff]

            with open(self.reports_file, 'w') as f:
                json.dump(self._reports, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving reports: {e}")

    def get_settings(self) -> Dict[str, Any]:
        """Get current email settings (without password)."""
        settings = asdict(self._settings)
        settings['smtp_password'] = '***' if self._settings.smtp_password else ''
        return settings

    def update_settings(
        self,
        enabled: Optional[bool] = None,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sender_email: Optional[str] = None,
        recipient_email: Optional[str] = None,
        use_tls: Optional[bool] = None,
        send_daily_report: Optional[bool] = None,
        send_monthly_report: Optional[bool] = None,
        send_yearly_report: Optional[bool] = None,
        daily_report_time: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Update email settings."""
        try:
            if enabled is not None:
                self._settings.enabled = enabled
            if smtp_server is not None:
                self._settings.smtp_server = smtp_server
            if smtp_port is not None:
                self._settings.smtp_port = smtp_port
            if smtp_username is not None:
                self._settings.smtp_username = smtp_username
            if smtp_password is not None and smtp_password != '***':
                self._settings.smtp_password = smtp_password
            if sender_email is not None:
                self._settings.sender_email = sender_email
            if recipient_email is not None:
                self._settings.recipient_email = recipient_email
            if use_tls is not None:
                self._settings.use_tls = use_tls
            if send_daily_report is not None:
                self._settings.send_daily_report = send_daily_report
            if send_monthly_report is not None:
                self._settings.send_monthly_report = send_monthly_report
            if send_yearly_report is not None:
                self._settings.send_yearly_report = send_yearly_report
            if daily_report_time is not None:
                self._settings.daily_report_time = daily_report_time

            self._save_settings()
            return True, "Settings updated successfully"
        except Exception as e:
            logger.error(f"Error updating email settings: {e}")
            return False, str(e)

    def test_connection(self) -> tuple[bool, str]:
        """Test SMTP connection."""
        if not self._settings.smtp_server:
            return False, "SMTP server not configured"

        try:
            if self._settings.use_tls:
                server = smtplib.SMTP(self._settings.smtp_server, self._settings.smtp_port, timeout=10)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self._settings.smtp_server, self._settings.smtp_port, timeout=10)

            if self._settings.smtp_username and self._settings.smtp_password:
                server.login(self._settings.smtp_username, self._settings.smtp_password)

            server.quit()
            logger.info("SMTP connection test successful")
            return True, "Connection successful"
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False, str(e)

    def send_test_email(self) -> tuple[bool, str]:
        """Send a test email."""
        if not self._settings.enabled:
            return False, "Email notifications are disabled"

        if not self._settings.recipient_email:
            return False, "Recipient email not configured"

        subject = "OptiFlow Pro - Test Email"
        body = """
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #1e293b; color: #e2e8f0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border-radius: 12px; padding: 30px;">
                <h1 style="color: #60a5fa; margin-bottom: 20px;">OptiFlow Pro</h1>
                <p style="font-size: 16px;">This is a test email from your OptiFlow Pro trading system.</p>
                <p style="font-size: 14px; color: #94a3b8;">If you received this email, your email notifications are configured correctly.</p>
                <hr style="border-color: #334155; margin: 20px 0;">
                <p style="font-size: 12px; color: #64748b;">
                    Sent at: {timestamp}
                </p>
            </div>
        </body>
        </html>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        return self._send_email(subject, body)

    def _send_email(self, subject: str, html_body: str) -> tuple[bool, str]:
        """Send an email."""
        if not self._settings.enabled:
            return False, "Email notifications are disabled"

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self._settings.sender_email or self._settings.smtp_username
            msg['To'] = self._settings.recipient_email

            # Create plain text version
            plain_text = html_body.replace('<br>', '\n').replace('</p>', '\n')
            import re
            plain_text = re.sub('<[^<]+?>', '', plain_text)

            part1 = MIMEText(plain_text, 'plain')
            part2 = MIMEText(html_body, 'html')

            msg.attach(part1)
            msg.attach(part2)

            if self._settings.use_tls:
                server = smtplib.SMTP(self._settings.smtp_server, self._settings.smtp_port, timeout=30)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self._settings.smtp_server, self._settings.smtp_port, timeout=30)

            if self._settings.smtp_username and self._settings.smtp_password:
                server.login(self._settings.smtp_username, self._settings.smtp_password)

            server.sendmail(
                msg['From'],
                [self._settings.recipient_email],
                msg.as_string()
            )
            server.quit()

            logger.info(f"Email sent successfully: {subject}")
            return True, "Email sent successfully"
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False, str(e)

    def generate_pnl_report(
        self,
        report_type: str,
        period_start: date,
        period_end: date,
        paper_trading_service
    ) -> PnLReport:
        """Generate P&L report from paper trading data."""
        from app.services.paper_trading import get_paper_trading_service

        if paper_trading_service is None:
            paper_trading_service = get_paper_trading_service()

        # Get order history for the period
        all_orders = paper_trading_service.get_order_history()

        # Filter orders within the period
        period_orders = []
        for order in all_orders:
            order_date = datetime.fromisoformat(order['exit_time']).date() if order.get('exit_time') else None
            if order_date and period_start <= order_date <= period_end:
                period_orders.append(order)

        # Calculate statistics
        total_pnl = sum(o.get('pnl', 0) for o in period_orders)
        total_trades = len(period_orders)
        winning_trades = len([o for o in period_orders if o.get('pnl', 0) > 0])
        losing_trades = len([o for o in period_orders if o.get('pnl', 0) < 0])

        pnls = [o.get('pnl', 0) for o in period_orders]
        best_trade = max(pnls) if pnls else 0
        worst_trade = min(pnls) if pnls else 0
        average_trade = sum(pnls) / len(pnls) if pnls else 0

        # Get capital info
        stats = paper_trading_service.get_stats()
        starting_capital = stats.get('starting_capital', 500000)
        ending_capital = starting_capital + total_pnl

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl_percent = (total_pnl / starting_capital * 100) if starting_capital > 0 else 0

        # Calculate max drawdown from the period
        max_drawdown = 0
        max_drawdown_percent = 0
        running_pnl = 0
        peak_pnl = 0
        for order in sorted(period_orders, key=lambda x: x.get('exit_time', '')):
            running_pnl += order.get('pnl', 0)
            peak_pnl = max(peak_pnl, running_pnl)
            drawdown = peak_pnl - running_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = (drawdown / starting_capital * 100) if starting_capital > 0 else 0

        return PnLReport(
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            starting_capital=starting_capital,
            ending_capital=ending_capital,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            best_trade=best_trade,
            worst_trade=worst_trade,
            average_trade=average_trade,
            trade_details=period_orders[:50]  # Limit to 50 trades in email
        )

    def _format_report_email(self, report: PnLReport) -> str:
        """Format P&L report as HTML email."""
        period_label = {
            'daily': 'Daily',
            'monthly': 'Monthly',
            'yearly': 'Yearly'
        }.get(report.report_type, 'P&L')

        pnl_color = '#10b981' if report.total_pnl >= 0 else '#ef4444'
        pnl_sign = '+' if report.total_pnl >= 0 else ''

        # Format trade rows
        trade_rows = ""
        for trade in report.trade_details[:20]:  # Show top 20 trades
            trade_pnl = trade.get('pnl', 0)
            trade_color = '#10b981' if trade_pnl >= 0 else '#ef4444'
            trade_rows += f"""
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 8px; color: #e2e8f0;">{trade.get('symbol', 'N/A')}</td>
                <td style="padding: 8px; color: #94a3b8;">{trade.get('option_type', '')}</td>
                <td style="padding: 8px; color: #e2e8f0;">{trade.get('lots', 0)} lots</td>
                <td style="padding: 8px; color: {trade_color}; font-weight: bold;">
                    {'+' if trade_pnl >= 0 else ''}₹{trade_pnl:,.0f}
                </td>
            </tr>
            """

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #1e293b; color: #e2e8f0; padding: 20px; margin: 0;">
            <div style="max-width: 700px; margin: 0 auto; background-color: #0f172a; border-radius: 12px; overflow: hidden;">
                <!-- Header -->
                <div style="background: linear-gradient(135deg, #3b82f6, #6366f1); padding: 30px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 24px;">OptiFlow Pro</h1>
                    <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">{period_label} P&L Report</p>
                </div>

                <!-- Period Info -->
                <div style="padding: 20px 30px; background-color: #1e293b; border-bottom: 1px solid #334155;">
                    <p style="margin: 0; color: #94a3b8; font-size: 14px;">
                        Period: <span style="color: #e2e8f0;">{report.period_start.strftime('%d %b %Y')} - {report.period_end.strftime('%d %b %Y')}</span>
                    </p>
                </div>

                <!-- Main Stats -->
                <div style="padding: 30px;">
                    <!-- P&L Highlight -->
                    <div style="text-align: center; margin-bottom: 30px; padding: 30px; background-color: #1e293b; border-radius: 12px; border: 1px solid #334155;">
                        <p style="margin: 0 0 10px 0; color: #94a3b8; font-size: 14px;">Total P&L</p>
                        <p style="margin: 0; font-size: 42px; font-weight: bold; color: {pnl_color};">
                            {pnl_sign}₹{report.total_pnl:,.0f}
                        </p>
                        <p style="margin: 10px 0 0 0; color: {pnl_color}; font-size: 16px;">
                            {pnl_sign}{report.total_pnl_percent:.2f}%
                        </p>
                    </div>

                    <!-- Stats Grid -->
                    <div style="display: table; width: 100%; margin-bottom: 30px;">
                        <div style="display: table-row;">
                            <div style="display: table-cell; width: 50%; padding: 10px; text-align: center; background-color: #1e293b; border-radius: 8px 0 0 0;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Starting Capital</p>
                                <p style="margin: 5px 0 0 0; color: #e2e8f0; font-size: 18px; font-weight: bold;">₹{report.starting_capital:,.0f}</p>
                            </div>
                            <div style="display: table-cell; width: 50%; padding: 10px; text-align: center; background-color: #1e293b; border-radius: 0 8px 0 0;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Ending Capital</p>
                                <p style="margin: 5px 0 0 0; color: #e2e8f0; font-size: 18px; font-weight: bold;">₹{report.ending_capital:,.0f}</p>
                            </div>
                        </div>
                        <div style="display: table-row;">
                            <div style="display: table-cell; width: 50%; padding: 10px; text-align: center; background-color: #1e293b;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Total Trades</p>
                                <p style="margin: 5px 0 0 0; color: #e2e8f0; font-size: 18px; font-weight: bold;">{report.total_trades}</p>
                            </div>
                            <div style="display: table-cell; width: 50%; padding: 10px; text-align: center; background-color: #1e293b;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Win Rate</p>
                                <p style="margin: 5px 0 0 0; color: {'#10b981' if report.win_rate >= 50 else '#ef4444'}; font-size: 18px; font-weight: bold;">{report.win_rate:.1f}%</p>
                            </div>
                        </div>
                        <div style="display: table-row;">
                            <div style="display: table-cell; width: 50%; padding: 10px; text-align: center; background-color: #1e293b; border-radius: 0 0 0 8px;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Winning Trades</p>
                                <p style="margin: 5px 0 0 0; color: #10b981; font-size: 18px; font-weight: bold;">{report.winning_trades}</p>
                            </div>
                            <div style="display: table-cell; width: 50%; padding: 10px; text-align: center; background-color: #1e293b; border-radius: 0 0 8px 0;">
                                <p style="margin: 0; color: #94a3b8; font-size: 12px;">Losing Trades</p>
                                <p style="margin: 5px 0 0 0; color: #ef4444; font-size: 18px; font-weight: bold;">{report.losing_trades}</p>
                            </div>
                        </div>
                    </div>

                    <!-- Additional Stats -->
                    <div style="background-color: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 30px;">
                        <h3 style="margin: 0 0 15px 0; color: #e2e8f0; font-size: 16px;">Performance Metrics</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px 0; color: #94a3b8;">Best Trade</td>
                                <td style="padding: 8px 0; color: #10b981; text-align: right; font-weight: bold;">+₹{report.best_trade:,.0f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #94a3b8;">Worst Trade</td>
                                <td style="padding: 8px 0; color: #ef4444; text-align: right; font-weight: bold;">₹{report.worst_trade:,.0f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #94a3b8;">Average Trade</td>
                                <td style="padding: 8px 0; color: #e2e8f0; text-align: right; font-weight: bold;">₹{report.average_trade:,.0f}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px 0; color: #94a3b8;">Max Drawdown</td>
                                <td style="padding: 8px 0; color: #ef4444; text-align: right; font-weight: bold;">₹{report.max_drawdown:,.0f} ({report.max_drawdown_percent:.2f}%)</td>
                            </tr>
                        </table>
                    </div>

                    <!-- Trade Details -->
                    {f'''
                    <div style="background-color: #1e293b; border-radius: 8px; padding: 20px;">
                        <h3 style="margin: 0 0 15px 0; color: #e2e8f0; font-size: 16px;">Recent Trades</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <thead>
                                <tr style="border-bottom: 2px solid #334155;">
                                    <th style="padding: 10px 8px; text-align: left; color: #94a3b8; font-size: 12px;">Symbol</th>
                                    <th style="padding: 10px 8px; text-align: left; color: #94a3b8; font-size: 12px;">Type</th>
                                    <th style="padding: 10px 8px; text-align: left; color: #94a3b8; font-size: 12px;">Qty</th>
                                    <th style="padding: 10px 8px; text-align: left; color: #94a3b8; font-size: 12px;">P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trade_rows}
                            </tbody>
                        </table>
                        {f'<p style="margin: 15px 0 0 0; color: #64748b; font-size: 12px;">Showing {min(20, len(report.trade_details))} of {len(report.trade_details)} trades</p>' if len(report.trade_details) > 20 else ''}
                    </div>
                    ''' if report.trade_details else ''}
                </div>

                <!-- Footer -->
                <div style="padding: 20px 30px; background-color: #1e293b; border-top: 1px solid #334155; text-align: center;">
                    <p style="margin: 0; color: #64748b; font-size: 12px;">
                        This is an automated report from OptiFlow Pro<br>
                        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_daily_report(self, paper_trading_service=None) -> tuple[bool, str]:
        """Send daily P&L report."""
        if not self._settings.enabled or not self._settings.send_daily_report:
            return False, "Daily reports are disabled"

        today = date.today()

        # Don't send if already sent today
        if self._last_daily_report == today:
            return False, "Daily report already sent today"

        report = self.generate_pnl_report(
            report_type='daily',
            period_start=today,
            period_end=today,
            paper_trading_service=paper_trading_service
        )

        subject = f"OptiFlow Pro - Daily P&L Report ({today.strftime('%d %b %Y')})"
        html_body = self._format_report_email(report)

        success, message = self._send_email(subject, html_body)

        if success:
            self._last_daily_report = today
            self._reports.append({
                'type': 'daily',
                'date': today.isoformat(),
                'pnl': report.total_pnl,
                'trades': report.total_trades,
                'created_at': datetime.now().isoformat()
            })
            self._save_settings()
            self._save_reports()

        return success, message

    def send_monthly_report(self, paper_trading_service=None) -> tuple[bool, str]:
        """Send monthly P&L report."""
        if not self._settings.enabled or not self._settings.send_monthly_report:
            return False, "Monthly reports are disabled"

        today = date.today()
        # First day of current month
        period_start = today.replace(day=1)
        # Last day of previous month (for end-of-month report)
        if today.day == 1:
            period_end = today - timedelta(days=1)
            period_start = period_end.replace(day=1)
        else:
            period_end = today

        # Don't send if already sent this month
        if self._last_monthly_report and self._last_monthly_report.month == today.month:
            return False, "Monthly report already sent this month"

        report = self.generate_pnl_report(
            report_type='monthly',
            period_start=period_start,
            period_end=period_end,
            paper_trading_service=paper_trading_service
        )

        month_name = period_start.strftime('%B %Y')
        subject = f"OptiFlow Pro - Monthly P&L Report ({month_name})"
        html_body = self._format_report_email(report)

        success, message = self._send_email(subject, html_body)

        if success:
            self._last_monthly_report = today
            self._reports.append({
                'type': 'monthly',
                'date': today.isoformat(),
                'month': month_name,
                'pnl': report.total_pnl,
                'trades': report.total_trades,
                'created_at': datetime.now().isoformat()
            })
            self._save_settings()
            self._save_reports()

        return success, message

    def send_yearly_report(self, paper_trading_service=None) -> tuple[bool, str]:
        """Send yearly P&L report."""
        if not self._settings.enabled or not self._settings.send_yearly_report:
            return False, "Yearly reports are disabled"

        today = date.today()
        # First day of current year
        period_start = today.replace(month=1, day=1)
        period_end = today

        # Don't send if already sent this year
        if self._last_yearly_report and self._last_yearly_report.year == today.year:
            return False, "Yearly report already sent this year"

        report = self.generate_pnl_report(
            report_type='yearly',
            period_start=period_start,
            period_end=period_end,
            paper_trading_service=paper_trading_service
        )

        year = period_start.year
        subject = f"OptiFlow Pro - Yearly P&L Report ({year})"
        html_body = self._format_report_email(report)

        success, message = self._send_email(subject, html_body)

        if success:
            self._last_yearly_report = today
            self._reports.append({
                'type': 'yearly',
                'date': today.isoformat(),
                'year': year,
                'pnl': report.total_pnl,
                'trades': report.total_trades,
                'created_at': datetime.now().isoformat()
            })
            self._save_settings()
            self._save_reports()

        return success, message

    def send_report_now(self, report_type: str, paper_trading_service=None) -> tuple[bool, str]:
        """Send a report immediately (manual trigger)."""
        today = date.today()

        if report_type == 'daily':
            period_start = today
            period_end = today
        elif report_type == 'monthly':
            period_start = today.replace(day=1)
            period_end = today
        elif report_type == 'yearly':
            period_start = today.replace(month=1, day=1)
            period_end = today
        else:
            return False, f"Invalid report type: {report_type}"

        report = self.generate_pnl_report(
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            paper_trading_service=paper_trading_service
        )

        period_label = {
            'daily': f"Daily ({today.strftime('%d %b %Y')})",
            'monthly': f"Monthly ({today.strftime('%B %Y')})",
            'yearly': f"Yearly ({today.year})"
        }.get(report_type, report_type)

        subject = f"OptiFlow Pro - {period_label} P&L Report"
        html_body = self._format_report_email(report)

        return self._send_email(subject, html_body)

    def get_report_history(self) -> List[Dict[str, Any]]:
        """Get history of sent reports."""
        return sorted(self._reports, key=lambda x: x.get('created_at', ''), reverse=True)[:50]

    def check_and_send_scheduled_reports(self, paper_trading_service=None):
        """Check if any scheduled reports need to be sent."""
        if not self._settings.enabled:
            return

        now = datetime.now()
        current_time = now.strftime("%H:%M")
        today = now.date()

        # Check daily report
        if (self._settings.send_daily_report and
            current_time >= self._settings.daily_report_time and
            self._last_daily_report != today):
            self.send_daily_report(paper_trading_service)

        # Check monthly report (send on 1st of month)
        if (self._settings.send_monthly_report and
            today.day == 1 and
            (not self._last_monthly_report or self._last_monthly_report.month != today.month)):
            self.send_monthly_report(paper_trading_service)

        # Check yearly report (send on Jan 1st)
        if (self._settings.send_yearly_report and
            today.month == 1 and today.day == 1 and
            (not self._last_yearly_report or self._last_yearly_report.year != today.year)):
            self.send_yearly_report(paper_trading_service)


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get singleton instance of EmailService."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
