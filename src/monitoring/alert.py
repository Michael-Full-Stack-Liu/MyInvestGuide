"""
Multi-Channel Alerting Module
Supports: Telegram, SMTP Email, Discord Webhook, W&B (if available)

Usage:
    from src.monitoring.alert import send_alert, run_drift_check
    
    # Send alert (will try available channels)
    send_alert("⚠️ Drift Detected", "Drift: 50%", level="warn")
"""

import os
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Union, Dict, List, Optional
import pandas as pd

from .drift import check_drift, generate_drift_report


# =============================================================================
# Configuration (from environment variables)
# =============================================================================

# Telegram Config (recommended - you already have this!)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# SMTP Email Config
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # For Gmail, use App Password
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")

# Discord Webhook
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# W&B (may require paid plan)
WANDB_ENABLED = os.getenv("WANDB_ALERTS_ENABLED", "false").lower() == "true"


# =============================================================================
# Alert Functions
# =============================================================================

def send_telegram_alert(title: str, text: str, level: str = "warn") -> bool:
    """Send alert via Telegram Bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False
    
    # Format message with emoji based on level
    emoji = {"info": "ℹ️", "warn": "⚠️", "error": "🚨"}.get(level, "📢")
    message = f"{emoji} *{title}*\n\n{text}"
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }, timeout=10)
        
        if response.status_code == 200:
            print(f"[Telegram] Alert sent: {title}")
            return True
        else:
            print(f"[Telegram] Failed: {response.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False


def send_email_alert(title: str, text: str, level: str = "warn") -> bool:
    """Send alert via SMTP email."""
    if not all([SMTP_USER, SMTP_PASSWORD, ALERT_EMAIL_TO]):
        print("[Email] Not configured (missing SMTP credentials)")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ALERT_EMAIL_TO
        msg['Subject'] = f"[Congress ML] {title}"
        
        # HTML body
        html = f"""
        <html>
        <body>
            <h2 style="color: {'#ff6b6b' if level == 'error' else '#ffa94d' if level == 'warn' else '#4dabf7'};">
                {title}
            </h2>
            <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
{text}
            </pre>
            <hr>
            <p style="color: #868e96; font-size: 12px;">
                Congress Trading ML Monitor
            </p>
        </body>
        </html>
        """
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"[Email] Alert sent to {ALERT_EMAIL_TO}: {title}")
        return True
    except Exception as e:
        print(f"[Email] Error: {e}")
        return False


def send_discord_alert(title: str, text: str, level: str = "warn") -> bool:
    """Send alert via Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        print("[Discord] Not configured (missing DISCORD_WEBHOOK_URL)")
        return False
    
    # Color based on level
    color = {"info": 0x3498db, "warn": 0xf39c12, "error": 0xe74c3c}.get(level, 0x95a5a6)
    
    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": text,
                "color": color,
                "footer": {"text": "Congress Trading ML Monitor"}
            }]
        }
        
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        
        if response.status_code in [200, 204]:
            print(f"[Discord] Alert sent: {title}")
            return True
        else:
            print(f"[Discord] Failed: {response.text}")
            return False
    except Exception as e:
        print(f"[Discord] Error: {e}")
        return False


def send_wandb_alert(title: str, text: str, level: str = "warn") -> bool:
    """Send alert via W&B (requires paid plan for alerts)."""
    if not WANDB_ENABLED:
        print("[W&B] Alerts disabled (set WANDB_ALERTS_ENABLED=true to enable)")
        return False
    
    try:
        import wandb
        from wandb import AlertLevel
        
        level_map = {
            "info": AlertLevel.INFO,
            "warn": AlertLevel.WARN,
            "error": AlertLevel.ERROR
        }
        
        with wandb.init(project="congress-trading-monitor", job_type="alert", reinit=True) as run:
            run.alert(title=title, text=text, level=level_map.get(level, AlertLevel.WARN))
        
        print(f"[W&B] Alert sent: {title}")
        return True
    except Exception as e:
        print(f"[W&B] Failed: {e}")
        return False


def send_alert(title: str, text: str, level: str = "warn", channels: List[str] = None) -> bool:
    """
    Send alert via multiple channels.
    
    Args:
        title: Alert title
        text: Alert message body
        level: "info", "warn", or "error"
        channels: List of channels to use. If None, tries all configured channels.
                  Options: ["telegram", "email", "discord", "wandb"]
    
    Returns:
        True if at least one channel succeeded
    """
    if channels is None:
        # Try all channels, prioritize Telegram
        channels = ["telegram", "email", "discord", "wandb"]
    
    channel_funcs = {
        "telegram": send_telegram_alert,
        "email": send_email_alert,
        "discord": send_discord_alert,
        "wandb": send_wandb_alert
    }
    
    success = False
    for channel in channels:
        if channel in channel_funcs:
            result = channel_funcs[channel](title, text, level)
            if result:
                success = True
                # If Telegram succeeds, that's usually enough
                if channel == "telegram":
                    break
    
    if not success:
        print(f"[Alert] Warning: No alert channels configured or all failed")
    
    return success


def run_drift_check(
    reference: Union[str, pd.DataFrame],
    current: Union[str, pd.DataFrame],
    threshold: float = 0.3,
    report_path: str = None
) -> Dict:
    """
    Run drift check and send alert if threshold exceeded.
    
    Args:
        reference: Training data (path or DataFrame)
        current: New data (path or DataFrame)
        threshold: Alert if drift_share > threshold (default: 0.3)
        report_path: Optional path to save HTML report
        
    Returns:
        Dict with results
    """
    # 1. Check drift using Evidently
    is_drifted, metrics = check_drift(reference, current)
    drift_share = metrics['drift_share']
    drifted_columns = metrics['drifted_columns']
    
    print(f"[Monitor] Drift: {drift_share:.1%}, Columns: {drifted_columns[:3]}")
    
    # 2. Save report if requested
    if report_path:
        generate_drift_report(reference, current, report_path)
    
    # 3. Send alert if threshold exceeded
    alert_sent = False
    if drift_share > threshold:
        alert_level = "error" if drift_share > 0.5 else "warn"
        alert_sent = send_alert(
            title="⚠️ Data Drift Detected",
            text=f"Drift: {drift_share:.1%}\nThreshold: {threshold:.0%}\nColumns: {', '.join(drifted_columns[:5])}",
            level=alert_level
        )
    
    return {
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "alert_sent": alert_sent
    }


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--report", default=None)
    parser.add_argument("--test-alert", action="store_true", help="Send test alert")
    args = parser.parse_args()
    
    if args.test_alert:
        send_alert("🧪 Test Alert", "This is a test alert from Congress ML Monitor", level="info")
    else:
        result = run_drift_check(args.reference, args.current, args.threshold, args.report)
        print(f"Result: {result}")
