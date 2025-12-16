"""
W&B Alerting Module
Send alerts when drift is detected.

Usage:
    from src.monitoring.alert import run_drift_check
    
    # Check drift and alert if threshold exceeded
    result = run_drift_check("data/train.parquet", "data/new.csv", threshold=0.3)
"""

import wandb
from wandb import AlertLevel
from typing import Union, Dict
import pandas as pd

from .drift import check_drift, generate_drift_report


def send_alert(title: str, text: str, level: str = "warn") -> bool:
    """Send alert via W&B (email/Slack)."""
    level_map = {
        "info": AlertLevel.INFO,
        "warn": AlertLevel.WARN, 
        "error": AlertLevel.ERROR
    }
    
    try:
        with wandb.init(project="congress-trading-monitor", job_type="alert", reinit=True) as run:
            run.alert(title=title, text=text, level=level_map.get(level, AlertLevel.WARN))
        print(f"[Alert] Sent: {title}")
        return True
    except Exception as e:
        print(f"[Alert] Failed: {e}")
        return False


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
        alert_sent = send_alert(
            title="⚠️ Data Drift Detected",
            text=f"Drift: {drift_share:.1%}\nColumns: {', '.join(drifted_columns[:5])}",
            level="warn"
        )
    
    return {
        "drift_share": drift_share,
        "drifted_columns": drifted_columns,
        "alert_sent": alert_sent
    }


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    
    result = run_drift_check(args.reference, args.current, args.threshold, args.report)
    print(f"Result: {result}")
