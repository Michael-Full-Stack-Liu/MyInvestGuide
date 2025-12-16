"""
Data Drift Monitoring using Evidently
Detects distribution changes between reference and current data.

Usage:
    from src.monitoring.drift import check_drift, generate_drift_report
    
    # Using file paths
    is_drifted, metrics = check_drift("data/train.parquet", "data/new.csv")
    generate_drift_report("data/train.parquet", "data/new.csv", "reports/drift.html")
    
    # Or using DataFrames
    is_drifted, metrics = check_drift(train_df, new_df)
"""

import pandas as pd
from typing import Tuple, Dict, List, Union
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset


def _load_data(data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Load data from file path or return DataFrame as-is."""
    if isinstance(data, pd.DataFrame):
        return data
    
    path = str(data)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def check_drift(
    reference: Union[str, pd.DataFrame],
    current: Union[str, pd.DataFrame],
    columns: List[str] = None
) -> Tuple[bool, Dict]:
    """
    Check for data drift between reference and current data.
    
    Args:
        reference: Training data (file path or DataFrame)
        current: New data to compare (file path or DataFrame)
        columns: Specific columns to check (default: all)
        
    Returns:
        Tuple of (is_drifted, metrics_dict)
    """
    ref = _load_data(reference)
    cur = _load_data(current)
    
    # Select columns if specified
    if columns:
        ref = ref[[c for c in columns if c in ref.columns]]
        cur = cur[[c for c in columns if c in cur.columns]]
    
    # Run drift report
    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur, reference_data=ref)
    
    # Extract metrics
    metrics = {
        'dataset_drift': False,
        'drift_share': 0.0,
        'drifted_columns': [],
        'n_columns': 0
    }
    
    for metric in result.dict().get('metrics', []):
        metric_result = metric.get('result', {})
        if 'dataset_drift' in metric_result:
            metrics['dataset_drift'] = metric_result.get('dataset_drift', False)
            metrics['drift_share'] = metric_result.get('drift_share', 0.0)
            metrics['n_columns'] = metric_result.get('number_of_columns', 0)
            
            for col, info in metric_result.get('drift_by_columns', {}).items():
                if info.get('drift_detected', False):
                    metrics['drifted_columns'].append(col)
    
    return metrics['dataset_drift'], metrics


def generate_drift_report(
    reference: Union[str, pd.DataFrame],
    current: Union[str, pd.DataFrame],
    output_path: str,
    columns: List[str] = None
) -> str:
    """
    Generate and save a drift report as HTML.
    
    Args:
        reference: Training data (file path or DataFrame)
        current: New data to compare (file path or DataFrame)
        output_path: Path to save HTML report
        columns: Specific columns to check (default: all)
        
    Returns:
        Path to saved report
    """
    ref = _load_data(reference)
    cur = _load_data(current)
    
    if columns:
        ref = ref[[c for c in columns if c in ref.columns]]
        cur = cur[[c for c in columns if c in cur.columns]]
    
    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur, reference_data=ref)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(output_path)
    
    print(f"[Drift] Report saved: {output_path}")
    return output_path


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check data drift")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", default="reports/drift_report.html")
    args = parser.parse_args()
    
    is_drifted, metrics = check_drift(args.reference, args.current)
    print(f"\nDrift detected: {is_drifted}")
    print(f"Drift share: {metrics['drift_share']:.1%}")
    print(f"Drifted columns: {metrics['drifted_columns']}")
    
    generate_drift_report(args.reference, args.current, args.output)
