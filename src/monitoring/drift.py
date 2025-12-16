"""
Data Drift Monitoring using Evidently
Simple module for detecting distribution changes between datasets.

Usage:
    from monitoring.drift import check_drift, generate_drift_report
    
    # Quick check
    is_drifted, metrics = check_drift(reference_df, current_df)
    
    # Generate HTML report
    generate_drift_report(reference_df, current_df, "reports/drift.html")
"""

import pandas as pd
from typing import Tuple, Dict, List, Optional

from evidently import Report
from evidently.presets import DataDriftPreset


def check_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: List[str] = None
) -> Tuple[bool, Dict]:
    """
    Check for data drift between reference and current datasets.
    
    Args:
        reference: Training/baseline data
        current: New data to compare
        columns: Specific columns to check (default: all)
        
    Returns:
        Tuple of (is_drifted, metrics_dict)
    """
    # Select columns if specified
    if columns:
        ref = reference[columns].copy()
        cur = current[[c for c in columns if c in current.columns]].copy()
    else:
        ref = reference.copy()
        cur = current.copy()
    
    # Run drift report
    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur, reference_data=ref)
    
    # Extract metrics
    result_dict = result.dict()
    metrics = {
        'dataset_drift': False,
        'drift_share': 0.0,
        'drifted_columns': [],
        'n_columns': 0
    }
    
    for metric in result_dict.get('metrics', []):
        metric_result = metric.get('result', {})
        if 'dataset_drift' in metric_result:
            metrics['dataset_drift'] = metric_result.get('dataset_drift', False)
            metrics['drift_share'] = metric_result.get('drift_share', 0.0)
            metrics['n_columns'] = metric_result.get('number_of_columns', 0)
            
            # Get drifted columns
            drift_by_col = metric_result.get('drift_by_columns', {})
            for col, info in drift_by_col.items():
                if info.get('drift_detected', False):
                    metrics['drifted_columns'].append(col)
    
    return metrics['dataset_drift'], metrics


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: str,
    columns: List[str] = None
) -> str:
    """
    Generate and save a drift report.
    
    Args:
        reference: Training/baseline data  
        current: New data to compare
        output_path: Path to save HTML report
        columns: Specific columns to check (default: all)
        
    Returns:
        Path to saved report
    """
    # Select columns if specified
    if columns:
        ref = reference[columns].copy()
        cur = current[[c for c in columns if c in current.columns]].copy()
    else:
        ref = reference.copy()
        cur = current.copy()
    
    # Run and save report
    report = Report([DataDriftPreset()])
    result = report.run(current_data=cur, reference_data=ref)
    result.save_html(output_path)
    
    print(f"[Drift] Report saved to: {output_path}")
    return output_path


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check data drift")
    parser.add_argument("--reference", required=True, help="Reference parquet file")
    parser.add_argument("--current", required=True, help="Current parquet/csv file")  
    parser.add_argument("--output", default="reports/drift_report.html")
    args = parser.parse_args()
    
    # Load data
    ref_df = pd.read_parquet(args.reference)
    cur_df = pd.read_csv(args.current) if args.current.endswith('.csv') else pd.read_parquet(args.current)
    
    # Check drift
    is_drifted, metrics = check_drift(ref_df, cur_df)
    
    print(f"\nDataset Drift: {is_drifted}")
    print(f"Drift Share: {metrics['drift_share']:.1%}")
    print(f"Drifted Columns: {metrics['drifted_columns']}")
    
    # Save report
    generate_drift_report(ref_df, cur_df, args.output)
