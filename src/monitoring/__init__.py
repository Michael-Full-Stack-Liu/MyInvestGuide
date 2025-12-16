# src/monitoring/__init__.py
"""Monitoring module - Data drift detection using Evidently."""

from .drift import check_drift, generate_drift_report

__all__ = ['check_drift', 'generate_drift_report']
