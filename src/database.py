"""
Database Module for Congress Trading Prediction API

Provides database connection management and repository functions for:
- Predictions logging
- Drift detection history
- Backtest results storage

Uses psycopg2 for PostgreSQL connection with connection pooling.
"""

import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool


# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mlflow:mlflow_password@localhost:5432/mlflow_db"
)

# Parse DATABASE_URL for individual components
def _parse_db_url(url: str) -> dict:
    """Parse PostgreSQL connection URL."""
    # Format: postgresql://user:password@host:port/database
    if url.startswith("postgresql://"):
        url = url[13:]
    
    user_pass, host_db = url.split("@")
    user, password = user_pass.split(":")
    host_port, database = host_db.split("/")
    
    if ":" in host_port:
        host, port = host_port.split(":")
    else:
        host, port = host_port, "5432"
    
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": int(port),
        "database": database,
    }


# =============================================================================
# Connection Pool
# =============================================================================

_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def init_pool(min_conn: int = 1, max_conn: int = 10):
    """Initialize the connection pool."""
    global _connection_pool
    
    if _connection_pool is not None:
        return
    
    try:
        config = _parse_db_url(DATABASE_URL)
        _connection_pool = pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            **config
        )
        print(f"[Database] Connection pool initialized: {config['host']}:{config['port']}/{config['database']}")
    except Exception as e:
        print(f"[Database] Failed to initialize pool: {e}")
        _connection_pool = None


def close_pool():
    """Close the connection pool."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        print("[Database] Connection pool closed")


@contextmanager
def get_connection():
    """Get a connection from the pool."""
    if _connection_pool is None:
        init_pool()
    
    if _connection_pool is None:
        raise RuntimeError("Database connection pool not available")
    
    conn = _connection_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        _connection_pool.putconn(conn)


def check_connection() -> bool:
    """Check if database is accessible."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
    except Exception as e:
        print(f"[Database] Connection check failed: {e}")
        return False


# =============================================================================
# Predictions Repository
# =============================================================================

def save_prediction(
    politician_name: str,
    ticker: str,
    trade_type: str,
    amount_min: float,
    filed_after: int,
    party: Optional[str],
    chamber: Optional[str],
    transaction_date: Optional[str],
    prediction: int,
    label: str,
    recommendation: str,
    model_version: Optional[str] = None,
    model_type: Optional[str] = None,
) -> int:
    """Save a single prediction to the database."""
    sql = """
    INSERT INTO predictions (
        politician_name, ticker, trade_type, amount_min, filed_after,
        party, chamber, transaction_date,
        prediction, label, recommendation,
        model_version, model_type
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) RETURNING id
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                politician_name, ticker, trade_type, amount_min, filed_after,
                party, chamber, transaction_date,
                prediction, label, recommendation,
                model_version, model_type
            ))
            return cur.fetchone()[0]


def save_predictions_batch(predictions: List[Dict[str, Any]]) -> int:
    """Save multiple predictions in a batch."""
    if not predictions:
        return 0
    
    sql = """
    INSERT INTO predictions (
        politician_name, ticker, trade_type, amount_min, filed_after,
        party, chamber, transaction_date,
        prediction, label, recommendation,
        model_version, model_type
    ) VALUES (
        %(politician_name)s, %(ticker)s, %(trade_type)s, %(amount_min)s, %(filed_after)s,
        %(party)s, %(chamber)s, %(transaction_date)s,
        %(prediction)s, %(label)s, %(recommendation)s,
        %(model_version)s, %(model_type)s
    )
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, predictions)
            return cur.rowcount


def get_recent_predictions(
    days: int = 7,
    limit: int = 1000
) -> List[Dict]:
    """Get recent predictions for drift detection."""
    sql = """
    SELECT 
        politician_name, ticker, trade_type, amount_min, filed_after,
        party, chamber, prediction, label, recommendation, created_at
    FROM predictions
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
    ORDER BY created_at DESC
    LIMIT %s
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (days, limit))
            return [dict(row) for row in cur.fetchall()]


def get_prediction_stats(days: int = 7) -> Dict:
    """Get prediction statistics for the last N days."""
    sql = """
    SELECT 
        COUNT(*) as total_predictions,
        SUM(CASE WHEN recommendation = 'FOLLOW' THEN 1 ELSE 0 END) as follow_count,
        SUM(CASE WHEN recommendation = 'SKIP' THEN 1 ELSE 0 END) as skip_count,
        AVG(prediction) as avg_prediction
    FROM predictions
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (days,))
            result = cur.fetchone()
            return dict(result) if result else {}


# =============================================================================
# Drift History Repository
# =============================================================================

def save_drift_result(
    drift_share: float,
    is_drifted: bool,
    drifted_columns: List[str],
    total_columns: int,
    threshold: float,
    alert_sent: bool,
    alert_level: Optional[str],
    check_type: str,
    reference_count: int,
    current_count: int,
) -> int:
    """Save a drift detection result."""
    sql = """
    INSERT INTO drift_history (
        drift_share, is_drifted, drifted_columns, total_columns,
        threshold, alert_sent, alert_level,
        check_type, reference_count, current_count
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) RETURNING id
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                drift_share, is_drifted, drifted_columns, total_columns,
                threshold, alert_sent, alert_level,
                check_type, reference_count, current_count
            ))
            return cur.fetchone()[0]


def get_drift_history(days: int = 30, limit: int = 100) -> List[Dict]:
    """Get drift detection history."""
    sql = """
    SELECT 
        id, drift_share, is_drifted, drifted_columns, total_columns,
        threshold, alert_sent, alert_level,
        check_type, reference_count, current_count, checked_at
    FROM drift_history
    WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
    ORDER BY checked_at DESC
    LIMIT %s
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (days, limit))
            return [dict(row) for row in cur.fetchall()]


def get_drift_trend(days: int = 30) -> List[Dict]:
    """Get daily drift trend."""
    sql = """
    SELECT 
        DATE(checked_at) as date,
        AVG(drift_share) as avg_drift_share,
        MAX(drift_share) as max_drift_share,
        COUNT(*) as check_count,
        SUM(CASE WHEN alert_sent THEN 1 ELSE 0 END) as alerts_sent
    FROM drift_history
    WHERE checked_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
    GROUP BY DATE(checked_at)
    ORDER BY date DESC
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (days,))
            return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Backtest Results Repository
# =============================================================================

def save_backtest_result(
    model_version: str,
    mlflow_run_id: Optional[str],
    model_type: Optional[str],
    backtest_start_date: Optional[str],
    backtest_end_date: Optional[str],
    train_size: int,
    test_size: int,
    # Classification metrics
    accuracy: Optional[float] = None,
    precision_macro: Optional[float] = None,
    recall_macro: Optional[float] = None,
    f1_macro: Optional[float] = None,
    roc_auc_macro: Optional[float] = None,
    # Business metrics
    avg_alpha: Optional[float] = None,
    std_alpha: Optional[float] = None,
    information_ratio: Optional[float] = None,
    win_rate: Optional[float] = None,
    total_trades: Optional[int] = None,
    # High conviction metrics
    high_conviction_trades: Optional[int] = None,
    high_conviction_avg_alpha: Optional[float] = None,
    high_conviction_win_rate: Optional[float] = None,
) -> int:
    """Save a backtest result."""
    sql = """
    INSERT INTO backtest_results (
        model_version, mlflow_run_id, model_type,
        backtest_start_date, backtest_end_date, train_size, test_size,
        accuracy, precision_macro, recall_macro, f1_macro, roc_auc_macro,
        avg_alpha, std_alpha, information_ratio, win_rate, total_trades,
        high_conviction_trades, high_conviction_avg_alpha, high_conviction_win_rate
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) RETURNING id
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                model_version, mlflow_run_id, model_type,
                backtest_start_date, backtest_end_date, train_size, test_size,
                accuracy, precision_macro, recall_macro, f1_macro, roc_auc_macro,
                avg_alpha, std_alpha, information_ratio, win_rate, total_trades,
                high_conviction_trades, high_conviction_avg_alpha, high_conviction_win_rate
            ))
            return cur.fetchone()[0]


def get_backtest_results(limit: int = 20) -> List[Dict]:
    """Get recent backtest results."""
    sql = """
    SELECT *
    FROM backtest_results
    ORDER BY created_at DESC
    LIMIT %s
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return [dict(row) for row in cur.fetchall()]


def get_best_backtest_by_metric(metric: str = "avg_alpha", limit: int = 5) -> List[Dict]:
    """Get best backtest results by a specific metric."""
    # Whitelist allowed metrics to prevent SQL injection
    allowed_metrics = [
        "avg_alpha", "win_rate", "information_ratio", "accuracy", 
        "roc_auc_macro", "f1_macro"
    ]
    if metric not in allowed_metrics:
        metric = "avg_alpha"
    
    sql = f"""
    SELECT *
    FROM backtest_results
    WHERE {metric} IS NOT NULL
    ORDER BY {metric} DESC
    LIMIT %s
    """
    
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return [dict(row) for row in cur.fetchall()]
