"""
Congress Trading Prediction API v2.1
With PostgreSQL database integration for predictions logging and drift detection.

Features:
- MLflow-wrapped model with built-in preprocessing
- PostgreSQL storage for predictions, drift history, and backtest results
- Scheduled drift detection with database persistence
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from apscheduler.schedulers.background import BackgroundScheduler


# =============================================================================
# Configuration
# =============================================================================

VERSION = "2.1.0"  # Database integration
MODEL_PATH = os.getenv("MODEL_PATH", "models/autogluon")
MLFLOW_MODEL_PATH = os.getenv("MLFLOW_MODEL_PATH", "models/mlflow_model")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/intermediate/01_raw_trades.parquet")
DRIFT_CHECK_INTERVAL_DAYS = int(os.getenv("DRIFT_CHECK_INTERVAL_DAYS", "7"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

# Database toggle (for graceful degradation)
USE_DATABASE = os.getenv("USE_DATABASE", "true").lower() == "true"

# Columns to monitor for drift (raw input features)
DRIFT_MONITOR_COLUMNS = ["amount_min", "filed_after", "trade_type", "party", "chamber"]


# =============================================================================
# Schemas
# =============================================================================

class TradeInput(BaseModel):
    """Input for prediction - matches MLflow model INPUT_SCHEMA"""
    # === Required Fields ===
    politician_name: str = Field(..., example="Nancy Pelosi")
    ticker: str = Field(..., example="AAPL")
    type: str = Field(..., example="Purchase")
    amount_min: float = Field(0, example=50000)
    filed_after: int = Field(..., example=14)
    
    # === Important Fields (needed for full feature set) ===
    amount_max: Optional[float] = Field(None, example=100000)  # If None, defaults to amount_min * 2
    owner: Optional[str] = Field(None, example="Self")  # "Self", "Spouse", "Child", "Joint"
    state: Optional[str] = Field(None, example="CA")  # State code
    notification_date: Optional[str] = Field(None, example="2024-01-15")  # Disclosure date
    
    # === Optional Fields ===
    party: Optional[str] = None
    chamber: Optional[str] = None
    transaction_date: Optional[str] = None
    price: Optional[float] = Field(None, example=150.50)  # Price at transaction
    disclosure_price: Optional[float] = Field(None, example=155.00)  # Price at disclosure
    buy_to_disclosure_pct: Optional[float] = Field(None, example=3.0)  # % change


class PredictionResponse(BaseModel):
    """Prediction result - matches MLflow model OUTPUT_SCHEMA"""
    prediction: int
    label: str
    recommendation: str


class BatchInput(BaseModel):
    """Batch of trades"""
    trades: List[TradeInput]


class BatchResponse(BaseModel):
    """Batch results"""
    predictions: List[PredictionResponse]
    follow_count: int
    skip_count: int


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages the MLflow-wrapped model lifecycle."""
    
    _model = None
    _model_type: str = None
    _model_version: str = None
    
    @classmethod
    def load(cls) -> bool:
        """Load the model. Tries MLflow model first, falls back to AutoGluon."""
        
        # Try MLflow model first
        if Path(MLFLOW_MODEL_PATH).exists():
            try:
                import mlflow.pyfunc
                cls._model = mlflow.pyfunc.load_model(MLFLOW_MODEL_PATH)
                cls._model_type = "mlflow"
                cls._model_version = VERSION
                print(f"[Model] Loaded MLflow model from: {MLFLOW_MODEL_PATH}")
                return True
            except Exception as e:
                print(f"[Model] MLflow model load failed: {e}")
        
        # Fall back to AutoGluon model with manual preprocessing
        if Path(MODEL_PATH).exists():
            try:
                import sys
                from io import StringIO
                
                # Suppress AutoGluon's mismatch warnings during load
                # (Python version mismatch, Windows->Linux system mismatch)
                suppress_warnings = os.getenv("AUTOGLUON_SUPPRESS_WARNINGS", "false").lower() == "true"
                
                if suppress_warnings:
                    # Capture stdout AND stderr to suppress warning messages
                    # Must be set BEFORE importing and calling load
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = StringIO()
                    sys.stderr = StringIO()
                    try:
                        from autogluon.tabular import TabularPredictor
                        cls._model = TabularPredictor.load(MODEL_PATH)
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                else:
                    from autogluon.tabular import TabularPredictor
                    cls._model = TabularPredictor.load(MODEL_PATH)
                
                cls._model_type = "autogluon"
                cls._model_version = VERSION
                print(f"[Model] Loaded AutoGluon model from: {MODEL_PATH}")
                return True
            except Exception as e:
                print(f"[Model] AutoGluon model load failed: {e}")
        
        print("[Model] No model found!")
        return False
    
    @classmethod
    def predict(cls, trades: List[TradeInput]) -> List[PredictionResponse]:
        """Run prediction on trades using the loaded model."""
        
        if cls._model_type == "mlflow":
            return cls._predict_mlflow(trades)
        else:
            return cls._predict_autogluon(trades)
    
    @classmethod
    def _predict_mlflow(cls, trades: List[TradeInput]) -> List[PredictionResponse]:
        """Predict using MLflow model (preprocessing is built-in)."""
        now = pd.Timestamp.now()
        
        df = pd.DataFrame([{
            # Required fields
            "politician_name": t.politician_name,
            "ticker": t.ticker,
            "type": t.type,
            "amount_min": t.amount_min,
            "filed_after": t.filed_after,
            # Important fields (with defaults)
            "amount_max": t.amount_max if t.amount_max is not None else t.amount_min * 2,
            "owner": t.owner or "Self",
            "state": t.state or "Unknown",
            "notification_date": t.notification_date or now.strftime("%Y-%m-%d"),
            # Optional fields
            "party": t.party or "Unknown",
            "chamber": t.chamber or "House",
            "transaction_date": t.transaction_date,
            "price": t.price or 0,
            "disclosure_price": t.disclosure_price or 0,
            "buy_to_disclosure_pct": t.buy_to_disclosure_pct or 0,
        } for t in trades])
        
        result_df = cls._model.predict(df)
        
        return [
            PredictionResponse(
                prediction=int(row["prediction"]),
                label=row["label"],
                recommendation=row["recommendation"]
            )
            for _, row in result_df.iterrows()
        ]
    
    @classmethod
    def _predict_autogluon(cls, trades: List[TradeInput]) -> List[PredictionResponse]:
        """Predict using AutoGluon model (requires manual preprocessing)."""
        from ..model.data_cleaner import clean_for_prediction
        from ..model.feature_engineer import engineer_features_for_prediction
        from ..model.schemas import PREDICTION_LABELS, get_recommendation
        
        now = pd.Timestamp.now()
        records = [{
            "Politician Name": t.politician_name,
            "Ticker": t.ticker,
            "Type": t.type,
            "Amount Min": t.amount_min,
            "Amount Max": t.amount_max if t.amount_max is not None else t.amount_min * 2,
            "Filed After": t.filed_after,
            "Party": t.party or "Unknown",
            "Chamber": t.chamber or "House",
            "State": t.state or "Unknown",
            "Owner": t.owner or "Self",
            "Asset Name": t.ticker,
            "Notes": "",
            "Transaction Date": pd.to_datetime(t.transaction_date) if t.transaction_date else now,
            "Notification Date": pd.to_datetime(t.notification_date) if t.notification_date else now,
            "Entry Date": now,
            "Price": t.price or 0,
            "Current Price": 0,
            "Entry Price": t.price or 0,
            "Disclosure Price": t.disclosure_price or 0,
            "Buy to Disclosure %": t.buy_to_disclosure_pct or 0,
            "Ticker_Clean": t.ticker.split(':')[0].replace('$', '').strip(),
        } for t in trades]
        
        df = pd.DataFrame(records)
        df = clean_for_prediction(df)
        df = engineer_features_for_prediction(df)
        
        features = cls._model.feature_metadata_in.get_features()
        for col in features:
            if col not in df.columns:
                df[col] = 0
        
        preds = cls._model.predict(df[features])
        
        return [
            PredictionResponse(
                prediction=int(p),
                label=PREDICTION_LABELS.get(int(p), "Unknown"),
                recommendation=get_recommendation(int(p))
            )
            for p in preds
        ]
    
    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None
    
    @classmethod
    def model_type(cls) -> Optional[str]:
        return cls._model_type
    
    @classmethod
    def model_version(cls) -> Optional[str]:
        return cls._model_version


# =============================================================================
# Prediction Logging
# =============================================================================

def log_predictions_to_db(trades: List[TradeInput], results: List[PredictionResponse]):
    """Log predictions to PostgreSQL database."""
    if not USE_DATABASE:
        return
    
    try:
        from ..database import save_predictions_batch, check_connection
        
        if not check_connection():
            print("[Database] Not available, skipping prediction logging")
            return
        
        predictions = [{
            "politician_name": t.politician_name,
            "ticker": t.ticker,
            "trade_type": t.type,
            "amount_min": t.amount_min,
            "filed_after": t.filed_after,
            "party": t.party or "Unknown",
            "chamber": t.chamber or "House",
            "transaction_date": t.transaction_date,
            "prediction": r.prediction,
            "label": r.label,
            "recommendation": r.recommendation,
            "model_version": ModelManager.model_version(),
            "model_type": ModelManager.model_type(),
        } for t, r in zip(trades, results)]
        
        count = save_predictions_batch(predictions)
        print(f"[Database] Logged {count} predictions")
        
    except Exception as e:
        print(f"[Database] Failed to log predictions: {e}")


# =============================================================================
# Drift Detection
# =============================================================================

def run_drift_check(check_type: str = "scheduled") -> dict:
    """
    Run comprehensive drift detection:
    1. Input feature drift (using Evidently)
    2. Prediction distribution drift (comparing recent vs historical predictions)
    
    Always sends notification:
    - Simple report if no drift
    - Detailed report if drift detected
    
    Stores results in drift_history table.
    """
    try:
        from ..monitoring.drift import check_drift
        from ..monitoring.alert import send_alert
        
        # Load reference data
        if not Path(REFERENCE_DATA_PATH).exists():
            print(f"[Drift] Reference data not found: {REFERENCE_DATA_PATH}")
            return {"error": "Reference data not found"}
        
        ref_df = pd.read_parquet(REFERENCE_DATA_PATH)
        
        # Load current data from database or fallback to file
        if USE_DATABASE:
            try:
                from ..database import get_recent_predictions, get_prediction_stats, check_connection
                
                if check_connection():
                    predictions = get_recent_predictions(days=30, limit=10000)
                    if not predictions:
                        # No recent predictions - send simple notification
                        send_alert(
                            title="✅ 漂移检测完成",
                            text=f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n状态: 无预测数据，跳过检测",
                            level="info"
                        )
                        return {"drift_share": 0, "message": "No recent predictions"}
                    cur_df = pd.DataFrame(predictions)
                    
                    # Get prediction distribution stats
                    pred_stats = get_prediction_stats(days=30)
                else:
                    return {"error": "Database not available"}
            except Exception as e:
                print(f"[Drift] Database error: {e}")
                return {"error": str(e)}
        else:
            # Fallback to file-based
            prediction_log_path = os.getenv("PREDICTION_LOG_PATH", "data/logs/prediction_log.parquet")
            if not Path(prediction_log_path).exists():
                return {"drift_share": 0, "message": "No predictions logged"}
            cur_df = pd.read_parquet(prediction_log_path)
            pred_stats = None
        
        # Standardize column names for comparison
        ref_renamed = ref_df.rename(columns={
            "Amount Min": "amount_min",
            "Filed After": "filed_after",
            "Type": "trade_type",
            "Party": "party",
            "Chamber": "chamber",
        })
        
        # Filter to common columns
        cols = [c for c in DRIFT_MONITOR_COLUMNS if c in ref_renamed.columns and c in cur_df.columns]
        
        if not cols:
            return {"error": "No common columns to compare"}
        
        # === 1. Input Feature Drift Detection ===
        is_drifted, metrics = check_drift(ref_renamed[cols], cur_df[cols])
        feature_drift_share = metrics.get('drift_share', 0)
        drifted_columns = metrics.get('drifted_columns', [])
        
        # === 2. Prediction Distribution Drift Detection ===
        pred_dist_drift = 0.0
        pred_dist_info = ""
        
        if 'prediction' in cur_df.columns:
            # Calculate current prediction distribution
            cur_pred_dist = cur_df['prediction'].value_counts(normalize=True).sort_index()
            
            # Expected distribution from training (approximate)
            # Class 0: ~60%, Class 1: ~20%, Class 2: ~15%, Class 3: ~5%
            expected_dist = {0: 0.60, 1: 0.20, 2: 0.15, 3: 0.05}
            
            # Calculate distribution drift (simple KL-divergence approximation)
            total_diff = 0
            for cls in range(4):
                cur_pct = cur_pred_dist.get(cls, 0)
                exp_pct = expected_dist.get(cls, 0.25)
                total_diff += abs(cur_pct - exp_pct)
            
            pred_dist_drift = total_diff / 2  # Normalize to 0-1
            
            # Format distribution info
            dist_parts = []
            for cls in range(4):
                pct = cur_pred_dist.get(cls, 0) * 100
                dist_parts.append(f"Class{cls}:{pct:.1f}%")
            pred_dist_info = " | ".join(dist_parts)
        
        # === Combined Drift Score ===
        # Weight: 70% feature drift, 30% prediction distribution drift
        combined_drift = feature_drift_share * 0.7 + pred_dist_drift * 0.3
        
        # Determine if we need to alert
        alert_sent = False
        alert_level = None
        has_drift = combined_drift > DRIFT_THRESHOLD
        
        # === Always Send Notification ===
        check_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        if has_drift:
            # Detailed report for drift detected
            alert_level = "warn" if combined_drift < 0.5 else "error"
            
            detail_text = f"""🔍 漂移检测报告 - 检测到异常
━━━━━━━━━━━━━━━━━━━━━━━
⏰ 检测时间: {check_time}
📊 检测类型: {check_type}

📈 特征漂移: {feature_drift_share:.1%}
📉 预测分布漂移: {pred_dist_drift:.1%}
⚠️ 综合得分: {combined_drift:.1%} (阈值: {DRIFT_THRESHOLD:.0%})

🔴 漂移特征: {', '.join(drifted_columns[:5]) if drifted_columns else '无'}

📊 当前预测分布:
{pred_dist_info}

📋 数据规模:
• 参考数据: {len(ref_renamed):,} 条
• 当前数据: {len(cur_df):,} 条

💡 建议: 考虑重新训练模型"""

            alert_sent = send_alert(
                title="⚠️ 漂移检测 - 发现异常",
                text=detail_text,
                level=alert_level
            )
        else:
            # Simple report for no drift
            simple_text = f"""✅ 漂移检测报告 - 正常
━━━━━━━━━━━━━━━━━━━━━━━
⏰ 检测时间: {check_time}
📊 检测类型: {check_type}

📈 特征漂移: {feature_drift_share:.1%}
📉 预测分布漂移: {pred_dist_drift:.1%}
✅ 综合得分: {combined_drift:.1%} (阈值: {DRIFT_THRESHOLD:.0%})

📋 数据: {len(ref_renamed):,} 参考 / {len(cur_df):,} 当前
📊 分布: {pred_dist_info}

状态: 一切正常 ✓"""

            send_alert(
                title="✅ 漂移检测 - 正常",
                text=simple_text,
                level="info"
            )
        
        # Save to database
        if USE_DATABASE:
            try:
                from ..database import save_drift_result
                save_drift_result(
                    drift_share=float(combined_drift),
                    is_drifted=bool(has_drift),
                    drifted_columns=list(drifted_columns),
                    total_columns=int(len(cols)),
                    threshold=float(DRIFT_THRESHOLD),
                    alert_sent=bool(alert_sent),
                    alert_level=alert_level,
                    check_type=check_type,
                    reference_count=int(len(ref_renamed)),
                    current_count=int(len(cur_df)),
                )
                print(f"[Drift] Result saved to database")
            except Exception as e:
                print(f"[Drift] Failed to save to database: {e}")
        
        print(f"[Drift] Check complete - Combined drift: {combined_drift:.1%} (Feature: {feature_drift_share:.1%}, Pred: {pred_dist_drift:.1%})")
        
        return {
            "drift_share": float(combined_drift),
            "feature_drift": float(feature_drift_share),
            "prediction_drift": float(pred_dist_drift),
            "is_drifted": bool(has_drift),
            "drifted_columns": list(drifted_columns),
            "prediction_distribution": str(pred_dist_info),
            "alert_sent": bool(alert_sent),
            "reference_count": int(len(ref_renamed)),
            "current_count": int(len(cur_df)),
        }
        
    except Exception as e:
        print(f"[Drift] Check failed: {e}")
        # Send error notification
        try:
            from ..monitoring.alert import send_alert
            send_alert(
                title="🚨 漂移检测失败",
                text=f"错误: {str(e)}\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                level="error"
            )
        except:
            pass
        return {"error": str(e)}


# =============================================================================
# Scheduler
# =============================================================================

scheduler = BackgroundScheduler()


def scheduled_drift_check():
    """Wrapper for scheduled drift check."""
    run_drift_check(check_type="scheduled")


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    ModelManager.load()
    
    # Initialize database connection pool
    if USE_DATABASE:
        try:
            from ..database import init_pool, check_connection
            init_pool()
            if check_connection():
                print("[Database] Connected successfully")
            else:
                print("[Database] Connection failed, running without database")
        except Exception as e:
            print(f"[Database] Initialization failed: {e}")
    
    # Start scheduler
    scheduler.add_job(scheduled_drift_check, 'interval', days=DRIFT_CHECK_INTERVAL_DAYS, id='drift')
    scheduler.start()
    
    yield
    
    # Shutdown
    scheduler.shutdown()
    
    if USE_DATABASE:
        try:
            from ..database import close_pool
            close_pool()
        except Exception:
            pass


app = FastAPI(
    title="Congress Trading Prediction API",
    version=VERSION,
    description="ML-powered prediction for congressional trades with PostgreSQL storage.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    db_status = "disabled"
    if USE_DATABASE:
        try:
            from ..database import check_connection
            db_status = "connected" if check_connection() else "disconnected"
        except Exception:
            db_status = "error"
    
    return {
        "status": "healthy",
        "model_loaded": ModelManager.is_loaded(),
        "model_type": ModelManager.model_type(),
        "database": db_status,
        "version": VERSION
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(trade: TradeInput, background: BackgroundTasks):
    """Predict trade quality for a single trade."""
    if not ModelManager.is_loaded():
        raise HTTPException(503, "Model not loaded")
    
    results = ModelManager.predict([trade])
    background.add_task(log_predictions_to_db, [trade], results)
    return results[0]


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchInput, background: BackgroundTasks):
    """Predict trade quality for multiple trades at once."""
    if not ModelManager.is_loaded():
        raise HTTPException(503, "Model not loaded")
    if not batch.trades:
        raise HTTPException(400, "Empty batch")
    
    results = ModelManager.predict(batch.trades)
    background.add_task(log_predictions_to_db, batch.trades, results)
    
    follow = sum(1 for r in results if r.recommendation == "FOLLOW")
    return BatchResponse(
        predictions=results,
        follow_count=follow,
        skip_count=len(results) - follow
    )


@app.post("/drift/check")
async def check_drift_endpoint():
    """Manually trigger a drift check."""
    result = run_drift_check(check_type="manual")
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.get("/drift/history")
async def get_drift_history(days: int = 30, limit: int = 50):
    """Get drift detection history."""
    if not USE_DATABASE:
        raise HTTPException(400, "Database not enabled")
    
    try:
        from ..database import get_drift_history
        history = get_drift_history(days=days, limit=limit)
        return {"history": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/drift/trend")
async def get_drift_trend(days: int = 30):
    """Get daily drift trend."""
    if not USE_DATABASE:
        raise HTTPException(400, "Database not enabled")
    
    try:
        from ..database import get_drift_trend
        trend = get_drift_trend(days=days)
        return {"trend": trend}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/predictions/stats")
async def get_prediction_stats(days: int = 7):
    """Get prediction statistics."""
    if not USE_DATABASE:
        raise HTTPException(400, "Database not enabled")
    
    try:
        from ..database import get_prediction_stats
        stats = get_prediction_stats(days=days)
        return stats
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    return {
        "loaded": ModelManager.is_loaded(),
        "type": ModelManager.model_type(),
        "version": ModelManager.model_version(),
        "model_path": MODEL_PATH if ModelManager.model_type() == "autogluon" else MLFLOW_MODEL_PATH,
    }


# =============================================================================
# Accuracy Validation Endpoint
# =============================================================================

class AccuracyValidationResult(BaseModel):
    """Result of accuracy validation."""
    total_records: int
    matched_records: int
    accuracy: float
    precision_by_class: dict
    recall_by_class: dict
    confusion_matrix: dict
    class_distribution_actual: dict
    class_distribution_predicted: dict
    follow_accuracy: float  # Accuracy for FOLLOW recommendations
    recommendations: List[str]


@app.post("/validate/accuracy", response_model=AccuracyValidationResult)
async def validate_accuracy(
    file_path: str = None,
    alpha_column: str = "Alpha_180",
    min_records: int = 100
):
    """
    Validate model prediction accuracy against actual outcomes.
    
    This is a POST-HOC validation - you provide a CSV file with actual Alpha values,
    and the system compares model predictions against actual outcomes.
    
    Args:
        file_path: Path to CSV file with actual outcomes (must contain Alpha column)
        alpha_column: Name of the Alpha column in the CSV (default: Alpha_180)
        min_records: Minimum records required for validation (default: 100)
    
    The CSV should contain:
    - Standard trade columns (Politician Name, Ticker, Type, Amount Min, Filed After, etc.)
    - An Alpha column with actual 180-day returns (Alpha_180)
    
    Returns:
        Detailed accuracy metrics and recommendations
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    if not file_path:
        raise HTTPException(400, "file_path is required. Provide path to CSV with actual outcomes.")
    
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    
    try:
        # Load the file (CSV or Parquet)
        if str(csv_path).endswith('.parquet'):
            df = pd.read_parquet(csv_path)
        else:
            df = pd.read_csv(csv_path)
        
        # If Alpha column not found, try to calculate it from original CSV columns
        if alpha_column not in df.columns:
            # Try to calculate Alpha_180 from Exit Price 180 and Entry Price
            if 'Exit Price 180' in df.columns and 'Entry Price' in df.columns:
                # Stock return = (Exit Price - Entry Price) / Entry Price
                df['Stock_Return_180'] = (df['Exit Price 180'] - df['Entry Price']) / df['Entry Price']
                
                # SPY return (if available)
                if 'SPY Exit 180' in df.columns and 'SPY Entry' in df.columns:
                    df['SPY_Return_180'] = (df['SPY Exit 180'] - df['SPY Entry']) / df['SPY Entry']
                    df['Alpha_180'] = df['Stock_Return_180'] - df['SPY_Return_180']
                else:
                    # Use stock return as alpha if SPY data not available
                    df['Alpha_180'] = df['Stock_Return_180']
                
                alpha_column = 'Alpha_180'
                print(f"[Validation] Calculated Alpha_180 from price columns")
            else:
                available_cols = [c for c in df.columns if 'price' in c.lower() or 'alpha' in c.lower() or 'return' in c.lower()]
                raise HTTPException(
                    400, 
                    f"Column '{alpha_column}' not found and cannot be calculated. "
                    f"Need 'Entry Price' and 'Exit Price 180' columns. "
                    f"Available related columns: {available_cols[:10]}"
                )
        
        # Filter valid records (drop NaN in Alpha column)
        df = df.dropna(subset=[alpha_column])
        
        if len(df) < min_records:
            raise HTTPException(400, f"Insufficient records with valid {alpha_column}: {len(df)} < {min_records} required")
        
        # Calculate actual classes from Alpha values
        def alpha_to_class(alpha):
            if alpha < 0:
                return 0  # Weak
            elif alpha < 0.10:
                return 1  # Fair
            elif alpha < 0.20:
                return 2  # Good
            else:
                return 3  # Excellent
        
        df['actual_class'] = df[alpha_column].apply(alpha_to_class)
        
        # Prepare for prediction
        trades = []
        for _, row in df.iterrows():
            trade = TradeInput(
                politician_name=row.get('Politician Name', 'Unknown'),
                ticker=row.get('Ticker', 'UNK'),
                type=row.get('Type', 'Purchase'),
                amount_min=float(row.get('Amount Min', 0) or 0),
                filed_after=int(row.get('Filed After', 0) or 0),
                party=row.get('Party'),
                chamber=row.get('Chamber'),
            )
            trades.append(trade)
        
        # Get predictions
        predictions = ModelManager.predict(trades)
        df['predicted_class'] = [p.prediction for p in predictions]
        df['recommendation'] = [p.recommendation for p in predictions]
        
        # Calculate metrics
        y_true = df['actual_class'].values
        y_pred = df['predicted_class'].values
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class precision and recall
        labels = [0, 1, 2, 3]
        precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        precision_by_class = {f"class_{i}": float(p) for i, p in enumerate(precision)}
        recall_by_class = {f"class_{i}": float(r) for i, r in enumerate(recall)}
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        confusion_dict = {
            f"actual_{i}": {f"pred_{j}": int(cm[i][j]) for j in range(4)}
            for i in range(4)
        }
        
        # Class distribution
        actual_dist = df['actual_class'].value_counts(normalize=True).to_dict()
        pred_dist = df['predicted_class'].value_counts(normalize=True).to_dict()
        
        actual_dist = {f"class_{k}": float(v) for k, v in actual_dist.items()}
        pred_dist = {f"class_{k}": float(v) for k, v in pred_dist.items()}
        
        # FOLLOW accuracy (how often FOLLOW recommendations are actually good)
        follow_df = df[df['recommendation'] == 'FOLLOW']
        if len(follow_df) > 0:
            follow_correct = (follow_df['actual_class'] >= 2).sum()
            follow_accuracy = follow_correct / len(follow_df)
        else:
            follow_accuracy = 0.0
        
        # Generate recommendations
        recommendations = []
        
        if accuracy < 0.3:
            recommendations.append("⚠️ 整体准确率较低，建议重新训练模型")
        elif accuracy < 0.5:
            recommendations.append("📊 准确率一般，考虑调整特征工程")
        else:
            recommendations.append("✅ 模型表现良好")
        
        if follow_accuracy < 0.5:
            recommendations.append("⚠️ FOLLOW 推荐准确率不足50%，需优化推荐阈值")
        elif follow_accuracy >= 0.7:
            recommendations.append("✅ FOLLOW 推荐可信度高")
        
        # Check for class imbalance in predictions
        pred_follow_rate = (df['predicted_class'] >= 2).mean()
        if pred_follow_rate < 0.1:
            recommendations.append("📉 模型过于保守，FOLLOW 推荐比例过低")
        elif pred_follow_rate > 0.5:
            recommendations.append("📈 模型过于激进，FOLLOW 推荐比例过高")
        
        return AccuracyValidationResult(
            total_records=len(df),
            matched_records=len(df),
            accuracy=float(accuracy),
            precision_by_class=precision_by_class,
            recall_by_class=recall_by_class,
            confusion_matrix=confusion_dict,
            class_distribution_actual=actual_dist,
            class_distribution_predicted=pred_dist,
            follow_accuracy=float(follow_accuracy),
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Validation failed: {str(e)}")


@app.post("/validate/accuracy/upload")
async def validate_accuracy_upload():
    """
    Validate accuracy by directly uploading a CSV file.
    
    Usage: Upload a CSV file via form-data with field name 'file'.
    The CSV should contain trade data with an 'Alpha_180' column for actual returns.
    
    Example using curl:
    ```
    curl -X POST http://localhost:8000/validate/accuracy/upload \
         -F "file=@congress_trading_2025.csv"
    ```
    
    Note: For large files, use the /validate/accuracy endpoint with a file path instead.
    """
    raise HTTPException(
        501, 
        "File upload not implemented. Please use /validate/accuracy with a file_path parameter. "
        "Place your CSV file in the mounted volume (e.g., /app/data/) and provide the path."
    )


# =============================================================================
# Run: uvicorn src.api.app:app --reload
# =============================================================================
