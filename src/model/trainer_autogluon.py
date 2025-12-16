"""
AutoGluon Trainer for Congress Trading Prediction
Uses AutoGluon's TabularPredictor for automatic model selection and tuning.
Integrated with MLflow for experiment tracking.

Usage:
    python trainer_autogluon.py
    python trainer_autogluon.py --time-limit 600  # 10 minutes
    python trainer_autogluon.py --preset best_quality

Output:
    models/autogluon/  - AutoGluon model artifacts
    models/autogluon_metrics.json - Evaluation metrics
    MLflow UI: http://localhost:5000
"""
import pandas as pd
import numpy as np
import argparse
import os
import json
import requests
from datetime import datetime

import mlflow
from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
DEFAULT_INPUT_PATH = "data/intermediate/03_featured_trades.parquet"
DEFAULT_OUTPUT_DIR = "models/autogluon"
DEFAULT_METRICS_PATH = "models/autogluon_metrics.json"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "congress-trading-autogluon"

# 78 Features - NO Alpha leakage
FEATURE_COLS = [
    # === Basic Features (19) ===
    'Politician_ID', 'Is_Republican', 'Is_Senate', 'State_ID',
    'Is_Purchase', 'Amount_Min', 'Amount_Max', 'Amount_Log', 'Owner_ID',
    'Filed_After', 'Transaction_Month', 'Transaction_DayOfWeek', 'Transaction_Quarter',
    'Price', 'Disclosure_Price', 'Buy_to_Disclosure_Pct',
    'Past_Trade_Count', 'Conviction_Score', 'Co_Trading_Count',
    
    # === Amount Derived (5) ===
    'Amount_Range', 'Amount_Mid', 'Amount_Ratio', 'Amount_Category', 'Is_Large_Trade',
    
    # === Time Derived (8) ===
    'Is_Year_End', 'Is_Q1', 'Is_Q4', 'Is_Monday', 'Is_Friday',
    'Filed_After_Log', 'Is_Quick_Disclosure', 'Is_Slow_Disclosure',
    
    # === Price Derived (4) ===
    'Price_Change_Abs', 'Price_Momentum', 'Is_Dip_Buy', 'Is_Run_Up',
    
    # === Historical Behavior (6) ===
    'Politician_Avg_Amount', 'Amount_vs_Avg', 'Is_Unusual_Size',
    'Days_Since_Last_Trade', 'Trade_Streak', 'Is_First_Trade',
    
    # === Signal Strength (2) ===
    'Signal_Strength', 'Cluster_Buy',
    
    # === Owner Derived (2) ===
    'Is_Self_Trade', 'Is_Family_Trade',
    
    # === Interaction Features (15) ===
    'Senate_Large_Trade', 'Senate_Purchase', 'Senate_Quick_Disclosure',
    'Republican_Purchase', 'Republican_Large_Trade', 'Quick_Large_Trade',
    'Self_Large_Trade', 'Self_Purchase', 'Dip_Buy_Large',
    'High_Conviction_Quick', 'Experienced_Large', 'Cluster_Large',
    'YearEnd_Sale', 'Monday_Purchase', 'Friday_Sale',
    
    # === Lag - Politician (5) ===
    'Politician_Last_Trade_Same_Type', 'Politician_Trade_Frequency_30D',
    'Politician_Trade_Frequency_90D', 'Politician_Avg_Filed_After', 'Politician_Purchase_Ratio',
    
    # === Lag - Ticker (6) ===
    'Ticker_Trade_Count_30D', 'Ticker_Trade_Count_90D', 'Ticker_Buy_Ratio_30D',
    'Ticker_Politician_Count_30D', 'Ticker_Avg_Amount_30D', 'Is_Ticker_Hot',
    
    # === Lag - Market (5) ===
    'Market_Buy_Ratio_7D', 'Market_Buy_Ratio_30D', 'Market_Trade_Volume_7D',
    'Market_Avg_Amount_7D', 'Is_High_Activity_Period',
]

TARGET_COL = 'Target'


def check_mlflow_connection() -> bool:
    """Check if MLflow server is accessible."""
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_available_features(df: pd.DataFrame) -> list:
    """Get list of features that exist in the dataframe."""
    available = [col for col in FEATURE_COLS if col in df.columns]
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    
    print(f"[AutoGluon] Available features: {len(available)}/{len(FEATURE_COLS)}")
    if missing:
        print(f"[AutoGluon] Missing features ({len(missing)}): {missing[:5]}...")
    
    return available


def calculate_backtest_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """Calculate backtest-style portfolio metrics."""
    df = df.copy()
    df['Predicted'] = y_pred
    
    alpha_col = 'Alpha_180' if 'Alpha_180' in df.columns else 'Alpha'
    
    def calc_metrics(subset, prefix=''):
        if len(subset) == 0 or alpha_col not in subset.columns:
            return {
                f'{prefix}trades': 0,
                f'{prefix}avg_alpha': 0,
                f'{prefix}std_alpha': 0,
                f'{prefix}information_ratio': 0,
                f'{prefix}win_rate': 0
            }
        
        alpha_values = subset[alpha_col].dropna()
        if len(alpha_values) == 0:
            return {
                f'{prefix}trades': len(subset),
                f'{prefix}avg_alpha': 0,
                f'{prefix}std_alpha': 0,
                f'{prefix}information_ratio': 0,
                f'{prefix}win_rate': 0
            }
        
        avg_alpha = alpha_values.mean()
        std_alpha = alpha_values.std()
        ir = avg_alpha / std_alpha if std_alpha > 0 else 0
        win_rate = (alpha_values > 0).mean()
        
        return {
            f'{prefix}trades': len(subset),
            f'{prefix}avg_alpha': float(avg_alpha),
            f'{prefix}std_alpha': float(std_alpha),
            f'{prefix}information_ratio': float(ir),
            f'{prefix}win_rate': float(win_rate)
        }
    
    metrics = {}
    
    # Follow all positive predictions (class 1, 2, or 3)
    followed_any = df[df['Predicted'] >= 1]
    metrics.update(calc_metrics(followed_any, 'backtest_'))
    
    # Follow only high conviction (class 2 or 3)
    followed_high = df[df['Predicted'] >= 2]
    metrics.update(calc_metrics(followed_high, 'backtest_high_'))
    
    # Follow only very high conviction (class 3)
    followed_very_high = df[df['Predicted'] >= 3]
    metrics.update(calc_metrics(followed_very_high, 'backtest_very_high_'))
    
    return metrics


# --- Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoGluon Trainer for Congress Trading")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--time-limit", type=int, default=7200, 
                        help="Training time limit in seconds (default: 7200 = 2 hours)")
    parser.add_argument("--preset", type=str, default="best_quality", 
                        choices=["best_quality", "high_quality", "good_quality", "medium_quality"],
                        help="AutoGluon preset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("[AutoGluon] AUTOMATIC MODEL TRAINING")
    print("=" * 60)
    print(f"[AutoGluon] Time limit: {args.time_limit} seconds")
    print(f"[AutoGluon] Preset: {args.preset}")
    
    # Check MLflow connection
    if not check_mlflow_connection():
        print(f"[MLflow] ERROR: Cannot connect to {MLFLOW_TRACKING_URI}")
        print("[MLflow] Please ensure MLflow server is running:")
        print("         cd docker && docker compose up -d mlflow")
        exit(1)
    
    print(f"[MLflow] Connected to {MLFLOW_TRACKING_URI}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    run_name = f"autogluon_{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Set tags
        mlflow.set_tag("model_type", "autogluon")
        mlflow.set_tag("preset", args.preset)
        
        # 1. Load data
        df = pd.read_parquet(args.input)
        print(f"\n[AutoGluon] Read {len(df)} rows from {args.input}")
        
        # 2. Get available features
        feature_cols = get_available_features(df)
        
        # 3. Prepare data
        df_clean = df.dropna(subset=feature_cols + [TARGET_COL])
        print(f"[AutoGluon] After dropping NaN: {len(df_clean)} rows")
        
        # Target distribution
        target_dist = df_clean[TARGET_COL].value_counts().sort_index()
        print(f"\n[AutoGluon] Target distribution:")
        for cls, count in target_dist.items():
            print(f"  Class {cls}: {count} ({100*count/len(df_clean):.1f}%)")
        
        # 4. Time-series split (80/20)
        df_clean = df_clean.sort_values('Notification Date').reset_index(drop=True)
        split_idx = int(len(df_clean) * 0.8)
        
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        print(f"\n[AutoGluon] Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Log parameters
        mlflow.log_params({
            "input_path": args.input,
            "time_limit": args.time_limit,
            "preset": args.preset,
            "num_features": len(feature_cols),
            "train_size": len(train_df),
            "test_size": len(test_df),
            "total_rows": len(df),
            "clean_rows": len(df_clean),
            "target_classes": len(target_dist),
        })
        
        # 5. Prepare AutoGluon format
        train_data = train_df[feature_cols + [TARGET_COL]]
        test_data = test_df[feature_cols + [TARGET_COL]]
        
        # 6. Train with AutoGluon
        print(f"\n[AutoGluon] Training started...")
        print(f"[AutoGluon] This will try multiple models: GBM, XGBoost, LightGBM, CatBoost, RF, NN, etc.")
        
        predictor = TabularPredictor(
            label=TARGET_COL,
            path=args.output_dir,
            problem_type='multiclass',
            eval_metric='roc_auc_ovo_macro',
        ).fit(
            train_data=train_data,
            time_limit=args.time_limit,
            presets=args.preset,
            verbosity=2,
        )
        
        # 7. Evaluate
        print(f"\n[AutoGluon] Evaluating on test set...")
        
        y_pred = predictor.predict(test_data.drop(columns=[TARGET_COL]))
        y_test = test_data[TARGET_COL]
        
        # Leaderboard
        leaderboard = predictor.leaderboard(test_data, silent=True)
        print(f"\n[AutoGluon] === MODEL LEADERBOARD ===")
        print(leaderboard.to_string())
        
        # Feature importance
        print(f"\n[AutoGluon] === FEATURE IMPORTANCE (Top 15) ===")
        try:
            importance = predictor.feature_importance(test_data)
            print(importance.head(15).to_string())
        except:
            importance = None
            print("Feature importance not available")
        
        # Get evaluation metrics
        eval_results = predictor.evaluate(test_data)
        print(f"\n[AutoGluon] === EVALUATION RESULTS ===")
        for metric, value in eval_results.items():
            print(f"  {metric}: {value:.4f}")
        
        # Backtest metrics
        backtest = calculate_backtest_metrics(test_df, y_pred.values)
        
        print(f"\n[AutoGluon] === BACKTEST RESULTS ===")
        print(f"  --- All Positive Predictions (Class >= 1) ---")
        print(f"  Trades:              {backtest['backtest_trades']}")
        print(f"  Avg Alpha (180d):    {backtest['backtest_avg_alpha']*100:.2f}%")
        print(f"  Information Ratio:   {backtest['backtest_information_ratio']:.4f}")
        print(f"  Win Rate:            {backtest['backtest_win_rate']:.2%}")
        
        print(f"\n  --- High Conviction (Class >= 2) ---")
        print(f"  Trades:              {backtest['backtest_high_trades']}")
        if backtest['backtest_high_trades'] > 0:
            print(f"  Avg Alpha (180d):    {backtest['backtest_high_avg_alpha']*100:.2f}%")
            print(f"  Information Ratio:   {backtest['backtest_high_information_ratio']:.4f}")
            print(f"  Win Rate:            {backtest['backtest_high_win_rate']:.2%}")
        
        print(f"\n  --- Very High Conviction (Class >= 3) ---")
        print(f"  Trades:              {backtest['backtest_very_high_trades']}")
        if backtest['backtest_very_high_trades'] > 0:
            print(f"  Avg Alpha (180d):    {backtest['backtest_very_high_avg_alpha']*100:.2f}%")
            print(f"  Information Ratio:   {backtest['backtest_very_high_information_ratio']:.4f}")
            print(f"  Win Rate:            {backtest['backtest_very_high_win_rate']:.2%}")
        
        # Log metrics to MLflow
        for metric, value in eval_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"eval_{metric}", value)
        
        mlflow.log_metrics({
            "backtest_trades": backtest['backtest_trades'],
            "backtest_avg_alpha": backtest['backtest_avg_alpha'],
            "backtest_information_ratio": backtest['backtest_information_ratio'],
            "backtest_win_rate": backtest['backtest_win_rate'],
            "backtest_high_trades": backtest['backtest_high_trades'],
            "backtest_high_avg_alpha": backtest['backtest_high_avg_alpha'],
        })
        
        # Log artifacts
        mlflow.log_artifacts(args.output_dir, "autogluon_model")
        
        # Log leaderboard as JSON
        mlflow.log_dict(leaderboard.to_dict('records'), "leaderboard.json")
        
        if importance is not None:
            mlflow.log_dict(importance.to_dict(), "feature_importance.json")
        
        # Get run info
        run_id = mlflow.active_run().info.run_id
        
        # 8. Save metrics locally
        metrics = {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'num_features': len(feature_cols),
            'time_limit': args.time_limit,
            'preset': args.preset,
            'evaluation': eval_results,
            'backtest': backtest,
            'best_model': predictor.model_best,
            'train_date': datetime.now().isoformat(),
            'mlflow_run_id': run_id,
        }
        
        metrics['leaderboard'] = leaderboard.to_dict('records')
        
        with open(args.metrics, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\n[AutoGluon] Model saved to: {args.output_dir}")
        print(f"[AutoGluon] Metrics saved to: {args.metrics}")
        print(f"[AutoGluon] Best model: {predictor.model_best}")
        print(f"[MLflow] Run ID: {run_id}")
        print(f"[MLflow] View at: {MLFLOW_TRACKING_URI}")
        print("[AutoGluon] Complete.\n")
