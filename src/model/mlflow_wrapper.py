"""
MLflow Custom Model Wrapper for Congress Trading Prediction

This module wraps the complete prediction pipeline (data cleaning + feature engineering + model)
into a single MLflow PythonModel, enabling:
1. One-step model deployment (no external code dependencies)
2. Input/output schema validation
3. Version-locked preprocessing with model

Usage:
    # During training (in trainer_autogluon.py):
    from mlflow_wrapper import CongressTradingModel, log_mlflow_model
    log_mlflow_model(predictor, model_dir)
    
    # During inference (in app.py):
    model = mlflow.pyfunc.load_model("models:/congress-trading/Production")
    result = model.predict(raw_trade_df)
"""

import pandas as pd
import numpy as np
from typing import Optional
import mlflow.pyfunc

from .schemas import PREDICTION_LABELS, get_recommendation, FEATURE_COLS


class CongressTradingModel(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow model that encapsulates the entire prediction pipeline:
    1. Data Cleaning (data_cleaner.clean_for_prediction)
    2. Feature Engineering (feature_engineer.engineer_features_for_prediction)
    3. AutoGluon Prediction (TabularPredictor.predict)
    
    This ensures that the preprocessing logic is always bundled with the model,
    eliminating version mismatch issues during deployment.
    """
    
    def __init__(self):
        """Initialize the model wrapper."""
        self.predictor = None
        self._features = None
    
    def load_context(self, context):
        """
        Load model artifacts when the model is loaded.
        
        This method is called automatically by MLflow when loading the model
        via mlflow.pyfunc.load_model().
        
        Args:
            context: MLflow PythonModelContext containing artifact paths
        """
        from autogluon.tabular import TabularPredictor
        
        # Load the AutoGluon model from artifacts
        model_path = context.artifacts["autogluon_model"]
        self.predictor = TabularPredictor.load(model_path)
        self._features = self.predictor.feature_metadata_in.get_features()
        
        print(f"[MLflow Model] Loaded AutoGluon model with {len(self._features)} features")
    
    def _prepare_raw_input(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw trade input to the format expected by data_cleaner.
        
        Args:
            model_input: DataFrame with columns matching INPUT_SCHEMA
            
        Returns:
            DataFrame with standardized column names for processing
        """
        now = pd.Timestamp.now()
        
        records = []
        for _, row in model_input.iterrows():
            # Get amount_min first for default amount_max calculation
            amount_min = row.get("amount_min", 15000)
            amount_max = row.get("amount_max") if pd.notna(row.get("amount_max")) else amount_min * 2
            
            # Parse notification_date
            notification_date = row.get("notification_date")
            if notification_date and pd.notna(notification_date):
                notification_date = pd.to_datetime(notification_date)
            else:
                notification_date = now
            
            # Parse transaction_date
            transaction_date = row.get("transaction_date")
            if transaction_date and pd.notna(transaction_date):
                transaction_date = pd.to_datetime(transaction_date)
            else:
                transaction_date = now
            
            record = {
                # Core fields
                "Politician Name": row.get("politician_name", "Unknown"),
                "Ticker": row.get("ticker", ""),
                "Type": row.get("type", "Purchase"),
                "Amount Min": amount_min,
                "Amount Max": amount_max,
                "Filed After": row.get("filed_after", 14),
                "Party": row.get("party") or "Unknown",
                "Chamber": row.get("chamber") or "House",
                "State": row.get("state") or "Unknown",
                "Owner": row.get("owner") or "Self",
                "Asset Name": row.get("ticker", ""),
                "Notes": "",
                
                # Date fields
                "Transaction Date": transaction_date,
                "Notification Date": notification_date,
                "Entry Date": now,
                
                # Price fields
                "Price": row.get("price") if pd.notna(row.get("price")) else 0,
                "Current Price": 0,
                "Entry Price": row.get("price") if pd.notna(row.get("price")) else 0,
                "Disclosure Price": row.get("disclosure_price") if pd.notna(row.get("disclosure_price")) else 0,
                "Buy to Disclosure %": row.get("buy_to_disclosure_pct") if pd.notna(row.get("buy_to_disclosure_pct")) else 0,
                
                # Derived field needed by feature_engineer
                "Ticker_Clean": str(row.get("ticker", "")).split(':')[0].replace('$', '').strip(),
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete prediction pipeline on raw trade data.
        
        This method:
        1. Converts raw input to internal format
        2. Cleans the data (data_cleaner.clean_for_prediction)
        3. Engineers features (feature_engineer.engineer_features_for_prediction)
        4. Runs prediction using AutoGluon
        5. Formats the output with labels and recommendations
        
        Args:
            context: MLflow PythonModelContext (unused after load_context)
            model_input: DataFrame with columns matching INPUT_SCHEMA
            
        Returns:
            DataFrame with columns: prediction, label, recommendation
        """
        # Import here to avoid circular dependencies and ensure code is bundled
        from .data_cleaner import clean_for_prediction
        from .feature_engineer import engineer_features_for_prediction
        
        # Step 1: Prepare raw input
        df = self._prepare_raw_input(model_input)
        
        # Step 2: Clean data
        df = clean_for_prediction(df)
        
        # Step 3: Engineer features
        df = engineer_features_for_prediction(df)
        
        # Step 4: Ensure all required features exist
        for col in self._features or FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0
        
        # Select only the features the model expects
        if self._features:
            feature_df = df[self._features]
        else:
            feature_df = df[[c for c in FEATURE_COLS if c in df.columns]]
        
        # Step 5: Predict
        predictions = self.predictor.predict(feature_df)
        
        # Step 6: Format output
        result = pd.DataFrame({
            "prediction": predictions.astype(int),
            "label": predictions.apply(lambda x: PREDICTION_LABELS.get(int(x), "Unknown")),
            "recommendation": predictions.apply(lambda x: get_recommendation(int(x))),
        })
        
        return result


def log_mlflow_model(
    predictor,
    model_dir: str,
    registered_model_name: Optional[str] = "congress-trading-model"
):
    """
    Log the wrapped model to MLflow with all artifacts and dependencies.
    
    This function is called at the end of training to register the complete
    model pipeline in MLflow.
    
    Args:
        predictor: Trained AutoGluon TabularPredictor
        model_dir: Path to the saved AutoGluon model directory
        registered_model_name: Name for Model Registry (optional)
        
    Returns:
        MLflow run info with model URI
    """
    import os
    from .schemas import MODEL_SIGNATURE
    
    # Create the wrapper instance
    wrapped_model = CongressTradingModel()
    
    # Get the source files to include with the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    code_paths = [
        os.path.join(current_dir, "data_cleaner.py"),
        os.path.join(current_dir, "feature_engineer.py"),
        os.path.join(current_dir, "schemas.py"),
        os.path.join(current_dir, "mlflow_wrapper.py"),
    ]
    
    # Filter to only existing files
    code_paths = [p for p in code_paths if os.path.exists(p)]
    
    # Log the model to MLflow
    model_info = mlflow.pyfunc.log_model(
        artifact_path="congress_trading_model",
        python_model=wrapped_model,
        artifacts={
            "autogluon_model": model_dir,
        },
        signature=MODEL_SIGNATURE,
        pip_requirements=[
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "autogluon.tabular>=1.0.0",
            "lightgbm",
            "catboost",
            "xgboost",
        ],
        code_paths=code_paths,
        registered_model_name=registered_model_name,
    )
    
    print(f"[MLflow] Model logged: {model_info.model_uri}")
    print(f"[MLflow] Registered as: {registered_model_name}")
    
    return model_info


def load_model_for_prediction(model_uri: str = "models/mlflow_model"):
    """
    Load the MLflow-wrapped model for prediction.
    
    This is a convenience function for loading the model in the API.
    
    Args:
        model_uri: Path to the MLflow model or Model Registry URI
                   Examples:
                   - "models/mlflow_model" (local path)
                   - "models:/congress-trading-model/Production" (registry)
                   - "runs:/<run_id>/congress_trading_model" (specific run)
    
    Returns:
        Loaded MLflow PythonModel ready for prediction
    """
    return mlflow.pyfunc.load_model(model_uri)
