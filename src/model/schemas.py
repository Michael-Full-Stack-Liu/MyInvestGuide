"""
MLflow Input/Output Schemas for Congress Trading Model

Defines the expected input and output schema for the MLflow-wrapped model.
This ensures type safety and documentation for API consumers.
"""

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


# =============================================================================
# Input Schema - Raw Trade Data
# =============================================================================
# These are the fields expected by the model's predict() method.
# The model internally handles data cleaning and feature engineering.

INPUT_SCHEMA = Schema([
    # === Required Fields ===
    ColSpec("string", "politician_name"),      # Name of the politician
    ColSpec("string", "ticker"),               # Stock ticker symbol (used for lag features)
    ColSpec("string", "type"),                 # "Purchase" or "Sale"
    ColSpec("double", "amount_min"),           # Minimum transaction amount
    ColSpec("integer", "filed_after"),         # Days between transaction and disclosure
    
    # === Important Fields (needed for full feature set) ===
    ColSpec("double", "amount_max"),           # Maximum transaction amount (for Amount_Range, Amount_Mid, Amount_Ratio)
    ColSpec("string", "owner"),                # Owner: "Self", "Spouse", "Child", "Joint", etc. (for Owner_ID, Is_Self_Trade, Is_Family_Trade)
    ColSpec("string", "state"),                # State code (for State_ID)
    ColSpec("string", "notification_date"),    # Disclosure/notification date ISO string (critical for lag features)
    
    # === Optional Fields (can use defaults if missing) ===
    ColSpec("string", "party"),                # Political party: "Republican" or "Democrat"
    ColSpec("string", "chamber"),              # "House" or "Senate"
    ColSpec("string", "transaction_date"),     # Transaction date ISO string
    ColSpec("double", "price"),                # Price at transaction (optional)
    ColSpec("double", "disclosure_price"),     # Price at disclosure (optional)
    ColSpec("double", "buy_to_disclosure_pct"), # Price change % from buy to disclosure (optional)
])


# =============================================================================
# Output Schema - Prediction Results
# =============================================================================
# The model returns a DataFrame with these columns.

OUTPUT_SCHEMA = Schema([
    ColSpec("integer", "prediction"),          # 0, 1, 2, or 3 (conviction class)
    ColSpec("string", "label"),                # "Weak", "Moderate", "Strong", "Very Strong"
    ColSpec("string", "recommendation"),       # "FOLLOW" or "SKIP"
])


# =============================================================================
# Model Signature
# =============================================================================
# Combined input/output signature for MLflow model registration.

MODEL_SIGNATURE = ModelSignature(
    inputs=INPUT_SCHEMA, 
    outputs=OUTPUT_SCHEMA
)


# =============================================================================
# Label Mappings
# =============================================================================

PREDICTION_LABELS = {
    0: "Weak",
    1: "Moderate",
    2: "Strong",
    3: "Very Strong"
}


def get_recommendation(prediction: int) -> str:
    """Get follow/skip recommendation based on prediction class."""
    return "FOLLOW" if prediction >= 2 else "SKIP"


# =============================================================================
# Feature Columns (72 features used by the model)
# =============================================================================

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
