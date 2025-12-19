"""
Stage 3: Feature Engineering
DVC Pipeline Step: Engineers 72 features for congress trading prediction.

NO Alpha-based features used (to avoid data leakage).
Alpha_180 is used ONLY to generate Target labels.

Usage:
    python feature_engineer.py --input <input_parquet> --output <output_parquet>

Features Created (78 total):
    - Basic features (20): Politician, Trade, Time, Price info
    - Amount derived (5): Range, Mid, Ratio, Category, Is_Large
    - Time derived (8): Year-end, Quarter, Day effects
    - Price derived (4): Change, Momentum, Dip/Run
    - Historical behavior (6): Avg amount, unusual size, streak
    - Signal strength (2): Signal score, Cluster
    - Owner derived (2): Self, Family
    - Interaction features (15): Cross features
    - Lag - Politician (5): Frequency, patterns
    - Lag - Ticker (6): Popularity, buy ratio
    - Lag - Market (5): Overall activity
"""
import pandas as pd
import numpy as np
import argparse
import os
from datetime import timedelta

# --- Constants ---
DEFAULT_INPUT_PATH = "data/intermediate/02_cleaned_trades.parquet"
DEFAULT_OUTPUT_PATH = "data/intermediate/03_featured_trades.parquet"

# Amount category thresholds
AMOUNT_THRESHOLDS = [15000, 50000, 250000]  # Small, Medium, Large, Very Large

# Alpha thresholds for multi-class target
ALPHA_THRESHOLD_1 = 0.0    # Beat market
ALPHA_THRESHOLD_2 = 0.10   # Beat by 10%
ALPHA_THRESHOLD_3 = 0.20   # Beat by 20%


# ============================================================
# BASIC FEATURES
# ============================================================

def create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create basic features from raw data."""
    print("[Stage 3] Creating basic features...")
    
    # Ensure datetime
    df['Notification Date'] = pd.to_datetime(df['Notification Date'])
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    
    # --- Politician ID (Label Encoding) ---
    df['Politician_ID'] = df['Politician Name'].astype('category').cat.codes
    
    # --- Party encoding ---
    df['Is_Republican'] = (df['Party'] == 'Republican').astype(int)
    
    # --- Chamber encoding ---
    df['Is_Senate'] = (df['Chamber'].str.lower() == 'senate').astype(int).fillna(0)
    
    # --- State ID (Label Encoding) ---
    df['State_ID'] = df['State'].astype('category').cat.codes
    
    # --- Trade type ---
    df['Is_Purchase'] = (df['Type'].str.lower() == 'purchase').astype(int)
    
    # --- Amount features ---
    df['Amount_Min'] = df['Amount Min'].fillna(0)
    df['Amount_Max'] = df['Amount Max'].fillna(df['Amount_Min'])
    df['Amount_Log'] = np.log1p(df['Amount_Min'])
    
    # --- Owner ID (Label Encoding) ---
    df['Owner'] = df['Owner'].fillna('Undisclosed')
    df['Owner_ID'] = df['Owner'].astype('category').cat.codes
    
    # --- Time features ---
    df['Filed_After'] = df['Filed After'].fillna(0)
    df['Transaction_Month'] = df['Transaction Date'].dt.month
    df['Transaction_DayOfWeek'] = df['Transaction Date'].dt.dayofweek
    df['Transaction_Quarter'] = df['Transaction Date'].dt.quarter
    
    # --- Price features ---
    df['Price'] = df['Price'].fillna(0)
    df['Disclosure_Price'] = df['Disclosure Price'].fillna(df['Price'])
    
    # Buy to Disclosure %
    if 'Buy to Disclosure %' in df.columns:
        df['Buy_to_Disclosure_Pct'] = df['Buy to Disclosure %'].fillna(0)
    else:
        df['Buy_to_Disclosure_Pct'] = 0
    
    return df


# ============================================================
# AMOUNT DERIVED FEATURES
# ============================================================

def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create amount-derived features."""
    print("[Stage 3] Creating amount-derived features...")
    
    df['Amount_Range'] = df['Amount_Max'] - df['Amount_Min']
    df['Amount_Mid'] = (df['Amount_Max'] + df['Amount_Min']) / 2
    df['Amount_Ratio'] = (df['Amount_Max'] / df['Amount_Min']).replace([np.inf, -np.inf], 1).fillna(1)
    
    # Amount category
    df['Amount_Category'] = pd.cut(
        df['Amount_Min'], 
        bins=[-np.inf] + AMOUNT_THRESHOLDS + [np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)
    
    df['Is_Large_Trade'] = (df['Amount_Min'] > 100000).astype(int)
    
    return df


# ============================================================
# TIME DERIVED FEATURES
# ============================================================

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-derived features."""
    print("[Stage 3] Creating time-derived features...")
    
    df['Is_Year_End'] = df['Transaction_Month'].isin([11, 12]).astype(int)
    df['Is_Q1'] = (df['Transaction_Quarter'] == 1).astype(int)
    df['Is_Q4'] = (df['Transaction_Quarter'] == 4).astype(int)
    df['Is_Monday'] = (df['Transaction_DayOfWeek'] == 0).astype(int)
    df['Is_Friday'] = (df['Transaction_DayOfWeek'] == 4).astype(int)
    df['Filed_After_Log'] = np.log1p(df['Filed_After'])
    df['Is_Quick_Disclosure'] = (df['Filed_After'] <= 7).astype(int)
    df['Is_Slow_Disclosure'] = (df['Filed_After'] > 45).astype(int)
    
    return df


# ============================================================
# PRICE DERIVED FEATURES
# ============================================================

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-derived features."""
    print("[Stage 3] Creating price-derived features...")
    
    df['Price_Change_Abs'] = np.abs(df['Disclosure_Price'] - df['Price'])
    df['Price_Momentum'] = (df['Buy_to_Disclosure_Pct'] > 0).astype(int)
    df['Is_Dip_Buy'] = ((df['Is_Purchase'] == 1) & (df['Buy_to_Disclosure_Pct'] < -5)).astype(int)
    df['Is_Run_Up'] = ((df['Is_Purchase'] == 1) & (df['Buy_to_Disclosure_Pct'] > 10)).astype(int)
    
    return df


# ============================================================
# HISTORICAL BEHAVIOR FEATURES
# ============================================================

def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical behavior features (Point-in-Time correct)."""
    print("[Stage 3] Creating historical behavior features...")
    
    df = df.sort_values(['Politician Name', 'Notification Date']).reset_index(drop=True)
    
    # Past trade count
    df['Past_Trade_Count'] = df.groupby('Politician Name').cumcount()
    
    # Politician average amount (expanding mean, shifted to avoid leakage)
    df['Politician_Avg_Amount'] = df.groupby('Politician Name')['Amount_Min'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(df['Amount_Min'].median())
    
    # Current amount vs average
    df['Amount_vs_Avg'] = (df['Amount_Min'] / df['Politician_Avg_Amount']).replace([np.inf, -np.inf], 1).fillna(1)
    df['Is_Unusual_Size'] = (df['Amount_vs_Avg'] > 2).astype(int)
    
    # Days since last trade
    df['Prev_Notification_Date'] = df.groupby('Politician Name')['Notification Date'].shift(1)
    df['Days_Since_Last_Trade'] = (df['Notification Date'] - df['Prev_Notification_Date']).dt.days.fillna(365)
    df = df.drop(columns=['Prev_Notification_Date'])
    
    # Trade streak (consecutive trades in 30 days)
    df['Trade_Streak'] = df.groupby('Politician Name')['Days_Since_Last_Trade'].transform(
        lambda x: (x <= 30).cumsum()
    )
    
    # Is first trade
    df['Is_First_Trade'] = (df['Past_Trade_Count'] == 0).astype(int)
    
    return df


# ============================================================
# CONVICTION AND SIGNAL FEATURES
# ============================================================

def create_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create conviction and signal strength features."""
    print("[Stage 3] Creating signal strength features...")
    
    # Conviction Score = Amount / Rolling Avg Amount
    df['Conviction_Score'] = df['Amount_vs_Avg'].clip(0, 10)
    
    # Signal Strength = Conviction / (Filed_After + 1)
    df['Signal_Strength'] = df['Conviction_Score'] / (df['Filed_After'] + 1)
    
    # Cluster Buy: Same ticker traded by multiple politicians in 7 days
    df['Trade_Date'] = df['Notification Date'].dt.date
    
    # Count trades per ticker in rolling 7 days (simplified: same day)
    co_trade = df.groupby(['Trade_Date', 'Ticker_Clean']).size().reset_index(name='Co_Trading_Count')
    df = df.merge(co_trade, on=['Trade_Date', 'Ticker_Clean'], how='left')
    df['Co_Trading_Count'] = (df['Co_Trading_Count'] - 1).clip(lower=0)
    
    # Cluster buy signal
    df['Cluster_Buy'] = ((df['Co_Trading_Count'] >= 2) & (df['Is_Purchase'] == 1)).astype(int)
    
    df = df.drop(columns=['Trade_Date'])
    
    return df


# ============================================================
# OWNER DERIVED FEATURES
# ============================================================

def create_owner_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create owner-derived features."""
    print("[Stage 3] Creating owner-derived features...")
    
    df['Is_Self_Trade'] = (df['Owner'] == 'Self').astype(int)
    df['Is_Family_Trade'] = df['Owner'].isin(['Spouse', 'Child', 'Joint']).astype(int)
    
    return df


# ============================================================
# INTERACTION FEATURES
# ============================================================

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features."""
    print("[Stage 3] Creating interaction features...")
    
    # Senate interactions
    df['Senate_Large_Trade'] = df['Is_Senate'] * df['Is_Large_Trade']
    df['Senate_Purchase'] = df['Is_Senate'] * df['Is_Purchase']
    df['Senate_Quick_Disclosure'] = df['Is_Senate'] * df['Is_Quick_Disclosure']
    
    # Republican interactions
    df['Republican_Purchase'] = df['Is_Republican'] * df['Is_Purchase']
    df['Republican_Large_Trade'] = df['Is_Republican'] * df['Is_Large_Trade']
    
    # Quick disclosure interactions
    df['Quick_Large_Trade'] = df['Is_Quick_Disclosure'] * df['Is_Large_Trade']
    
    # Self trade interactions
    df['Self_Large_Trade'] = df['Is_Self_Trade'] * df['Is_Large_Trade']
    df['Self_Purchase'] = df['Is_Self_Trade'] * df['Is_Purchase']
    
    # Price pattern interactions
    df['Dip_Buy_Large'] = df['Is_Dip_Buy'] * df['Is_Large_Trade']
    
    # Conviction interactions
    conviction_median = df['Conviction_Score'].median()
    df['High_Conviction_Quick'] = ((df['Conviction_Score'] > conviction_median) & (df['Is_Quick_Disclosure'] == 1)).astype(int)
    
    # Experience interactions
    df['Experienced_Large'] = ((df['Past_Trade_Count'] > 10) & (df['Is_Large_Trade'] == 1)).astype(int)
    
    # Cluster interactions
    df['Cluster_Large'] = ((df['Co_Trading_Count'] > 5) & (df['Is_Large_Trade'] == 1)).astype(int)
    
    # Time-type interactions
    df['YearEnd_Sale'] = df['Is_Year_End'] * (1 - df['Is_Purchase'])
    df['Monday_Purchase'] = df['Is_Monday'] * df['Is_Purchase']
    df['Friday_Sale'] = df['Is_Friday'] * (1 - df['Is_Purchase'])
    
    return df


# ============================================================
# LAG FEATURES - POLITICIAN DIMENSION
# ============================================================

def create_politician_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features based on politician's history (no Alpha)."""
    print("[Stage 3] Creating politician lag features...")
    
    df = df.sort_values(['Politician Name', 'Notification Date']).reset_index(drop=True)
    
    # Last trade same type
    df['Politician_Last_Trade_Type'] = df.groupby('Politician Name')['Is_Purchase'].shift(1)
    df['Politician_Last_Trade_Same_Type'] = (df['Is_Purchase'] == df['Politician_Last_Trade_Type']).astype(int)
    df = df.drop(columns=['Politician_Last_Trade_Type'])
    
    # Trade frequency (count-based, no Alpha)
    # 30-day frequency
    df['Notification_Date_Num'] = (df['Notification Date'] - df['Notification Date'].min()).dt.days
    
    def count_trades_in_window(group, days):
        result = []
        dates = group['Notification Date'].values
        for i, current_date in enumerate(dates):
            window_start = current_date - np.timedelta64(days, 'D')
            count = ((dates[:i] >= window_start) & (dates[:i] < current_date)).sum()
            result.append(count)
        return pd.Series(result, index=group.index)
    
    df['Politician_Trade_Frequency_30D'] = df.groupby('Politician Name', group_keys=False).apply(
        lambda x: count_trades_in_window(x, 30)
    ).fillna(0)
    
    df['Politician_Trade_Frequency_90D'] = df.groupby('Politician Name', group_keys=False).apply(
        lambda x: count_trades_in_window(x, 90)
    ).fillna(0)
    
    # Politician average filed_after (disclosure speed tendency)
    df['Politician_Avg_Filed_After'] = df.groupby('Politician Name')['Filed_After'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(df['Filed_After'].median())
    
    # Politician purchase ratio
    df['Politician_Purchase_Ratio'] = df.groupby('Politician Name')['Is_Purchase'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(0.5)
    
    df = df.drop(columns=['Notification_Date_Num'])
    
    return df


# ============================================================
# LAG FEATURES - TICKER DIMENSION
# ============================================================

def create_ticker_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features based on ticker history (no Alpha)."""
    print("[Stage 3] Creating ticker lag features...")
    
    df = df.sort_values('Notification Date').reset_index(drop=True)
    
    def count_ticker_trades_in_window(group, df_full, days):
        result = []
        for idx, row in group.iterrows():
            ticker = row['Ticker_Clean']
            current_date = row['Notification Date']
            window_start = current_date - timedelta(days=days)
            
            mask = (
                (df_full['Ticker_Clean'] == ticker) & 
                (df_full['Notification Date'] >= window_start) & 
                (df_full['Notification Date'] < current_date)
            )
            result.append(mask.sum())
        return pd.Series(result, index=group.index)
    
    # Pre-compute for efficiency
    print("[Stage 3]   Computing ticker trade counts...")
    ticker_date_counts = df.groupby(['Ticker_Clean', pd.Grouper(key='Notification Date', freq='D')]).size().reset_index(name='daily_count')
    
    # Simpler approach: use merge with date ranges
    df['Ticker_Trade_Count_30D'] = 0
    df['Ticker_Trade_Count_90D'] = 0
    df['Ticker_Buy_Ratio_30D'] = 0.5
    df['Ticker_Politician_Count_30D'] = 0
    df['Ticker_Avg_Amount_30D'] = df['Amount_Min'].median()
    
    # Compute per ticker (simplified for performance)
    for ticker in df['Ticker_Clean'].unique():
        ticker_mask = df['Ticker_Clean'] == ticker
        ticker_df = df[ticker_mask].copy()
        
        for i, (idx, row) in enumerate(ticker_df.iterrows()):
            current_date = row['Notification Date']
            
            # Past 30 days
            past_30 = ticker_df[
                (ticker_df['Notification Date'] >= current_date - timedelta(days=30)) &
                (ticker_df['Notification Date'] < current_date)
            ]
            
            df.loc[idx, 'Ticker_Trade_Count_30D'] = len(past_30)
            if len(past_30) > 0:
                df.loc[idx, 'Ticker_Buy_Ratio_30D'] = past_30['Is_Purchase'].mean()
                df.loc[idx, 'Ticker_Politician_Count_30D'] = past_30['Politician Name'].nunique()
                df.loc[idx, 'Ticker_Avg_Amount_30D'] = past_30['Amount_Min'].mean()
            
            # Past 90 days
            past_90 = ticker_df[
                (ticker_df['Notification Date'] >= current_date - timedelta(days=90)) &
                (ticker_df['Notification Date'] < current_date)
            ]
            df.loc[idx, 'Ticker_Trade_Count_90D'] = len(past_90)
    
    df['Is_Ticker_Hot'] = (df['Ticker_Trade_Count_30D'] > 5).astype(int)
    
    return df


# ============================================================
# LAG FEATURES - MARKET DIMENSION
# ============================================================

def create_market_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features based on overall market activity (no Alpha)."""
    print("[Stage 3] Creating market lag features...")
    
    df = df.sort_values('Notification Date').reset_index(drop=True)
    
    # Pre-compute daily aggregates
    daily_stats = df.groupby(df['Notification Date'].dt.date).agg({
        'Is_Purchase': 'mean',
        'Amount_Min': 'mean',
        'Politician Name': 'count'
    }).reset_index()
    daily_stats.columns = ['Date', 'Daily_Buy_Ratio', 'Daily_Avg_Amount', 'Daily_Trade_Count']
    
    # Rolling 7-day and 30-day
    daily_stats['Market_Buy_Ratio_7D'] = daily_stats['Daily_Buy_Ratio'].rolling(7, min_periods=1).mean().shift(1)
    daily_stats['Market_Buy_Ratio_30D'] = daily_stats['Daily_Buy_Ratio'].rolling(30, min_periods=1).mean().shift(1)
    daily_stats['Market_Trade_Volume_7D'] = daily_stats['Daily_Trade_Count'].rolling(7, min_periods=1).sum().shift(1)
    daily_stats['Market_Avg_Amount_7D'] = daily_stats['Daily_Avg_Amount'].rolling(7, min_periods=1).mean().shift(1)
    
    # Merge back
    df['Trade_Date_Only'] = df['Notification Date'].dt.date
    df = df.merge(
        daily_stats[['Date', 'Market_Buy_Ratio_7D', 'Market_Buy_Ratio_30D', 'Market_Trade_Volume_7D', 'Market_Avg_Amount_7D']],
        left_on='Trade_Date_Only',
        right_on='Date',
        how='left'
    )
    df = df.drop(columns=['Date', 'Trade_Date_Only'])
    
    # Fill NaN
    df['Market_Buy_Ratio_7D'] = df['Market_Buy_Ratio_7D'].fillna(0.5)
    df['Market_Buy_Ratio_30D'] = df['Market_Buy_Ratio_30D'].fillna(0.5)
    df['Market_Trade_Volume_7D'] = df['Market_Trade_Volume_7D'].fillna(0)
    df['Market_Avg_Amount_7D'] = df['Market_Avg_Amount_7D'].fillna(df['Amount_Min'].median())
    
    # High activity period
    volume_median = df['Market_Trade_Volume_7D'].median()
    df['Is_High_Activity_Period'] = (df['Market_Trade_Volume_7D'] > volume_median).astype(int)
    
    return df


# ============================================================
# TARGET CREATION (Based on Alpha_180)
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create multi-class target based on Alpha_180."""
    print("[Stage 3] Creating multi-class target from Alpha_180...")
    
    # Calculate Alpha_180 if not exists
    if 'Alpha_180' not in df.columns:
        if 'Stock_Return_180' in df.columns and 'SPY_Return_180' in df.columns:
            df['Alpha_180'] = df['Stock_Return_180'] - df['SPY_Return_180']
        else:
            print("[Stage 3] WARNING: Cannot calculate Alpha_180, using Alpha (60-day)")
            df['Alpha_180'] = df.get('Alpha', 0)
    
    # Multi-class target
    df['Target'] = 0
    df.loc[df['Alpha_180'] > ALPHA_THRESHOLD_1, 'Target'] = 1
    df.loc[df['Alpha_180'] > ALPHA_THRESHOLD_2, 'Target'] = 2
    df.loc[df['Alpha_180'] > ALPHA_THRESHOLD_3, 'Target'] = 3
    
    target_dist = df['Target'].value_counts().sort_index()
    print(f"[Stage 3] Target distribution:")
    print(f"  Class 0 (Alpha <= 0):       {target_dist.get(0, 0)}")
    print(f"  Class 1 (0 < Alpha <= 10%): {target_dist.get(1, 0)}")
    print(f"  Class 2 (10% < Alpha <= 20%): {target_dist.get(2, 0)}")
    print(f"  Class 3 (Alpha > 20%):      {target_dist.get(3, 0)}")
    
    return df


# ============================================================
# MASTER FUNCTIONS
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: Apply all feature engineering steps.
    Used for TRAINING (includes Target creation).
    """
    df = create_basic_features(df)
    df = create_amount_features(df)
    df = create_time_features(df)
    df = create_price_features(df)
    df = create_historical_features(df)
    df = create_signal_features(df)
    df = create_owner_features(df)
    df = create_interaction_features(df)
    df = create_politician_lag_features(df)
    df = create_ticker_lag_features(df)
    df = create_market_lag_features(df)
    df = create_target(df)
    
    return df


def engineer_features_for_prediction(df: pd.DataFrame, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Apply feature engineering for PREDICTION (no Target creation).
    
    Args:
        df: New trades to predict (cleaned data)
        historical_df: Historical training data for lag features (optional)
        
    Returns:
        DataFrame with all features ready for model prediction
    """
    print("[Features] Engineering features for prediction...")
    
    # If historical data provided, combine for proper lag feature calculation
    if historical_df is not None:
        # Mark rows
        df['_is_new'] = True
        historical_df['_is_new'] = False
        
        # Combine
        combined = pd.concat([historical_df, df], ignore_index=True)
        combined = combined.sort_values('Notification Date').reset_index(drop=True)
    else:
        combined = df.copy()
        combined['_is_new'] = True
    
    # Apply feature engineering (without Target)
    combined = create_basic_features(combined)
    combined = create_amount_features(combined)
    combined = create_time_features(combined)
    combined = create_price_features(combined)
    combined = create_historical_features(combined)
    combined = create_signal_features(combined)
    combined = create_owner_features(combined)
    combined = create_interaction_features(combined)
    combined = create_politician_lag_features(combined)
    combined = create_ticker_lag_features(combined)
    combined = create_market_lag_features(combined)
    
    # Extract only new rows
    result = combined[combined['_is_new'] == True].drop(columns=['_is_new'])
    
    print(f"[Features] Created {len(result)} rows with features")
    return result


# ============================================================
# DVC STAGE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Feature Engineering (72 features)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("[Stage 3] FEATURE ENGINEERING (72 features, no Alpha leakage)")
    print("=" * 60)
    
    # Read input artifact
    df = pd.read_parquet(args.input)
    print(f"[Stage 3] Read {len(df)} rows from {args.input}")
    
    # Process
    featured_df = engineer_features(df)
    
    # Summary of features
    feature_cols = [
        # Basic (20)
        'Politician_ID', 'Is_Republican', 'Is_Senate', 'State_ID',
        'Is_Purchase', 'Amount_Min', 'Amount_Max', 'Amount_Log', 'Owner_ID',
        'Filed_After', 'Transaction_Month', 'Transaction_DayOfWeek', 'Transaction_Quarter',
        'Price', 'Disclosure_Price', 'Buy_to_Disclosure_Pct',
        'Past_Trade_Count', 'Conviction_Score', 'Co_Trading_Count',
        # Amount derived (5)
        'Amount_Range', 'Amount_Mid', 'Amount_Ratio', 'Amount_Category', 'Is_Large_Trade',
        # Time derived (8)
        'Is_Year_End', 'Is_Q1', 'Is_Q4', 'Is_Monday', 'Is_Friday',
        'Filed_After_Log', 'Is_Quick_Disclosure', 'Is_Slow_Disclosure',
        # Price derived (4)
        'Price_Change_Abs', 'Price_Momentum', 'Is_Dip_Buy', 'Is_Run_Up',
        # Historical (6)
        'Politician_Avg_Amount', 'Amount_vs_Avg', 'Is_Unusual_Size',
        'Days_Since_Last_Trade', 'Trade_Streak', 'Is_First_Trade',
        # Signal (2)
        'Signal_Strength', 'Cluster_Buy',
        # Owner (2)
        'Is_Self_Trade', 'Is_Family_Trade',
        # Interaction (15)
        'Senate_Large_Trade', 'Senate_Purchase', 'Senate_Quick_Disclosure',
        'Republican_Purchase', 'Republican_Large_Trade', 'Quick_Large_Trade',
        'Self_Large_Trade', 'Self_Purchase', 'Dip_Buy_Large',
        'High_Conviction_Quick', 'Experienced_Large', 'Cluster_Large',
        'YearEnd_Sale', 'Monday_Purchase', 'Friday_Sale',
        # Lag - Politician (5)
        'Politician_Last_Trade_Same_Type', 'Politician_Trade_Frequency_30D',
        'Politician_Trade_Frequency_90D', 'Politician_Avg_Filed_After', 'Politician_Purchase_Ratio',
        # Lag - Ticker (6)
        'Ticker_Trade_Count_30D', 'Ticker_Trade_Count_90D', 'Ticker_Buy_Ratio_30D',
        'Ticker_Politician_Count_30D', 'Ticker_Avg_Amount_30D', 'Is_Ticker_Hot',
        # Lag - Market (5)
        'Market_Buy_Ratio_7D', 'Market_Buy_Ratio_30D', 'Market_Trade_Volume_7D',
        'Market_Avg_Amount_7D', 'Is_High_Activity_Period',
    ]
    
    existing_features = [c for c in feature_cols if c in featured_df.columns]
    missing_features = [c for c in feature_cols if c not in featured_df.columns]
    
    print(f"\n[Stage 3] Features created: {len(existing_features)}/{len(feature_cols)}")
    if missing_features:
        print(f"[Stage 3] Missing features: {missing_features}")
    
    # Save
    featured_df.to_parquet(args.output, index=False)
    print(f"\n[Stage 3] Output saved to: {args.output}")
    print(f"[Stage 3] Final shape: {featured_df.shape}")
    print("[Stage 3] Complete.\n")
