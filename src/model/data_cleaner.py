"""
Stage 2: Data Cleaning (Point-in-Time Validation)
DVC Pipeline Step: Validates PIT correctness, filters invalid data.

Usage:
    python data_cleaner.py --input <input_parquet> --output <output_parquet>

Input Artifact:
    data/intermediate/01_raw_trades.parquet
Output Artifact:
    data/intermediate/02_cleaned_trades.parquet
"""
import pandas as pd
import numpy as np
import argparse
import os

# --- Constants ---
DEFAULT_INPUT_PATH = "data/intermediate/01_raw_trades.parquet"
DEFAULT_OUTPUT_PATH = "data/intermediate/02_cleaned_trades.parquet"


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning and strict Point-in-Time (PIT) validation.
    """
    print("[Stage 2] Starting data cleaning...")
    initial_count = len(df)
    cleaned = df.copy()
    
    # 1. Ensure date columns are datetime
    date_cols = ['Transaction Date', 'Notification Date', 'Entry_Date', 'Exit_Date']
    for col in date_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col], errors='coerce')
    
    # 2. Drop rows with missing critical dates
    cleaned = cleaned.dropna(subset=['Transaction Date', 'Notification Date'])
    print(f"[Stage 2] After date validation: {len(cleaned)} rows")
    
    # 3. PIT Validation: Notification Date >= Transaction Date
    mask_valid_pit = cleaned['Notification Date'] >= cleaned['Transaction Date']
    invalid_pit = (~mask_valid_pit).sum()
    if invalid_pit > 0:
        print(f"[Stage 2] WARNING: Dropping {invalid_pit} rows where Notification < Transaction (PIT violation)")
        cleaned = cleaned[mask_valid_pit]
    
    # 4. Calculate Reporting Lag
    cleaned['Reporting_Lag'] = (cleaned['Notification Date'] - cleaned['Transaction Date']).dt.days
    
    # 5. Filter extreme reporting lags (> 60 days is suspicious)
    extreme_lag = cleaned['Reporting_Lag'] > 60
    if extreme_lag.sum() > 0:
        print(f"[Stage 2] Dropping {extreme_lag.sum()} rows with reporting lag > 60 days")
        cleaned = cleaned[~extreme_lag]
    
    # 6. Filter out trades without price data (can't calculate return)
    has_prices = cleaned['Entry_Price'].notna() & cleaned['Exit_Price'].notna()
    no_price_count = (~has_prices).sum()
    if no_price_count > 0:
        print(f"[Stage 2] Dropping {no_price_count} rows without price data")
        cleaned = cleaned[has_prices]
    
    # 7. Filter out trades with SPY data missing
    has_spy = cleaned['SPY_Entry'].notna() & cleaned['SPY_Exit'].notna()
    no_spy = (~has_spy).sum()
    if no_spy > 0:
        print(f"[Stage 2] Dropping {no_spy} rows without SPY data")
        cleaned = cleaned[has_spy]
    
    # 8. Filter extreme returns (> 500% is likely data error)
    extreme_return = cleaned['Stock_Return'].abs() > 5.0
    if extreme_return.sum() > 0:
        print(f"[Stage 2] Dropping {extreme_return.sum()} rows with extreme returns (>500%)")
        cleaned = cleaned[~extreme_return]
    
    # 9. Standardize Ticker column
    if 'Ticker_Clean' not in cleaned.columns:
        cleaned['Ticker_Clean'] = cleaned['Ticker'].astype(str).str.split(':').str[0].str.strip()
    
    # 10. Filter bad tickers
    bad_tickers = ['N/A', 'nan', 'SHEET', '', 'None']
    cleaned = cleaned[~cleaned['Ticker_Clean'].isin(bad_tickers)]
    
    # Summary
    final_count = len(cleaned)
    dropped = initial_count - final_count
    print(f"[Stage 2] Cleaning complete: {final_count} rows remaining ({dropped} dropped, {100*dropped/initial_count:.1f}%)")
    
    return cleaned


# --- DVC Stage Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Clean and Validate Data (PIT)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("[Stage 2] DATA CLEANING (Point-in-Time)")
    print("=" * 60)
    
    # Read input artifact
    df = pd.read_parquet(args.input)
    print(f"[Stage 2] Read {len(df)} rows from {args.input}")
    
    # Process
    cleaned_df = clean_and_validate_data(df)
    
    # Save
    cleaned_df.to_parquet(args.output, index=False)
    print(f"[Stage 2] Output saved to: {args.output}")
    print("[Stage 2] Complete.\n")
