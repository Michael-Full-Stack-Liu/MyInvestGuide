"""
Stage 1: Data Loading
DVC Pipeline Step: Reads raw CSV with pre-calculated price data, outputs combined data.
Usage:
    python data_loader.py --data-dir <path> --output <output_path>

Output Artifact:
    data/intermediate/01_raw_trades.parquet
"""
import pandas as pd
import glob
import os
import argparse
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
DEFAULT_DATA_DIR = "."
DEFAULT_OUTPUT_PATH = "data/intermediate/01_raw_trades.parquet"
FORWARD_DAYS = 60  # Default forward period for return calculation


def load_latest_congress_data(data_dir: str = ".") -> pd.DataFrame:
    """Finds and loads the most recent congress_trading CSV file."""
    pattern = os.path.join(data_dir, "congress_trading_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No congress trading files found matching '{pattern}'")
    
    latest_file = sorted(files)[-1]
    print(f"[Stage 1] Loading data from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    return df


def clean_ticker(ticker: str) -> str:
    """Clean ticker symbol by removing :US suffix and special characters."""
    if pd.isna(ticker):
        return ""
    clean = str(ticker).split(':')[0].replace('$', '').strip()
    if clean in ['N/A', 'nan', 'SHEET', '']:
        return ""
    return clean


# --- DVC Stage Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Load Raw Trade Data with Pre-calculated Prices")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 60)
    print("[Stage 1] DATA LOADING (Pre-calculated Prices)")
    print("=" * 60)
    
    # 1. Load raw trades with pre-calculated prices
    df = load_latest_congress_data(args.data_dir)
    print(f"[Stage 1] Loaded {len(df)} trades")
    
    # 2. Parse dates
    df['Notification Date'] = pd.to_datetime(df['Notification Date'], errors='coerce')
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
    df = df.dropna(subset=['Notification Date', 'Transaction Date'])
    print(f"[Stage 1] After date cleaning: {len(df)} trades")
    
    # 3. Clean ticker column for matching
    df['Ticker_Clean'] = df['Ticker'].apply(clean_ticker)
    
    # 4. Ensure Entry_Date and Exit_Date columns exist
    # If they exist in the CSV, use them; otherwise calculate from Notification Date
    if 'Entry Date' in df.columns:
        df['Entry_Date'] = pd.to_datetime(df['Entry Date'], errors='coerce')
    else:
        df['Entry_Date'] = df['Notification Date'] + timedelta(days=1)
    
    # Map CSV column names to expected internal names
    # The CSV uses spaces, we use underscores
    column_mapping = {
        'Entry Price': 'Entry_Price',
        'Exit Price 60': 'Exit_Price_60',
        'Exit Price 180': 'Exit_Price_180',
        'SPY Entry': 'SPY_Entry',
        'SPY Exit 60': 'SPY_Exit_60', 
        'SPY Exit 180': 'SPY_Exit_180',
        'Disclosure Price': 'Disclosure_Price',
        'Exit Date 60': 'Exit_Date_60',
        'Exit Date 180': 'Exit_Date_180',
        'Buy to Disclosure %': 'Buy_to_Disclosure_Pct',
        'Buy to Sell %': 'Buy_to_Sell_Pct',
        'Filed After': 'Filed_After',
        'Amount Min': 'Amount_Min',
        'Amount Max': 'Amount_Max',
        'Current Price': 'Current_Price',
        'Politician Name': 'Politician_Name',
        'Asset Name': 'Asset_Name'
    }
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    # 5. Set default exit date for backward compatibility
    df['Exit_Date'] = df['Entry_Date'] + timedelta(days=FORWARD_DAYS)
    
    # Use Exit_Price_60 as the default Exit_Price (60-day return)
    if 'Exit_Price_60' in df.columns:
        df['Exit_Price'] = df['Exit_Price_60']
    elif 'Exit Price 60' in df.columns:
        df['Exit_Price'] = df['Exit Price 60']
    else:
        df['Exit_Price'] = None
    
    # Use SPY_Exit_60 as the default SPY_Exit
    if 'SPY_Exit_60' in df.columns:
        df['SPY_Exit'] = df['SPY_Exit_60']
    elif 'SPY Exit 60' in df.columns:
        df['SPY_Exit'] = df['SPY Exit 60']
    else:
        df['SPY_Exit'] = None
    
    # 6. Calculate returns using pre-fetched prices
    # Stock return
    df['Stock_Return'] = (df['Exit_Price'] - df['Entry_Price']) / df['Entry_Price']
    
    # SPY return (benchmark)
    df['SPY_Return'] = (df['SPY_Exit'] - df['SPY_Entry']) / df['SPY_Entry']
    
    # Alpha (excess return over market)
    df['Alpha'] = df['Stock_Return'] - df['SPY_Return']
    
    # 7. Calculate 180-day returns as well
    if 'Exit_Price_180' in df.columns and 'SPY_Exit_180' in df.columns:
        df['Stock_Return_180'] = (df['Exit_Price_180'] - df['Entry_Price']) / df['Entry_Price']
        df['SPY_Return_180'] = (df['SPY_Exit_180'] - df['SPY_Entry']) / df['SPY_Entry']
        df['Alpha_180'] = df['Stock_Return_180'] - df['SPY_Return_180']
    
    # 8. Save
    df.to_parquet(args.output, index=False)
    
    # Summary
    valid_returns = df['Stock_Return'].notna().sum()
    valid_spy = df['SPY_Return'].notna().sum()
    valid_alpha = df['Alpha'].notna().sum()
    
    print(f"\n[Stage 1] Summary:")
    print(f"  - Total trades: {len(df)}")
    print(f"  - Trades with valid stock return: {valid_returns} ({100*valid_returns/len(df):.1f}%)")
    print(f"  - Trades with valid SPY return: {valid_spy} ({100*valid_spy/len(df):.1f}%)")
    print(f"  - Trades with valid Alpha: {valid_alpha} ({100*valid_alpha/len(df):.1f}%)")
    
    # Show some statistics
    if valid_returns > 0:
        print(f"\n[Stage 1] Return Statistics (60-day):")
        print(f"  - Mean Stock Return: {df['Stock_Return'].mean()*100:.2f}%")
        print(f"  - Mean SPY Return: {df['SPY_Return'].mean()*100:.2f}%")
        print(f"  - Mean Alpha: {df['Alpha'].mean()*100:.2f}%")
    
    print(f"\n[Stage 1] Output saved to: {args.output}")
    print("[Stage 1] Complete.\n")
