"""
Prediction Pipeline for Congress Trading
Processes new trade data and generates predictions using trained AutoGluon model.

Handles:
- Data cleaning
- Feature engineering
- Missing value imputation
- Model prediction

Usage:
    python predict.py --input <new_trades.csv> --output <predictions.csv>
    python predict.py --input data/new_trades.csv --output data/predictions.csv
"""
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
from autogluon.tabular import TabularPredictor
import warnings
warnings.filterwarnings('ignore')

# --- Constants ---
DEFAULT_MODEL_PATH = "models/autogluon"
DEFAULT_OUTPUT_PATH = "data/predictions.csv"

# Feature columns expected by model
FEATURE_COLS = [
    'Politician_ID', 'Is_Republican', 'Is_Senate', 'State_ID',
    'Is_Purchase', 'Amount_Min', 'Amount_Max', 'Amount_Log', 'Owner_ID',
    'Filed_After', 'Transaction_Month', 'Transaction_DayOfWeek', 'Transaction_Quarter',
    'Price', 'Disclosure_Price', 'Buy_to_Disclosure_Pct',
    'Past_Trade_Count', 'Conviction_Score', 'Co_Trading_Count',
    'Amount_Range', 'Amount_Mid', 'Amount_Ratio', 'Amount_Category', 'Is_Large_Trade',
    'Is_Year_End', 'Is_Q1', 'Is_Q4', 'Is_Monday', 'Is_Friday',
    'Filed_After_Log', 'Is_Quick_Disclosure', 'Is_Slow_Disclosure',
    'Price_Change_Abs', 'Price_Momentum', 'Is_Dip_Buy', 'Is_Run_Up',
    'Politician_Avg_Amount', 'Amount_vs_Avg', 'Is_Unusual_Size',
    'Days_Since_Last_Trade', 'Trade_Streak', 'Is_First_Trade',
    'Signal_Strength', 'Cluster_Buy',
    'Is_Self_Trade', 'Is_Family_Trade',
    'Senate_Large_Trade', 'Senate_Purchase', 'Senate_Quick_Disclosure',
    'Republican_Purchase', 'Republican_Large_Trade', 'Quick_Large_Trade',
    'Self_Large_Trade', 'Self_Purchase', 'Dip_Buy_Large',
    'High_Conviction_Quick', 'Experienced_Large', 'Cluster_Large',
    'YearEnd_Sale', 'Monday_Purchase', 'Friday_Sale',
    'Politician_Last_Trade_Same_Type', 'Politician_Trade_Frequency_30D',
    'Politician_Trade_Frequency_90D', 'Politician_Avg_Filed_After', 'Politician_Purchase_Ratio',
    'Ticker_Trade_Count_30D', 'Ticker_Trade_Count_90D', 'Ticker_Buy_Ratio_30D',
    'Ticker_Politician_Count_30D', 'Ticker_Avg_Amount_30D', 'Is_Ticker_Hot',
    'Market_Buy_Ratio_7D', 'Market_Buy_Ratio_30D', 'Market_Trade_Volume_7D',
    'Market_Avg_Amount_7D', 'Is_High_Activity_Period',
]

# Default values for missing data
DEFAULT_VALUES = {
    'Amount_Min': 15000,           # Median value
    'Amount_Max': 50000,
    'Filed_After': 14,             # Median filing delay
    'Price': 100,                  # Placeholder
    'Disclosure_Price': 100,
    'Buy_to_Disclosure_Pct': 0,
    'Past_Trade_Count': 0,
    'Conviction_Score': 1.0,
    'Co_Trading_Count': 0,
    'Days_Since_Last_Trade': 365,
    'Trade_Streak': 0,
    'Politician_Avg_Amount': 50000,
    'Politician_Avg_Filed_After': 14,
    'Politician_Purchase_Ratio': 0.5,
    'Ticker_Trade_Count_30D': 0,
    'Ticker_Trade_Count_90D': 0,
    'Ticker_Buy_Ratio_30D': 0.5,
    'Ticker_Politician_Count_30D': 0,
    'Ticker_Avg_Amount_30D': 50000,
    'Market_Buy_Ratio_7D': 0.5,
    'Market_Buy_Ratio_30D': 0.5,
    'Market_Trade_Volume_7D': 50,
    'Market_Avg_Amount_7D': 50000,
}

# Label encodings from training (you may need to update these based on your training data)
# These should match the training data encodings
PARTY_MAP = {'Democrat': 0, 'Republican': 1}
CHAMBER_MAP = {'house': 0, 'senate': 1}
OWNER_MAP = {'Child': 0, 'Joint': 1, 'Self': 2, 'Spouse': 3, 'Undisclosed': 4}

# Amount category thresholds
AMOUNT_THRESHOLDS = [15000, 50000, 250000]


class PredictionPipeline:
    """Complete pipeline for predicting congress trading outcomes."""
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """Initialize the prediction pipeline."""
        print(f"[Predict] Loading model from {model_path}...")
        self.predictor = TabularPredictor.load(model_path)
        self.model_features = self.predictor.features()
        print(f"[Predict] Model loaded. Expects {len(self.model_features)} features.")
        
        # Load historical data for context-aware features
        self.historical_data = None
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical data for computing lag features."""
        historical_path = "data/intermediate/03_featured_trades.parquet"
        if os.path.exists(historical_path):
            self.historical_data = pd.read_parquet(historical_path)
            print(f"[Predict] Loaded {len(self.historical_data)} historical trades for context.")
        else:
            print("[Predict] WARNING: No historical data found. Lag features will use defaults.")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 1: Clean raw input data."""
        print(f"[Predict] Cleaning {len(df)} rows...")
        
        # Parse dates
        if 'Notification Date' in df.columns:
            df['Notification Date'] = pd.to_datetime(df['Notification Date'], errors='coerce')
        if 'Transaction Date' in df.columns:
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
        
        # Clean ticker
        if 'Ticker' in df.columns:
            df['Ticker_Clean'] = df['Ticker'].astype(str).str.split(':').str[0].str.replace('$', '').str.strip()
        
        return df
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Create basic features."""
        print("[Predict] Creating basic features...")
        
        # --- Politician ID (Label Encoding) ---
        # For new politicians, assign a new ID
        if 'Politician Name' in df.columns:
            if self.historical_data is not None and 'Politician Name' in self.historical_data.columns:
                # Get existing politician mapping
                known_politicians = self.historical_data['Politician Name'].unique()
                politician_to_id = {name: i for i, name in enumerate(sorted(known_politicians))}
                max_id = len(politician_to_id)
                
                # Map known politicians, assign new IDs to unknown
                df['Politician_ID'] = df['Politician Name'].apply(
                    lambda x: politician_to_id.get(x, max_id + hash(str(x)) % 1000)
                )
            else:
                df['Politician_ID'] = df['Politician Name'].astype('category').cat.codes
        else:
            df['Politician_ID'] = 0
        
        # --- Party encoding ---
        df['Is_Republican'] = df.get('Party', 'Unknown').map(
            lambda x: 1 if x == 'Republican' else 0
        )
        
        # --- Chamber encoding ---
        df['Is_Senate'] = df.get('Chamber', 'house').apply(
            lambda x: 1 if str(x).lower() == 'senate' else 0
        )
        
        # --- State ID ---
        if 'State' in df.columns:
            df['State_ID'] = df['State'].astype('category').cat.codes
        else:
            df['State_ID'] = 0
        
        # --- Trade type ---
        if 'Type' in df.columns:
            df['Is_Purchase'] = (df['Type'].str.lower() == 'purchase').astype(int)
        else:
            df['Is_Purchase'] = 1  # Default to purchase
        
        # --- Amount features with default handling ---
        df['Amount_Min'] = pd.to_numeric(df.get('Amount Min', DEFAULT_VALUES['Amount_Min']), errors='coerce')
        df['Amount_Min'] = df['Amount_Min'].fillna(DEFAULT_VALUES['Amount_Min'])
        
        df['Amount_Max'] = pd.to_numeric(df.get('Amount Max', df['Amount_Min']), errors='coerce')
        df['Amount_Max'] = df['Amount_Max'].fillna(df['Amount_Min'])
        
        df['Amount_Log'] = np.log1p(df['Amount_Min'])
        
        # --- Owner ID ---
        if 'Owner' in df.columns:
            df['Owner'] = df['Owner'].fillna('Undisclosed')
            df['Owner_ID'] = df['Owner'].map(OWNER_MAP).fillna(4).astype(int)
        else:
            df['Owner_ID'] = 4  # Undisclosed
        
        # --- Time features ---
        df['Filed_After'] = pd.to_numeric(df.get('Filed After', DEFAULT_VALUES['Filed_After']), errors='coerce')
        df['Filed_After'] = df['Filed_After'].fillna(DEFAULT_VALUES['Filed_After'])
        
        if 'Transaction Date' in df.columns:
            df['Transaction_Month'] = df['Transaction Date'].dt.month.fillna(6)
            df['Transaction_DayOfWeek'] = df['Transaction Date'].dt.dayofweek.fillna(2)
            df['Transaction_Quarter'] = df['Transaction Date'].dt.quarter.fillna(2)
        else:
            df['Transaction_Month'] = 6
            df['Transaction_DayOfWeek'] = 2
            df['Transaction_Quarter'] = 2
        
        # --- Price features ---
        df['Price'] = pd.to_numeric(df.get('Price', DEFAULT_VALUES['Price']), errors='coerce')
        df['Price'] = df['Price'].fillna(DEFAULT_VALUES['Price'])
        
        df['Disclosure_Price'] = pd.to_numeric(
            df.get('Disclosure Price', df['Price']), errors='coerce'
        )
        df['Disclosure_Price'] = df['Disclosure_Price'].fillna(df['Price'])
        
        if 'Buy to Disclosure %' in df.columns:
            df['Buy_to_Disclosure_Pct'] = pd.to_numeric(
                df['Buy to Disclosure %'], errors='coerce'
            ).fillna(0)
        else:
            df['Buy_to_Disclosure_Pct'] = 0
        
        return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 3: Create derived features."""
        print("[Predict] Creating derived features...")
        
        # Amount derived
        df['Amount_Range'] = df['Amount_Max'] - df['Amount_Min']
        df['Amount_Mid'] = (df['Amount_Max'] + df['Amount_Min']) / 2
        df['Amount_Ratio'] = (df['Amount_Max'] / df['Amount_Min']).replace([np.inf, -np.inf], 1).fillna(1)
        df['Amount_Category'] = pd.cut(
            df['Amount_Min'], 
            bins=[-np.inf] + AMOUNT_THRESHOLDS + [np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)
        df['Is_Large_Trade'] = (df['Amount_Min'] > 100000).astype(int)
        
        # Time derived
        df['Is_Year_End'] = df['Transaction_Month'].isin([11, 12]).astype(int)
        df['Is_Q1'] = (df['Transaction_Quarter'] == 1).astype(int)
        df['Is_Q4'] = (df['Transaction_Quarter'] == 4).astype(int)
        df['Is_Monday'] = (df['Transaction_DayOfWeek'] == 0).astype(int)
        df['Is_Friday'] = (df['Transaction_DayOfWeek'] == 4).astype(int)
        df['Filed_After_Log'] = np.log1p(df['Filed_After'])
        df['Is_Quick_Disclosure'] = (df['Filed_After'] <= 7).astype(int)
        df['Is_Slow_Disclosure'] = (df['Filed_After'] > 45).astype(int)
        
        # Price derived
        df['Price_Change_Abs'] = np.abs(df['Disclosure_Price'] - df['Price'])
        df['Price_Momentum'] = (df['Buy_to_Disclosure_Pct'] > 0).astype(int)
        df['Is_Dip_Buy'] = ((df['Is_Purchase'] == 1) & (df['Buy_to_Disclosure_Pct'] < -5)).astype(int)
        df['Is_Run_Up'] = ((df['Is_Purchase'] == 1) & (df['Buy_to_Disclosure_Pct'] > 10)).astype(int)
        
        # Owner derived
        df['Is_Self_Trade'] = (df['Owner_ID'] == 2).astype(int)
        df['Is_Family_Trade'] = df['Owner_ID'].isin([0, 1, 3]).astype(int)
        
        return df
    
    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 4: Create features based on historical data."""
        print("[Predict] Creating historical features...")
        
        # Initialize with defaults
        df['Past_Trade_Count'] = DEFAULT_VALUES['Past_Trade_Count']
        df['Conviction_Score'] = DEFAULT_VALUES['Conviction_Score']
        df['Co_Trading_Count'] = DEFAULT_VALUES['Co_Trading_Count']
        df['Days_Since_Last_Trade'] = DEFAULT_VALUES['Days_Since_Last_Trade']
        df['Trade_Streak'] = DEFAULT_VALUES['Trade_Streak']
        df['Is_First_Trade'] = 1
        df['Politician_Avg_Amount'] = DEFAULT_VALUES['Politician_Avg_Amount']
        df['Amount_vs_Avg'] = 1.0
        df['Is_Unusual_Size'] = 0
        df['Signal_Strength'] = 1.0
        df['Cluster_Buy'] = 0
        
        # Politician lag features
        df['Politician_Last_Trade_Same_Type'] = 0
        df['Politician_Trade_Frequency_30D'] = 0
        df['Politician_Trade_Frequency_90D'] = 0
        df['Politician_Avg_Filed_After'] = DEFAULT_VALUES['Politician_Avg_Filed_After']
        df['Politician_Purchase_Ratio'] = DEFAULT_VALUES['Politician_Purchase_Ratio']
        
        # Ticker lag features
        df['Ticker_Trade_Count_30D'] = DEFAULT_VALUES['Ticker_Trade_Count_30D']
        df['Ticker_Trade_Count_90D'] = DEFAULT_VALUES['Ticker_Trade_Count_90D']
        df['Ticker_Buy_Ratio_30D'] = DEFAULT_VALUES['Ticker_Buy_Ratio_30D']
        df['Ticker_Politician_Count_30D'] = DEFAULT_VALUES['Ticker_Politician_Count_30D']
        df['Ticker_Avg_Amount_30D'] = DEFAULT_VALUES['Ticker_Avg_Amount_30D']
        df['Is_Ticker_Hot'] = 0
        
        # Market lag features
        df['Market_Buy_Ratio_7D'] = DEFAULT_VALUES['Market_Buy_Ratio_7D']
        df['Market_Buy_Ratio_30D'] = DEFAULT_VALUES['Market_Buy_Ratio_30D']
        df['Market_Trade_Volume_7D'] = DEFAULT_VALUES['Market_Trade_Volume_7D']
        df['Market_Avg_Amount_7D'] = DEFAULT_VALUES['Market_Avg_Amount_7D']
        df['Is_High_Activity_Period'] = 0
        
        # If we have historical data, compute proper features
        if self.historical_data is not None and len(self.historical_data) > 0:
            for idx, row in df.iterrows():
                current_date = row.get('Notification Date', pd.Timestamp.now())
                if pd.isna(current_date):
                    current_date = pd.Timestamp.now()
                
                politician = row.get('Politician Name', '')
                ticker = row.get('Ticker_Clean', '')
                
                # Get historical data before current date
                hist = self.historical_data[
                    self.historical_data['Notification Date'] < current_date
                ]
                
                if len(hist) > 0:
                    # Politician history
                    politician_hist = hist[hist['Politician Name'] == politician] if 'Politician Name' in hist.columns else pd.DataFrame()
                    
                    if len(politician_hist) > 0:
                        df.loc[idx, 'Past_Trade_Count'] = len(politician_hist)
                        df.loc[idx, 'Is_First_Trade'] = 0
                        df.loc[idx, 'Politician_Avg_Amount'] = politician_hist['Amount_Min'].mean()
                        df.loc[idx, 'Amount_vs_Avg'] = row['Amount_Min'] / politician_hist['Amount_Min'].mean()
                        df.loc[idx, 'Is_Unusual_Size'] = 1 if df.loc[idx, 'Amount_vs_Avg'] > 2 else 0
                        
                        # Recent history (30 days)
                        recent_30 = politician_hist[
                            politician_hist['Notification Date'] >= current_date - timedelta(days=30)
                        ]
                        df.loc[idx, 'Politician_Trade_Frequency_30D'] = len(recent_30)
                        
                        # Recent history (90 days)
                        recent_90 = politician_hist[
                            politician_hist['Notification Date'] >= current_date - timedelta(days=90)
                        ]
                        df.loc[idx, 'Politician_Trade_Frequency_90D'] = len(recent_90)
                        
                        # Last trade
                        last_trade = politician_hist.iloc[-1]
                        last_date = last_trade['Notification Date']
                        df.loc[idx, 'Days_Since_Last_Trade'] = (current_date - last_date).days
                        
                        if 'Is_Purchase' in politician_hist.columns:
                            df.loc[idx, 'Politician_Purchase_Ratio'] = politician_hist['Is_Purchase'].mean()
                            df.loc[idx, 'Politician_Last_Trade_Same_Type'] = 1 if last_trade.get('Is_Purchase', 0) == row['Is_Purchase'] else 0
                    
                    # Ticker history
                    if ticker and 'Ticker_Clean' in hist.columns:
                        ticker_hist_30 = hist[
                            (hist['Ticker_Clean'] == ticker) &
                            (hist['Notification Date'] >= current_date - timedelta(days=30))
                        ]
                        ticker_hist_90 = hist[
                            (hist['Ticker_Clean'] == ticker) &
                            (hist['Notification Date'] >= current_date - timedelta(days=90))
                        ]
                        
                        df.loc[idx, 'Ticker_Trade_Count_30D'] = len(ticker_hist_30)
                        df.loc[idx, 'Ticker_Trade_Count_90D'] = len(ticker_hist_90)
                        if len(ticker_hist_30) > 0:
                            df.loc[idx, 'Ticker_Buy_Ratio_30D'] = ticker_hist_30['Is_Purchase'].mean() if 'Is_Purchase' in ticker_hist_30.columns else 0.5
                            df.loc[idx, 'Ticker_Politician_Count_30D'] = ticker_hist_30['Politician Name'].nunique() if 'Politician Name' in ticker_hist_30.columns else 0
                            df.loc[idx, 'Ticker_Avg_Amount_30D'] = ticker_hist_30['Amount_Min'].mean()
                        df.loc[idx, 'Is_Ticker_Hot'] = 1 if len(ticker_hist_30) > 5 else 0
                    
                    # Market overall (last 7 days)
                    market_7d = hist[hist['Notification Date'] >= current_date - timedelta(days=7)]
                    market_30d = hist[hist['Notification Date'] >= current_date - timedelta(days=30)]
                    
                    if len(market_7d) > 0 and 'Is_Purchase' in market_7d.columns:
                        df.loc[idx, 'Market_Buy_Ratio_7D'] = market_7d['Is_Purchase'].mean()
                        df.loc[idx, 'Market_Trade_Volume_7D'] = len(market_7d)
                        df.loc[idx, 'Market_Avg_Amount_7D'] = market_7d['Amount_Min'].mean()
                    
                    if len(market_30d) > 0 and 'Is_Purchase' in market_30d.columns:
                        df.loc[idx, 'Market_Buy_Ratio_30D'] = market_30d['Is_Purchase'].mean()
                    
                    median_volume = self.historical_data.groupby(
                        self.historical_data['Notification Date'].dt.date
                    ).size().median()
                    df.loc[idx, 'Is_High_Activity_Period'] = 1 if df.loc[idx, 'Market_Trade_Volume_7D'] > median_volume else 0
        
        # Conviction Score
        df['Conviction_Score'] = (df['Amount_vs_Avg']).clip(0, 10)
        df['Signal_Strength'] = df['Conviction_Score'] / (df['Filed_After'] + 1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 5: Create interaction features."""
        print("[Predict] Creating interaction features...")
        
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
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final feature matrix for prediction."""
        print("[Predict] Preparing feature matrix...")
        
        # Get only the features the model expects
        available_features = [f for f in self.model_features if f in df.columns]
        missing_features = [f for f in self.model_features if f not in df.columns]
        
        if missing_features:
            print(f"[Predict] WARNING: Missing {len(missing_features)} features: {missing_features[:5]}...")
            # Fill missing features with defaults
            for f in missing_features:
                df[f] = DEFAULT_VALUES.get(f, 0)
        
        return df[self.model_features]
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full prediction pipeline."""
        original_df = df.copy()
        
        # Run pipeline
        df = self.clean_data(df)
        df = self.create_basic_features(df)
        df = self.create_derived_features(df)
        df = self.create_historical_features(df)
        df = self.create_interaction_features(df)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Predict
        print(f"[Predict] Running predictions on {len(X)} rows...")
        predictions = self.predictor.predict(X)
        probabilities = self.predictor.predict_proba(X)
        
        # Add results to original dataframe
        original_df['Predicted_Class'] = predictions.values
        original_df['Class_Label'] = original_df['Predicted_Class'].map({
            0: 'Underperform (Alpha <= 0)',
            1: 'Beat Market (0 < Alpha <= 10%)',
            2: 'Strong (10% < Alpha <= 20%)',
            3: 'Very Strong (Alpha > 20%)'
        })
        
        # Add probabilities
        for i in range(4):
            original_df[f'Prob_Class_{i}'] = probabilities[i].values
        
        # Add recommendation
        original_df['Recommendation'] = 'SKIP'
        original_df.loc[original_df['Predicted_Class'] >= 2, 'Recommendation'] = 'WATCH'
        original_df.loc[original_df['Predicted_Class'] >= 3, 'Recommendation'] = 'FOLLOW'
        
        return original_df


def main():
    parser = argparse.ArgumentParser(description="Predict Congress Trading Outcomes")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with new trades")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output CSV file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("[Predict] CONGRESS TRADING PREDICTION PIPELINE")
    print("=" * 60)
    
    # Load new trades
    print(f"\n[Predict] Loading new trades from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"[Predict] Loaded {len(df)} trades")
    
    # Initialize pipeline
    pipeline = PredictionPipeline(model_path=args.model)
    
    # Run predictions
    results = pipeline.predict(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("[Predict] PREDICTION SUMMARY")
    print("=" * 60)
    
    class_counts = results['Predicted_Class'].value_counts().sort_index()
    for cls in range(4):
        count = class_counts.get(cls, 0)
        print(f"  Class {cls}: {count} trades ({100*count/len(results):.1f}%)")
    
    rec_counts = results['Recommendation'].value_counts()
    print(f"\n  SKIP:   {rec_counts.get('SKIP', 0)}")
    print(f"  WATCH:  {rec_counts.get('WATCH', 0)}")
    print(f"  FOLLOW: {rec_counts.get('FOLLOW', 0)}")
    
    # Show FOLLOW recommendations
    follow_trades = results[results['Recommendation'] == 'FOLLOW']
    if len(follow_trades) > 0:
        print(f"\n[Predict] === TOP RECOMMENDATIONS (Class 3) ===")
        cols_to_show = ['Politician Name', 'Ticker', 'Type', 'Amount Min', 'Predicted_Class', 'Prob_Class_3']
        cols_available = [c for c in cols_to_show if c in follow_trades.columns]
        print(follow_trades[cols_available].head(10).to_string())
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"\n[Predict] Results saved to: {args.output}")
    print("[Predict] Complete.\n")


if __name__ == "__main__":
    main()
