"""
Data Preprocessing Module

This module contains functions for cleaning and preprocessing cryptocurrency market data
and Fear & Greed Index data.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the historical price data and Fear & Greed Index data.
    
    Args:
        data_dir (str): Path to the directory containing the data files
        
    Returns:
        tuple: A tuple containing (historical_data, fear_greed_data)
    """
    data_path = Path(data_dir)
    
    # Load datasets
    historical_data = pd.read_csv(data_path / 'raw/historical_data.csv')
    fear_greed_data = pd.read_csv(data_path / 'raw/fear_greed_index.csv')
    
    return historical_data, fear_greed_data

def clean_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the historical price data.
    
    Args:
        df (pd.DataFrame): Raw historical price data
        
    Returns:
        pd.DataFrame: Cleaned historical price data
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Handle missing values (forward fill for OHLC data, 0 for volume if needed)
    df = df.ffill()
    
    # Calculate daily returns if not present
    if 'close' in df.columns and 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()
    
    return df

def clean_fear_greed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Fear & Greed Index data.
    
    Args:
        df (pd.DataFrame): Raw Fear & Greed Index data
        
    Returns:
        pd.DataFrame: Cleaned Fear & Greed Index data
    """
    df = df.copy()
    
    # Convert date column to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Forward fill missing values
    df = df.ffill()
    
    return df

def merge_datasets(historical_df: pd.DataFrame, fg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical price data with Fear & Greed Index data.
    
    Args:
        historical_df (pd.DataFrame): Historical price data
        fg_df (pd.DataFrame): Fear & Greed Index data
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    if 'date' in historical_df.columns and 'date' in fg_df.columns:
        # Merge on date column
        merged_df = pd.merge(historical_df, fg_df, on='date', how='left')
        return merged_df
    return pd.DataFrame()

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for the model.
    
    Args:
        df (pd.DataFrame): Input dataframe with price and Fear & Greed data
        
    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    df = df.copy()
    
    # Technical Indicators
    if 'close' in df.columns:
        # Moving Averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_21'] = df['close'].rolling(window=21).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=21).std() * np.sqrt(252)
        
        # Daily price range
        if all(col in df.columns for col in ['high', 'low']):
            df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    return df

def prepare_data(data_dir: str) -> pd.DataFrame:
    """
    Complete data preparation pipeline.
    
    Args:
        data_dir (str): Path to the directory containing the data files
        
    Returns:
        pd.DataFrame: Processed and feature-engineered dataset
    """
    # Load data
    historical_data, fear_greed_data = load_data(data_dir)
    
    # Clean data
    historical_clean = clean_historical_data(historical_data)
    fg_clean = clean_fear_greed_data(fear_greed_data)
    
    # Merge datasets
    merged_data = merge_datasets(historical_clean, fg_clean)
    
    # Create features
    final_data = create_features(merged_data)
    
    return final_data

if __name__ == "__main__":
    # Example usage
    data_path = Path("../data")
    processed_data = prepare_data(data_path)
    
    # Save processed data
    output_path = data_path / 'processed/processed_data.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
