import pandas as pd
import numpy as np
import datetime
from typing import Dict, Tuple, Optional, List

# Import necessary functions
from utils import fetch_and_process_data, generate_signals_with_indicators
from attached_assets.backtest_individual import calculate_performance_ranking as backtest_calculate_ranking
from attached_assets.fetch import fetch_historical_data

def get_price_data(symbol: str, timeframe: str, lookback_days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch and process price data for a given symbol
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe ('1h', '4h', '1d')
        lookback_days: Number of days to look back
        
    Returns:
        Tuple containing the processed DataFrame and error message (if any)
    """
    return fetch_and_process_data(symbol, timeframe, lookback_days)

def get_multi_asset_data(symbols_config: Dict, timeframe: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple assets
    
    Args:
        symbols_config: Dictionary of symbol configurations
        timeframe: Data timeframe
        lookback_days: Number of days to look back
        
    Returns:
        Dictionary of DataFrames keyed by symbol
    """
    prices_dataset = {}
    for symbol, config in symbols_config.items():
        yf_symbol = config['yfinance']
        if '/' in yf_symbol:
            yf_symbol = yf_symbol.replace('/', '-')
        
        try:
            data = fetch_historical_data(symbol, interval=timeframe, days=lookback_days)
            if not data.empty:
                prices_dataset[symbol] = data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
    
    return prices_dataset

def generate_trading_signals(df: pd.DataFrame, params: Optional[Dict] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Generate trading signals for the given data
    
    Args:
        df: DataFrame with OHLCV data
        params: Parameters for signal generation
        
    Returns:
        Tuple containing signals DataFrame, daily data, and weekly data
    """
    return generate_signals_with_indicators(df, params)

def calculate_performance_ranking(prices_dataset: Dict[str, pd.DataFrame], 
                                 current_time: datetime.datetime, 
                                 lookback_days: int) -> pd.DataFrame:
    """
    Calculate performance ranking of all symbols
    
    Args:
        prices_dataset: Dictionary of price dataframes keyed by symbol
        current_time: Current time to use as the reference point
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with performance and rank columns
    """
    return backtest_calculate_ranking(prices_dataset, current_time, lookback_days)