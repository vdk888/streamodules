import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union, List
import traceback

# Import functions from provided files
from attached_assets.fetch import fetch_historical_data, get_latest_data, is_market_open
from attached_assets.indicators import generate_signals, get_default_params
from attached_assets.config import TRADING_SYMBOLS, default_interval_yahoo
import attached_assets.config as config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_and_process_data(symbol: str, timeframe: str, lookback_days: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch and process data for the given symbol and timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe ('1h', '4h', '1d')
        lookback_days: Number of days to look back
    
    Returns:
        Tuple containing the processed DataFrame and error message (if any)
    """
    try:
        # Map streamlit timeframe to yfinance format
        yf_interval = timeframe
        if timeframe == '1h':
            yf_interval = '60m'
        elif timeframe == '4h':
            yf_interval = '1h'  # Will be resampled later
        elif timeframe == '1d':
            yf_interval = '1d'
        
        # Fetch data
        logger.info(f"Fetching {lookback_days} days of {timeframe} data for {symbol}")
        
        # Use the fetch_historical_data function from fetch.py
        df = fetch_historical_data(symbol, yf_interval, days=lookback_days)
        
        # Resample if needed (for 4h timeframe)
        if timeframe == '4h' and yf_interval == '1h':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # Verify data was fetched correctly
        if df is None or df.empty:
            logger.error(f"No data retrieved for {symbol} with {timeframe} timeframe")
            return None, "No data available"
        
        logger.info(f"Successfully fetched {len(df)} bars of {timeframe} data for {symbol}")
        return df, None
        
    except Exception as e:
        error_msg = f"Error fetching data: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, error_msg

def generate_signals_with_indicators(data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Generate signals and indicators for the given data.
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Tuple containing signals DataFrame, daily data, and weekly data
    """
    try:
        if data is None or data.empty:
            logger.error("No data available for signal generation")
            return None, None, None
        
        # Get default parameters
        params = get_default_params()
        
        # Use the generate_signals function from indicators.py
        signals, daily_data, weekly_data = generate_signals(data, params)
        
        return signals, daily_data, weekly_data
        
    except Exception as e:
        error_msg = f"Error generating signals: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None, None, None

def get_indicator_values(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate various indicator values for the given data.
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Dictionary of indicator names to Series of values
    """
    from attached_assets.indicators import (
        calculate_macd, calculate_rsi, calculate_stochastic, 
        calculate_fractal_complexity
    )
    
    indicators = {}
    
    # Calculate MACD
    indicators['macd'] = calculate_macd(data)
    
    # Calculate RSI
    indicators['rsi'] = calculate_rsi(data)
    
    # Calculate Stochastic
    indicators['stochastic'] = calculate_stochastic(data)
    
    # Calculate Fractal Complexity
    indicators['fractal'] = calculate_fractal_complexity(data)
    
    return indicators
