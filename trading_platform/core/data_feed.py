"""
Data feed functionality for fetching and processing market data
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import time

from .config import default_interval_yahoo, get_max_days

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_historical_data(symbol: str, interval: str = default_interval_yahoo, days: int = 3) -> pd.DataFrame:
    """
    Fetch historical data from Yahoo Finance
    
    Args:
        symbol: Stock symbol
        interval: Data interval ('1m', '5m', '15m', '30m', '60m', '1d')
        days: Number of days of historical data to fetch (default: 3)
    
    Returns:
        DataFrame with OHLCV data
    """
    # Convert crypto trading symbol to Yahoo Finance format if needed
    if '/' in symbol:
        # Extract the base currency (before the slash)
        yf_symbol = symbol.split('/')[0] + '-' + symbol.split('/')[1]
    else:
        yf_symbol = symbol
    
    # Calculate start and end dates
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Try to fetch data with retries
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            
            # Log successful data fetch
            logger.info(f"Fetched {len(df)} bars of {interval} data for {symbol} ({yf_symbol})")
            logger.info(f"Date range: {df.index[0] if not df.empty else start} to {df.index[-1] if not df.empty else end}")
            
            return df
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Attempt {attempt} failed: {error_msg}")
            
            # If we've reached max retries, raise the exception
            if attempt == max_retries:
                logger.error(f"Failed to fetch data for {symbol} ({yf_symbol}): {error_msg}")
                raise Exception(f"Too Many Requests. Rate limited. Try after a while.")
            
            # Wait before retrying (exponential backoff)
            time.sleep(attempt * 2)

def get_latest_data(symbol: str, interval: str = default_interval_yahoo, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Get the most recent data points
    
    Args:
        symbol: Stock symbol
        interval: Data interval
        limit: Number of data points to return (default: None = all available data)
    
    Returns:
        DataFrame with the most recent data points
    """
    max_days = get_max_days(interval)
    df = fetch_historical_data(symbol, interval, days=max_days)
    
    if limit and len(df) > limit:
        return df.tail(limit)
    return df

def is_market_open(symbol: str = 'BTC/USD') -> bool:
    """
    Check if market is currently open for the given symbol
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        Boolean indicating if market is currently open
    """
    # For crypto, market is always open
    if symbol.endswith('/USD') or symbol.endswith('-USD'):
        return True
    
    # For stocks, check market hours (implementation omitted for brevity)
    return False

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
        # Map timeframe to yfinance interval
        yf_interval = timeframe
        
        # Fetch data
        df = fetch_historical_data(symbol, yf_interval, days=lookback_days)
        
        if df.empty:
            return None, f"No data available for {symbol} with timeframe {timeframe}"
        
        # Process data
        df = df.copy()
        
        # Add additional columns
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Calculate volume-based indicators
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        
        return df, None
        
    except Exception as e:
        error_msg = f"Error fetching historical data for {symbol}: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def get_available_symbols() -> List[str]:
    """
    Get list of all available trading symbols
    
    Returns:
        List of trading symbol strings
    """
    from ..modules.crypto.config import TRADING_SYMBOLS
    return list(TRADING_SYMBOLS.keys())

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with symbol information
    """
    from ..modules.crypto.config import TRADING_SYMBOLS
    
    if symbol in TRADING_SYMBOLS:
        return TRADING_SYMBOLS[symbol]
    else:
        return {"error": f"Symbol {symbol} not found"}

def fetch_multi_symbol_data(symbols: List[str], timeframe: str, lookback_days: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols in parallel
    
    Args:
        symbols: List of trading symbols
        timeframe: Data timeframe
        lookback_days: Number of days to look back
        
    Returns:
        Dictionary mapping symbol strings to DataFrame objects
    """
    results = {}
    
    for symbol in symbols:
        df, error = fetch_and_process_data(symbol, timeframe, lookback_days)
        if df is not None:
            results[symbol] = df
    
    return results