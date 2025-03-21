"""
Data feed functionality for fetching and processing market data
"""
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import datetime
import time
from concurrent.futures import ThreadPoolExecutor

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
    # Validate inputs
    valid_intervals = ['1m', '5m', '15m', '30m', '60m', '1h', '1d']
    if interval not in valid_intervals:
        logger.warning(f"Invalid interval: {interval}. Using default: {default_interval_yahoo}")
        interval = default_interval_yahoo
    
    # Convert '1h' to '60m' for Yahoo Finance
    if interval == '1h':
        yf_interval = '60m'
    else:
        yf_interval = interval
    
    # Calculate period based on interval and days
    max_days = get_max_days(interval)
    if days > max_days:
        logger.warning(f"Days requested ({days}) exceeds maximum for interval {interval} ({max_days}). Using maximum.")
        days = max_days
    
    # Calculate start and end dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Convert dates to strings
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        # Prepare symbol for Yahoo Finance (replace / with -)
        yf_symbol = symbol.replace('/', '-')
        
        # Fetch data from Yahoo Finance
        data = yf.download(
            tickers=yf_symbol,
            start=start_str,
            end=end_str,
            interval=yf_interval,
            auto_adjust=True
        )
        
        # Clean and preprocess data
        if data.empty:
            logger.warning(f"No data retrieved for {symbol}")
            return pd.DataFrame()
        
        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Make sure we have all required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"Missing column: {col} for {symbol}")
                if col == 'volume':
                    data['volume'] = 0
                else:
                    return pd.DataFrame()
        
        # Log data retrieved
        logger.info(f"Fetched {len(data)} bars of {interval} data for {symbol} ({yf_symbol})")
        if not data.empty:
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

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
    # Fetch data
    days = get_max_days(interval)
    data = fetch_historical_data(symbol, interval, days)
    
    # Return most recent data points if limit specified
    if limit is not None and limit > 0 and limit < len(data):
        return data.tail(limit)
    else:
        return data

def is_market_open(symbol: str = 'BTC/USD') -> bool:
    """
    Check if market is currently open for the given symbol
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        Boolean indicating if market is currently open
    """
    # For crypto markets, they're always open
    if '/' in symbol and (symbol.startswith('BTC') or symbol.endswith('BTC') or
                          symbol.startswith('ETH') or symbol.endswith('ETH') or
                          'USD' in symbol):
        return True
    
    # Get current time in UTC
    now = datetime.datetime.utcnow()
    current_hour = now.hour
    current_day = now.weekday()
    
    # Check for stock market hours (simplified - assumes US market)
    if '/' not in symbol:
        # Check if weekend
        if current_day >= 5:  # 5,6 = Saturday, Sunday
            return False
        
        # Check if during market hours (9:30 AM - 4:00 PM EST, which is UTC-5)
        # So in UTC: 14:30 - 21:00
        if current_hour < 14 or current_hour >= 21:
            return False
        if current_hour == 14 and now.minute < 30:
            return False
            
        return True
    
    # Default to open (could be refined with exchange-specific logic)
    return True

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
        # Fetch data from Yahoo Finance
        data = fetch_historical_data(symbol, timeframe, lookback_days)
        
        # Check if data is empty
        if data.empty:
            return None, f"No data available for {symbol} with {timeframe} timeframe"
        
        # Process data (additional preprocessing if needed)
        # ...
        
        return data, None
        
    except Exception as e:
        logger.error(f"Error in fetch_and_process_data for {symbol}: {str(e)}")
        return None, str(e)

def get_available_symbols() -> List[str]:
    """
    Get list of all available trading symbols
    
    Returns:
        List of trading symbol strings
    """
    # For now, return a default list of popular crypto symbols
    return [
        'BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'DOT/USD',
        'LINK/USD', 'DOGE/USD', 'AAVE/USD', 'UNI/USD', 'LTC/USD'
    ]

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with symbol information
    """
    # Basic implementation - could be expanded with more details
    info = {
        'symbol': symbol,
        'name': symbol.split('/')[0],
        'type': 'crypto' if '/' in symbol else 'stock',
        'exchange': 'Coinbase' if '/' in symbol else 'NYSE/NASDAQ',
        'market_hours': {
            'open': True,  # Default to always open for crypto
            'next_open': None,
            'next_close': None
        }
    }
    
    # Check if market is open
    info['market_hours']['open'] = is_market_open(symbol)
    
    return info

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
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
        # Create fetch tasks
        future_to_symbol = {
            executor.submit(fetch_historical_data, symbol, timeframe, lookback_days): symbol
            for symbol in symbols
        }
        
        # Process completed tasks
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"Empty data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return results