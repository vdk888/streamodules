import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List, Any, Union
import logging
import pytz
import json
from .config import TRADING_SYMBOLS, DEFAULT_INTERVAL, DEFAULT_INTERVAL_WEEKLY, default_interval_yahoo

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
    # Get the correct Yahoo Finance symbol
    yf_symbol = TRADING_SYMBOLS[symbol]['yfinance']
    ticker = yf.Ticker(yf_symbol)
    
    # Calculate start and end dates
    end = datetime.now(pytz.UTC)
    start = end - timedelta(days=days)
    
    # Debug logging
    logger.debug(f"Attempting to fetch {interval} data for {symbol} ({yf_symbol})")
    logger.debug(f"Date range: {start} to {end}")
    logger.debug(f"Requested days: {days}")
    
    # Fetch data with retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = ticker.history(start=start, end=end, interval=interval)
            logger.debug(f"Successfully fetched {len(df)} bars of {interval} data")
            if not df.empty:
                break
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to fetch data for {symbol} ({yf_symbol}): {str(e)}")
                raise e
            continue
    
    if df.empty:
        logger.error(f"No data available for {symbol} ({yf_symbol})")
        raise ValueError(f"No data available for {symbol} ({yf_symbol})")
    
    # Ensure we have enough data
    min_required_bars = 100  # Minimum bars needed for signals
    if len(df) < min_required_bars:
        logger.debug(f"Only {len(df)} bars found, fetching more data")
        start = end - timedelta(days=days * 2)
        df = ticker.history(start=start, end=end, interval=interval)
    
    # Clean and format the data
    df.columns = [col.lower() for col in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    # Add logging for data quality
    logger.info(f"Fetched {len(df)} bars of {interval} data for {symbol} ({yf_symbol})")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df

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
    # Get the correct interval from config
    config_interval = TRADING_SYMBOLS[symbol].get('interval', interval)
    
    try:
        # Fetch at least 3 days of data for proper signal calculation
        df = fetch_historical_data(symbol, config_interval, days=3)
        
        # Filter for market hours
        market_hours = TRADING_SYMBOLS[symbol]['market_hours']
        if market_hours['start'] != '00:00' or market_hours['end'] != '23:59':
            # Convert index to market timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            market_tz = market_hours['timezone']
            df.index = df.index.tz_convert(market_tz)
            
            # Create time masks for market hours
            start_time = pd.Timestamp.strptime(market_hours['start'], '%H:%M').time()
            end_time = pd.Timestamp.strptime(market_hours['end'], '%H:%M').time()
            
            # Filter for market hours
            df = df[
                (df.index.time >= start_time) & 
                (df.index.time <= end_time) &
                (df.index.weekday < 5)  # Monday = 0, Friday = 4
            ]
        
        # Apply limit if specified
        if limit is not None:
            return df.tail(limit)
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

def is_market_open(symbol: str = 'BTC/USD') -> bool:
    """
    Check if market is currently open for the given symbol
    
    Args:
        symbol: Trading symbol to check
        
    Returns:
        Boolean indicating if market is currently open
    """
    try:
        market_hours = TRADING_SYMBOLS[symbol]['market_hours']
        now = datetime.now(pytz.UTC)
        
        # For 24/7 markets
        if market_hours['start'] == '00:00' and market_hours['end'] == '23:59':
            return True
            
        # Convert current time to market timezone
        market_tz = market_hours['timezone']
        market_time = now.astimezone(pytz.timezone(market_tz))
        
        # Check if it's a weekday
        if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Parse market hours
        start_time = datetime.strptime(market_hours['start'], '%H:%M').time()
        end_time = datetime.strptime(market_hours['end'], '%H:%M').time()
        current_time = market_time.time()
        
        return start_time <= current_time <= end_time
        
    except Exception as e:
        logger.error(f"Error checking market hours for {symbol}: {str(e)}")
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
        # Map timeframe to yfinance format
        yf_interval = timeframe
        if timeframe == '1h':
            yf_interval = '60m'
        elif timeframe == '4h':
            yf_interval = '1h'  # Will be resampled later
        elif timeframe == '1d':
            yf_interval = '1d'

        # Fetch data
        logger.info(f"Fetching {lookback_days} days of {timeframe} data for {symbol}")

        # Fetch the historical data
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
        return None, error_msg

def get_available_symbols() -> List[str]:
    """
    Get list of all available trading symbols
    
    Returns:
        List of trading symbol strings
    """
    return list(TRADING_SYMBOLS.keys())

def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with symbol information
    """
    if symbol in TRADING_SYMBOLS:
        return TRADING_SYMBOLS[symbol]
    else:
        return {}

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
        try:
            df, error = fetch_and_process_data(symbol, timeframe, lookback_days)
            if df is not None:
                results[symbol] = df
            else:
                logger.warning(f"Could not fetch data for {symbol}: {error}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
    
    return results