import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Union, List
import traceback
import json
import datetime

# Import functions from provided files
from attached_assets.fetch import fetch_historical_data, get_latest_data, is_market_open
from attached_assets.indicators import generate_signals, get_default_params
from attached_assets.config import TRADING_SYMBOLS, default_interval_yahoo
import attached_assets.config as config
from replit import db

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

def generate_signals_with_indicators(data: pd.DataFrame, params: Optional[Dict] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Generate signals and indicators for the given data.

    Args:
        data: DataFrame with OHLCV data
        params: Optional parameters for signal generation, uses defaults if None

    Returns:
        Tuple containing signals DataFrame, daily data, and weekly data
    """
    try:
        if data is None or data.empty:
            logger.error("No data available for signal generation")
            return None, None, None

        # Use provided parameters or get defaults
        if params is None:
            params = get_default_params()

        # Use the generate_signals function from indicators.py
        signals, daily_data, weekly_data = generate_signals(data, params)
        
        # Calculate technical indicators manually
        from attached_assets.indicators import calculate_rsi, calculate_stochastic, calculate_fractal_complexity
        
        # Add close price for reference
        signals['close'] = data['close']
        
        # Calculate MACD
        fast = params['macd_fast']
        slow = params['macd_slow']
        signal_period = params['macd_signal']
        
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        
        signals['macd'] = macd_line
        signals['macd_signal'] = macd_signal
        signals['macd_hist'] = macd_hist
        
        # Add RSI
        signals['rsi'] = calculate_rsi(data, period=params['rsi_period'])
        
        # Add Stochastic
        stoch_k = calculate_stochastic(data, k_period=params['stochastic_k_period'], d_period=params['stochastic_d_period'])
        signals['stoch_k'] = stoch_k
        signals['stoch_d'] = stoch_k.rolling(params['stochastic_d_period']).mean()
        
        # Add Fractal Complexity
        signals['fractal'] = calculate_fractal_complexity(data)

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
        calculate_rsi, calculate_stochastic, 
        calculate_fractal_complexity
    )

    indicators = {}

    # Calculate MACD manually
    fast = 12
    slow = 26
    signal_period = 9
    
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    indicators['macd'] = macd_line

    # Calculate RSI
    indicators['rsi'] = calculate_rsi(data)

    # Calculate Stochastic
    indicators['stochastic'] = calculate_stochastic(data)

    # Calculate Fractal Complexity
    indicators['fractal'] = calculate_fractal_complexity(data)

    return indicators

def load_best_params(symbol: str, client) -> Dict:
    try:
        # Check if file exists in Object Storage, if not create it
        if "best_params.json" not in client.list():
            default_params = get_default_params()
            initial_data = {symbol: {'best_params': default_params, 'date': datetime.datetime.now().strftime("%Y-%m-%d")}}
            client.upload_from_text("best_params.json", json.dumps(initial_data))
            logger.info("Created initial parameters file in Object Storage")
            return default_params

        # Load existing parameters
        json_content = client.download_as_text("best_params.json")
        best_params_data = json.loads(json_content)
        
        # Return symbol-specific or default parameters
        if symbol in best_params_data:
            params = best_params_data[symbol]['best_params']
            logger.info(f"Using best parameters for {symbol} from Object Storage: {params}")
            return params
        else:
            default_params = get_default_params()
            # Add default params for this symbol
            best_params_data[symbol] = {
                'best_params': default_params,
                'date': datetime.datetime.now().strftime("%Y-%m-%d")
            }
            client.upload_from_text("best_params.json", json.dumps(best_params_data))
            logger.info(f"Added default parameters for {symbol} to Object Storage")
            return default_params

    except Exception as e:
        logger.error(f"Object Storage error: {str(e)}")
        raise  # Re-raise the exception to handle it at a higher level