"""
Technical indicators for market analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_hurst_exponent(data: pd.Series, lags: List[int]) -> float:
    """Calculate Hurst exponent using R/S analysis"""
    # Convert to numpy array for faster computation
    ts = data.values
    
    # Calculate R/S for different lag values
    rs_values = []
    
    for lag in lags:
        # Convert lag to int to ensure it works with array slicing
        lag = int(lag)
        
        # Skip if lag is too large for the data
        if lag >= len(ts):
            continue
            
        # Calculate returns    
        returns = np.diff(np.log(ts))
        
        # Skip if not enough returns
        if len(returns) < lag:
            continue
            
        # Calculate R/S value for this lag
        tau = []
        
        # Split returns into chunks of size 'lag'
        for i in range(0, len(returns), lag):
            chunk = returns[i:i+lag]
            
            # Skip if chunk is too small
            if len(chunk) < lag:
                continue
                
            # Calculate R/S for this chunk
            mean = np.mean(chunk)
            std = np.std(chunk)
            
            # Avoid division by zero
            if std == 0:
                continue
                
            # Calculate cumulative deviation
            x_t = np.cumsum(chunk - mean)
            r = np.max(x_t) - np.min(x_t)
            s = std
            
            # Store R/S value
            if s != 0:
                tau.append(r/s)
        
        # Average R/S values for this lag
        if tau:
            rs_values.append((lag, np.mean(tau)))
    
    # Calculate Hurst exponent (slope of log-log plot)
    if len(rs_values) > 1:
        x = np.log10([lag for lag, _ in rs_values])
        y = np.log10([rs for _, rs in rs_values])
        
        # Linear regression
        slope, _ = np.polyfit(x, y, 1)
        return slope
    else:
        return 0.5  # Default to random walk

def calculate_fractal_complexity(data: pd.DataFrame, lags: List[int] = [10, 20, 40], window: int = 100) -> pd.Series:
    """Calculate fractal complexity indicator using Hurst exponent."""
    if len(data) < window:
        return pd.Series(0.5, index=data.index)
    
    # Initialize results series
    complexity = pd.Series(index=data.index)
    
    # Calculate moving Hurst exponent
    for i in range(window, len(data) + 1):
        window_data = data['Close'].iloc[i-window:i]
        hurst = calculate_hurst_exponent(window_data, lags)
        complexity.iloc[i-1] = 1 - hurst  # Subtract from 1 to get complexity (anti-persistence)
    
    # Fill NaN values
    complexity = complexity.fillna(0.5)
    
    return complexity

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Ensure integers for moving average periods
    fast, slow, signal = int(fast), int(slow), int(signal)
    
    # Calculate EMAs
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line and signal line
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return histogram

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    period = int(period)
    delta = data['Close'].diff()
    
    # Make two series: one for gains and one for losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss.where(avg_loss != 0, 1)  # Avoid division by zero
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
    """Calculate Stochastic Oscillator"""
    k_period, d_period = int(k_period), int(d_period)
    
    # Calculate %K
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    # Avoid division by zero
    denom = high_max - low_min
    denom = denom.where(denom != 0, 1)
    
    k = 100 * ((data['Close'] - low_min) / denom)
    
    # Calculate %D (signal line)
    d = k.rolling(window=d_period).mean()
    
    return d

def calculate_composite_indicator(data: pd.DataFrame, params: Dict[str, Union[float, int]], 
                                 reactivity: float = 1.0, is_weekly: bool = False, 
                                 default_interval_weekly: str = '4h') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Calculate composite indicator combining multiple technical indicators
    
    Args:
        data: DataFrame with OHLCV data
        params: Dictionary of parameters for indicator calculation
        reactivity: Reactivity multiplier for thresholds
        is_weekly: Whether calculating for weekly timeframe
        default_interval_weekly: Default interval for weekly calculations
        
    Returns:
        DataFrame with composite indicator and threshold bands, raw composite values, and rolling std
    """
    # Extract parameters with safe fallbacks
    rsi_length = int(params.get('rsi_length', 14))
    macd_fast = int(params.get('macd_fast', 12))
    macd_slow = int(params.get('macd_slow', 26))
    macd_signal = int(params.get('macd_signal', 9))
    stoch_k = int(params.get('stoch_k', 14))
    stoch_d = int(params.get('stoch_d', 3))
    volatility_window = int(params.get('volatility_window', 20))
    complexity_window = int(params.get('complexity_window', 100))
    
    # Calculate indicators
    rsi = calculate_rsi(data, period=rsi_length)
    macd_hist = calculate_macd(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    stoch = calculate_stochastic(data, k_period=stoch_k, d_period=stoch_d)
    
    # Normalize indicators to -1 to 1 scale
    rsi_norm = (rsi - 50) / 50  # RSI: 0-100 -> -1 to 1
    
    # Normalize MACD (using historical standard deviation)
    macd_std = macd_hist.rolling(window=volatility_window).std()
    macd_norm = macd_hist / (macd_std * 2).replace(0, 1)  # Avoid division by zero
    macd_norm = macd_norm.clip(-1, 1)  # Clip to -1 to 1
    
    # Normalize Stochastic
    stoch_norm = (stoch - 50) / 50  # Stochastic: 0-100 -> -1 to 1
    
    # Calculate fractal complexity if we have enough data
    if len(data) >= complexity_window:
        complexity = calculate_fractal_complexity(data, window=complexity_window)
        complexity_norm = (complexity - 0.5) * 2  # 0-1 -> -1 to 1
    else:
        complexity_norm = pd.Series(0, index=data.index)
    
    # Combine indicators with equal weights (can be adjusted)
    composite = (rsi_norm * 0.25 + macd_norm * 0.3 + stoch_norm * 0.25 + complexity_norm * 0.2)
    
    # Calculate adaptive thresholds based on volatility
    rolling_std = composite.rolling(window=volatility_window).std()
    upper_threshold = rolling_std * 1.5 * reactivity
    lower_threshold = -rolling_std * 1.5 * reactivity
    
    # Create result DataFrame
    result = pd.DataFrame({
        'composite': composite,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold
    })
    
    return result, composite, rolling_std

def generate_signals(data: pd.DataFrame, params: Dict[str, Union[float, int]], 
                    reactivity: float = 1.0, default_interval_weekly: str = '4h') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate trading signals based on technical indicators
    
    Args:
        data: DataFrame with OHLCV data
        params: Dictionary of parameters for signal generation
        reactivity: Reactivity multiplier for thresholds
        default_interval_weekly: Default interval for weekly calculations
        
    Returns:
        Tuple containing signals DataFrame, daily composite data, and weekly composite data
    """
    if data.empty:
        logger.warning("Empty data provided, unable to generate signals")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Calculate composite indicators
    daily_result, daily_composite, daily_std = calculate_composite_indicator(
        data, params, reactivity, is_weekly=False
    )
    
    # No need for separate weekly calculation in this simplified version
    # Just duplicate daily for the structure
    weekly_result = daily_result.copy()
    
    # Generate signals
    signals = pd.DataFrame(index=data.index)
    signals['timestamp'] = data.index
    signals['price'] = data['Close']
    
    # Signal logic: 1 for buy, -1 for sell, 0 for hold
    signals['signal'] = 0
    
    # Buy signal when composite crosses below lower threshold
    buy_condition = (daily_result['composite'] < daily_result['lower_threshold']) & \
                    (daily_result['composite'].shift(1) >= daily_result['lower_threshold'].shift(1))
    signals.loc[buy_condition, 'signal'] = 1
    
    # Sell signal when composite crosses above upper threshold
    sell_condition = (daily_result['composite'] > daily_result['upper_threshold']) & \
                     (daily_result['composite'].shift(1) <= daily_result['upper_threshold'].shift(1))
    signals.loc[sell_condition, 'signal'] = -1
    
    # Add indicator values for reference
    signals['composite'] = daily_result['composite']
    signals['upper_threshold'] = daily_result['upper_threshold']
    signals['lower_threshold'] = daily_result['lower_threshold']
    
    return signals, daily_result, weekly_result

def get_default_params():
    """Get default parameters for indicator calculations"""
    return {
        'rsi_length': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'stoch_k': 14,
        'stoch_d': 3,
        'volatility_window': 20,
        'complexity_window': 100
    }