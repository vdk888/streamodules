"""
Technical indicators for market analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_hurst_exponent(data: pd.Series, lags: List[int]) -> float:
    """Calculate Hurst exponent using R/S analysis"""
    # Convert to numpy array for faster processing
    data_np = np.array(data)
    
    # Return values
    tau = []
    lagvec = []
    
    # Step through the different lags
    for lag in lags:
        # Construct a vector of data point differences with lag
        pp = []
        
        # Step from lag to length of data point array
        for i in range(lag, len(data_np)):
            # Calculate percentage returns on price data
            pp.append(np.log(data_np[i] / data_np[i - lag]))
        
        # Calculate the variance of the percentage returns
        pp = np.array(pp)
        lagvec.append(lag)
        
        # Calculate the range from min to max of pp
        rs_range = np.max(pp) - np.min(pp)
        
        # Calculate the standard deviation of pp
        rs_std = np.std(pp)
        
        # Calculate R/S ratio
        rs = rs_range / rs_std if rs_std > 0 else 0
        tau.append(rs)
    
    # Calculate Hurst exponent using OLS regression of log(R/S) against log(lag)
    lag_log = np.log(lagvec)
    tau_log = np.log(tau)
    
    # OLS regression with gradient as Hurst exponent
    if len(tau_log) > 1 and len(lag_log) > 1:
        # Use numpy's polyfit to estimate the Hurst exponent
        hurst = np.polyfit(lag_log, tau_log, 1)[0]
        return hurst
    else:
        return 0.5  # Default to 0.5 (random walk) if not enough data

def calculate_fractal_complexity(data: pd.DataFrame, lags: List[int] = [10, 20, 40], window: int = 100) -> pd.Series:
    """Calculate fractal complexity indicator using Hurst exponent."""
    # Initialize the result series
    fractal_complexity = pd.Series(index=data.index, dtype=float)
    
    # Use closing prices for calculations
    prices = data['close']
    
    # Calculate rolling window Hurst exponent
    for i in range(window, len(prices)):
        window_data = prices.iloc[i-window:i]
        fractal_complexity.iloc[i] = 1 - calculate_hurst_exponent(window_data, lags)
    
    return fractal_complexity

def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Calculate the fast and slow EMA
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Return MACD values as a normalized combined series
    # (gives more weight to the histogram for signal generation)
    macd_normalized = (macd_line / data['close'].rolling(window=slow).mean()) * 0.4 + \
                      (histogram / data['close'].rolling(window=slow).mean()) * 0.6
    
    return macd_normalized

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    # Calculate price changes
    delta = data['close'].diff()
    
    # Create gain and loss series
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize RSI to -1 to 1 range
    rsi_normalized = (rsi - 50) / 50
    
    return rsi_normalized

def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
    """Calculate Stochastic Oscillator"""
    # Calculate %K
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    
    # Handle division by zero
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)
    
    k = 100 * ((data['close'] - lowest_low) / denominator)
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    # Normalize to -1 to 1 range
    stoch_normalized = (k - 50) / 50
    
    return stoch_normalized

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
    # Extract parameters
    rsi_period = params.get('rsi_period', 14)
    macd_fast = params.get('macd_fast', 12)
    macd_slow = params.get('macd_slow', 26)
    macd_signal = params.get('macd_signal', 9)
    stoch_k = params.get('stoch_k', 14)
    stoch_d = params.get('stoch_d', 3)
    hurst_lags = params.get('hurst_lags', [10, 20, 40, 80])
    volatility_window = params.get('volatility_window', 20)
    rsi_weight = params.get('rsi_weight', 0.3)
    macd_weight = params.get('macd_weight', 0.3) 
    stoch_weight = params.get('stoch_weight', 0.2)
    fractal_weight = params.get('fractal_weight', 0.2)
    
    # Calculate indicators
    rsi = calculate_rsi(data, rsi_period)
    macd = calculate_macd(data, macd_fast, macd_slow, macd_signal)
    stoch = calculate_stochastic(data, stoch_k, stoch_d)
    fractal = calculate_fractal_complexity(data, hurst_lags)
    
    # Normalize and weight the indicators
    composite = (
        rsi * rsi_weight +
        macd * macd_weight +
        stoch * stoch_weight +
        (fractal - 0.5) * fractal_weight  # Centered around 0
    )
    
    # Calculate volatility
    returns = np.log(data['close'] / data['close'].shift(1))
    volatility = returns.rolling(window=volatility_window).std()
    
    # Calculate adaptive threshold based on volatility
    threshold_multiplier = 1.0 + volatility * reactivity
    
    if is_weekly:
        # For weekly timeframe, increase threshold
        threshold_multiplier = threshold_multiplier * 1.2
    
    # Generate thresholds
    upper_threshold = 0.1 * threshold_multiplier
    lower_threshold = -0.1 * threshold_multiplier
    
    # Create result dataframe
    result = pd.DataFrame({
        'composite': composite,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold
    })
    
    # Return the result dataframe and additional data
    return result, composite, volatility

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
    if data is None or len(data) < 50:
        logger.warning("Insufficient data for signal generation")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Calculate daily composite indicator
    daily_composite, raw_daily, daily_vol = calculate_composite_indicator(
        data, params, reactivity, is_weekly=False
    )
    
    # Weekly analysis - resample to 4h if smaller timeframe
    if 'h' in data.index.freq.name if hasattr(data.index, 'freq') and data.index.freq else True:
        # Resample to 4h data for weekly analysis
        try:
            weekly_data = data.resample(default_interval_weekly).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            weekly_composite, raw_weekly, weekly_vol = calculate_composite_indicator(
                weekly_data, params, reactivity, is_weekly=True
            )
        except Exception as e:
            logger.error(f"Error in weekly resampling: {str(e)}")
            weekly_composite = pd.DataFrame({
                'composite': pd.Series(dtype=float),
                'upper_threshold': pd.Series(dtype=float),
                'lower_threshold': pd.Series(dtype=float)
            })
            raw_weekly = pd.Series(dtype=float)
    else:
        # Already at or above 4h timeframe, just use as is
        weekly_composite, raw_weekly, weekly_vol = calculate_composite_indicator(
            data, params, reactivity, is_weekly=True
        )
    
    # Generate signal DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['close']
    signals['daily_composite'] = daily_composite['composite']
    signals['daily_upper'] = daily_composite['upper_threshold']
    signals['daily_lower'] = daily_composite['lower_threshold']
    
    # Calculate signal conditions
    signals['buy_signal'] = (daily_composite['composite'] < daily_composite['lower_threshold']) & \
                            (daily_composite['composite'].shift(1) >= daily_composite['lower_threshold'].shift(1))
    
    signals['sell_signal'] = (daily_composite['composite'] > daily_composite['upper_threshold']) & \
                             (daily_composite['composite'].shift(1) <= daily_composite['upper_threshold'].shift(1))
    
    # Add additional columns for position management
    signals['potential_buy'] = daily_composite['composite'] < daily_composite['lower_threshold']
    signals['potential_sell'] = daily_composite['composite'] > daily_composite['upper_threshold']
    
    # Add volatility for position sizing
    signals['volatility'] = daily_vol
    
    return signals, daily_composite, weekly_composite

def get_default_params():
    """Get default parameters for indicator calculations"""
    return {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'stoch_k': 14,
        'stoch_d': 3,
        'hurst_lags': [10, 20, 40, 80],
        'volatility_window': 20,
        'rsi_weight': 0.3,
        'macd_weight': 0.3,
        'stoch_weight': 0.2,
        'fractal_weight': 0.2
    }