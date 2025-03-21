import pandas as pd
import numpy as np
from typing import Dict, Union, List, Tuple
import json
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
    ts = np.array(data)
    # Calculate R/S ratio for different lags
    rs_values = []
    
    for lag in lags:
        # Split time series into chunks
        chunks = len(ts) // lag
        if chunks < 1:
            continue
            
        # Calculate R/S for each chunk
        rs_chunk = []
        for i in range(chunks):
            chunk = ts[i*lag:(i+1)*lag]
            if len(chunk) < 2:  # Need at least 2 points
                continue
            # Mean-adjust the chunk
            mean_adj = chunk - chunk.mean()
            # Range and standard deviation
            r = max(mean_adj) - min(mean_adj)
            s = np.std(chunk)
            if s == 0:  # Avoid division by zero
                continue
            rs_chunk.append(r/s)
            
        if rs_chunk:  # If we have valid R/S values
            rs_values.append(np.mean(rs_chunk))
    
    if len(rs_values) < 2:  # Need at least 2 points for regression
        return 0.5
    
    # Calculate Hurst exponent from log-log regression
    x = np.log(lags[:len(rs_values)])
    y = np.log(rs_values)
    hurstExp = np.polyfit(x, y, 1)[0]
    return min(max(hurstExp, 0), 1)  # Bound between 0 and 1

def calculate_fractal_complexity(data: pd.DataFrame, lags: List[int] = [10, 20, 40], window: int = 100) -> pd.Series:
    """Calculate fractal complexity indicator using Hurst exponent."""
    returns = np.log(data['close']).diff().dropna()
    complexity = pd.Series(index=data.index, dtype=float)

    for i in range(window, len(returns) + 1):
        window_data = returns.iloc[i - window:i]
        h = calculate_hurst_exponent(window_data, lags)
        complexity.iloc[i - 1] = 2 * abs(h - 0.5)

    return complexity.fillna(0)


def calculate_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line  # MACD histogram

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.Series:
    """Calculate Stochastic Oscillator"""
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    k = 100 * (data['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k - d  # Similar to MACD, we return K-D

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
    # Calculate fractal complexity
    complexity = calculate_fractal_complexity(data, lags=params.get('fractal_lags', [10, 20, 40]), 
                                            window=params.get('fractal_window', 100))
    
    # Log input data stats
    logger.debug(f"{'Weekly' if is_weekly else 'Daily'} composite indicator calculation")
    logger.debug(f"Data points: {len(data)}")
    logger.debug(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate indicators
    macd = calculate_macd(data, fast=params.get('macd_fast', 12), 
                          slow=params.get('macd_slow', 26), 
                          signal=params.get('macd_signal', 9))
    
    rsi = calculate_rsi(data, period=params.get('rsi_period', 14))
    
    stoch = calculate_stochastic(data, k_period=params.get('stochastic_k_period', 14), 
                                d_period=params.get('stochastic_d_period', 3))
    
    # Handle NaN values safely
    macd = macd.fillna(0)
    rsi = rsi.fillna(50)  # Neutral RSI value
    stoch = stoch.fillna(0)
    
    # Normalize each indicator with safe division
    norm_macd = macd / (macd.std() if macd.std() != 0 else 1)
    norm_rsi = (rsi - 50) / 25  # Center around 0 and scale
    norm_stoch = stoch / (100 if stoch.max() != 0 else 1)  # Scale to [-1, 1]
    
    # Normalize fractal complexity
    norm_complexity = (complexity - complexity.mean()) / (complexity.std() if complexity.std() != 0 else 1)
    
    # Extract weights from params
    weights = params.get('weights', {
        'macd_weight': 0.25, 
        'rsi_weight': 0.25, 
        'stoch_weight': 0.25, 
        'complexity_weight': 0.25, 
        'weekly_macd_weight': 0.25, 
        'weekly_rsi_weight': 0.25, 
        'weekly_stoch_weight': 0.25, 
        'weekly_complexity_weight': 0.25
    })

    # Combine indicators with weighted approach
    if is_weekly:
        composite = (weights.get('weekly_macd_weight', 0.25) * norm_macd + 
                    weights.get('weekly_rsi_weight', 0.25) * norm_rsi + 
                    weights.get('weekly_stoch_weight', 0.25) * norm_stoch + 
                    weights.get('weekly_complexity_weight', 0.25) * norm_complexity)
    else:
        composite = (weights.get('macd_weight', 0.25) * norm_macd + 
                    weights.get('rsi_weight', 0.25) * norm_rsi + 
                    weights.get('stoch_weight', 0.25) * norm_stoch + 
                    weights.get('complexity_weight', 0.25) * norm_complexity)

    composite = composite.fillna(0)
    
    # Calculate the rolling standard deviation for thresholds
    window = int(params.get('sell_rolling_std', 20) if is_weekly else params.get('buy_rolling_std', 20))
    rolling_std = composite.rolling(window=window, min_periods=5).std()
    rolling_std = rolling_std.fillna(composite.std())  # Use overall std for initial values
    
    # Ensure rolling_std is never zero to avoid division issues
    rolling_std = rolling_std.clip(lower=0.0001)
    
    # Calculate dynamic threshold lines
    if is_weekly:
        # Weekly thresholds are more conservative
        down_lim_line = -rolling_std * reactivity
        up_lim_line = rolling_std * reactivity
        # Add 2 std lines for weekly
        down_lim_line_2std = 2 * params.get('sell_down_lim', 2.0) * rolling_std * reactivity
        up_lim_line_2std = 2 * params.get('buy_up_lim', -2.0) * rolling_std * reactivity
    else:
        # Daily thresholds use parameter-based multipliers
        down_lim_line = params.get('sell_down_lim', 2.0) * rolling_std * reactivity
        up_lim_line = params.get('buy_up_lim', -2.0) * rolling_std * reactivity
        # Add 2 std lines for daily
        down_lim_line_2std = 2 * params.get('sell_down_lim', 2.0) * rolling_std * reactivity
        up_lim_line_2std = 2 * params.get('buy_up_lim', -2.0) * rolling_std * reactivity

    # Return DataFrame with all calculated values
    return pd.DataFrame({
        'Composite': composite,
        'Down_Lim': down_lim_line,
        'Up_Lim': up_lim_line,
        'Down_Lim_2STD': down_lim_line_2std,
        'Up_Lim_2STD': up_lim_line_2std
    }), composite, rolling_std

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
    logger.debug(f"Starting signal generation with {len(data)} data points")
    
    # Ensure index is datetime
    if data.empty:
        raise ValueError("No data available for signal generation")
        
    # Convert index to datetime if it's not already
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Calculate daily composite and thresholds (primary timeframe)
    daily_data, daily_composite, daily_std = calculate_composite_indicator(data, params, reactivity)
    
    # Calculate weekly composite (resampled timeframe)
    try:
        weekly_data = data.resample(default_interval_weekly).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()  # Remove any NaN rows
    except Exception as e:
        logger.error(f"Error resampling data: {str(e)}")
        logger.error(f"Data index type: {type(data.index)}")
        logger.error(f"Data index sample: {data.index[:5]}")
        raise
    
    # Verify we have enough data for weekly calculations
    min_weekly_bars = 20  # Minimum bars needed for reliable signals
    if len(weekly_data) < min_weekly_bars:
        logger.warning(f"Insufficient weekly data. Have {len(weekly_data)} bars, need {min_weekly_bars}")
    
    # Calculate weekly indicators
    weekly_indicators, weekly_composite, weekly_std = calculate_composite_indicator(
        weekly_data, params, reactivity, is_weekly=True, default_interval_weekly=default_interval_weekly
    )
    
    # Forward fill the weekly data to match the primary timeframe
    weekly_resampled = pd.DataFrame(index=data.index)
    for col in weekly_indicators.columns:
        weekly_resampled[col] = weekly_indicators[col].reindex(data.index, method='ffill')
    
    # Initialize signals DataFrame with zeros
    signals = pd.DataFrame(0, index=data.index, columns=[
        'signal', 'daily_composite', 'daily_down_lim', 'daily_up_lim', 
        'weekly_composite', 'weekly_down_lim', 'weekly_up_lim', 
        'weekly_down_lim_2std', 'weekly_up_lim_2std'
    ])
    
    # Assign values without chaining
    signals = signals.assign(
        daily_composite=daily_data['Composite'],
        daily_down_lim=daily_data['Down_Lim'],
        daily_up_lim=daily_data['Up_Lim'],
        daily_down_lim_2std=daily_data['Down_Lim_2STD'],
        daily_up_lim_2std=daily_data['Up_Lim_2STD'],
        weekly_composite=weekly_resampled['Composite'],
        weekly_down_lim=weekly_resampled['Down_Lim'],
        weekly_up_lim=weekly_resampled['Up_Lim'],
        weekly_down_lim_2std=weekly_resampled['Down_Lim_2STD'],
        weekly_up_lim_2std=weekly_resampled['Up_Lim_2STD']
    )
    
    # Generate buy signals (daily crossing above upper limit or crossing above -2std)
    buy_mask = ((daily_data['Composite'] > daily_data['Up_Lim']) & 
               (daily_data['Composite'].shift(1) <= daily_data['Up_Lim'].shift(1))) | \
              ((daily_data['Composite'] > daily_data['Up_Lim_2STD']) & 
               (daily_data['Composite'].shift(1) <= daily_data['Up_Lim_2STD'].shift(1)))
    signals.loc[buy_mask, 'signal'] = 1
    
    # Generate sell signals (weekly crossing below either upper or lower limit, or crossing below ±2std)
    sell_mask = ((weekly_resampled['Composite'] < weekly_resampled['Up_Lim']) & 
                (weekly_resampled['Composite'].shift(1) >= weekly_resampled['Up_Lim'].shift(1))) | \
                ((weekly_resampled['Composite'] < weekly_resampled['Down_Lim']) & 
                (weekly_resampled['Composite'].shift(1) >= weekly_resampled['Down_Lim'].shift(1))) | \
                ((weekly_resampled['Composite'] < weekly_resampled['Up_Lim_2STD']) & 
                (weekly_resampled['Composite'].shift(1) >= weekly_resampled['Up_Lim_2STD'].shift(1))) | \
                ((weekly_resampled['Composite'] < weekly_resampled['Down_Lim_2STD']) & 
                (weekly_resampled['Composite'].shift(1) >= weekly_resampled['Down_Lim_2STD'].shift(1)))
    signals.loc[sell_mask, 'signal'] = -1
    
    # Log final signal counts
    buy_count = (signals['signal'] == 1).sum()
    sell_count = (signals['signal'] == -1).sum()
    logger.debug(f"Generated {buy_count} buy signals and {sell_count} sell signals")
    
    return signals, daily_data, weekly_resampled


def get_default_params():
    """Get default parameters for indicator calculations"""
    return {
        'percent_increase_buy': 0.02,
        'percent_decrease_sell': 0.02,
        'sell_down_lim': 2.0,
        'sell_rolling_std': 20,
        'buy_up_lim': -2.0,
        'buy_rolling_std': 20,
        'rsi_period': 14,
        'stochastic_k_period': 14,
        'stochastic_d_period': 3,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'fractal_window': 100,  
        'fractal_lags': [10, 20, 40],  
        'reactivity': 1.0,
        'weights': {
            'macd_weight': 0.25,
            'rsi_weight': 0.25,
            'stoch_weight': 0.25,
            'complexity_weight': 0.25,
            'weekly_macd_weight': 0.25,
            'weekly_rsi_weight': 0.25,
            'weekly_stoch_weight': 0.25,
            'weekly_complexity_weight': 0.25
        }
    }