"""
Core configuration settings for the trading platform
"""
from typing import Dict, List, Optional, Any, Union

# Default intervals and parameters
default_interval_yahoo = '1h'  # Default Yahoo Finance interval
default_lookback_days = 15     # Default number of days to look back
default_backtest_interval = 15  # Default number of days for backtesting

# Parameter grid for optimization
param_grid = {
    'rsi_period': [7, 10, 14, 21],
    'macd_fast': [8, 12, 16],
    'macd_slow': [20, 26, 30],
    'macd_signal': [7, 9, 11],
    'stoch_k': [9, 14, 21],
    'stoch_d': [3, 5, 7],
    'hurst_lags': [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64]],
    'volatility_window': [10, 15, 20, 30]
}

# Timeframe to update interval mapping (in seconds)
UPDATE_INTERVALS = {
    '1m': 60,
    '5m': 300,
    '15m': 900,
    '30m': 1800,
    '1h': 3600,
    '4h': 14400,
    '1d': 86400
}

# Max data days per interval
MAX_DAYS = {
    '1m': 1,
    '5m': 5,
    '15m': 7,
    '30m': 14,
    '1h': 30,
    '4h': 60,
    '1d': 365
}

def get_update_interval(timeframe: str) -> int:
    """Get the appropriate update interval in seconds for a given timeframe"""
    return UPDATE_INTERVALS.get(timeframe, 3600)  # Default to 1 hour

def get_max_days(interval: str) -> int:
    """
    Calculate maximum number of days for a given interval
    
    Args:
        interval: Data interval
    
    Returns:
        Maximum number of days
    """
    return MAX_DAYS.get(interval, 30)  # Default to 30 days

def calculate_capital_multiplier(lookback_days=default_backtest_interval/2):
    """
    Calculate a dynamic capital multiplier based on asset performance.
    
    Args:
        lookback_days: Number of days to look back for performance calculation
        
    Returns:
        float: Capital multiplier between 0.5 and 3.0
    """
    # In a real implementation, this would analyze market conditions
    # and adjust capital allocation accordingly
    
    # For now, return a default value
    return 1.0