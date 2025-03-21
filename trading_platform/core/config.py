"""
Core configuration settings for the trading platform
"""
from typing import Dict, List, Any

# Time-related constants
default_interval_yahoo = '1h'
default_interval_weekly = '4h'
default_backtest_interval = 15  # days

# Parameter grid for optimization
param_grid = {
    'rsi_length': [7, 14, 21],
    'macd_fast': [8, 12, 16],
    'macd_slow': [21, 26, 30],
    'stoch_k': [9, 14, 20],
    'stoch_d': [3, 5, 7],
    'volatility_window': [10, 20, 30]
}

# Trading costs
default_trading_costs = {
    'trading_fee': 0.001,  # 0.1% per trade
    'slippage': 0.001,     # 0.1% slippage
}

# Default risk parameters
default_risk_parameters = {
    'max_position_size': 0.25,  # Maximum position size as fraction of portfolio
    'stop_loss': 0.05,          # Stop loss as fraction of entry price
    'take_profit': 0.15,        # Take profit as fraction of entry price
    'max_drawdown': 0.25,       # Maximum allowed drawdown
}

def get_update_interval(timeframe: str) -> int:
    """Get the appropriate update interval in seconds for a given timeframe"""
    intervals = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800
    }
    return intervals.get(timeframe, 3600)  # Default to 1h if not found

def get_max_days(interval: str) -> int:
    """
    Calculate maximum number of days for a given interval
    
    Args:
        interval: Data interval
    
    Returns:
        Maximum number of days
    """
    interval_limits = {
        '1m': 7,
        '5m': 30,
        '15m': 30,
        '30m': 60,
        '1h': 730,
        '4h': 730,
        '1d': 1825,
        '1w': 1825
    }
    return interval_limits.get(interval, 30)  # Default to 30 days if not found

def calculate_capital_multiplier(lookback_days=default_backtest_interval/2):
    """
    Calculate a dynamic capital multiplier based on asset performance.
    
    Args:
        lookback_days: Number of days to look back for performance calculation
        
    Returns:
        float: Capital multiplier between 0.5 and 3.0
    """
    # This would typically involve analyzing asset volatility, correlation, etc.
    # For simplicity, we're returning a default value
    return 1.0