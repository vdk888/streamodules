import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default trading parameters
DEFAULT_RISK_PERCENT = 0.95
DEFAULT_INTERVAL = '1h'  # available intervals: 1h, 4h, 1d
DEFAULT_INTERVAL_WEEKLY = '4h'
default_interval_yahoo = '1h'  # available intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d
default_backtest_interval = 15  # Default number of days for backtesting
lookback_days_param = 5  # Default lookback for performance ranking

# Bars per day for each interval
BARS_PER_DAY = {
    '1m': 1440,
    '5m': 288,
    '15m': 96,
    '30m': 48,
    '60m': 24,
    '1h': 24,
    '1d': 1
}

# Update intervals in seconds for each timeframe
UPDATE_INTERVALS = {
    '1m': 60,    # Check every minute
    '5m': 300,   # Check every 5 minutes
    '15m': 900,  # Check every 15 minutes
    '30m': 1800, # Check every 30 minutes
    '60m': 3600, # Check every hour
    '1h': 3600,  # Check every hour
    '1d': 86400  # Check every day
}

# Get update interval based on timeframe
def get_update_interval(timeframe: str) -> int:
    """Get the appropriate update interval in seconds for a given timeframe"""
    return UPDATE_INTERVALS.get(timeframe, 3600)  # Default to 1 hour if timeframe not found

# Maximum data points per request
MAX_DATA_POINTS = 2000

# Calculate maximum number of days for a given interval
def get_max_days(interval: str) -> int:
    """
    Calculate maximum number of days for a given interval
    
    Args:
        interval: Data interval
    
    Returns:
        Maximum number of days
    """
    bars_per_day = BARS_PER_DAY.get(interval, 24)  # Default to 24 bars/day
    max_days = MAX_DATA_POINTS // bars_per_day
    if interval == '1h':
        return min(730, max_days)
    return min(60, max_days)

# Interval to maximum days mapping
INTERVAL_MAX_DAYS = {interval: get_max_days(interval) for interval in BARS_PER_DAY}

# Trading costs configuration
TRADING_COSTS = {
    'DEFAULT': {
        'trading_fee': 0.001,  # 0.1% trading fee
        'spread': 0.006,  # 0.6% spread (bid-ask)
    },
    'CRYPTO': {
        'trading_fee': 0.003,  # 0.3% taker fee
        'spread': 0.002,  # 0.2% maker fee
    },
    'STOCK': {
        'trading_fee': 0.0005,  # 0.05% trading fee
        'spread': 0.001,  # 0.1% spread (bid-ask)
    }
}

# Parameter grid for optimization
param_grid = {
    'percent_increase_buy': [0.02],
    'percent_decrease_sell': [0.02],
    'sell_down_lim': [2.0],
    'sell_rolling_std': [20],
    'buy_up_lim': [-2.0],
    'buy_rolling_std': [20],
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],
    'rsi_period': [14],
    'stochastic_k_period': [14],
    'stochastic_d_period': [3],
    'fractal_window': [50, 100, 150],
    'fractal_lags': [[5, 10, 20], [10, 20, 40], [15, 30, 60]],
    'reactivity': [0.8, 0.9, 1.0, 1.1, 1.2],
    'weights': [
        {'weekly_macd_weight': 0.1, 'weekly_rsi_weight': 0.1, 'weekly_stoch_weight': 0.1, 'weekly_complexity_weight': 0.7,
         'macd_weight': 0.1, 'rsi_weight': 0.1, 'stoch_weight': 0.1, 'complexity_weight': 0.7},
        {'weekly_macd_weight': 0.15, 'weekly_rsi_weight': 0.15, 'weekly_stoch_weight': 0.15, 'weekly_complexity_weight': 0.55,
         'macd_weight': 0.15, 'rsi_weight': 0.15, 'stoch_weight': 0.15, 'complexity_weight': 0.55},
        {'weekly_macd_weight': 0.2, 'weekly_rsi_weight': 0.2, 'weekly_stoch_weight': 0.2, 'weekly_complexity_weight': 0.4,
         'macd_weight': 0.2, 'rsi_weight': 0.2, 'stoch_weight': 0.2, 'complexity_weight': 0.4},
        {'weekly_macd_weight': 0.25, 'weekly_rsi_weight': 0.25, 'weekly_stoch_weight': 0.25, 'weekly_complexity_weight': 0.25,
         'macd_weight': 0.25, 'rsi_weight': 0.25, 'stoch_weight': 0.25, 'complexity_weight': 0.25},
        {'weekly_macd_weight': 0.3, 'weekly_rsi_weight': 0.3, 'weekly_stoch_weight': 0.3, 'weekly_complexity_weight': 0.1,
         'macd_weight': 0.3, 'rsi_weight': 0.3, 'stoch_weight': 0.3, 'complexity_weight': 0.1},
        {'weekly_macd_weight': 0.3, 'weekly_rsi_weight': 0.3, 'weekly_stoch_weight': 0.3, 'weekly_complexity_weight': 0.1,
         'macd_weight': 0.1, 'rsi_weight': 0.1, 'stoch_weight': 0.1, 'complexity_weight': 0.7},
    ]
}

def calculate_capital_multiplier(lookback_days=default_backtest_interval/2):
    """
    Calculate a dynamic capital multiplier based on asset performance.
    
    Args:
        lookback_days: Number of days to look back for performance calculation
        
    Returns:
        float: Capital multiplier between 0.5 and 3.0
    """
    # Default multiplier if calculation fails
    default_multiplier = 1.5
    
    try:
        import yfinance as yf
        from ..modules.crypto.config import TRADING_SYMBOLS
        
        # Get end and start dates
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=lookback_days * 2)  # Double lookback for MA calculation
        
        # Collect daily performance data for all assets
        daily_performances = []
        
        for symbol, config in TRADING_SYMBOLS.items():
            try:
                # Get the yfinance symbol
                yf_symbol = config['yfinance']
                
                # Fetch historical data
                ticker = yf.Ticker(yf_symbol)
                data = ticker.history(start=start_date, end=end_date, interval=default_interval_yahoo)
                
                if len(data) >= 10:  # Need enough data for meaningful calculation
                    # Calculate daily returns
                    data['return'] = data['Close'].pct_change() * 100  # as percentage
                    
                    # Add to our collection, dropping NaN values
                    returns = data['return'].dropna().values
                    if len(returns) > 0:
                        daily_performances.append(returns)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not daily_performances or len(daily_performances) < 3:
            return default_multiplier
        
        # Calculate average daily performance across all assets for each day
        # Pad or truncate arrays to ensure they have the same length
        min_length = min(len(perfs) for perfs in daily_performances)
        if min_length < 10:  # Need enough data points
            return default_multiplier
            
        aligned_performances = [perfs[-min_length:] for perfs in daily_performances]
        daily_avg_performance = np.mean(aligned_performances, axis=0)
        
        # Calculate moving average with a window of 7 days
        window = min(7, len(daily_avg_performance)//2)
        if window < 3:
            return default_multiplier
            
        # Calculate moving average
        ma = np.convolve(daily_avg_performance, np.ones(window)/window, mode='valid')
        
        if len(ma) < 2:
            return default_multiplier
        
        # Get current performance (average of last 3 days) and MA
        current_perf = np.mean(daily_avg_performance[-3:])
        current_ma = ma[-1]
        
        # Calculate differences between performance and MA over time
        diffs = daily_avg_performance[-len(ma):] - ma
        
        # Calculate standard deviation of these differences
        std_diff = np.std(diffs)
        if std_diff == 0:
            std_diff = 0.1  # Avoid division by zero
        
        # Calculate current difference
        current_diff = current_perf - current_ma
        
        # Normalize the difference with bounds at -2std and 2std
        normalized_diff = max(min(current_diff / (2 * std_diff), 2), -2)
        
        # Apply sigmoid function to get a value between 0 and 1
        sigmoid = 1 / (1 + np.exp(-normalized_diff))
        
        # Scale to range [0.5, 3.0]
        multiplier = 0.5 + 2.5 * sigmoid
        
        logger.debug(f"Capital multiplier calculation:")
        logger.debug(f"Current performance: {current_perf:.2f}%")
        logger.debug(f"Moving average: {current_ma:.2f}%")
        logger.debug(f"Difference: {current_diff:.2f}%")
        logger.debug(f"Std of differences: {std_diff:.2f}")
        logger.debug(f"Normalized diff: {normalized_diff:.2f}")
        logger.debug(f"Final multiplier: {multiplier:.2f}x")
        
        return multiplier
        
    except Exception as e:
        logger.error(f"Error calculating capital multiplier: {str(e)}")
        return default_multiplier

# Trading symbols will be defined in each module's config file (e.g., crypto/config.py)
TRADING_SYMBOLS = {}