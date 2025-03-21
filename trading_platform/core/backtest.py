"""
Backtesting functionality for trading strategies
"""
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import itertools
import json

from .data_feed import fetch_historical_data, fetch_multi_symbol_data
from .indicators import generate_signals, get_default_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_market_hours(timestamp, market_hours):
    """
    Check if given timestamp is within market hours
    
    Args:
        timestamp: Datetime to check
        market_hours: Dictionary with market hours configuration
        
    Returns:
        Boolean indicating if timestamp is within market hours
    """
    # For crypto assets, market is always open
    if market_hours.get('always_open', False):
        return True
    
    # Get timezone from market hours or default to UTC
    timezone = market_hours.get('timezone', 'UTC')
    
    # Get start and end times
    start_time_str = market_hours.get('start', '00:00')
    end_time_str = market_hours.get('end', '23:59')
    
    # Parse times
    start_hour, start_minute = map(int, start_time_str.split(':'))
    end_hour, end_minute = map(int, end_time_str.split(':'))
    
    # Get current hour and minute
    current_hour = timestamp.hour
    current_minute = timestamp.minute
    
    # Get day of week (0 is Monday, 6 is Sunday)
    day_of_week = timestamp.weekday()
    
    # Check if weekend
    weekend_closed = market_hours.get('weekend_closed', True)
    if weekend_closed and day_of_week >= 5:  # 5,6 = Saturday, Sunday
        return False
    
    # Check time
    current_time = current_hour * 60 + current_minute
    start_time = start_hour * 60 + start_minute
    end_time = end_hour * 60 + end_minute
    
    # Return True if within market hours
    return start_time <= current_time <= end_time

def run_backtest(symbol: str,
                days: int = 15,
                params: Optional[Dict[str, Any]] = None,
                is_simulating: bool = False,
                lookback_days_param: int = 5,
                initial_capital: float = 100000.0) -> Dict[str, Any]:
    """
    Run a single backtest simulation for a given symbol and parameter set
    
    Args:
        symbol: Trading symbol
        days: Number of days to backtest
        params: Dictionary of parameters for backtesting
        is_simulating: Whether running in simulation mode
        lookback_days_param: Number of days to look back for performance calculation
        initial_capital: Initial capital amount
        
    Returns:
        Dictionary with backtest results
    """
    try:
        # Fetch historical data
        data = fetch_historical_data(symbol, '1h', days)
        
        if data.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Use default parameters if none provided
        if params is None:
            params = get_default_params()
        
        # Generate signals
        signals_df, daily_data, weekly_data = generate_signals(data, params)
        
        if signals_df.empty:
            return {"error": f"Failed to generate signals for {symbol}"}
        
        # Initialize variables for tracking portfolio
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Set up market hours
        market_hours = {
            'always_open': True,  # For crypto
            'timezone': 'UTC',
            'start': '00:00',
            'end': '23:59',
            'weekend_closed': False
        }
        
        # Multi-asset position sizing (only used if is_simulating=True)
        ranking_data = {}
        
        # Track portfolio value through time
        portfolio_values = pd.Series(index=signals_df.index, dtype=float)
        portfolio_values.iloc[0] = initial_capital
        
        # Iterate through signals
        for i in range(1, len(signals_df)):
            # Get current timestamp and check if market is open
            timestamp = signals_df.index[i]
            if not is_market_hours(timestamp, market_hours):
                portfolio_values.iloc[i] = portfolio_values.iloc[i-1]
                continue
            
            # Get current price and signals
            current_price = signals_df['price'].iloc[i]
            buy_signal = signals_df['buy_signal'].iloc[i]
            sell_signal = signals_df['sell_signal'].iloc[i]
            
            # Current portfolio value (before any new trades)
            current_value = capital + position * current_price
            
            # Performance-based position sizing for multi-asset simulations
            buy_size_pct = None
            sell_size_pct = None
            
            if is_simulating:
                # Get performance ranking at this timestamp
                rank, total_assets = get_historical_ranking(symbol, timestamp)
                
                if rank is not None:
                    # Calculate position sizing based on performance ranking
                    buy_size_pct = calculate_buy_percentage(rank, total_assets)
                    sell_size_pct = calculate_sell_percentage(rank, total_assets)
                    
                    ranking_data[timestamp.strftime('%Y-%m-%d %H:%M:%S')] = {
                        'rank': rank,
                        'total_assets': total_assets,
                        'buy_size': buy_size_pct,
                        'sell_size': sell_size_pct
                    }
            
            # Process buy signal
            if buy_signal and position == 0:
                # Calculate position size
                if buy_size_pct is not None:
                    # Use performance-based sizing
                    position_size = (capital * buy_size_pct) / current_price
                else:
                    # Use fixed position sizing (50% of capital)
                    position_size = (capital * 0.5) / current_price
                
                # Execute trade
                cost = position_size * current_price
                position = position_size
                capital -= cost
                entry_price = current_price
                
                # Record trade
                trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': current_price,
                    'size': position_size,
                    'cost': cost,
                    'capital': capital,
                    'position_value': position * current_price,
                    'total_value': capital + position * current_price
                })
                
            # Process sell signal
            elif sell_signal and position > 0:
                # Calculate position size to sell
                if sell_size_pct is not None:
                    # Use performance-based sizing
                    position_size = position * sell_size_pct
                else:
                    # Sell entire position
                    position_size = position
                
                # Execute trade
                revenue = position_size * current_price
                position -= position_size
                capital += revenue
                
                # Calculate profit/loss
                pl_pct = (current_price / entry_price - 1) * 100
                
                # Record trade
                trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': current_price,
                    'size': position_size,
                    'revenue': revenue,
                    'capital': capital,
                    'position_value': position * current_price,
                    'total_value': capital + position * current_price,
                    'pl_pct': pl_pct
                })
                
                # Reset entry price if position is fully closed
                if position == 0:
                    entry_price = 0
            
            # Update portfolio value
            portfolio_values.iloc[i] = capital + position * current_price
            
            # Record equity curve point
            equity_curve.append({
                'timestamp': timestamp,
                'price': current_price,
                'capital': capital,
                'position': position,
                'position_value': position * current_price,
                'total_value': capital + position * current_price
            })
        
        # Close any open position at the end
        final_timestamp = signals_df.index[-1]
        final_price = signals_df['price'].iloc[-1]
        final_value = capital
        
        if position > 0:
            revenue = position * final_price
            pl_pct = (final_price / entry_price - 1) * 100
            
            trades.append({
                'timestamp': final_timestamp,
                'action': 'CLOSE',
                'price': final_price,
                'size': position,
                'revenue': revenue,
                'capital': capital + revenue,
                'position_value': 0,
                'total_value': capital + revenue,
                'pl_pct': pl_pct
            })
            
            final_value = capital + revenue
        
        # Calculate statistics
        initial_price = signals_df['price'].iloc[0]
        final_price = signals_df['price'].iloc[-1]
        
        # Buy and hold return
        buy_hold_return = (final_price / initial_price - 1) * 100
        
        # Strategy return
        strategy_return = (final_value / initial_capital - 1) * 100
        
        # Calculate drawdown
        portfolio_values = portfolio_values.fillna(method='ffill')
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Number of trades
        num_trades = len([t for t in trades if t['action'] in ['BUY', 'SELL']])
        
        # Winning trades
        winning_trades = [t for t in trades if t['action'] == 'SELL' and t.get('pl_pct', 0) > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        # Calculate Sharpe ratio (annualized)
        if len(portfolio_values) > 1:
            returns = portfolio_values.pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365 * 24 / days)
        else:
            sharpe_ratio = 0
        
        # Prepare result
        result = {
            'symbol': symbol,
            'days': days,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': strategy_return - buy_hold_return,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades[-10:],  # Only include most recent trades
            'equity_curve': equity_curve[-100:],  # Only include most recent equity curve points
            'params': params
        }
        
        # Add ranking data for simulations
        if is_simulating and ranking_data:
            result['ranking_data'] = ranking_data
        
        return result
        
    except Exception as e:
        logger.error(f"Error in run_backtest for {symbol}: {str(e)}")
        return {"error": str(e)}

def find_best_params(symbol: str,
                    param_grid: Dict[str, List[Any]],
                    days: int = 15,
                    output_file: str = "best_params.json") -> Dict[str, Any]:
    """
    Find the best parameter set by running a backtest for each combination.
    
    Args:
        symbol: Trading symbol
        param_grid: Dictionary of parameter names to lists of parameter values
        days: Number of days to backtest
        output_file: Name of file to save results
        
    Returns:
        Dictionary with best parameters and backtest results
    """
    try:
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations for {symbol}")
        
        # Run backtest for each combination
        results = []
        
        for i, combo in enumerate(combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combo))
            
            # Run backtest
            result = run_backtest(symbol, days, params)
            
            # Check for errors
            if "error" in result:
                logger.warning(f"Error in combination {i+1}/{len(combinations)}: {result['error']}")
                continue
            
            # Store result
            results.append({
                'params': params,
                'return': result['return'],
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result['sharpe_ratio']
            })
            
            # Log progress
            if (i+1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(combinations)} combinations")
        
        # Find best parameters
        if not results:
            return {"error": "No valid parameter combinations found"}
        
        # Sort by return, drawdown, and Sharpe ratio
        # First prioritize Sharpe ratio
        sharpe_sorted = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        top_sharpe = sharpe_sorted[:10]
        
        # Among top Sharpe ratios, prioritize return and drawdown
        final_sorted = sorted(top_sharpe, key=lambda x: x['return'] * 0.7 - abs(x['max_drawdown']) * 0.3, reverse=True)
        
        best_result = final_sorted[0]
        
        # Run a final backtest with the best parameters
        final_backtest = run_backtest(symbol, days, best_result['params'])
        
        # Save best parameters to file
        try:
            with open(output_file, 'w') as f:
                json.dump({symbol: best_result['params']}, f, indent=4)
            logger.info(f"Best parameters saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving parameters to file: {str(e)}")
        
        # Return results
        return {
            'symbol': symbol,
            'best_params': best_result['params'],
            'backtest_result': final_backtest,
            'all_results': results
        }
        
    except Exception as e:
        logger.error(f"Error in find_best_params for {symbol}: {str(e)}")
        return {"error": str(e)}

def calculate_performance_ranking(prices_dataset=None, timestamp=None, lookback_days=15):
    """
    Calculate performance ranking across assets for a specific point in time.
    
    Args:
        prices_dataset: Dictionary of symbol -> DataFrame with price data
        timestamp: The timestamp to calculate ranking for (if None, use latest data)
        lookback_days: Number of days to look back for performance calculation
    
    Returns:
        DataFrame with performance and rank data
    """
    try:
        # If no prices dataset provided, fetch data for top cryptocurrencies
        if prices_dataset is None:
            from .data_feed import get_available_symbols
            symbols = get_available_symbols()
            prices_dataset = fetch_multi_symbol_data(symbols, '1d', lookback_days)
        
        # Initialize results
        performances = []
        
        # Process each symbol
        for symbol, data in prices_dataset.items():
            if data.empty:
                continue
            
            # Determine timestamp to use
            if timestamp is None:
                ts = data.index[-1]  # Use latest timestamp
            else:
                # Find closest timestamp
                ts = data.index[data.index.get_indexer([timestamp], method='nearest')[0]]
            
            # Calculate lookback timestamp
            lookback_ts = ts - pd.Timedelta(days=lookback_days)
            
            # Get data in range
            filtered_data = data[(data.index >= lookback_ts) & (data.index <= ts)]
            
            if len(filtered_data) < 2:
                continue
            
            # Get prices
            start_price = filtered_data['close'].iloc[0]
            end_price = filtered_data['close'].iloc[-1]
            
            # Calculate performance
            performance = (end_price / start_price - 1) * 100
            
            # Add to results
            performances.append({
                'symbol': symbol,
                'timestamp': ts,
                'start_price': start_price,
                'end_price': end_price,
                'performance': performance
            })
        
        # Convert to DataFrame
        if not performances:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(performances)
        
        # Calculate ranks (1 = best performer)
        result_df['rank'] = result_df['performance'].rank(ascending=False)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in calculate_performance_ranking: {str(e)}")
        return pd.DataFrame()

def get_historical_ranking(symbol, timestamp):
    """
    Get performance ranking for a specific symbol at a specific timestamp.
    
    Args:
        symbol: The symbol to get rank for
        timestamp: The timestamp to calculate ranking for
    
    Returns:
        tuple: (rank, performance) or (None, None) if not available
    """
    try:
        # Fetch data and calculate ranking
        from .data_feed import get_available_symbols
        symbols = get_available_symbols()
        
        # Limit lookback days to avoid excessive data fetching
        lookback_days = 5
        
        # Fetch data
        prices_dataset = fetch_multi_symbol_data(symbols, '1d', lookback_days)
        
        # Calculate ranking
        ranking_df = calculate_performance_ranking(prices_dataset, timestamp, lookback_days)
        
        if ranking_df.empty or symbol not in ranking_df['symbol'].values:
            return None, None
        
        # Get rank for the symbol
        symbol_rank = ranking_df.loc[ranking_df['symbol'] == symbol, 'rank'].values[0]
        total_assets = len(ranking_df)
        
        return int(symbol_rank), total_assets
        
    except Exception as e:
        logger.error(f"Error in get_historical_ranking for {symbol}: {str(e)}")
        return None, None

def calculate_buy_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate buy percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.0 and 0.2 representing buy percentage
    """
    # Top performers get higher allocation
    # rank 1 (best) = 0.2, rank = total (worst) = 0.01
    if rank <= 0 or total_assets <= 1:
        return 0.1  # default value
    
    normalized_rank = rank / total_assets
    
    # Exponential decay formula to favor top performers
    buy_pct = 0.20 * np.exp(-2.5 * normalized_rank) + 0.01
    
    # Ensure result is between 0.01 and 0.2
    return min(max(buy_pct, 0.01), 0.2)

def calculate_sell_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate sell percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.1 and 1.0 representing sell percentage
    """
    # Worst performers get higher sell percentage
    # rank 1 (best) = 0.1, rank = total (worst) = 1.0
    if rank <= 0 or total_assets <= 1:
        return 0.5  # default value
    
    normalized_rank = rank / total_assets
    
    # Linear scaling formula to sell more of poor performers
    sell_pct = 0.1 + 0.9 * normalized_rank
    
    # Ensure result is between 0.1 and 1.0
    return min(max(sell_pct, 0.1), 1.0)