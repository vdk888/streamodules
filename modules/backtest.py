import pandas as pd
import numpy as np
import datetime
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any
import json

from utils import fetch_and_process_data, generate_signals_with_indicators
from modules.storage import save_best_params
from attached_assets.indicators import get_default_params

def run_backtest(symbol: str, 
                days: int = 5, 
                params: Optional[Dict] = None, 
                is_simulating: bool = False,
                lookback_days_param: int = 5) -> Optional[Dict]:
    """
    Run a backtest for a given symbol and parameters
    
    Args:
        symbol: Trading symbol
        days: Number of days to backtest
        params: Parameters for the backtest
        is_simulating: Whether this is a simulation or real backtest
        lookback_days_param: Number of days to look back for performance calculations
        
    Returns:
        Dictionary with backtest results
    """
    # Use default parameters if none provided
    if params is None:
        params = get_default_params()
    
    # Fetch historical data
    timeframe = "1h"  # Default timeframe
    data, error = fetch_and_process_data(symbol, timeframe, days)
    
    if error or data is None or data.empty:
        st.error(f"Error fetching data for backtest: {error}")
        return None
    
    # Generate signals
    signals_df, daily_composite, weekly_composite = generate_signals_with_indicators(data, params)
    
    if signals_df is None:
        st.error("Failed to generate signals for backtest")
        return None
    
    # Initialize portfolio DataFrame
    portfolio = signals_df.copy()
    
    # Make sure 'close' column exists in portfolio
    if 'close' not in portfolio.columns:
        # Add close price from original data to portfolio DataFrame
        if 'close' in data.columns:
            common_dates = portfolio.index.intersection(data.index)
            portfolio.loc[common_dates, 'close'] = data.loc[common_dates, 'close']
        else:
            st.error("Error: Price data does not contain 'close' column")
            return None
    
    portfolio['position'] = 0.0  # Number of shares/units held
    portfolio['cash'] = 100000.0  # Initial cash
    portfolio['position_value'] = 0.0  # Value of position
    portfolio['portfolio_value'] = portfolio['cash']  # Total portfolio value
    portfolio['returns'] = 0.0  # Daily returns
    portfolio['drawdown'] = 0.0  # Drawdown from peak
    
    # Trading cost as a percentage
    trading_cost_percentage = 0.0025  # 0.25% per trade
    
    # Value to track the peak portfolio value for drawdown calculation
    peak_value = portfolio['cash'].iloc[0]
    
    # Variables to track trades
    trades = []
    
    # Process each day
    for i in range(1, len(portfolio)):
        # Get current day and previous day
        current_day = portfolio.index[i]
        prev_day = portfolio.index[i-1]
        
        # Carry over position from previous day
        portfolio.loc[current_day, 'position'] = portfolio.loc[prev_day, 'position']
        portfolio.loc[current_day, 'cash'] = portfolio.loc[prev_day, 'cash']
        
        # Process buy signal
        if portfolio.loc[current_day, 'signal'] == 1:
            # Calculate buy amount as 20% of available cash
            buy_percentage = 0.2
            buy_amount = portfolio.loc[current_day, 'cash'] * buy_percentage
            
            # Calculate trading cost
            trading_cost = buy_amount * trading_cost_percentage
            buy_amount -= trading_cost
            
            # Update position and cash
            new_position = portfolio.loc[current_day, 'position'] + (buy_amount / portfolio.loc[current_day, 'close'])
            new_cash = portfolio.loc[current_day, 'cash'] - buy_amount - trading_cost
            
            portfolio.loc[current_day, 'position'] = new_position
            portfolio.loc[current_day, 'cash'] = new_cash
            
            # Record trade
            trades.append({
                'date': current_day,
                'type': 'buy',
                'price': portfolio.loc[current_day, 'close'],
                'amount': buy_amount,
                'units': buy_amount / portfolio.loc[current_day, 'close'],
                'fee': trading_cost
            })
        
        # Process sell signal
        elif portfolio.loc[current_day, 'signal'] == -1 and portfolio.loc[current_day, 'position'] > 0:
            # Calculate sell units as 50% of current position
            sell_percentage = 0.5
            sell_units = portfolio.loc[current_day, 'position'] * sell_percentage
            sell_amount = sell_units * portfolio.loc[current_day, 'close']
            
            # Calculate trading cost
            trading_cost = sell_amount * trading_cost_percentage
            sell_amount -= trading_cost
            
            # Update position and cash
            new_position = portfolio.loc[current_day, 'position'] - sell_units
            new_cash = portfolio.loc[current_day, 'cash'] + sell_amount
            
            portfolio.loc[current_day, 'position'] = new_position
            portfolio.loc[current_day, 'cash'] = new_cash
            
            # Record trade
            trades.append({
                'date': current_day,
                'type': 'sell',
                'price': portfolio.loc[current_day, 'close'],
                'amount': sell_amount,
                'units': sell_units,
                'fee': trading_cost
            })
        
        # Calculate position value and portfolio value
        portfolio.loc[current_day, 'position_value'] = portfolio.loc[current_day, 'position'] * portfolio.loc[current_day, 'close']
        portfolio.loc[current_day, 'portfolio_value'] = portfolio.loc[current_day, 'cash'] + portfolio.loc[current_day, 'position_value']
        
        # Calculate daily returns
        prev_value = portfolio.loc[prev_day, 'portfolio_value']
        current_value = portfolio.loc[current_day, 'portfolio_value']
        portfolio.loc[current_day, 'returns'] = (current_value / prev_value) - 1 if prev_value > 0 else 0
        
        # Update peak value and calculate drawdown
        if current_value > peak_value:
            peak_value = current_value
        
        if peak_value > 0:
            portfolio.loc[current_day, 'drawdown'] = (peak_value - current_value) / peak_value
    
    # Calculate performance metrics
    initial_value = portfolio['portfolio_value'].iloc[0]
    final_value = portfolio['portfolio_value'].iloc[-1]
    
    # Calculate total return
    total_return = ((final_value / initial_value) - 1) * 100
    
    # Calculate max drawdown
    max_drawdown = portfolio['drawdown'].max() * 100
    
    # Calculate Sharpe ratio (assuming risk-free rate = 0)
    daily_returns = portfolio['returns']
    annualized_return = daily_returns.mean() * 252  # 252 trading days in a year
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Calculate win rate
    num_trades = len(trades)
    win_count = 0
    
    # For each sell trade, check if it was profitable
    for trade in trades:
        if trade['type'] == 'sell':
            # Find the most recent buy trade before this sell
            buy_trades = [t for t in trades if t['type'] == 'buy' and t['date'] < trade['date']]
            if buy_trades:
                latest_buy = max(buy_trades, key=lambda x: x['date'])
                if trade['price'] > latest_buy['price']:
                    win_count += 1
    
    win_rate = (win_count / num_trades) * 100 if num_trades > 0 else 0
    
    # Prepare results dictionary
    results = {
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'parameters': params,
        'portfolio': portfolio,
        'trades': trades,
        'metrics': {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'final_value': final_value
        }
    }
    
    return results

def find_best_params(symbol: str, 
                    param_grid: Dict[str, List], 
                    days: int = 5) -> Optional[Dict]:
    """
    Find the best parameters for a given symbol
    
    Args:
        symbol: Trading symbol
        param_grid: Grid of parameters to search
        days: Number of days to use for the parameter search
        
    Returns:
        Dictionary with the best parameters
    """
    # Create a progress bar
    progress_text = f"Optimizing parameters for {symbol}..."
    progress_bar = st.progress(0)
    
    # Generate all possible parameter combinations
    from itertools import product
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Track best parameters and results
    best_return = -float('inf')
    best_params = None
    best_result = None
    
    # Test each parameter combination
    total_combinations = len(param_combinations)
    for i, values in enumerate(param_combinations):
        # Update progress bar
        progress = (i + 1) / total_combinations
        progress_bar.progress(progress)
        
        # Create parameters dictionary
        params = get_default_params()
        for j, key in enumerate(param_keys):
            params[key] = values[j]
        
        # Run backtest with these parameters
        result = run_backtest(symbol, days=days, params=params)
        
        # Check if this is the best result so far
        if result and result['metrics']['total_return'] > best_return:
            best_return = result['metrics']['total_return']
            best_params = params.copy()
            best_result = result
    
    # Clear progress bar
    progress_bar.empty()
    
    # Save best parameters if found
    if best_params:
        save_best_params(symbol, best_params)
        
        # Show a success message
        st.success(f"Found best parameters for {symbol}: {best_params}")
        st.write(f"Total Return: {best_result['metrics']['total_return']:.2f}%")
        st.write(f"Max Drawdown: {best_result['metrics']['max_drawdown']:.2f}%")
        st.write(f"Sharpe Ratio: {best_result['metrics']['sharpe_ratio']:.2f}")
        st.write(f"Win Rate: {best_result['metrics']['win_rate']:.2f}%")
    else:
        st.warning(f"Could not find optimal parameters for {symbol}")
    
    return best_params

def backtest_calculate_ranking(prices_dataset: Dict[str, pd.DataFrame], 
                              current_time: pd.Timestamp, 
                              lookback_days_param: int) -> pd.DataFrame:
    """
    Calculate performance ranking of all symbols over the last N days,
    using the same logic as in backtest_individual.py
    
    Args:
        prices_dataset: Dictionary of price dataframes keyed by symbol
        current_time: Current time to use as the reference point
        lookback_days_param: Number of days to look back
        
    Returns:
        DataFrame with performance and rank columns
    """
    performances = []
    
    # Calculate lookback start time
    lookback_delta = pd.Timedelta(days=lookback_days_param)
    lookback_start_time = current_time - lookback_delta
    
    # Calculate performance for each symbol
    for symbol, df in prices_dataset.items():
        # Filter data within lookback period
        mask = (df.index >= lookback_start_time) & (df.index <= current_time)
        lookback_df = df[mask]
        
        if not lookback_df.empty and len(lookback_df) > 1:
            start_price = lookback_df['close'].iloc[0]
            end_price = lookback_df['close'].iloc[-1]
            
            # Calculate simple price performance
            performance = (end_price / start_price) - 1
            
            performances.append({
                'symbol': symbol,
                'performance': performance
            })
    
    # Create DataFrame with all performances
    if not performances:
        return pd.DataFrame()
    
    performance_df = pd.DataFrame(performances)
    performance_df.set_index('symbol', inplace=True)
    
    # Add rank column (0-1 scale, 1 being best)
    performance_df['rank'] = performance_df['performance'].rank(pct=True)
    
    return performance_df

def calculate_performance_ranking(prices_dataset: Optional[Dict[str, pd.DataFrame]] = None, 
                                 lookback_days: int = 15) -> pd.DataFrame:
    """
    Calculate simple performance ranking across assets for display
    
    Args:
        prices_dataset: Dictionary of price dataframes keyed by symbol
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with performance metrics
    """
    if prices_dataset is None or not prices_dataset:
        # Fetch data for all symbols if not provided
        from attached_assets.config import TRADING_SYMBOLS
        
        prices_dataset = {}
        for symbol, config in TRADING_SYMBOLS.items():
            data, error = fetch_and_process_data(symbol, "1h", lookback_days)
            if not error and data is not None and not data.empty:
                prices_dataset[symbol] = data
    
    if not prices_dataset:
        return pd.DataFrame()
    
    # Get current time (use the latest timestamp from the first DataFrame)
    first_df = next(iter(prices_dataset.values()))
    current_time = first_df.index[-1]
    
    # Calculate performance ranking
    return backtest_calculate_ranking(prices_dataset, current_time, lookback_days)