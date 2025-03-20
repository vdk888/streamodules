import pandas as pd
import numpy as np
import json
import streamlit as st
from typing import Dict, Optional, List, Tuple, Any

# Import backtest functions from backtest_individual.py
from attached_assets.backtest_individual import run_backtest as run_individual_backtest
from attached_assets.backtest_individual import find_best_params as find_optimal_params

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
    try:
        # Use the run_backtest function from backtest_individual.py
        backtest_result = run_individual_backtest(
            symbol=symbol, 
            days=days,
            params=params,
            is_simulating=is_simulating,
            lookback_days_param=lookback_days_param
        )
        return backtest_result
    except Exception as e:
        st.error(f"Error during backtest: {str(e)}")
        return None

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
    try:
        # Use the find_best_params function from backtest_individual.py
        best_params = find_optimal_params(symbol, param_grid, days=days)
        return best_params
    except Exception as e:
        st.error(f"Error during parameter optimization: {str(e)}")
        return None

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
    # Create a dictionary to store performance metrics
    performance_metrics = {}
    
    # Calculate the start time from the lookback period
    start_time = current_time - pd.Timedelta(days=lookback_days_param)
    
    # Iterate through each symbol and calculate performance
    for symbol, data in prices_dataset.items():
        try:
            # Filter the data to the lookback period
            mask = (data.index >= start_time) & (data.index <= current_time)
            filtered_data = data.loc[mask]
            
            if len(filtered_data) < 2:
                # Not enough data points
                continue
            
            # Calculate the simple performance (percentage change)
            start_price = filtered_data['close'].iloc[0]
            end_price = filtered_data['close'].iloc[-1]
            performance = (end_price - start_price) / start_price
            
            # Store the performance
            performance_metrics[symbol] = performance
        except Exception as e:
            # Skip this symbol on error
            print(f"Error calculating performance for {symbol}: {str(e)}")
            continue
    
    if not performance_metrics:
        # No valid metrics calculated
        return None
    
    # Create a DataFrame from the metrics
    df = pd.DataFrame.from_dict(performance_metrics, orient='index', columns=['performance'])
    
    # Calculate percentile ranks (0-1 scale, higher is better)
    df['rank'] = df['performance'].rank(method='min') / len(df)
    
    return df

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
    if not prices_dataset:
        return None
    
    # Calculate the current time (most recent data point across all datasets)
    current_time = max([data.index[-1] for data in prices_dataset.values()])
    
    # Use the backtest ranking function
    return backtest_calculate_ranking(prices_dataset, current_time, lookback_days)