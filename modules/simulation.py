import pandas as pd
import numpy as np
import datetime
from typing import Dict, Optional, List, Tuple

def calculate_buy_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate buy percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    
    Returns:
        float between 0.0 and 0.5 representing buy percentage
    """
    # For best performing asset (rank 1), we want to allocate more capital
    # For worst performing asset (rank = total_assets), we want to allocate less
    
    if rank <= 3:  # Top 3 performers get higher allocation
        return 0.2 - ((rank - 1) * 0.05)  # 0.2, 0.15, 0.1
    elif rank <= 5:  # Next 2 performers get medium allocation
        return 0.075
    else:  # Rest get lower allocation
        return 0.05

def calculate_sell_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate sell percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    
    Returns:
        float between 0.1 and 1.0 representing sell percentage
    """
    # For worst performing assets (higher rank), we want to sell more
    # For best performing assets (lower rank), we want to hold more
    
    # Calculate normalized rank (0 to 1)
    normalized_rank = (rank - 1) / (total_assets - 1) if total_assets > 1 else 0
    
    # Lower performers (higher normalized rank) get higher sell percentage
    # Map normalized rank 0->0.1, 1->1.0
    sell_percentage = 0.1 + (0.9 * normalized_rank)
    
    return sell_percentage

def simulate_portfolio(signals_df: pd.DataFrame, 
                      price_data: pd.DataFrame, 
                      prices_dataset: Dict[str, pd.DataFrame],
                      selected_symbol: str,
                      lookback_days: int,
                      initial_capital: float = 100000) -> Optional[pd.DataFrame]:
    """
    Simulate portfolio performance based on trading signals
    
    Args:
        signals_df: DataFrame with trading signals
        price_data: DataFrame with price data
        prices_dataset: Dictionary of price dataframes for all assets
        selected_symbol: Current symbol being traded
        lookback_days: Number of days to look back for performance ranking
        initial_capital: Initial capital in USD
        
    Returns:
        DataFrame with portfolio performance
    """
    if signals_df is None or signals_df.empty:
        return None
    
    # Initialize portfolio DataFrame
    portfolio = signals_df.copy()
    
    # Make sure we have the price data columns we need
    if 'close' not in portfolio.columns and 'close' in price_data.columns:
        portfolio['close'] = price_data['close']
    
    portfolio['position'] = 0.0  # Number of shares/units held
    portfolio['cash'] = initial_capital  # Cash on hand
    portfolio['position_value'] = 0.0  # Value of position
    portfolio['portfolio_value'] = initial_capital  # Total portfolio value
    portfolio['returns'] = 0.0  # Daily returns
    portfolio['drawdown'] = 0.0  # Drawdown from peak
    
    # Trading cost as a percentage
    trading_cost_percentage = 0.0025  # 0.25% per trade
    
    # Value to track the peak portfolio value for drawdown calculation
    peak_value = initial_capital
    
    # Process each day
    for i in range(1, len(portfolio)):
        # Get current day and previous day
        current_day = portfolio.index[i]
        prev_day = portfolio.index[i-1]
        
        # Carry over position from previous day
        portfolio.loc[current_day, 'position'] = portfolio.loc[prev_day, 'position']
        portfolio.loc[current_day, 'cash'] = portfolio.loc[prev_day, 'cash']
        
        # Get current ranking of assets
        try:
            current_ranking = calculate_performance_ranking(prices_dataset, current_day, lookback_days)
            total_assets = len(current_ranking)
            asset_rank = int(current_ranking.loc[selected_symbol, 'rank'] * total_assets) if selected_symbol in current_ranking.index else total_assets
        except:
            # If ranking calculation fails, use default values
            total_assets = len(prices_dataset)
            asset_rank = total_assets  # Worst rank by default
        
        # Process buy signal
        if portfolio.loc[current_day, 'signal'] == 1:
            # Calculate buy amount based on asset rank
            buy_percentage = calculate_buy_percentage(asset_rank, total_assets)
            buy_amount = portfolio.loc[current_day, 'cash'] * buy_percentage
            
            # Calculate trading cost
            trading_cost = buy_amount * trading_cost_percentage
            buy_amount -= trading_cost
            
            # Update position and cash
            new_position = portfolio.loc[current_day, 'position'] + (buy_amount / portfolio.loc[current_day, 'close'])
            new_cash = portfolio.loc[current_day, 'cash'] - buy_amount - trading_cost
            
            portfolio.loc[current_day, 'position'] = new_position
            portfolio.loc[current_day, 'cash'] = new_cash
        
        # Process sell signal
        elif portfolio.loc[current_day, 'signal'] == -1 and portfolio.loc[current_day, 'position'] > 0:
            # Calculate sell amount based on asset rank
            sell_percentage = calculate_sell_percentage(asset_rank, total_assets)
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
    
    return portfolio

def calculate_performance_ranking(prices_dataset: Dict[str, pd.DataFrame], 
                                 current_time: datetime.datetime, 
                                 lookback_days: int) -> pd.DataFrame:
    """
    Calculate performance ranking of assets over the given lookback period
    
    Args:
        prices_dataset: Dictionary of price dataframes keyed by symbol
        current_time: Current time to use as the reference point
        lookback_days: Number of days to look back
        
    Returns:
        DataFrame with performance metrics
    """
    from modules.backtest import backtest_calculate_ranking
    
    # Use the function from backtest.py
    return backtest_calculate_ranking(prices_dataset, current_time, lookback_days)