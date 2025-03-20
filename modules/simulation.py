import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import datetime

def calculate_buy_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate buy percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    
    Returns:
        float between 0.0 and 0.5 representing buy percentage
    """
    # Calculate cutoff points
    bottom_third = int(total_assets * (1 / 3))
    top_two_thirds = total_assets - bottom_third
    
    # If in bottom third, buy 0%
    if rank > top_two_thirds:
        return 0.0
    
    # For top two-thirds, use inverted wave function
    x = (rank - 1) / (top_two_thirds - 1)
    wave = 0.02 * np.sin(2 * np.pi * x)  # Smaller oscillation
    linear = 0.48 - 0.48 * x  # Linear decrease from 0.48 to 0.0
    return max(0.0, min(0.5, linear + wave))  # Clamp between 0.0 and 0.5

def calculate_sell_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate sell percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    
    Returns:
        float between 0.1 and 1.0 representing sell percentage
    """
    # Calculate cutoff points
    bottom_third = int(total_assets * (1 / 3))
    top_two_thirds = total_assets - bottom_third
    
    # If in bottom third, sell 100%
    if rank > top_two_thirds:
        return 1.0
    
    # For top two-thirds, use a wave function
    x = (rank - 1) / (top_two_thirds - 1)
    wave = 0.1 * np.sin(2 * np.pi * x)  # Oscillation between -0.1 and 0.1
    linear = 0.3 + 0.7 * x  # Linear increase from 0.3 to 1.0
    return max(0.1, min(1.0, linear + wave))  # Clamp between 0.1 and 1.0

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
    from modules.data import calculate_performance_ranking
    
    if signals_df is None or signals_df.empty:
        return None
    
    # Initialize portfolio tracking
    position = 0  # Current position in shares
    cash = initial_capital
    portfolio_value = []  # Portfolio value over time
    shares_owned = []  # Shares owned over time
    trade_ranks = []  # Store the rank for each trade
    
    # For each row in the signals DataFrame
    for idx, row in signals_df.iterrows():
        signal = row.get('signal', 0)
        price = price_data.loc[idx, 'close'] if idx in price_data.index else 0
        current_rank = None
        
        if price == 0:
            continue
        
        # Calculate performance ranking at this timestamp if we have enough data
        if prices_dataset:
            # Use the backtest's ranking algorithm
            current_time = idx
            perf_rankings = calculate_performance_ranking(prices_dataset, current_time, lookback_days)
            
            if perf_rankings is not None and selected_symbol in perf_rankings.index:
                # Get the raw rank value (percentile)
                raw_rank = perf_rankings.loc[selected_symbol, 'rank']
                
                # Calculate integer rank (1 is best, total_assets is worst)
                total_assets = len(perf_rankings)
                rank = 1 + sum(1 for other_metric in perf_rankings['rank'].values if other_metric > raw_rank)
                current_rank = rank
        
        # Process buy signal
        if signal == 1 and cash > 0:
            if current_rank is not None:
                # Use the ranking-based buy percentage
                total_assets = len(prices_dataset)
                buy_percentage = calculate_buy_percentage(current_rank, total_assets)
                amount_to_use = cash * buy_percentage
            else:
                # Fallback to the original fixed percentage
                amount_to_use = cash * 0.95
            
            # Calculate shares to buy
            shares_to_buy = amount_to_use / price if price > 0 else 0
            
            # Crypto can be fractional
            shares_to_buy = round(shares_to_buy, 8)
            
            # Update position
            position += shares_to_buy
            cash -= shares_to_buy * price
        
        # Process sell signal    
        elif signal == -1 and position > 0:
            if current_rank is not None:
                # Use the ranking-based sell percentage
                total_assets = len(prices_dataset)
                sell_percentage = calculate_sell_percentage(current_rank, total_assets)
                shares_to_sell = position * sell_percentage
                cash += shares_to_sell * price
                position -= shares_to_sell
            else:
                # Fallback to selling all shares
                cash += position * price
                position = 0
        
        # Calculate portfolio value
        current_value = cash + (position * price)
        portfolio_value.append(current_value)
        shares_owned.append(position)
        trade_ranks.append(current_rank)  # Add rank information
    
    # Create a DataFrame for the portfolio history
    portfolio_df = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'shares_owned': shares_owned,
        'rank': trade_ranks
    }, index=signals_df.index)
    
    # Add additional calculated columns
    portfolio_df['price'] = price_data['close']
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
    portfolio_df['cumulative_returns'] = (1 + portfolio_df['returns']).cumprod() - 1
    
    # Calculate some portfolio metrics
    portfolio_df['drawdown'] = 1 - portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].cummax()
    
    return portfolio_df