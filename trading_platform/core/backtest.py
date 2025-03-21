"""
Backtesting functionality for trading strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import itertools
import json

from .data_feed import fetch_and_process_data, fetch_multi_symbol_data
from .indicators import generate_signals, get_default_params
from .config import default_backtest_interval

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
    # For crypto, market is always open (24/7)
    if market_hours['start'] == '00:00' and market_hours['end'] == '23:59':
        return True
    
    # For markets with specific hours
    hour_start = int(market_hours['start'].split(':')[0])
    minute_start = int(market_hours['start'].split(':')[1])
    hour_end = int(market_hours['end'].split(':')[0])
    minute_end = int(market_hours['end'].split(':')[1])
    
    market_open = time(hour=hour_start, minute=minute_start)
    market_close = time(hour=hour_end, minute=minute_end)
    
    # Convert timestamp to target timezone if specified
    if 'timezone' in market_hours:
        # In a real implementation, we would apply timezone conversion here
        pass
    
    current_time = timestamp.time()
    
    # Handle overnight markets (e.g., 22:00 - 04:00)
    if market_open > market_close:
        return current_time >= market_open or current_time <= market_close
    else:
        return market_open <= current_time <= market_close

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
        # Load symbol info
        from ..modules.crypto.config import TRADING_SYMBOLS, TRADING_COSTS
        
        if symbol not in TRADING_SYMBOLS:
            return {"error": f"Symbol {symbol} not found"}
        
        symbol_info = TRADING_SYMBOLS[symbol]
        market_hours = symbol_info['market_hours']
        
        # Use default parameters if none provided
        if params is None:
            params = get_default_params()
        
        # Fetch historical data
        lookback_days = max(days, lookback_days_param) + 10  # Add buffer for indicators
        data, error = fetch_and_process_data(symbol, '1h', lookback_days)
        
        if data is None or error:
            return {"error": error or "Failed to fetch data"}
        
        # Generate signals
        signals_df, daily_data, weekly_data = generate_signals(data, params)
        
        # Trim data to the requested backtest period
        start_date = datetime.now() - timedelta(days=days)
        backtest_data = signals_df[signals_df.index >= start_date]
        
        if backtest_data.empty:
            return {"error": "No data available for backtest period"}
        
        # Simulate trades
        portfolio_value = initial_capital
        cash = initial_capital
        position = 0
        trades = []
        portfolio_values = []
        
        # Trading costs
        trading_fee = TRADING_COSTS.get('trading_fee', 0.001)  # 0.1% by default
        spread = TRADING_COSTS.get('spread', 0.001)            # 0.1% by default
        
        # Performance ranking for position sizing
        prices_dataset = None
        total_assets = len(TRADING_SYMBOLS)
        
        for idx, row in backtest_data.iterrows():
            price = row['price']
            signal = row['signal']
            timestamp = idx
            
            # Check if market is open
            if not is_market_hours(timestamp, market_hours):
                continue
            
            # Calculate current portfolio value
            current_value = cash + (position * price)
            
            # Add to portfolio values
            portfolio_values.append({
                'timestamp': timestamp,
                'value': current_value,
                'cash': cash,
                'position': position
            })
            
            # Skip if no signal
            if signal == 0:
                continue
            
            # Calculate performance ranking for this symbol at current timestamp
            if is_simulating:
                if prices_dataset is None:
                    # Fetch data for all symbols for calculating rankings
                    assets = list(TRADING_SYMBOLS.keys())
                    prices_dataset = fetch_multi_symbol_data(assets, '1d', lookback_days_param)
                
                # Calculate ranking
                ranking_df = calculate_performance_ranking(prices_dataset, timestamp, lookback_days_param)
                
                # Get rank for this symbol
                if symbol in ranking_df.index:
                    rank = ranking_df.loc[symbol, 'rank']
                else:
                    rank = total_assets  # Default to worst rank
                
                # Calculate position sizing based on rank
                if signal > 0:  # Buy signal
                    buy_pct = calculate_buy_percentage(rank, total_assets)
                    buy_amount = cash * buy_pct
                    
                    # Calculate shares to buy
                    shares_to_buy = buy_amount / (price * (1 + trading_fee + spread/2))
                    
                    # Update position and cash
                    position += shares_to_buy
                    cash -= shares_to_buy * price * (1 + trading_fee + spread/2)
                    
                    # Record trade
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'amount': shares_to_buy * price,
                        'shares': shares_to_buy,
                        'fees': shares_to_buy * price * trading_fee,
                        'ranking': rank,
                        'buy_pct': buy_pct
                    })
                    
                elif signal < 0 and position > 0:  # Sell signal with existing position
                    sell_pct = calculate_sell_percentage(rank, total_assets)
                    shares_to_sell = position * sell_pct
                    
                    # Update position and cash
                    position -= shares_to_sell
                    cash += shares_to_sell * price * (1 - trading_fee - spread/2)
                    
                    # Record trade
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': price,
                        'amount': shares_to_sell * price,
                        'shares': shares_to_sell,
                        'fees': shares_to_sell * price * trading_fee,
                        'ranking': rank,
                        'sell_pct': sell_pct
                    })
            else:
                # Simple fixed position sizing without ranking
                if signal > 0:  # Buy signal
                    # Use 20% of cash for each buy
                    buy_amount = cash * 0.2
                    shares_to_buy = buy_amount / (price * (1 + trading_fee + spread/2))
                    
                    # Update position and cash
                    position += shares_to_buy
                    cash -= shares_to_buy * price * (1 + trading_fee + spread/2)
                    
                    # Record trade
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'amount': shares_to_buy * price,
                        'shares': shares_to_buy,
                        'fees': shares_to_buy * price * trading_fee
                    })
                    
                elif signal < 0 and position > 0:  # Sell signal with existing position
                    # Sell 50% of position
                    shares_to_sell = position * 0.5
                    
                    # Update position and cash
                    position -= shares_to_sell
                    cash += shares_to_sell * price * (1 - trading_fee - spread/2)
                    
                    # Record trade
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'SELL',
                        'price': price,
                        'amount': shares_to_sell * price,
                        'shares': shares_to_sell,
                        'fees': shares_to_sell * price * trading_fee
                    })
        
        # Liquidate final position
        final_price = backtest_data.iloc[-1]['price']
        final_value = cash + (position * final_price * (1 - trading_fee - spread/2))
        
        # Calculate metrics
        start_date = backtest_data.index[0]
        end_date = backtest_data.index[-1]
        n_days = (end_date - start_date).days
        
        roi = (final_value - initial_capital) / initial_capital
        roi_percent = roi * 100
        annualized_roi = ((1 + roi) ** (365 / max(1, n_days)) - 1) * 100 if n_days > 0 else 0
        
        # Drawdown calculation
        portfolio_values_df = pd.DataFrame(portfolio_values)
        if not portfolio_values_df.empty:
            portfolio_values_df['peak'] = portfolio_values_df['value'].cummax()
            portfolio_values_df['drawdown'] = (portfolio_values_df['value'] - portfolio_values_df['peak']) / portfolio_values_df['peak']
            max_drawdown = portfolio_values_df['drawdown'].min() * 100  # as percentage
        else:
            max_drawdown = 0
        
        return {
            "symbol": symbol,
            "n_days": n_days,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "roi": roi,
            "roi_percent": roi_percent,
            "annualized_roi": annualized_roi,
            "max_drawdown": max_drawdown,
            "n_trades": len(trades),
            "trades": trades,
            "is_simulating": is_simulating,
            "params": params,
            "portfolio_values": portfolio_values
        }
    
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
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
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    best_roi = -float('inf')
    best_params = None
    best_result = None
    
    # Run backtest for each combination
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        # Log parameter set being tested
        logger.info(f"Testing parameters for {symbol}: {params}")
        
        # Run backtest
        result = run_backtest(symbol, days, params)
        
        # Skip if error
        if 'error' in result:
            logger.error(f"Error with parameters {params}: {result['error']}")
            continue
        
        # Update best result if better
        roi = result.get('roi', -float('inf'))
        if roi > best_roi:
            best_roi = roi
            best_params = params
            best_result = result
            
            # Log improvement
            logger.info(f"New best ROI for {symbol}: {best_roi:.2%} with params: {best_params}")
    
    # Save best parameters to file
    if best_params:
        try:
            with open(output_file, 'r') as f:
                all_params = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_params = {}
        
        all_params[symbol] = best_params
        
        with open(output_file, 'w') as f:
            json.dump(all_params, f, indent=4)
    
    return {
        "symbol": symbol,
        "best_params": best_params,
        "best_roi": best_roi,
        "backtest_result": best_result
    }

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
    if prices_dataset is None or not prices_dataset:
        return pd.DataFrame()
    
    # Use current time if timestamp not provided
    if timestamp is None:
        # Find the latest timestamp across all symbols
        latest_timestamps = {symbol: data.index[-1] for symbol, data in prices_dataset.items() if len(data) > 0}
        if not latest_timestamps:
            return pd.DataFrame()
        timestamp = max(latest_timestamps.values())
    
    # Calculate performance for each symbol
    performances = []
    
    for symbol, df in prices_dataset.items():
        if df.empty:
            continue
        
        # Find closest timestamp in the data
        if timestamp in df.index:
            end_idx = df.index.get_loc(timestamp)
        else:
            # Find the closest timestamp less than the given timestamp
            end_idx = df.index.searchsorted(timestamp) - 1
            if end_idx < 0:
                continue
        
        # Calculate lookback period
        start_date = timestamp - timedelta(days=lookback_days)
        start_idx = df.index.searchsorted(start_date)
        
        # Skip if not enough data
        if start_idx >= end_idx:
            continue
        
        # Calculate performance
        start_price = df.iloc[start_idx]['Close']
        end_price = df.iloc[end_idx]['Close']
        
        performance = (end_price - start_price) / start_price
        
        performances.append({
            'symbol': symbol,
            'performance': performance,
            'start_price': start_price,
            'end_price': end_price,
            'timestamp': timestamp
        })
    
    # Create and sort DataFrame
    if not performances:
        return pd.DataFrame()
    
    perf_df = pd.DataFrame(performances)
    perf_df = perf_df.sort_values('performance', ascending=False)
    
    # Add rank
    perf_df['rank'] = range(1, len(perf_df) + 1)
    
    # Set symbol as index for easy lookup
    perf_df = perf_df.set_index('symbol')
    
    return perf_df

def get_historical_ranking(symbol, timestamp):
    """
    Get performance ranking for a specific symbol at a specific timestamp.
    
    Args:
        symbol: The symbol to get rank for
        timestamp: The timestamp to calculate ranking for
    
    Returns:
        tuple: (rank, performance) or (None, None) if not available
    """
    from ..modules.crypto.config import TRADING_SYMBOLS
    
    # Fetch data for all symbols
    symbols = list(TRADING_SYMBOLS.keys())
    
    # Calculate lookback period
    lookback_days = 15
    start_date = timestamp - timedelta(days=lookback_days)
    
    # Fetch data spanning the period
    prices_dataset = fetch_multi_symbol_data(symbols, '1d', lookback_days)
    
    # Calculate ranking
    ranking_df = calculate_performance_ranking(prices_dataset, timestamp, lookback_days)
    
    # Get rank for the specified symbol
    if symbol in ranking_df.index:
        rank = ranking_df.loc[symbol, 'rank']
        performance = ranking_df.loc[symbol, 'performance']
        return rank, performance
    else:
        return None, None

def calculate_buy_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate buy percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.0 and 0.2 representing buy percentage
    """
    # Normalize rank to 0-1 range
    normalized_rank = 1 - ((rank - 1) / max(1, (total_assets - 1)))
    
    # Calculate percentage (0.05 to 0.2)
    percentage = 0.05 + (normalized_rank * 0.15)
    
    return percentage

def calculate_sell_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate sell percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.1 and 1.0 representing sell percentage
    """
    # Normalize rank to 0-1 range (reverse: lower rank = higher selling)
    normalized_rank = (rank - 1) / max(1, (total_assets - 1))
    
    # Calculate percentage (0.1 to 1.0)
    percentage = 0.1 + (normalized_rank * 0.9)
    
    return percentage