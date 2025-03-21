import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import json
from datetime import datetime, timedelta
import itertools
import pytz
from .indicators import generate_signals, get_default_params
from .data_feed import fetch_historical_data

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
    # For 24/7 markets
    if market_hours['start'] == '00:00' and market_hours['end'] == '23:59':
        return True
        
    # Convert timestamp to market timezone
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    market_tz = market_hours['timezone']
    market_time = timestamp.astimezone(pytz.timezone(market_tz))
    
    # Check if it's a weekday
    if market_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Parse market hours
    start_time = datetime.strptime(market_hours['start'], '%H:%M').time()
    end_time = datetime.strptime(market_hours['end'], '%H:%M').time()
    current_time = market_time.time()
    
    return start_time <= current_time <= end_time

def run_backtest(symbol: str,
                days: int = 15,
                params: Optional[Dict[str, Any]] = None,
                is_simulating: bool = False,
                lookback_days_param: int = 5) -> Dict[str, Any]:
    """
    Run a single backtest simulation for a given symbol and parameter set
    
    Args:
        symbol: Trading symbol
        days: Number of days to backtest
        params: Dictionary of parameters for backtesting
        is_simulating: Whether running in simulation mode
        lookback_days_param: Number of days to look back for performance calculation
        
    Returns:
        Dictionary with backtest results
    """
    # Use default parameters if none provided
    if params is None:
        params = get_default_params()
    
    # Determine how many days of data to fetch
    lookback_days = max(days * 2, 30)  # Ensure enough data for indicators
    
    try:
        # Fetch data from Yahoo Finance
        from ..modules.crypto.config import TRADING_SYMBOLS
        market_info = TRADING_SYMBOLS.get(symbol, {})
        
        # Extract market hours configuration
        market_hours = market_info.get('market_hours', {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        })
        
        # Get data for backtesting
        df = fetch_historical_data(symbol, interval='1h', days=lookback_days)
        
        # Generate signals using the provided parameters
        signals_df, daily_data, weekly_data = generate_signals(df, params)
        
        # Get starting timestamp for backtesting
        if is_simulating:
            # For simulation, use the full dataset
            start_ts = df.index[0]
        else:
            # For backtesting, use the last X days
            start_ts = df.index[-1] - timedelta(days=days)
            
        # Filter data for the backtest period
        backtest_mask = df.index >= start_ts
        signals_df = signals_df[backtest_mask]
        df = df[backtest_mask]
        
        # Calculate performance metrics
        capital = 100000  # Starting capital
        crypto_holdings = 0  # Starting crypto holdings
        cash = capital  # Starting cash
        
        # Track trades and portfolio value
        trades = []
        portfolio_values = []
        
        # Initialize current price
        current_price = df['close'].iloc[0]
        
        # Process each bar
        for idx, row in df.iterrows():
            # Skip if outside market hours
            if not is_market_hours(idx, market_hours):
                continue
                
            # Update current price
            current_price = row['close']
            
            # Calculate current portfolio value
            portfolio_value = cash + (crypto_holdings * current_price)
            portfolio_values.append({
                'timestamp': idx,
                'value': portfolio_value,
                'cash': cash,
                'holdings': crypto_holdings,
                'price': current_price
            })
            
            # Get signal for this bar
            if idx in signals_df.index:
                signal = signals_df.loc[idx, 'signal']
                
                # Process buy signal
                if signal == 1 and cash > 100:  # Ensure we have cash to buy
                    # Check performance ranking to determine position size
                    rank, performance = get_historical_ranking(symbol, idx)
                    if rank is not None:
                        buy_pct = calculate_buy_percentage(rank, len(TRADING_SYMBOLS))
                    else:
                        buy_pct = 0.1  # Default if no ranking available
                        
                    # Calculate position size
                    buy_amount = cash * buy_pct
                    fee = buy_amount * 0.001  # 0.1% fee
                    buy_amount -= fee
                    
                    # Execute trade
                    amount_bought = buy_amount / current_price
                    crypto_holdings += amount_bought
                    cash -= (buy_amount + fee)
                    
                    # Record trade
                    trades.append({
                        'timestamp': idx,
                        'action': 'BUY',
                        'price': current_price,
                        'amount': amount_bought,
                        'value': buy_amount,
                        'fee': fee,
                        'cash': cash,
                        'holdings': crypto_holdings,
                        'portfolio_value': cash + (crypto_holdings * current_price)
                    })
                
                # Process sell signal
                elif signal == -1 and crypto_holdings > 0:  # Ensure we have holdings to sell
                    # Check performance ranking to determine position size
                    rank, performance = get_historical_ranking(symbol, idx)
                    if rank is not None:
                        sell_pct = calculate_sell_percentage(rank, len(TRADING_SYMBOLS))
                    else:
                        sell_pct = 0.2  # Default if no ranking available
                        
                    # Calculate position size
                    amount_sold = crypto_holdings * sell_pct
                    sell_amount = amount_sold * current_price
                    fee = sell_amount * 0.001  # 0.1% fee
                    sell_amount -= fee
                    
                    # Execute trade
                    crypto_holdings -= amount_sold
                    cash += sell_amount
                    
                    # Record trade
                    trades.append({
                        'timestamp': idx,
                        'action': 'SELL',
                        'price': current_price,
                        'amount': amount_sold,
                        'value': sell_amount,
                        'fee': fee,
                        'cash': cash,
                        'holdings': crypto_holdings,
                        'portfolio_value': cash + (crypto_holdings * current_price)
                    })
        
        # Calculate final portfolio value
        final_value = cash + (crypto_holdings * current_price)
        
        # Calculate performance metrics
        roi = (final_value - capital) / capital * 100
        
        # Calculate daily returns
        if len(portfolio_values) >= 2:
            values = [p['value'] for p in portfolio_values]
            daily_returns = []
            
            # Calculate daily returns using daily close values
            daily_data = pd.DataFrame(portfolio_values)
            daily_data.set_index('timestamp', inplace=True)
            daily_data = daily_data.resample('1D').last().dropna()
            
            if len(daily_data) >= 2:
                for i in range(1, len(daily_data)):
                    prev_value = daily_data['value'].iloc[i-1]
                    curr_value = daily_data['value'].iloc[i]
                    
                    if prev_value > 0:
                        daily_return = (curr_value - prev_value) / prev_value
                        daily_returns.append(daily_return)
            
            # Calculate metrics
            if daily_returns:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
                max_drawdown = 0
                peak = values[0]
                
                for value in values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            else:
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Return backtest results
        return {
            'symbol': symbol,
            'start_date': start_ts,
            'end_date': df.index[-1],
            'days': days,
            'initial_capital': capital,
            'final_value': final_value,
            'roi_percent': roi,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': len(trades),
            'params': params,
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'error': str(e),
            'success': False
        }

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
        logger.info(f"Starting parameter optimization for {symbol}")
        
        # Check if we need to expand the weights parameter
        if 'weights' in param_grid and isinstance(param_grid['weights'], list):
            weight_options = param_grid.pop('weights')
            
            # Create all combinations of parameters excluding weights
            keys = list(param_grid.keys())
            combinations = list(itertools.product(*[param_grid[key] for key in keys]))
            
            # Generate list of parameter dictionaries
            all_params = []
            for combo in combinations:
                param_dict = {keys[i]: combo[i] for i in range(len(keys))}
                
                # Add each weight option
                for weight_option in weight_options:
                    param_copy = param_dict.copy()
                    param_copy['weights'] = weight_option
                    all_params.append(param_copy)
        else:
            # Create all combinations of parameters
            keys = list(param_grid.keys())
            combinations = list(itertools.product(*[param_grid[key] for key in keys]))
            
            # Generate list of parameter dictionaries
            all_params = [{keys[i]: combo[i] for i in range(len(keys))} for combo in combinations]
        
        logger.info(f"Testing {len(all_params)} parameter combinations")
        
        # Run backtest with each parameter set
        results = []
        for i, params in enumerate(all_params):
            logger.info(f"Testing combination {i+1} of {len(all_params)}")
            
            # Run backtest
            backtest_result = run_backtest(symbol, days, params)
            
            # Save result if successful
            if 'error' not in backtest_result:
                results.append({
                    'params': params,
                    'roi': backtest_result.get('roi_percent', 0),
                    'sharpe': backtest_result.get('sharpe_ratio', 0),
                    'drawdown': backtest_result.get('max_drawdown', 1),
                    'trades': backtest_result.get('n_trades', 0)
                })
        
        # Find best parameter set based on ROI and Sharpe ratio
        if results:
            # Sort by ROI (primary) and Sharpe ratio (secondary)
            results.sort(key=lambda x: (x['roi'], x['sharpe']), reverse=True)
            best_result = results[0]
            
            # Create result dictionary
            result = {
                'symbol': symbol,
                'best_params': best_result['params'],
                'performance': {
                    'roi': best_result['roi'],
                    'sharpe': best_result['sharpe'],
                    'drawdown': best_result['drawdown'],
                    'trades': best_result['trades']
                },
                'date': datetime.now().strftime('%Y-%m-%d'),
                'all_results': results[:10]  # Save top 10 results
            }
            
            # Save to file
            try:
                with open(output_file, 'r') as f:
                    all_results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_results = {}
                
            all_results[symbol] = result
            
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Optimization complete for {symbol}")
            logger.info(f"Best ROI: {best_result['roi']:.2f}%, Sharpe: {best_result['sharpe']:.2f}")
            
            return result
        else:
            logger.warning(f"No valid results for {symbol}")
            return {'symbol': symbol, 'error': 'No valid results', 'success': False}
    
    except Exception as e:
        logger.error(f"Error optimizing parameters for {symbol}: {str(e)}")
        return {'symbol': symbol, 'error': str(e), 'success': False}

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
        from ..modules.crypto.config import TRADING_SYMBOLS
        
        if prices_dataset is None:
            # Fetch data for all symbols
            prices_dataset = {}
            for symbol in TRADING_SYMBOLS.keys():
                try:
                    df = fetch_historical_data(symbol, interval='1d', days=lookback_days*2)
                    prices_dataset[symbol] = df
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        if timestamp is None:
            # Use the latest timestamp available across all datasets
            timestamps = [df.index[-1] for df in prices_dataset.values() if not df.empty]
            if not timestamps:
                return pd.DataFrame()
            timestamp = min(timestamps)  # Use earliest of latest timestamps
        
        # Calculate performance for each symbol
        performances = []
        for symbol, df in prices_dataset.items():
            if df.empty:
                continue
                
            # Find closest timestamp
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            if idx < 0 or idx >= len(df):
                continue
            
            current_ts = df.index[idx]
            
            # Get starting point (lookback days before current timestamp)
            start_ts = current_ts - timedelta(days=lookback_days)
            start_idx = df.index.get_indexer([start_ts], method='nearest')[0]
            
            if start_idx < 0 or start_idx >= len(df) or start_idx == idx:
                continue
            
            # Calculate performance
            start_price = df['close'].iloc[start_idx]
            current_price = df['close'].iloc[idx]
            
            if start_price <= 0:
                continue
                
            performance = (current_price - start_price) / start_price * 100
            
            performances.append({
                'symbol': symbol,
                'timestamp': current_ts,
                'performance': performance
            })
        
        if not performances:
            return pd.DataFrame()
            
        # Create DataFrame and calculate ranks
        ranking_df = pd.DataFrame(performances)
        ranking_df.sort_values('performance', ascending=False, inplace=True)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        ranking_df['rank_normalized'] = ranking_df['rank'] / len(ranking_df)
        
        return ranking_df
    
    except Exception as e:
        logger.error(f"Error calculating performance ranking: {str(e)}")
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
        # Calculate ranking
        ranking_df = calculate_performance_ranking(timestamp=timestamp)
        
        if ranking_df.empty or symbol not in ranking_df['symbol'].values:
            return None, None
            
        # Get symbol's row
        symbol_row = ranking_df[ranking_df['symbol'] == symbol].iloc[0]
        
        return symbol_row['rank'], symbol_row['performance']
        
    except Exception as e:
        logger.error(f"Error getting historical ranking for {symbol}: {str(e)}")
        return None, None

def calculate_buy_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate buy percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.0 and 0.2 representing buy percentage
    """
    if total_assets <= 1:
        return 0.2  # Default percentage if only one asset
    
    # Normalize rank to [0, 1] where 0 is best performer
    normalized_rank = (rank - 1) / (total_assets - 1) if total_assets > 1 else 0
    
    # Calculate percentage (higher for better performers)
    # Best performer (rank 1) gets 20%, worst performer gets 5%
    percentage = 0.2 - (normalized_rank * 0.15)
    
    return percentage

def calculate_sell_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate sell percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.1 and 1.0 representing sell percentage
    """
    if total_assets <= 1:
        return 0.2  # Default percentage if only one asset
    
    # Normalize rank to [0, 1] where 0 is best performer
    normalized_rank = (rank - 1) / (total_assets - 1) if total_assets > 1 else 0
    
    # Calculate percentage (higher for worse performers)
    # Best performer (rank 1) gets 20%, worst performer gets 100%
    percentage = 0.2 + (normalized_rank * 0.8)
    
    return percentage