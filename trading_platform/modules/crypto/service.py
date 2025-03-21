"""
Cryptocurrency-specific service functions
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from .config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LOOKBACK_DAYS, TRADING_SYMBOLS
from ...core.data_feed import fetch_and_process_data, fetch_multi_symbol_data
from ...core.indicators import generate_signals, get_default_params
from ...core.backtest import run_backtest as core_run_backtest
from ...core.backtest import find_best_params as core_find_best_params
from ...core.backtest import calculate_performance_ranking

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoService:
    """Service for cryptocurrency data processing and trading"""
    
    @staticmethod
    def get_available_symbols() -> List[Dict[str, str]]:
        """
        Get list of all available crypto symbols
        
        Returns:
            List of dictionaries with symbol information
        """
        symbols = []
        
        for symbol, info in TRADING_SYMBOLS.items():
            symbols.append({
                'symbol': symbol,
                'name': info['name'],
                'description': info['description'],
                'exchange': info['exchange']
            })
        
        return symbols
    
    @staticmethod
    def get_symbol_data(
        symbol: str = DEFAULT_SYMBOL, 
        timeframe: str = DEFAULT_TIMEFRAME, 
        lookback_days: int = DEFAULT_LOOKBACK_DAYS
    ) -> Dict[str, Any]:
        """
        Get data for a specific crypto symbol
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with price data, signals, and indicators
        """
        # Fetch and process data
        price_data, error = fetch_and_process_data(symbol, timeframe, lookback_days)
        
        if error:
            return {'error': error}
        
        if price_data is None or price_data.empty:
            return {'error': f'No data available for {symbol}'}
        
        # Generate signals
        params = get_default_params()
        signals_df, daily_data, weekly_data = generate_signals(price_data, params)
        
        # Return data
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'lookback_days': lookback_days,
            'price_data': price_data,
            'signals_data': signals_df,
            'daily_data': daily_data,
            'weekly_data': weekly_data
        }
    
    @staticmethod
    def get_multi_symbol_data(
        symbols: List[str] = None, 
        timeframe: str = DEFAULT_TIMEFRAME, 
        lookback_days: int = DEFAULT_LOOKBACK_DAYS
    ) -> Dict[str, Any]:
        """
        Get data for multiple crypto symbols
        
        Args:
            symbols: List of trading symbols (default: all symbols)
            timeframe: Data timeframe
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with data for all requested symbols
        """
        # Use all symbols if none provided
        if symbols is None or len(symbols) == 0:
            symbols = list(TRADING_SYMBOLS.keys())
        
        # Fetch data for all symbols
        price_dataset = fetch_multi_symbol_data(symbols, timeframe, lookback_days)
        
        # Process data for each symbol
        results = {}
        for symbol in symbols:
            if symbol in price_dataset:
                price_data = price_dataset[symbol]
                
                # Skip if empty
                if price_data is None or price_data.empty:
                    results[symbol] = {'error': f'No data available for {symbol}'}
                    continue
                
                # Generate signals
                params = get_default_params()
                signals_df, daily_data, weekly_data = generate_signals(price_data, params)
                
                # Store results
                results[symbol] = {
                    'price_data': price_data,
                    'signals_data': signals_df,
                    'daily_data': daily_data,
                    'weekly_data': weekly_data
                }
            else:
                results[symbol] = {'error': f'Failed to fetch data for {symbol}'}
        
        return {
            'symbols': symbols,
            'timeframe': timeframe,
            'lookback_days': lookback_days,
            'data': results
        }
    
    @staticmethod
    def run_backtest(
        symbol: str = DEFAULT_SYMBOL,
        days: int = DEFAULT_LOOKBACK_DAYS,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run backtest for a crypto symbol
        
        Args:
            symbol: Trading symbol
            days: Number of days to backtest
            params: Dictionary of parameters for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        return core_run_backtest(symbol, days, params)
    
    @staticmethod
    def optimize_parameters(
        symbol: str = DEFAULT_SYMBOL,
        days: int = DEFAULT_LOOKBACK_DAYS
    ) -> Dict[str, Any]:
        """
        Find optimal parameters for a crypto symbol
        
        Args:
            symbol: Trading symbol
            days: Number of days to backtest
            
        Returns:
            Dictionary with optimization results
        """
        from ...core.config import param_grid
        return core_find_best_params(symbol, param_grid, days)
    
    @staticmethod
    def get_performance_ranking(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, Any]:
        """
        Get performance ranking for all crypto assets
        
        Args:
            lookback_days: Number of days to look back for performance calculation
            
        Returns:
            Dictionary with ranking data
        """
        # Fetch data for all symbols
        symbols = list(TRADING_SYMBOLS.keys())
        price_dataset = fetch_multi_symbol_data(symbols, '1d', lookback_days)
        
        # Calculate performance ranking
        ranking_df = calculate_performance_ranking(price_dataset, None, lookback_days)
        
        # Convert to list of dictionaries for API response
        ranking_list = []
        for symbol, row in ranking_df.iterrows():
            ranking_list.append({
                'symbol': symbol,
                'performance': float(row['performance']),
                'rank': int(row['rank']),
                'start_price': float(row['start_price']),
                'end_price': float(row['end_price']),
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
            })
        
        return {
            'lookback_days': lookback_days,
            'ranking': ranking_list
        }
    
    @staticmethod
    def simulate_portfolio(
        symbols: List[str] = None,
        days: int = DEFAULT_LOOKBACK_DAYS,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Simulate portfolio performance across multiple assets
        
        Args:
            symbols: List of symbols to include (default: all symbols)
            days: Number of days to simulate
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary with portfolio simulation results
        """
        # Use all symbols if none provided
        if symbols is None or len(symbols) == 0:
            symbols = list(TRADING_SYMBOLS.keys())
        
        # Run backtests for each symbol with simulation mode
        backtest_results = {}
        total_value = 0.0
        
        for symbol in symbols:
            # Allocate equal capital to each symbol
            symbol_capital = initial_capital / len(symbols)
            
            # Run backtest with performance-based position sizing
            result = core_run_backtest(
                symbol, 
                days, 
                None,  # Use default parameters
                is_simulating=True,  # Enable simulation mode with performance ranking
                lookback_days_param=days // 3,  # Use 1/3 of backtest period for lookback
                initial_capital=symbol_capital
            )
            
            backtest_results[symbol] = result
            
            if 'error' not in result:
                total_value += result['final_value']
        
        # Calculate portfolio metrics
        roi = (total_value - initial_capital) / initial_capital
        roi_percent = roi * 100
        
        return {
            'symbols': symbols,
            'days': days,
            'initial_capital': initial_capital,
            'final_value': total_value,
            'roi': roi,
            'roi_percent': roi_percent,
            'backtest_results': backtest_results
        }