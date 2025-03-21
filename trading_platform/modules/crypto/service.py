"""
Cryptocurrency-specific service functions
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import time
import traceback

from trading_platform.core.data_feed import fetch_historical_data, fetch_and_process_data, fetch_multi_symbol_data, get_available_symbols
from trading_platform.core.indicators import generate_signals, get_default_params
from trading_platform.core.backtest import run_backtest, find_best_params, calculate_performance_ranking

from .config import (
    DEFAULT_SYMBOL,
    DEFAULT_TIMEFRAME,
    DEFAULT_LOOKBACK_DAYS,
    AVAILABLE_SYMBOLS
)

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
        try:
            # Get base symbols from config
            symbols = AVAILABLE_SYMBOLS
            
            # Format as list of objects
            formatted_symbols = []
            for symbol in symbols:
                name = symbol.split('/')[0] if '/' in symbol else symbol
                formatted_symbols.append({
                    'symbol': symbol,
                    'name': name,
                    'type': 'crypto',
                    'exchange': 'Coinbase'
                })
                
            return formatted_symbols
        except Exception as e:
            logger.error(f"Error getting available symbols: {str(e)}")
            return []
    
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
        try:
            # Fetch and process data
            data, error = fetch_and_process_data(symbol, timeframe, lookback_days)
            
            if error or data is None or data.empty:
                return {
                    'success': False,
                    'error': error or f"No data available for {symbol}"
                }
            
            # Generate signals
            params = get_default_params()
            signals_df, daily_data, weekly_data = generate_signals(data, params)
            
            if signals_df.empty:
                return {
                    'success': False,
                    'error': f"Failed to generate signals for {symbol}"
                }
            
            # Process data for API response
            price_data = []
            for idx, row in data.iterrows():
                price_data.append({
                    'timestamp': idx.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
            
            # Process signals for API response
            signals_data = []
            for idx, row in signals_df.iterrows():
                signals_data.append({
                    'timestamp': idx.isoformat(),
                    'price': float(row['price']),
                    'daily_composite': float(row['daily_composite']),
                    'daily_upper': float(row['daily_upper']),
                    'daily_lower': float(row['daily_lower']),
                    'buy_signal': bool(row['buy_signal']),
                    'sell_signal': bool(row['sell_signal']),
                    'potential_buy': bool(row['potential_buy']),
                    'potential_sell': bool(row['potential_sell']),
                    'volatility': float(row['volatility']) if 'volatility' in row else None
                })
            
            return {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'lookback_days': lookback_days,
                'price_data': price_data,
                'signals_data': signals_data
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol data: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
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
        try:
            # If no symbols provided, use all available symbols
            if not symbols:
                symbols = [s['symbol'] for s in CryptoService.get_available_symbols()]
            
            # Fetch data for all symbols
            start_time = time.time()
            logger.info(f"Fetching data for {len(symbols)} symbols...")
            
            prices_dataset = fetch_multi_symbol_data(symbols, timeframe, lookback_days)
            
            logger.info(f"Data fetched in {time.time() - start_time:.2f} seconds")
            
            # Process each symbol's data
            result_data = {}
            for symbol, data in prices_dataset.items():
                # Skip if no data
                if data.empty:
                    result_data[symbol] = {
                        'success': False,
                        'error': f"No data available for {symbol}"
                    }
                    continue
                
                # Generate signals
                params = get_default_params()
                signals_df, daily_data, weekly_data = generate_signals(data, params)
                
                # Skip if no signals
                if signals_df.empty:
                    result_data[symbol] = {
                        'success': False,
                        'error': f"Failed to generate signals for {symbol}"
                    }
                    continue
                
                # Process data for API response
                price_data = []
                for idx, row in data.iterrows():
                    price_data.append({
                        'timestamp': idx.isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume'])
                    })
                
                # Process signals for API response
                signals_data = []
                for idx, row in signals_df.iterrows():
                    signals_data.append({
                        'timestamp': idx.isoformat(),
                        'price': float(row['price']),
                        'daily_composite': float(row['daily_composite']),
                        'daily_upper': float(row['daily_upper']),
                        'daily_lower': float(row['daily_lower']),
                        'buy_signal': bool(row['buy_signal']),
                        'sell_signal': bool(row['sell_signal']),
                        'potential_buy': bool(row['potential_buy']),
                        'potential_sell': bool(row['potential_sell']),
                        'volatility': float(row['volatility']) if 'volatility' in row else None
                    })
                
                result_data[symbol] = {
                    'success': True,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'lookback_days': lookback_days,
                    'price_data': price_data,
                    'signals_data': signals_data
                }
            
            return {
                'success': True,
                'data': result_data
            }
            
        except Exception as e:
            logger.error(f"Error getting multi-symbol data: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
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
        try:
            # Run backtest
            results = run_backtest(symbol, days, params)
            
            # Check for errors
            if 'error' in results:
                return {
                    'success': False,
                    'error': results['error']
                }
            
            # Format dates for JSON serialization
            if 'trades' in results:
                for trade in results['trades']:
                    if 'timestamp' in trade and not isinstance(trade['timestamp'], str):
                        trade['timestamp'] = trade['timestamp'].isoformat()
            
            if 'equity_curve' in results:
                for point in results['equity_curve']:
                    if 'timestamp' in point and not isinstance(point['timestamp'], str):
                        point['timestamp'] = point['timestamp'].isoformat()
            
            return {
                'success': True,
                'backtest_results': results
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
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
        try:
            # Get parameter grid from configuration
            from trading_platform.core.config import param_grid
            
            # Run optimization
            results = find_best_params(symbol, param_grid, days)
            
            # Check for errors
            if 'error' in results:
                return {
                    'success': False,
                    'error': results['error']
                }
            
            # Format dates for JSON serialization in backtest results
            backtest_result = results.get('backtest_result', {})
            if 'trades' in backtest_result:
                for trade in backtest_result['trades']:
                    if 'timestamp' in trade and not isinstance(trade['timestamp'], str):
                        trade['timestamp'] = trade['timestamp'].isoformat()
            
            if 'equity_curve' in backtest_result:
                for point in backtest_result['equity_curve']:
                    if 'timestamp' in point and not isinstance(point['timestamp'], str):
                        point['timestamp'] = point['timestamp'].isoformat()
            
            return {
                'success': True,
                'best_params': results.get('best_params', {}),
                'backtest_result': backtest_result
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    @staticmethod
    def get_performance_ranking(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, Any]:
        """
        Get performance ranking for all crypto assets
        
        Args:
            lookback_days: Number of days to look back for performance calculation
            
        Returns:
            Dictionary with ranking data
        """
        try:
            # Get all available symbols
            symbols = [s['symbol'] for s in CryptoService.get_available_symbols()]
            
            # Fetch data
            prices_dataset = fetch_multi_symbol_data(symbols, '1d', lookback_days)
            
            # Calculate ranking
            ranking_df = calculate_performance_ranking(prices_dataset)
            
            if ranking_df.empty:
                return {
                    'success': False,
                    'error': "Failed to calculate performance ranking"
                }
            
            # Process ranking for API response
            ranking_data = []
            for idx, row in ranking_df.iterrows():
                ranking_data.append({
                    'symbol': row['symbol'],
                    'rank': int(row['rank']),
                    'performance': float(row['performance']),
                    'start_price': float(row['start_price']),
                    'end_price': float(row['end_price']),
                    'timestamp': row['timestamp'].isoformat() if not isinstance(row['timestamp'], str) else row['timestamp']
                })
            
            return {
                'success': True,
                'ranking': ranking_data
            }
            
        except Exception as e:
            logger.error(f"Error getting performance ranking: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
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
        try:
            # If no symbols provided, use top 5 by performance
            if not symbols:
                ranking_result = CryptoService.get_performance_ranking(lookback_days=days)
                
                if not ranking_result.get('success', False):
                    return {
                        'success': False,
                        'error': ranking_result.get('error', "Failed to get performance ranking")
                    }
                
                # Sort by rank and take top 5
                ranking_data = sorted(ranking_result.get('ranking', []), key=lambda x: x['rank'])
                symbols = [item['symbol'] for item in ranking_data[:5]]
            
            # Initialize portfolio results
            portfolio_results = {
                'initial_capital': initial_capital,
                'final_value': initial_capital,
                'return': 0.0,
                'drawdown': 0.0,
                'symbol_allocations': {},
                'trades': [],
                'equity_curve': []
            }
            
            # Divide initial capital equally among symbols
            symbol_capital = initial_capital / len(symbols)
            
            # Run backtest for each symbol with simulation mode enabled
            total_final_value = 0.0
            symbol_results = {}
            
            for symbol in symbols:
                # Run backtest with simulation mode
                result = run_backtest(
                    symbol=symbol,
                    days=days,
                    params=None,
                    is_simulating=True,
                    initial_capital=symbol_capital
                )
                
                if 'error' in result:
                    logger.warning(f"Error in portfolio simulation for {symbol}: {result['error']}")
                    continue
                
                # Add to total final value
                total_final_value += result.get('final_value', symbol_capital)
                
                # Store symbol result
                symbol_results[symbol] = result
                
                # Add allocation
                portfolio_results['symbol_allocations'][symbol] = {
                    'initial_allocation': symbol_capital,
                    'final_value': result.get('final_value', symbol_capital),
                    'return': result.get('return', 0.0),
                    'trades': len(result.get('trades', [])),
                }
                
                # Add trades
                for trade in result.get('trades', []):
                    if 'timestamp' in trade and not isinstance(trade['timestamp'], str):
                        trade['timestamp'] = trade['timestamp'].isoformat()
                    
                    trade['symbol'] = symbol
                    portfolio_results['trades'].append(trade)
                
                # Add to equity curve (simplified - in reality would need time alignment)
                for point in result.get('equity_curve', []):
                    if 'timestamp' in point and not isinstance(point['timestamp'], str):
                        point['timestamp'] = point['timestamp'].isoformat()
                    
                    point['symbol'] = symbol
                    portfolio_results['equity_curve'].append(point)
            
            # Sort trades and equity curve by timestamp
            portfolio_results['trades'].sort(key=lambda x: x['timestamp'])
            portfolio_results['equity_curve'].sort(key=lambda x: x['timestamp'])
            
            # Calculate portfolio metrics
            portfolio_results['final_value'] = total_final_value
            portfolio_results['return'] = (total_final_value / initial_capital - 1) * 100
            
            return {
                'success': True,
                'portfolio_results': portfolio_results,
                'symbol_results': symbol_results
            }
            
        except Exception as e:
            logger.error(f"Error simulating portfolio: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }