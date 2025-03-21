"""
Cryptocurrency-specific service functions
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

from ...core.data_feed import fetch_and_process_data, fetch_multi_symbol_data
from ...core.indicators import generate_signals, get_default_params
from ...core.backtest import run_backtest, find_best_params, calculate_performance_ranking
from .config import TRADING_SYMBOLS, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LOOKBACK_DAYS

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
        return [
            {"symbol": symbol, "name": info["name"]} 
            for symbol, info in TRADING_SYMBOLS.items()
        ]
    
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
            # Fetch price data
            df, error = fetch_and_process_data(symbol, timeframe, lookback_days)
            
            if df is None or error:
                return {
                    "success": False,
                    "error": error or "Failed to fetch data",
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            
            # Generate signals and indicators
            params = get_default_params()
            signals_df, daily_data, weekly_data = generate_signals(df, params)
            
            # Process data for response
            price_data = df.reset_index()
            price_data['timestamp'] = price_data['index'].astype(str)
            price_data = price_data.drop('index', axis=1).to_dict('records')
            
            signals_data = signals_df.reset_index()
            signals_data['timestamp'] = signals_data['index'].astype(str)
            signals_data = signals_data.drop('index', axis=1).to_dict('records')
            
            # Return formatted data
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "price_data": price_data,
                "signals_data": signals_data,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol,
                "timeframe": timeframe
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
        if symbols is None:
            symbols = list(TRADING_SYMBOLS.keys())
        
        results = {}
        for symbol in symbols:
            results[symbol] = CryptoService.get_symbol_data(symbol, timeframe, lookback_days)
        
        return results
    
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
            result = run_backtest(symbol, days, params)
            
            # Process results for response
            if 'trades' in result:
                # Convert datetime objects to strings
                for trade in result['trades']:
                    trade['timestamp'] = trade['timestamp'].isoformat()
                
                # Convert portfolio values
                if 'portfolio_values' in result:
                    for value in result['portfolio_values']:
                        value['timestamp'] = value['timestamp'].isoformat()
            
            # Convert timestamp fields in result
            if 'start_date' in result:
                result['start_date'] = result['start_date'].isoformat()
            if 'end_date' in result:
                result['end_date'] = result['end_date'].isoformat()
            
            return {
                "success": True,
                "backtest_result": result
            }
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
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
            from ...core.config import param_grid
            
            # Run parameter optimization
            result = find_best_params(symbol, param_grid, days)
            
            return {
                "success": True,
                "symbol": symbol,
                "optimization_result": result
            }
            
        except Exception as e:
            logger.error(f"Error optimizing parameters for {symbol}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
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
            # Fetch data for all symbols
            symbols = list(TRADING_SYMBOLS.keys())
            prices_dataset = {}
            
            for symbol in symbols:
                try:
                    df, error = fetch_and_process_data(symbol, '1d', lookback_days)
                    if df is not None and not error:
                        prices_dataset[symbol] = df
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
            # Calculate ranking
            ranking_df = calculate_performance_ranking(prices_dataset)
            
            if ranking_df.empty:
                return {
                    "success": False,
                    "error": "Could not calculate performance ranking",
                    "symbols": symbols
                }
            
            # Process results
            ranking_df['timestamp'] = ranking_df['timestamp'].astype(str)
            ranking_data = ranking_df.to_dict('records')
            
            return {
                "success": True,
                "ranking": ranking_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance ranking: {str(e)}")
            return {
                "success": False,
                "error": str(e)
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
        if symbols is None:
            symbols = list(TRADING_SYMBOLS.keys())[:5]  # Limit to top 5 symbols for performance
            
        try:
            # Run backtests for each symbol
            results = {}
            total_roi = 0.0
            total_final_value = 0.0
            
            for symbol in symbols:
                per_symbol_capital = initial_capital / len(symbols)
                
                # Run backtest with equal allocation
                result = run_backtest(symbol, days, initial_capital=per_symbol_capital)
                
                if 'error' not in result:
                    results[symbol] = {
                        'roi_percent': result.get('roi_percent', 0),
                        'final_value': result.get('final_value', per_symbol_capital),
                        'n_trades': result.get('n_trades', 0)
                    }
                    
                    total_roi += result.get('roi_percent', 0)
                    total_final_value += result.get('final_value', per_symbol_capital)
            
            # Calculate portfolio metrics
            avg_roi = total_roi / len(results) if results else 0
            portfolio_roi = (total_final_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
            
            return {
                "success": True,
                "portfolio": {
                    "symbols": list(results.keys()),
                    "initial_capital": initial_capital,
                    "final_value": total_final_value,
                    "roi_percent": portfolio_roi,
                    "avg_symbol_roi": avg_roi
                },
                "symbol_results": results
            }
            
        except Exception as e:
            logger.error(f"Error simulating portfolio: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "symbols": symbols
            }