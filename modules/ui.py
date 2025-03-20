import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from typing import Dict, Any, Tuple, Optional

from modules.backtest import run_backtest, find_best_params
from modules.config import TRADING_SYMBOLS, DEFAULT_PARAM_GRID
from attached_assets.indicators import get_default_params

def create_sidebar() -> Dict[str, Any]:
    """
    Create sidebar components and return user settings
    
    Returns:
        Dictionary of user settings
    """
    with st.sidebar:
        st.title("Crypto Price Monitor")
        
        # Trading symbol selection
        symbol_options = list(TRADING_SYMBOLS.keys())
        selected_symbol = st.selectbox("Select Symbol", symbol_options, index=0)
        
        # Timeframe selection
        timeframe_options = ["1h", "4h", "1d"]
        timeframe_labels = {"1h": "1 Hour", "4h": "4 Hours", "1d": "1 Day"}
        selected_timeframe = st.selectbox(
            "Select Timeframe",
            timeframe_options,
            index=0,
            format_func=lambda x: timeframe_labels.get(x, x)
        )
        
        # Lookback period
        lookback_days = st.slider("Lookback Period (Days)", min_value=1, max_value=30, value=5)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-Refresh", value=True)
        
        # Update interval selection
        update_interval = st.slider(
            "Update Interval (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10,
            disabled=not auto_refresh
        )
        
        # Simulation toggle
        enable_simulation = st.checkbox("Enable Portfolio Simulation", value=True)
        
        # Initial capital for simulation
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000,
            disabled=not enable_simulation
        )
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            # Toggle indicators
            show_macd = st.checkbox("Show MACD", value=True)
            show_rsi = st.checkbox("Show RSI", value=True)
            show_stochastic = st.checkbox("Show Stochastic", value=True)
            show_fractal = st.checkbox("Show Fractal Complexity", value=True)
            show_signals = st.checkbox("Show Signals", value=True)
        
        # Return all settings as a dictionary
        return {
            "symbol": selected_symbol,
            "timeframe": selected_timeframe,
            "lookback_days": lookback_days,
            "auto_refresh": auto_refresh,
            "update_interval": update_interval,
            "enable_simulation": enable_simulation,
            "initial_capital": initial_capital,
            "show_macd": show_macd,
            "show_rsi": show_rsi,
            "show_stochastic": show_stochastic,
            "show_fractal": show_fractal,
            "show_signals": show_signals
        }

def display_performance_metrics(metrics: Dict[str, float]) -> None:
    """
    Display key performance metrics in columns
    
    Args:
        metrics: Dictionary with performance metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics['total_return']:.2f}%",
            delta=f"{metrics['total_return']:.2f}%" if metrics['total_return'] != 0 else None
        )
    
    with col2:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2f}%",
            delta=f"-{metrics['max_drawdown']:.2f}%" if metrics['max_drawdown'] != 0 else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta=f"{metrics['sharpe_ratio']:.2f}" if metrics['sharpe_ratio'] > 1 else None
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.2f}%",
            delta=f"{metrics['win_rate']:.2f}%" if metrics['win_rate'] > 50 else None
        )

def create_header(selected_symbol: str) -> None:
    """
    Create the app header section
    
    Args:
        selected_symbol: The selected trading symbol
    """
    st.title(f"{selected_symbol} Price Monitor")
    st.markdown(
        """
        This app visualizes cryptocurrency price data with technical indicators and generates trading signals.
        Use the sidebar to customize the view and settings.
        """
    )

def setup_auto_refresh(update_interval: int) -> Tuple[st.empty, st.empty]:
    """
    Set up auto-refresh container and progress bar
    
    Args:
        update_interval: Update interval in seconds
        
    Returns:
        Tuple of placeholder and progress bar objects
    """
    refresh_container = st.empty()
    progress_bar = st.progress(0)
    
    with refresh_container.container():
        st.markdown(f"**Auto-refreshing every {update_interval} seconds**")
    
    return refresh_container, progress_bar

def update_progress_bar(progress_bar: st.progress, start_time: float, update_interval: int) -> float:
    """
    Update the progress bar based on elapsed time
    
    Args:
        progress_bar: Streamlit progress bar object
        start_time: Start time in seconds
        update_interval: Update interval in seconds
        
    Returns:
        Current progress value (0-1)
    """
    current_time = time.time()
    elapsed_time = current_time - start_time
    progress = min(elapsed_time / update_interval, 1.0)
    progress_bar.progress(progress)
    return progress

def handle_backtest_button(selected_symbol: str, lookback_days: int) -> None:
    """
    Handle the 'Run Full Backtest' button logic
    
    Args:
        selected_symbol: The selected trading symbol
        lookback_days: Number of days to look back
    """
    if st.button("Run Full Backtest"):
        with st.spinner(f"Running backtest for {selected_symbol}..."):
            # Get default parameters
            params = get_default_params()
            
            # Run backtest
            backtest_result = run_backtest(selected_symbol, days=lookback_days, params=params)
            
            if backtest_result:
                # Display performance metrics
                st.subheader("Backtest Results")
                display_performance_metrics(backtest_result['metrics'])
                
                # Display trades
                st.subheader("Trades")
                
                trades_df = pd.DataFrame(backtest_result['trades'])
                if not trades_df.empty:
                    # Format date column
                    trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d %H:%M')
                    
                    # Round numeric columns
                    for col in ['price', 'amount', 'units', 'fee']:
                        if col in trades_df.columns:
                            trades_df[col] = trades_df[col].round(4)
                    
                    # Display the trades table
                    st.dataframe(trades_df, use_container_width=True)
                else:
                    st.info("No trades were executed during the backtest period")
            else:
                st.error("Failed to run backtest. Please try a different symbol or time period.")

def handle_optimization_button(selected_symbol: str, lookback_days: int) -> None:
    """
    Handle the 'Find Best Parameters' button logic
    
    Args:
        selected_symbol: The selected trading symbol
        lookback_days: Number of days to look back
    """
    if st.button("Find Best Parameters"):
        with st.spinner(f"Finding optimal parameters for {selected_symbol}..."):
            # Get parameter grid for optimization
            param_grid = DEFAULT_PARAM_GRID
            
            # Run parameter optimization
            best_params = find_best_params(selected_symbol, param_grid, days=lookback_days)
            
            if best_params:
                st.success(f"Optimization completed for {selected_symbol}")
            else:
                st.error("Failed to find optimal parameters. Please try a different symbol or time period.")