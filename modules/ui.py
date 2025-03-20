import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional, List, Any
import datetime
import time

from attached_assets.config import TRADING_SYMBOLS, TRADING_COSTS, DEFAULT_RISK_PERCENT
from attached_assets.indicators import get_default_params

def create_sidebar() -> Dict[str, Any]:
    """
    Create sidebar components and return user settings
    
    Returns:
        Dictionary of user settings
    """
    # Sidebar for controls
    st.sidebar.header("Settings")
    
    # Symbol selection
    symbol_options = list(TRADING_SYMBOLS.keys())
    selected_symbol = st.sidebar.selectbox(
        "Symbol",
        symbol_options,
        index=0  # BTC/USD is the first
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1h", "4h", "1d"],
        index=0
    )
    
    # Lookback period selection
    lookback_days = st.sidebar.slider(
        "Lookback Period (days)",
        min_value=3,
        max_value=60,
        value=15,
        step=1
    )
    
    # Update interval selection
    update_interval = st.sidebar.selectbox(
        "Update Interval",
        [5, 15, 30, 60],
        index=1  # 15 seconds default
    )
    
    # Indicator display options
    with st.sidebar.expander("Indicator Settings", expanded=True):
        show_macd = st.checkbox("Show MACD", value=True)
        show_rsi = st.checkbox("Show RSI", value=True)
        show_stochastic = st.checkbox("Show Stochastic", value=True)
        show_fractal = st.checkbox("Show Fractal Complexity", value=True)
        show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    
    # Simulation settings
    with st.sidebar.expander("Portfolio Simulation", expanded=True):
        enable_simulation = st.checkbox("Enable Portfolio Simulation", value=True)
        initial_capital = st.number_input("Initial Capital ($)", value=100000, step=10000)
    
    # Parameter management
    with st.sidebar.expander("Parameter Management", expanded=True):
        st.text("Composite Indicator Parameters")
        use_best_params = st.checkbox("Use Optimized Parameters", value=True)
    
    # Button to force refresh data
    force_refresh = st.sidebar.button("Refresh Data Now")
    if force_refresh:
        st.sidebar.success("Data refreshed!")
    
    # Display information about the app
    with st.sidebar.expander("About", expanded=False):
        st.markdown("""
        This app displays real-time cryptocurrency price data with technical indicators and generates buy/sell signals.
        
        **Features:**
        - Real-time price data from Yahoo Finance for multiple crypto assets
        - Technical indicators (MACD, RSI, Stochastic, Fractal Complexity)
        - Daily and Weekly Composite indicators with threshold bands
        - Automated buy/sell signals based on indicator combinations
        - Portfolio simulation with performance tracking
        - Asset performance ranking and comparison
        - Parameter optimization for trading strategies
        - Full backtest capabilities with performance metrics
        
        *Data updates automatically based on the selected interval.*
        """)
    
    # Return all settings in a dictionary
    return {
        "selected_symbol": selected_symbol,
        "timeframe": timeframe,
        "lookback_days": lookback_days,
        "update_interval": update_interval,
        "show_macd": show_macd,
        "show_rsi": show_rsi,
        "show_stochastic": show_stochastic,
        "show_fractal": show_fractal,
        "show_signals": show_signals,
        "enable_simulation": enable_simulation,
        "initial_capital": initial_capital,
        "use_best_params": use_best_params,
        "force_refresh": force_refresh
    }

def display_performance_metrics(metrics: Dict[str, float]) -> None:
    """
    Display key performance metrics in columns
    
    Args:
        metrics: Dictionary with performance metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
    with col2:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
    with col3:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
    with col4:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")

def create_header(selected_symbol: str) -> None:
    """
    Create the app header section
    
    Args:
        selected_symbol: The selected trading symbol
    """
    # Header
    st.title(f"{selected_symbol} Price Monitor with Technical Indicators")
    st.markdown("Real-time price data with signal alerts based on technical indicators")

def setup_auto_refresh(update_interval: int) -> Tuple[st.empty, st.empty]:
    """
    Set up auto-refresh container and progress bar
    
    Args:
        update_interval: Update interval in seconds
        
    Returns:
        Tuple of placeholder and progress bar objects
    """
    # Set up placeholder for auto-refresh status
    auto_refresh_placeholder = st.empty()
    progress_bar = auto_refresh_placeholder.progress(0)
    
    return auto_refresh_placeholder, progress_bar

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
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    # Calculate progress (0-1)
    progress = min(1.0, elapsed_time / update_interval)
    # Update progress bar
    progress_bar.progress(progress)
    
    return progress

def handle_backtest_button(selected_symbol: str, lookback_days: int) -> None:
    """
    Handle the 'Run Full Backtest' button logic
    
    Args:
        selected_symbol: The selected trading symbol
        lookback_days: Number of days to look back
    """
    from modules.backtest import run_backtest
    
    if st.button("Run Full Backtest"):
        with st.spinner("Running backtest simulation... This may take a minute..."):
            try:
                # Use the run_backtest function
                backtest_result = run_backtest(
                    symbol=selected_symbol, 
                    days=lookback_days,
                    params=None,  # Use default params unless optimized
                    is_simulating=True
                )
                
                if backtest_result:
                    st.success(f"Backtest complete for {selected_symbol}!")
                    
                    # Display key performance metrics
                    metrics = backtest_result.get('metrics', {})
                    display_performance_metrics(metrics)
                    
                    # If there's a plot available in the result, display it
                    plot_data = backtest_result.get('plot_data', None)
                    if plot_data:
                        st.plotly_chart(plot_data, use_container_width=True)
                else:
                    st.warning("Could not complete backtest. Try a different timeframe.")
            except Exception as e:
                st.error(f"Error during backtest: {str(e)}")
                st.info("Check the logs for more details.")

def handle_optimization_button(selected_symbol: str, lookback_days: int) -> None:
    """
    Handle the 'Find Best Parameters' button logic
    
    Args:
        selected_symbol: The selected trading symbol
        lookback_days: Number of days to look back
    """
    from modules.backtest import find_best_params
    
    if st.button("Find Best Parameters"):
        with st.spinner("Running parameter optimization... This may take a minute..."):
            # Use the find_best_params function imported above
            try:
                # Define parameter grid based on current symbol
                param_grid = {
                    'macd_weight': [0.25, 0.5, 0.75, 1.0],
                    'rsi_weight': [0.25, 0.5, 0.75, 1.0],
                    'stoch_weight': [0.25, 0.5, 0.75, 1.0],
                    'fractal_weight': [0.1, 0.25, 0.5],
                    'reactivity': [0.5, 1.0, 1.5, 2.0]
                }
                
                # Run the parameter optimization
                best_params = find_best_params(selected_symbol, param_grid, days=lookback_days)
                
                if best_params:
                    st.success(f"Optimization complete! Found best parameters for {selected_symbol}")
                    st.json(best_params)
                else:
                    st.warning("Could not determine optimal parameters. Using defaults.")
            except Exception as e:
                st.error(f"Error during parameter optimization: {str(e)}")
                st.info("Using default parameters instead.")