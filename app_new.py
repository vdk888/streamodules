"""
Cryptocurrency Price Monitor
A Streamlit application for visualizing cryptocurrency price data, 
technical indicators, and trading signals.

Uses Plotly for interactive charts and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import datetime
import logging
from typing import Dict, Optional, Tuple, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to sys.path
sys.path.append('.')

# Import modules for data fetching and processing
from modules.data import get_price_data, get_multi_asset_data, generate_trading_signals
from modules.storage import load_best_params
from modules.simulation import simulate_portfolio
from modules.backtest import calculate_performance_ranking, run_backtest, find_best_params
import modules.config as config

# Import the new visualization module
from new_visualization import (
    create_price_chart, 
    create_performance_ranking_chart, 
    create_portfolio_performance_chart,
    create_multi_asset_chart
)

# Page configuration
st.set_page_config(
    page_title="Crypto Price Monitor",
    page_icon="📈",
    layout="wide"
)

# Define state container for app state management
class StateContainer:
    def __init__(self):
        self.settings = {}
        self.df = None
        self.signals_df = None
        self.daily_composite = None
        self.weekly_composite = None
        self.portfolio_df = None
        self.performance_df = None
        self.last_update_time = None
        self.start_time = time.time()

# Initialize session state
if 'state_container' not in st.session_state:
    st.session_state.state_container = StateContainer()

state = st.session_state.state_container

def create_sidebar() -> Dict:
    """
    Create sidebar UI components and handle user inputs
    
    Returns:
        Dictionary with user settings
    """
    with st.sidebar:
        st.title("Settings")
        
        # Symbol selection with custom format
        symbol_options = list(config.TRADING_SYMBOLS.keys())
        selected_symbol = st.selectbox(
            "Select Symbol",
            options=symbol_options,
            index=0,
            format_func=lambda x: f"{x} - {config.TRADING_SYMBOLS[x].get('description', '')}"
        )
        
        # Timeframe selection
        timeframe_options = ["1h", "4h", "1d"]
        timeframe = st.selectbox(
            "Timeframe",
            options=timeframe_options,
            index=0
        )
        
        # Lookback days slider
        lookback_days = st.slider(
            "Lookback Days",
            min_value=1,
            max_value=30,
            value=5,
            step=1
        )
        
        # Create an expander for indicator settings
        with st.expander("Indicator Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                show_macd = st.checkbox("Show MACD", value=True)
                show_rsi = st.checkbox("Show RSI", value=True)
            
            with col2:
                show_stochastic = st.checkbox("Show Stochastic", value=True)
                show_fractal = st.checkbox("Show Fractal", value=True)
            
            show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
        
        # Create an expander for simulation settings
        with st.expander("Simulation Settings", expanded=False):
            enable_simulation = st.checkbox("Enable Portfolio Simulation", value=True)
            
            if enable_simulation:
                initial_capital = st.number_input(
                    "Initial Capital (USD)",
                    min_value=1000,
                    max_value=1000000,
                    value=100000,
                    step=10000
                )
            else:
                initial_capital = 100000
        
        # Create an expander for auto-refresh settings
        with st.expander("Auto-Refresh Settings", expanded=False):
            auto_refresh = st.checkbox("Auto Refresh", value=True)
            
            if auto_refresh:
                update_interval = st.slider(
                    "Update Interval (seconds)",
                    min_value=30,
                    max_value=300,
                    value=60,
                    step=30
                )
            else:
                update_interval = 0
                
            if st.button("Refresh Now"):
                force_refresh = True
            else:
                force_refresh = False
        
        # Collect all settings in a dictionary
        settings = {
            "symbol": selected_symbol,
            "timeframe": timeframe,
            "lookback_days": lookback_days,
            "show_macd": show_macd,
            "show_rsi": show_rsi,
            "show_stochastic": show_stochastic,
            "show_fractal": show_fractal,
            "show_signals": show_signals,
            "enable_simulation": enable_simulation,
            "initial_capital": initial_capital,
            "auto_refresh": auto_refresh,
            "update_interval": update_interval,
            "force_refresh": force_refresh
        }
        
        return settings

def create_header(title: str) -> None:
    """
    Create the application header
    
    Args:
        title: The title to display
    """
    st.title(title)
    st.markdown("""
    This application monitors cryptocurrency prices and provides technical analysis 
    indicators and trading signals.
    """)
    st.markdown("---")

def setup_auto_refresh(update_interval: int) -> Tuple[st.empty, st.empty]:
    """
    Set up auto-refresh components
    
    Args:
        update_interval: Update interval in seconds
        
    Returns:
        Tuple of containers for auto-refresh display and progress bar
    """
    if update_interval > 0:
        auto_refresh_container = st.empty()
        progress_bar = st.progress(0)
        return auto_refresh_container, progress_bar
    else:
        return st.empty(), st.empty()

def update_progress_bar(progress_bar: st.progress, start_time: float, update_interval: int) -> float:
    """
    Update the auto-refresh progress bar
    
    Args:
        progress_bar: Streamlit progress bar
        start_time: Start time in seconds
        update_interval: Update interval in seconds
        
    Returns:
        Current progress value (0-1)
    """
    if update_interval <= 0:
        return 0
    
    elapsed_time = time.time() - start_time
    progress = min(elapsed_time / update_interval, 1.0)
    progress_bar.progress(progress)
    return progress

def handle_backtest_button(symbol: str, lookback_days: int) -> None:
    """
    Handle the 'Run Backtest' button action
    
    Args:
        symbol: Selected trading symbol
        lookback_days: Number of days to lookback
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Run Full Backtest", key="run_backtest"):
            with st.spinner(f"Running backtest for {symbol}..."):
                results = run_backtest(symbol, days=lookback_days)
                
                if results:
                    st.success("Backtest completed successfully!")
                    
                    # Display key metrics
                    metrics = results.get('metrics', {})
                    if metrics:
                        st.subheader("Backtest Results")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                        
                        with metric_col2:
                            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                        
                        with metric_col3:
                            st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
                            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                            
                else:
                    st.error("Backtest failed. Check logs for details.")

def handle_optimization_button(symbol: str, lookback_days: int) -> None:
    """
    Handle the 'Find Best Parameters' button action
    
    Args:
        symbol: Selected trading symbol
        lookback_days: Number of days to lookback
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("Find Best Parameters", key="find_params"):
            with st.spinner(f"Finding best parameters for {symbol}..."):
                # Define parameter grid
                param_grid = {
                    "macd_weight": [0.1, 0.2, 0.3],
                    "rsi_weight": [0.1, 0.2, 0.3],
                    "stoch_weight": [0.1, 0.2, 0.3],
                    "fractal_weight": [0.1, 0.2, 0.3],
                    "rsi_period": [9, 14, 21],
                    "macd_fast": [8, 12, 16],
                    "macd_slow": [21, 26, 30],
                    "stoch_k": [9, 14, 21],
                    "stoch_d": [3, 5, 7]
                }
                
                best_params = find_best_params(symbol, param_grid, days=lookback_days)
                
                if best_params:
                    st.success("Optimization completed successfully!")
                    st.json(best_params)
                else:
                    st.error("Optimization failed. Check logs for details.")

def update_charts() -> None:
    """
    Update all charts and data displays using containers
    """
    settings = state.settings
    
    # Fetch data if needed (first time, forced refresh, or time interval passed)
    current_time = time.time()
    time_since_update = float('inf') if state.last_update_time is None else current_time - state.last_update_time
    
    need_update = (
        state.df is None or 
        settings["force_refresh"] or 
        (settings["auto_refresh"] and time_since_update >= settings["update_interval"])
    )
    
    if need_update:
        # Reset the timer
        state.start_time = current_time
        state.last_update_time = current_time
        
        # Fetch price data
        df, error_message = get_price_data(
            symbol=settings["symbol"],
            timeframe=settings["timeframe"],
            lookback_days=settings["lookback_days"]
        )
        
        if error_message:
            st.error(f"Error fetching data: {error_message}")
            return
        
        if df is None or df.empty:
            st.warning(f"No data available for {settings['symbol']} with the current settings.")
            return
        
        # Load best parameters if available
        params = load_best_params(settings["symbol"])
        
        # Generate signals and indicators
        signals_df, daily_composite, weekly_composite = generate_trading_signals(df, params)
        
        # Initialize portfolio dataframe
        portfolio_df = None
        performance_df = None
        
        # Run portfolio simulation if enabled
        if settings["enable_simulation"] and signals_df is not None:
            # Fetch data for multiple assets (for ranking)
            prices_dataset = get_multi_asset_data(
                config.TRADING_SYMBOLS, 
                settings["timeframe"], 
                settings["lookback_days"]
            )
            
            # Calculate performance ranking
            performance_df = calculate_performance_ranking(
                prices_dataset,
                lookback_days=settings["lookback_days"]
            )
            
            # Run portfolio simulation
            portfolio_df = simulate_portfolio(
                signals_df=signals_df,
                price_data=df,
                prices_dataset=prices_dataset,
                selected_symbol=settings["symbol"],
                lookback_days=settings["lookback_days"],
                initial_capital=settings["initial_capital"]
            )
        
        # Update state with new data
        state.df = df
        state.signals_df = signals_df
        state.daily_composite = daily_composite
        state.weekly_composite = weekly_composite
        state.portfolio_df = portfolio_df
        state.performance_df = performance_df
    
    # Create charts with all indicators
    if state.df is not None:
        main_chart = create_price_chart(
            price_data=state.df,
            signals_df=state.signals_df,
            daily_composite=state.daily_composite,
            weekly_composite=state.weekly_composite,
            portfolio_df=state.portfolio_df,
            performance_df=state.performance_df,
            show_macd=settings["show_macd"],
            show_rsi=settings["show_rsi"],
            show_stochastic=settings["show_stochastic"],
            show_fractal=settings["show_fractal"],
            show_signals=settings["show_signals"]
        )
        
        # Display the main chart
        st.plotly_chart(main_chart, use_container_width=True)
        
        # Create additional charts
        cols = st.columns(2)
        
        # Portfolio performance chart
        if state.portfolio_df is not None:
            with cols[0]:
                st.subheader("Portfolio Performance")
                portfolio_chart = create_portfolio_performance_chart(state.portfolio_df)
                st.plotly_chart(portfolio_chart, use_container_width=True)
        
        # Performance ranking chart
        if state.performance_df is not None:
            with cols[1]:
                st.subheader("Asset Performance Ranking")
                ranking_chart = create_performance_ranking_chart(state.performance_df)
                st.plotly_chart(ranking_chart, use_container_width=True)
        
        # Display metrics about the current data
        with st.expander("Data Details", expanded=False):
            st.write(f"Data Range: {state.df.index[0]} to {state.df.index[-1]}")
            st.write(f"Number of Data Points: {len(state.df)}")
            
            if state.signals_df is not None:
                buy_signals = len(state.signals_df[state.signals_df['signal'] == 1])
                sell_signals = len(state.signals_df[state.signals_df['signal'] == -1])
                st.write(f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}")
            
            if settings["auto_refresh"]:
                st.write(f"Next Update: in {settings['update_interval'] - (time.time() - state.last_update_time):.0f} seconds")

def main():
    # Create the header
    create_header("Cryptocurrency Price Monitor")
    
    # Initialize UI components and get settings
    settings = create_sidebar()
    
    # Update the state container with the new settings
    state.settings = settings
    
    # Set up auto-refresh components
    auto_refresh_placeholder, progress_bar = setup_auto_refresh(settings["update_interval"])
    
    # Display backtest button and handle logic
    handle_backtest_button(settings["symbol"], settings["lookback_days"])
    
    # Display optimization button and handle logic
    handle_optimization_button(settings["symbol"], settings["lookback_days"])
    
    # Update charts and displays
    update_charts()
    
    # Handle auto-refresh if enabled
    if settings["auto_refresh"] and settings["update_interval"] > 0:
        # Update the progress text
        next_update = settings["update_interval"] - (time.time() - state.start_time)
        auto_refresh_placeholder.info(f"Auto-refreshing in {max(0, int(next_update))} seconds")
        
        # Update the progress bar
        progress = update_progress_bar(
            progress_bar, 
            state.start_time, 
            settings["update_interval"]
        )
        
        # Check if it's time to refresh
        if progress >= 1.0:
            state.start_time = time.time()
            st.experimental_rerun()

if __name__ == "__main__":
    main()