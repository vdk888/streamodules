import streamlit as st
import pandas as pd
import time
import sys
import datetime
from typing import Dict, Optional, Tuple

# Add the current directory to sys.path
sys.path.append('.')

# Import all modules for easier access
from modules import (
    create_sidebar, create_header, setup_auto_refresh, update_progress_bar,
    handle_backtest_button, handle_optimization_button,
    get_price_data, get_multi_asset_data, generate_trading_signals,
    load_best_params,
    simulate_portfolio,
    create_price_chart, create_performance_ranking_chart, create_portfolio_performance_chart,
    calculate_performance_ranking
)

# Import configuration
from modules.config import TRADING_SYMBOLS

# Page configuration
st.set_page_config(
    page_title="Crypto Price Monitor - BTC-USD",
    page_icon="📈",
    layout="wide"
)

def update_charts():
    """
    Update all charts and data displays using containers
    """
    # Initialize containers if they don't exist
    if 'chart_container' not in st.session_state:
        st.session_state.chart_container = st.empty()
    if 'portfolio_container' not in st.session_state:
        st.session_state.portfolio_container = st.empty()
    if 'metrics_container' not in st.session_state:
        st.session_state.metrics_container = st.empty()
    if 'ranking_container' not in st.session_state:
        st.session_state.ranking_container = st.empty()

    # Get UI settings
    settings = state_container.settings

    # Load optimal parameters if enabled
    params = None
    if settings["use_best_params"]:
        params = load_best_params(settings["symbol"])  # Use "symbol" key

    # Fetch and process the data
    df, error = get_price_data(settings["symbol"], settings["timeframe"], settings["lookback_days"])

    if error:
        st.error(f"Error fetching data: {error}")
        return

    if df is None or df.empty:
        st.warning("No data available. Please try a different timeframe or lookback period.")
        return

    # Generate signals based on the data
    signals_df, daily_composite, weekly_composite = generate_trading_signals(df, params)

    # If portfolio simulation is enabled, calculate portfolio performance
    portfolio_df = None
    if settings["enable_simulation"] and signals_df is not None:
        # Fetch data for multiple assets (for ranking)
        prices_dataset = get_multi_asset_data(TRADING_SYMBOLS, settings["timeframe"], settings["lookback_days"])

        # Run portfolio simulation
        portfolio_df = simulate_portfolio(
            signals_df=signals_df,
            price_data=df,
            prices_dataset=prices_dataset,
            selected_symbol=settings["symbol"],  # Use "symbol" key
            lookback_days=settings["lookback_days"],
            initial_capital=settings["initial_capital"]
        )

    # Create charts with all the indicator displays
    main_chart = create_price_chart(
        price_data=df,
        signals_df=signals_df,
        daily_composite=daily_composite,
        weekly_composite=weekly_composite,
        portfolio_df=portfolio_df,
        show_macd=settings["show_macd"],
        show_rsi=settings["show_rsi"],
        show_stochastic=settings["show_stochastic"],
        show_fractal=settings["show_fractal"],
        show_signals=settings["show_signals"]
    )

    # Create a new chart container
    st.session_state.chart_container = st.empty()
    
    # Update the main chart with a unique key based on symbol, timeframe and timestamp
    # Using import time at the top of file
    import time
    timestamp = int(time.time())
    unique_chart_key = f"main_chart_{settings['symbol']}_{settings['timeframe']}_{timestamp}"
    
    # Add the chart to the container
    with st.session_state.chart_container:
        st.plotly_chart(main_chart, use_container_width=True, key=unique_chart_key)

    # Update portfolio performance if available
    if portfolio_df is not None:
        # Create a new container for portfolio
        st.session_state.portfolio_container = st.empty()
        
        # Add content to the container
        with st.session_state.portfolio_container:
            st.subheader("Portfolio Performance")
            portfolio_chart = create_portfolio_performance_chart(portfolio_df)
            if portfolio_chart:
                timestamp = int(time.time())
                unique_portfolio_key = f"portfolio_chart_{settings['symbol']}_{settings['timeframe']}_{timestamp}"
                st.plotly_chart(portfolio_chart, use_container_width=True, key=unique_portfolio_key)

        # Create a new metrics container
        st.session_state.metrics_container = st.empty()
        
        # Add content to the metrics container
        with st.session_state.metrics_container:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                initial_value = portfolio_df['portfolio_value'].iloc[0]
                final_value = portfolio_df['portfolio_value'].iloc[-1]
                total_return = ((final_value / initial_value) - 1) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            with col2:
                max_drawdown = portfolio_df['drawdown'].max() * 100
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
            with col3:
                sharpe_ratio = portfolio_df['returns'].mean() / portfolio_df['returns'].std() if portfolio_df['returns'].std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            with col4:
                profit_days = (portfolio_df['returns'] > 0).sum()
                total_days = len(portfolio_df)
                win_rate = (profit_days / total_days) * 100 if total_days > 0 else 0
                st.metric("Win Rate", f"{win_rate:.2f}%")

    # Update asset performance ranking
    # Create a new container for ranking
    st.session_state.ranking_container = st.empty()
    
    # Add content to the ranking container
    with st.session_state.ranking_container:
        st.subheader("Asset Performance Ranking")
        
        # Fetch ranking data
        prices_dataset = get_multi_asset_data(TRADING_SYMBOLS, settings["timeframe"], settings["lookback_days"])
        performance_df = calculate_performance_ranking(prices_dataset, settings["lookback_days"])

        if performance_df is not None:
            ranking_chart = create_performance_ranking_chart(performance_df)
            if ranking_chart:
                timestamp = int(time.time())
                unique_ranking_key = f"ranking_chart_{settings['timeframe']}_{settings['lookback_days']}_{timestamp}"
                st.plotly_chart(ranking_chart, use_container_width=True, key=unique_ranking_key)
        else:
            st.warning("Could not calculate performance ranking. Try a different timeframe or lookback period.")

# Create a class to hold state
class StateContainer:
    def __init__(self):
        self.settings = {
            "symbol": "BTC/USD",  # Default symbol
            "timeframe": "1h",    # Default timeframe
            "lookback_days": 5,   # Default lookback days
            "auto_refresh": True, # Auto-refresh enabled by default
            "update_interval": 60, # 60 seconds update interval
            "enable_simulation": True, # Enable portfolio simulation
            "initial_capital": 100000, # Initial capital for simulation
            "show_macd": True,    # Show MACD indicator
            "show_rsi": True,     # Show RSI indicator
            "show_stochastic": True, # Show Stochastic indicator
            "show_fractal": True, # Show Fractal Complexity indicator
            "show_signals": True, # Show trading signals
            "selected_symbol": "BTC/USD", # Selected trading symbol
            "use_best_params": False, # Use best parameters from optimization
            "force_refresh": False # Whether to force a refresh
        }

# Initialize session state
if 'state_container' not in st.session_state:
    st.session_state.state_container = StateContainer()

state_container = st.session_state.state_container

# Main application flow
def main():
    # Create the header
    create_header("Cryptocurrency Price Monitor")

    # Initialize UI components and get settings
    settings = create_sidebar()

    # Update the state_container settings with the values from the UI
    # Make sure we have a 'selected_symbol' key that matches 'symbol'
    settings['selected_symbol'] = settings['symbol']

    # Add the missing keys that our state_container expects
    if 'force_refresh' not in settings:
        settings['force_refresh'] = False
    if 'use_best_params' not in settings:
        settings['use_best_params'] = False

    state_container.settings = settings

    # Set up auto-refresh components
    auto_refresh_placeholder, progress_bar = setup_auto_refresh(settings["update_interval"])

    # Display backtest button and handle logic
    handle_backtest_button(settings["symbol"], settings["lookback_days"])

    # Display optimization button and handle logic
    handle_optimization_button(settings["symbol"], settings["lookback_days"])

    # Update charts
    update_charts()

    # Auto-refresh logic
    if settings["auto_refresh"]:  # Only run auto-refresh if enabled
        start_time = time.time()
        update_interval = settings["update_interval"]

        while True:
            progress = update_progress_bar(progress_bar, start_time, update_interval)

            if progress >= 1.0:
                # Time to refresh
                update_charts()

                # Reset progress bar
                progress_bar.empty()

                # Show updated time
                auto_refresh_placeholder.success(f"Data refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
                time.sleep(2)  # Show the success message for 2 seconds
                progress_bar = auto_refresh_placeholder.progress(0)  # Reset the progress bar

                # Reset start time
                start_time = time.time()

            # Pause for a moment
            time.sleep(0.1)

if __name__ == "__main__":
    main()