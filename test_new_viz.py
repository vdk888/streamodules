"""
Test script for the new visualization module
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
from typing import Dict, Optional, Tuple

# Add current directory to sys.path
sys.path.append('.')

# Import required modules
from modules.data import get_price_data, get_multi_asset_data, generate_trading_signals
from modules.storage import load_best_params
from modules.simulation import simulate_portfolio
from modules.backtest import calculate_performance_ranking
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
    page_title="New Visualization Test",
    page_icon="📊",
    layout="wide"
)

# Main app
st.title("New Visualization Test")
st.markdown("This is a test for the new visualization module using Plotly.")

# Symbol selection
symbol_options = list(config.TRADING_SYMBOLS.keys())
selected_symbol = st.selectbox(
    "Select Symbol", 
    options=symbol_options,
    index=0
)

# Timeframe and lookback days
col1, col2 = st.columns(2)
with col1:
    timeframe = st.selectbox(
        "Timeframe",
        options=["1h", "4h", "1d"],
        index=0
    )
with col2:
    lookback_days = st.slider(
        "Lookback Days",
        min_value=3,
        max_value=30,
        value=5
    )

# Indicator toggles
col1, col2, col3, col4 = st.columns(4)
with col1:
    show_macd = st.checkbox("Show MACD", value=True)
with col2:
    show_rsi = st.checkbox("Show RSI", value=True)
with col3:
    show_stochastic = st.checkbox("Show Stochastic", value=True)
with col4:
    show_fractal = st.checkbox("Show Fractal", value=True)

# Enable portfolio simulation
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

# Fetch button
if st.button("Fetch Data and Generate Charts"):
    with st.spinner("Fetching data..."):
        # Fetch price data
        df, error_message = get_price_data(
            symbol=selected_symbol,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        
        if error_message:
            st.error(f"Error fetching data: {error_message}")
        elif df is None or df.empty:
            st.warning(f"No data available for {selected_symbol}.")
        else:
            st.success(f"Successfully fetched {len(df)} data points for {selected_symbol}.")
            
            # Load best parameters if available
            params = load_best_params(selected_symbol)
            
            # Generate signals and indicators
            signals_df, daily_composite, weekly_composite = generate_trading_signals(df, params)
            
            # Initialize portfolio and performance dataframes
            portfolio_df = None
            performance_df = None
            
            # Run portfolio simulation if enabled
            if enable_simulation and signals_df is not None:
                # Fetch data for multiple assets (for ranking)
                with st.spinner("Fetching market data for portfolio simulation..."):
                    prices_dataset = get_multi_asset_data(
                        config.TRADING_SYMBOLS,
                        timeframe,
                        lookback_days
                    )
                    
                    # Calculate performance ranking
                    performance_df = calculate_performance_ranking(
                        prices_dataset,
                        lookback_days=lookback_days
                    )
                    
                    # Run portfolio simulation
                    portfolio_df = simulate_portfolio(
                        signals_df=signals_df,
                        price_data=df,
                        prices_dataset=prices_dataset,
                        selected_symbol=selected_symbol,
                        lookback_days=lookback_days,
                        initial_capital=initial_capital
                    )
            
            # Create main chart with indicators
            st.subheader("Price Chart with Indicators")
            main_chart = create_price_chart(
                price_data=df,
                signals_df=signals_df,
                daily_composite=daily_composite,
                weekly_composite=weekly_composite,
                portfolio_df=portfolio_df,
                performance_df=performance_df,
                show_macd=show_macd,
                show_rsi=show_rsi,
                show_stochastic=show_stochastic,
                show_fractal=show_fractal,
                show_signals=True
            )
            st.plotly_chart(main_chart, use_container_width=True)
            
            # Create additional charts if data is available
            if portfolio_df is not None and performance_df is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Portfolio Performance")
                    portfolio_chart = create_portfolio_performance_chart(portfolio_df)
                    st.plotly_chart(portfolio_chart, use_container_width=True)
                
                with col2:
                    st.subheader("Asset Performance Ranking")
                    ranking_chart = create_performance_ranking_chart(performance_df)
                    st.plotly_chart(ranking_chart, use_container_width=True)
                
                # Create multi-asset chart
                st.subheader("Multi-Asset Comparison")
                # Take top 5 assets for comparison
                top_assets = performance_df.sort_values('performance', ascending=False).head(5).index.tolist()
                multi_chart_data = {sym: prices_dataset[sym] for sym in top_assets if sym in prices_dataset}
                
                if multi_chart_data:
                    multi_chart = create_multi_asset_chart(multi_chart_data, lookback_days)
                    st.plotly_chart(multi_chart, use_container_width=True)
            
            # Show data details
            with st.expander("Data Details"):
                st.write(f"Data Range: {df.index[0]} to {df.index[-1]}")
                st.write(f"Number of Data Points: {len(df)}")
                
                if signals_df is not None:
                    buy_signals = len(signals_df[signals_df['signal'] == 1])
                    sell_signals = len(signals_df[signals_df['signal'] == -1])
                    st.write(f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}")
                
                if portfolio_df is not None:
                    final_value = portfolio_df['portfolio_value'].iloc[-1]
                    roi = (final_value / initial_capital - 1) * 100
                    st.metric("Final Portfolio Value", f"${final_value:.2f}", f"{roi:.2f}%")