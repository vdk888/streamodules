import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import time
import sys

# Add the attached_assets directory to sys.path
sys.path.append('.')

from utils import fetch_and_process_data, generate_signals_with_indicators

# Page configuration
st.set_page_config(
    page_title="Crypto Price Monitor - BTC-USD",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("BTC-USD Price Monitor with Technical Indicators")
st.markdown("Real-time price data with signal alerts based on technical indicators")

# Sidebar for controls
st.sidebar.header("Settings")

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

# Button to force refresh data
if st.sidebar.button("Refresh Data Now"):
    st.sidebar.success("Data refreshed!")

# Display information about the app
with st.sidebar.expander("About", expanded=False):
    st.markdown("""
    This app displays real-time BTC-USD price data with technical indicators and generates buy/sell signals.
    
    **Features:**
    - Real-time price data from Yahoo Finance
    - Technical indicators (MACD, RSI, Stochastic)
    - Fractal complexity analysis
    - Buy/sell signals based on indicator combinations
    
    *Data updates automatically based on the selected interval.*
    """)

# Create placeholders for charts
price_chart_placeholder = st.empty()
indicators_placeholder = st.empty()
signals_placeholder = st.empty()

# Function to update the charts
def update_charts():
    # Fetch and process the data
    df, error = fetch_and_process_data('BTC/USD', timeframe, lookback_days)
    
    if error:
        st.error(f"Error fetching data: {error}")
        return
    
    if df is None or df.empty:
        st.warning("No data available. Please try a different timeframe or lookback period.")
        return
    
    # Generate signals and indicators
    signals_df, daily_data, weekly_data = generate_signals_with_indicators(df)
    
    if signals_df is None or signals_df.empty:
        st.warning("Could not generate signals. Please try a different timeframe or lookback period.")
        return
    
    # Create price chart with a secondary y-axis for indicators
    fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])
    
    # Add candlestick chart
    fig1.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'], 
            high=df['high'],
            low=df['low'], 
            close=df['close'],
            name="Price"
        ),
        secondary_y=False
    )
    
    # Add buy signals if available
    if show_signals and 'signal' in signals_df.columns:
        # Get points where signal is 1 (buy)
        buy_signals = signals_df[signals_df['signal'] == 1]
        if not buy_signals.empty:
            fig1.add_trace(
                go.Scatter(
                    x=buy_signals.index, 
                    y=df.loc[buy_signals.index, 'high'] * 1.01,  # Slightly above the candle
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name="Buy Signal"
                )
            )
        
        # Get points where signal is -1 (sell)
        sell_signals = signals_df[signals_df['signal'] == -1]
        if not sell_signals.empty:
            fig1.add_trace(
                go.Scatter(
                    x=sell_signals.index, 
                    y=df.loc[sell_signals.index, 'low'] * 0.99,  # Slightly below the candle
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name="Sell Signal"
                )
            )
    
    # Daily composite line
    if not signals_df.empty and 'daily_composite' in signals_df.columns:
        fig1.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['daily_composite'],
                mode='lines',
                name="Daily Composite",
                line=dict(color='royalblue', width=2.5)
            ),
            secondary_y=True
        )
        
        # Daily threshold lines
        if 'daily_up_lim' in signals_df.columns:
            fig1.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['daily_up_lim'],
                    mode='lines',
                    name="Daily Upper Threshold",
                    line=dict(color='green', width=1.5, dash='dash')
                ),
                secondary_y=True
            )
        
        if 'daily_down_lim' in signals_df.columns:
            fig1.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['daily_down_lim'],
                    mode='lines',
                    name="Daily Lower Threshold",
                    line=dict(color='red', width=1.5, dash='dash')
                ),
                secondary_y=True
            )
    
    # Weekly composite line
    if not signals_df.empty and 'weekly_composite' in signals_df.columns:
        fig1.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['weekly_composite'],
                mode='lines',
                name="Weekly Composite",
                line=dict(color='purple', width=2.5)
            ),
            secondary_y=True
        )
        
        # Weekly threshold lines
        if 'weekly_up_lim' in signals_df.columns:
            fig1.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['weekly_up_lim'],
                    mode='lines',
                    name="Weekly Upper Threshold",
                    line=dict(color='darkgreen', width=1.5, dash='dot')
                ),
                secondary_y=True
            )
        
        if 'weekly_down_lim' in signals_df.columns:
            fig1.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['weekly_down_lim'],
                    mode='lines',
                    name="Weekly Lower Threshold",
                    line=dict(color='darkred', width=1.5, dash='dot')
                ),
                secondary_y=True
            )
    
    # Set y-axis titles
    fig1.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig1.update_yaxes(title_text="Indicator Value", secondary_y=True)
    
    # Calculate dynamic y-axis range to better fit the price data
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    price_padding = price_range * 0.05  # Add 5% padding
    
    # Update layout with better y-axis scaling
    fig1.update_layout(
        title="BTC-USD Price with Signal Indicators",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            range=[price_min - price_padding, price_max + price_padding],
            autorange=False
        )
    )
    
    # Display the price chart with a unique key to avoid duplicate ID error
    price_chart_placeholder.plotly_chart(fig1, use_container_width=True, key="price_chart")
    
    # Create technical indicators chart
    fig2 = make_subplots(
        rows=4, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=("MACD", "RSI", "Stochastic", "Fractal Complexity")
    )
    
    # Add MACD
    if show_macd:
        from attached_assets.indicators import calculate_macd
        macd = calculate_macd(df)
        fig2.add_trace(
            go.Scatter(
                x=df.index, 
                y=macd, 
                name="MACD",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        # Add zero line
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[0, 0], 
                name="Zero Line",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Add RSI
    if show_rsi:
        from attached_assets.indicators import calculate_rsi
        rsi = calculate_rsi(df)
        fig2.add_trace(
            go.Scatter(
                x=df.index, 
                y=rsi, 
                name="RSI",
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )
        # Add overbought/oversold lines
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[70, 70], 
                name="Overbought",
                line=dict(color='red', width=1, dash='dash')
            ),
            row=2, col=1
        )
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[30, 30], 
                name="Oversold",
                line=dict(color='green', width=1, dash='dash')
            ),
            row=2, col=1
        )
    
    # Add Stochastic
    if show_stochastic:
        from attached_assets.indicators import calculate_stochastic
        stoch = calculate_stochastic(df)
        fig2.add_trace(
            go.Scatter(
                x=df.index, 
                y=stoch, 
                name="Stochastic",
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        # Add zero line
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[0, 0], 
                name="Zero Line",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=3, col=1
        )
    
    # Add Fractal Complexity
    if show_fractal:
        from attached_assets.indicators import calculate_fractal_complexity
        complexity = calculate_fractal_complexity(df)
        fig2.add_trace(
            go.Scatter(
                x=df.index, 
                y=complexity, 
                name="Fractal Complexity",
                line=dict(color='teal', width=1)
            ),
            row=4, col=1
        )
    
    # Update layout
    fig2.update_layout(
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    # Display the indicators chart with a unique key to avoid duplicate ID error
    indicators_placeholder.plotly_chart(fig2, use_container_width=True, key="indicators_chart")
    
    # Display signal information
    recent_signals = pd.DataFrame()
    if not signals_df.empty:
        # Get recent signals
        recent_signals = signals_df[signals_df['signal'] != 0].tail(5)
    
    if not recent_signals.empty:
        with signals_placeholder.container():
            st.subheader("Recent Signals")
            
            # Format the signals for display
            signal_display = []
            for idx, row in recent_signals.iterrows():
                signal_type = "🟢 BUY" if row['signal'] == 1 else "🔴 SELL"
                price = df.loc[idx, 'close']
                daily_composite = row.get('daily_composite', 'N/A')
                weekly_composite = row.get('weekly_composite', 'N/A')
                
                signal_display.append({
                    "Time": idx.strftime("%Y-%m-%d %H:%M"),
                    "Signal": signal_type,
                    "Price": f"${price:.2f}",
                    "Daily Composite": f"{daily_composite:.4f}" if isinstance(daily_composite, (int, float)) else daily_composite,
                    "Weekly Composite": f"{weekly_composite:.4f}" if isinstance(weekly_composite, (int, float)) else weekly_composite
                })
            
            st.table(pd.DataFrame(signal_display))
    else:
        signals_placeholder.info("No signals generated in the selected timeframe.")

    # Show last update time
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main app logic - update continuously
update_charts()

# Auto-update the charts
if st.checkbox("Enable auto-refresh", value=True):
    auto_refresh_placeholder = st.empty()
    progress_bar = auto_refresh_placeholder.progress(0)
    
    while True:
        for i in range(update_interval):
            # Update progress bar
            progress = (i + 1) / update_interval
            progress_bar.progress(progress)
            
            # Sleep for 1 second
            time.sleep(1)
        
        # Reset progress bar
        progress_bar.empty()
        
        # Update charts
        update_charts()
        
        # Show updated time
        auto_refresh_placeholder.success(f"Data refreshed at {datetime.datetime.now().strftime('%H:%M:%S')}")
        time.sleep(2)  # Show the success message for 2 seconds
        progress_bar = auto_refresh_placeholder.progress(0)  # Reset the progress bar
