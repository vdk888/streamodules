import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import datetime
import time
import sys
import json
import pytz
from datetime import timedelta

# Add the attached_assets directory to sys.path
sys.path.append('.')

from utils import fetch_and_process_data, generate_signals_with_indicators
from attached_assets.config import TRADING_SYMBOLS, TRADING_COSTS, DEFAULT_RISK_PERCENT
from attached_assets.indicators import get_default_params
# Import backtest functions from backtest_individual.py
from attached_assets.backtest_individual import run_backtest, find_best_params, calculate_performance_ranking as backtest_calculate_ranking
try:
    from replit.object_storage import Client
    object_storage_available = True
except ImportError:
    object_storage_available = False

# Page configuration
st.set_page_config(
    page_title="Crypto Price Monitor - BTC-USD",
    page_icon="📈",
    layout="wide"
)

# Sidebar for controls
st.sidebar.header("Settings")

# Symbol selection
symbol_options = list(TRADING_SYMBOLS.keys())
selected_symbol = st.sidebar.selectbox(
    "Symbol",
    symbol_options,
    index=0  # BTC/USD is the first
)

# Header
st.title(f"{selected_symbol} Price Monitor with Technical Indicators")
st.markdown("Real-time price data with signal alerts based on technical indicators")

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
    [55, 150, 300, 600],
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

    # Add button to run a backtest using the function from backtest_individual.py
    if st.button("Run Full Backtest"):
        with st.spinner("Running backtest simulation... This may take a minute..."):
            try:
                # Get parameter preference
                use_best_params = st.session_state.get('use_best_params', True)
                params = get_default_params() if not use_best_params else None
                backtest_result = run_backtest(
                    symbol=selected_symbol, 
                    days=lookback_days,
                    params=params,  # Pass default params if not using optimized ones
                    is_simulating=True,
                    lookback_days_param=lookback_days
                )

                if backtest_result:
                    st.success(f"Backtest complete for {selected_symbol}!")

                    # Display key performance metrics
                    stats = backtest_result.get('stats', {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{stats.get('total_return', 0):.2f}%")
                    with col2:
                        st.metric("Max Drawdown", f"{stats.get('max_drawdown', 0):.2f}%")
                    with col3:
                        st.metric("Win Rate", f"{stats.get('win_rate', 0):.2f}%")
                    with col4:
                        st.metric("Sharpe Ratio", f"{stats.get('sharpe_ratio', 0):.2f}")

                    # If there's a plot available in the result, display it
                    plot_data = backtest_result.get('plot_data', None)
                    if plot_data:
                        st.plotly_chart(plot_data, use_container_width=True)
                else:
                    st.warning("Could not complete backtest. Try a different timeframe.")
            except Exception as e:
                st.error(f"Error during backtest: {str(e)}")
                st.info("Check the logs for more details.")

# Best parameters management
with st.sidebar.expander("Parameter Management", expanded=True):
    st.text("Composite Indicator Parameters")
    # Initialize session state if needed
    if 'use_best_params' not in st.session_state:
        st.session_state['use_best_params'] = True
    use_best_params = st.checkbox("Use Optimized Parameters", key='use_best_params')

    # Add button to optimize parameters
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

# Button to force refresh data
if st.sidebar.button("Refresh Data Now"):
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

# Function to load best parameters from Replit Object Storage
def load_best_params(symbol):
    """
    Load best parameters for a symbol from Replit Object Storage

    Args:
        symbol: Trading symbol (e.g., 'BTC/USD')

    Returns:
        Dict of parameters or None if not found
    """
    best_params_file = "best_params.json"

    try:
        # Try to get parameters from Replit Object Storage first
        if object_storage_available:
            # Initialize Object Storage client and verify connection
            try:
                client = Client()
                # Test connection by listing contents
                client.list()
                st.success("Successfully connected to Replit Object Storage")
            except Exception as e:
                st.error(f"Failed to connect to Replit Object Storage: {str(e)}")
                st.stop()

            try:
                json_content = client.download_as_text(best_params_file)
                best_params_data = json.loads(json_content)
                st.success("Successfully loaded parameters from Replit Object Storage")

                if symbol in best_params_data:
                    params = best_params_data[symbol]['best_params']
                    return params
                else:
                    st.warning(f"No optimized parameters found for {symbol} in Replit Object Storage. Using defaults.")
                    return None
            except Exception as e:
                st.warning(f"Error accessing Replit Object Storage: {str(e)}. Falling back to local file.")

        # Fallback to local file if needed
        with open(best_params_file, "r") as f:
            best_params_data = json.load(f)
            st.info("Loaded parameters from local file as fallback")

            if symbol in best_params_data:
                params = best_params_data[symbol]['best_params']
                return params
            else:
                st.warning(f"No optimized parameters found for {symbol}. Using defaults.")
                return None
    except Exception as e:
        st.warning(f"Error loading parameters: {str(e)}. Using defaults.")
        return None

# Function to simulate portfolio based on signals
def simulate_portfolio(signals_df, price_data, initial_capital=100000):
    """
    Simulate portfolio performance based on trading signals

    Args:
        signals_df: DataFrame with trading signals
        price_data: DataFrame with price data
        initial_capital: Initial capital in USD

    Returns:
        DataFrame with portfolio performance
    """
    if signals_df is None or signals_df.empty:
        return None

    # Initialize portfolio tracking
    position = 0  # Current position in shares
    cash = initial_capital
    portfolio_value = []  # Portfolio value over time
    shares_owned = []  # Shares owned over time

    # For each row in the signals DataFrame
    for idx, row in signals_df.iterrows():
        signal = row.get('signal', 0)
        price = price_data.loc[idx, 'close'] if idx in price_data.index else 0

        if price == 0:
            continue

        # Process buy signal
        if signal == 1 and cash > 0:
            # Calculate how many shares to buy (use 95% of available cash)
            amount_to_use = cash * 0.95
            shares_to_buy = amount_to_use / price

            # Crypto can be fractional, stocks might need to be whole numbers
            shares_to_buy = round(shares_to_buy, 8)

            # Update position
            position += shares_to_buy
            cash -= shares_to_buy * price

        # Process sell signal    
        elif signal == -1 and position > 0:
            # Sell all shares
            cash += position * price
            position = 0

        # Calculate portfolio value
        current_value = cash + (position * price)
        portfolio_value.append(current_value)
        shares_owned.append(position)

    # Create a DataFrame for the portfolio history
    portfolio_df = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'shares_owned': shares_owned
    }, index=signals_df.index)

    return portfolio_df

# Create placeholders for charts
price_chart_placeholder = st.empty()
indicators_placeholder = st.empty()
signals_placeholder = st.empty()
portfolio_placeholder = st.empty()

# Calculate performance ranking for assets
def calculate_performance_ranking(prices_dataset=None, lookback_days=15):
    """Calculate simple performance ranking across assets for display"""
    if prices_dataset is None:
        prices_dataset = {}
        # Fetch data for multiple assets
        for symbol, config in TRADING_SYMBOLS.items():
            yf_symbol = config['yfinance']
            if '/' in yf_symbol:
                yf_symbol = yf_symbol.replace('/', '-')

            try:
                from attached_assets.fetch import fetch_historical_data
                data = fetch_historical_data(symbol, interval=timeframe, days=lookback_days)
                if not data.empty:
                    prices_dataset[symbol] = data
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {str(e)}")

    # Calculate performance for each asset
    performance_dict = {}
    for symbol, data in prices_dataset.items():
        try:
            if len(data) >= 2:
                # Make column names lowercase if they aren't already
                if data.columns[0].isupper():
                    data.columns = data.columns.str.lower()

                start_price = data['close'].iloc[0]
                end_price = data['close'].iloc[-1]
                performance = ((end_price - start_price) / start_price) * 100
                performance_dict[symbol] = performance
        except Exception as e:
            st.warning(f"Error calculating performance for {symbol}: {str(e)}")

    # Convert to DataFrame and calculate rankings
    if performance_dict:
        perf_df = pd.DataFrame.from_dict(performance_dict, 
                                        orient='index', 
                                        columns=['performance'])
        perf_df['rank'] = perf_df['performance'].rank(method='dense', pct=True)
        return perf_df
    return None

# Function to update the charts
def update_charts():
    # Load optimal parameters if enabled
    params = None
    if st.session_state.get('use_best_params', True):
        params = load_best_params(selected_symbol)
    else:
        params = get_default_params()

    # Fetch and process the data
    df, error = fetch_and_process_data(selected_symbol, timeframe, lookback_days)

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

    # Create price chart (without the composite indicators)
    fig1 = make_subplots(rows=1, cols=1)

    # Add candlestick chart
    fig1.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'], 
            high=df['high'],
            low=df['low'], 
            close=df['close'],
            name="Price"
        )
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

    # Composite indicators will be placed in separate subplots

    # Set y-axis title
    fig1.update_yaxes(title_text="Price (USD)")

    # Calculate dynamic y-axis range to better fit the price data
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    price_padding = price_range * 0.05  # Add 5% padding

    # Update layout with better y-axis scaling
    fig1.update_layout(
        title=f"{selected_symbol} Price with Signal Indicators",
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

    # Display the price chart with a timestamp-based key to avoid duplicate ID error
    price_chart_placeholder.plotly_chart(fig1, use_container_width=True, key=f"price_chart_{int(time.time())}")

    # Create technical indicators chart with 6 subplots (composite indicators first)
    fig2 = make_subplots(
        rows=6, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.18, 0.18, 0.16, 0.16, 0.16, 0.16],
        subplot_titles=("Daily Composite", "Weekly Composite", "MACD", "RSI", "Stochastic", "Fractal Complexity")
    )

    # Add Daily Composite indicator first
    if not signals_df.empty and 'daily_composite' in signals_df.columns:
        # Daily composite line
        fig2.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['daily_composite'],
                mode='lines',
                name="Daily Composite",
                line=dict(color='royalblue', width=2.5)
            ),
            row=1, col=1
        )

        # Daily threshold lines (1 STD)
        if 'daily_up_lim' in signals_df.columns:
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['daily_up_lim'],
                    mode='lines',
                    name="Daily Upper 1σ",
                    line=dict(color='green', width=1.5, dash='dash')
                ),
                row=1, col=1
            )

            # Daily 2 STD upper
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['daily_up_lim_2std'],
                    mode='lines',
                    name="Daily Upper 2σ",
                    line=dict(color='darkgreen', width=1, dash='dot')
                ),
                row=1, col=1
            )

        if 'daily_down_lim' in signals_df.columns:
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['daily_down_lim'],
                    mode='lines',
                    name="Daily Lower 1σ",
                    line=dict(color='red', width=1.5, dash='dash')
                ),
                row=1, col=1
            )

            # Daily 2 STD lower
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['daily_down_lim_2std'],
                    mode='lines',
                    name="Daily Lower 2σ",
                    line=dict(color='darkred', width=1, dash='dot')
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

    # Add Weekly Composite indicator
    if not signals_df.empty and 'weekly_composite' in signals_df.columns:
        # Weekly composite line
        fig2.add_trace(
            go.Scatter(
                x=signals_df.index,
                y=signals_df['weekly_composite'],
                mode='lines',
                name="Weekly Composite",
                line=dict(color='purple', width=2.5)
            ),
            row=2, col=1
        )

        # Weekly threshold lines (1 STD)
        if 'weekly_up_lim' in signals_df.columns:
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['weekly_up_lim'],
                    mode='lines',
                    name="Weekly Upper 1σ",
                    line=dict(color='darkgreen', width=1.5, dash='dot')
                ),
                row=2, col=1
            )

            # Weekly 2 STD upper
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['weekly_up_lim_2std'],
                    mode='lines',
                    name="Weekly Upper 2σ",
                    line=dict(color='green', width=1, dash='dashdot')
                ),
                row=2, col=1
            )

        if 'weekly_down_lim' in signals_df.columns:
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['weekly_down_lim'],
                    mode='lines',
                    name="Weekly Lower 1σ",
                    line=dict(color='darkred', width=1.5, dash='dot')
                ),
                row=2, col=1
            )

            # Weekly 2 STD lower
            fig2.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['weekly_down_lim_2std'],
                    mode='lines',
                    name="Weekly Lower 2σ",
                    line=dict(color='red', width=1, dash='dashdot')
                ),
                row=2, col=1
            )

        # Add zero line
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]],
                y=[0, 0],
                name="Zero Line",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=2, col=1
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
            row=4, col=1
        )
        # Add overbought/oversold lines
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[70, 70], 
                name="Overbought",
                line=dict(color='red', width=1, dash='dash')
            ),
            row=4, col=1
        )
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[30, 30], 
                name="Oversold",
                line=dict(color='green', width=1, dash='dash')
            ),
            row=4, col=1
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
            row=5, col=1
        )
        # Add zero line
        fig2.add_trace(
            go.Scatter(
                x=[df.index[0], df.index[-1]], 
                y=[0, 0], 
                name="Zero Line",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=5, col=1
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
            row=6, col=1
        )

    # Update layout
    fig2.update_layout(
        height=1000,  # Increased height to accommodate the additional subplots
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    # Display the indicators chart with a timestamp-based key to avoid duplicate ID error
    indicators_placeholder.plotly_chart(fig2, use_container_width=True, key=f"indicators_chart_{int(time.time())}")

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

    # Add portfolio simulation if enabled
    if enable_simulation:
        with portfolio_placeholder.container():
            st.subheader("Portfolio Simulation")

            # Simulate the portfolio
            portfolio_df = simulate_portfolio(signals_df, df, initial_capital=initial_capital)

            if portfolio_df is not None and not portfolio_df.empty:
                # Create portfolio value chart
                fig_portfolio = make_subplots(rows=2, cols=1, 
                                             shared_xaxes=True,
                                             vertical_spacing=0.05,
                                             row_heights=[0.7, 0.3],
                                             subplot_titles=("Portfolio Value", "Shares Owned"))

                # Add portfolio value line
                fig_portfolio.add_trace(
                    go.Scatter(
                        x=portfolio_df.index,
                        y=portfolio_df['portfolio_value'],
                        mode='lines',
                        name="Portfolio Value",
                        line=dict(color='green', width=2)
                    ),
                    row=1, col=1
                )

                # Add horizontal line for initial capital
                fig_portfolio.add_trace(
                    go.Scatter(
                        x=[portfolio_df.index[0], portfolio_df.index[-1]],
                        y=[initial_capital, initial_capital],
                        mode='lines',
                        name="Initial Capital",
                        line=dict(color='gray', width=1, dash='dash')
                    ),
                    row=1, col=1
                )

                # Add shares owned line
                fig_portfolio.add_trace(
                    go.Scatter(
                        x=portfolio_df.index,
                        y=portfolio_df['shares_owned'],
                        mode='lines',
                        name="Shares Owned",
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=1
                )

                # Update layout
                fig_portfolio.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=True
                )

                # Set y-axis titles
                fig_portfolio.update_yaxes(title_text="Value ($)", row=1, col=1)
                fig_portfolio.update_yaxes(title_text="Shares", row=2, col=1)

                # Display the portfolio chart
                st.plotly_chart(fig_portfolio, use_container_width=True, key=f"portfolio_chart_{int(time.time())}")

                # Show portfolio statistics
                if len(portfolio_df) > 0:
                    final_value = portfolio_df['portfolio_value'].iloc[-1]
                    max_value = portfolio_df['portfolio_value'].max()
                    min_value = portfolio_df['portfolio_value'].min()
                    max_drawdown = ((max_value - min_value) / max_value) * 100
                    roi = ((final_value - initial_capital) / initial_capital) * 100

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Value", f"${final_value:,.2f}", f"{roi:.2f}%")
                    col2.metric("Maximum Value", f"${max_value:,.2f}")
                    col3.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
            else:
                st.info("No signals available to simulate portfolio. Try a different timeframe or symbol.")

    # Add asset performance comparison
    st.subheader("Asset Performance Comparison")
    performance_df = calculate_performance_ranking(lookback_days=lookback_days)

    if performance_df is not None and not performance_df.empty:
        # Create a bar chart for the performance ranking
        fig_perf = go.Figure()

        # Sort by performance descending
        performance_df = performance_df.sort_values('performance', ascending=False)

        # Add bars
        fig_perf.add_trace(
            go.Bar(
                x=performance_df.index,
                y=performance_df['performance'],
                marker_color=['green' if x > 0 else 'red' for x in performance_df['performance']],
                text=[f"{x:.2f}%" for x in performance_df['performance']],
                textposition='outside'
            )
        )

        # Update layout
        fig_perf.update_layout(
            title="Performance Over Selected Period",
            yaxis_title="% Change",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        # Display the performance chart
        st.plotly_chart(fig_perf, use_container_width=True, key=f"perf_chart_{int(time.time())}")

        # Display a table with more details
        performance_df['performance'] = performance_df['performance'].apply(lambda x: f"{x:.2f}%")
        performance_df['rank'] = performance_df['rank'].apply(lambda x: f"{x*100:.1f}%")
        performance_df.columns = ['Performance', 'Percentile Rank']

        st.dataframe(performance_df, use_container_width=True)
    else:
        st.info("Unable to calculate performance ranking. Try a different timeframe.")

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