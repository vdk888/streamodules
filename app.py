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

    # Add button to run a backtest using the function from backtest_individual.py
    if st.button("Run Full Backtest"):
        with st.spinner("Running backtest simulation... This may take a minute..."):
            try:
                # Use the run_backtest function from backtest_individual.py
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
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
                    with col2:
                        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                    with col3:
                        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
                    with col4:
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")

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
    use_best_params = st.checkbox("Use Optimized Parameters", value=True)

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
                st.write("Attempting to initialize Replit Object Storage client...")
                client = Client()
                st.write("Client initialized successfully")

                # Try to get parameters from Object Storage
                try:
                    st.write("Attempting to download best_params.json...")
                    json_content = client.download_as_text("best_params.json")
                    st.write(f"Downloaded content: {json_content[:100]}...")  # Show first 100 chars
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
    trade_ranks = []  # Store the rank for each trade

    # For iterating through crypto assets, fetch data for all
    prices_dataset = {}
    # Fetch data for multiple assets (only once at the beginning)
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

    # For each row in the signals DataFrame
    for idx, row in signals_df.iterrows():
        signal = row.get('signal', 0)
        price = price_data.loc[idx, 'close'] if idx in price_data.index else 0
        current_rank = None

        if price == 0:
            continue

        # Calculate performance ranking at this timestamp if we have enough data
        if prices_dataset:
            # Use the backtest's ranking algorithm
            current_time = idx
            lookback_period = lookback_days  # Use the app's lookback slider
            perf_rankings = backtest_calculate_ranking(prices_dataset, current_time, lookback_period)

            if perf_rankings is not None and selected_symbol in perf_rankings.index:
                # Get the raw rank value (percentile)
                raw_rank = perf_rankings.loc[selected_symbol, 'rank']

                # Calculate integer rank (1 is best, total_assets is worst)
                total_assets = len(perf_rankings)
                rank = 1 + sum(1 for other_metric in perf_rankings['rank'].values if other_metric > raw_rank)
                current_rank = rank

        # Process buy signal
        if signal == 1 and cash > 0:
            if current_rank is not None:
                # Use the ranking-based buy percentage from backtest_individual.py
                total_assets = len(prices_dataset)

                # Calculate buy percentage using the same formula as in backtest_individual.py
                def calculate_buy_percentage(rank, total_assets):
                    # Calculate cutoff points
                    bottom_third = int(total_assets * (1 / 3))
                    top_two_thirds = total_assets - bottom_third

                    # If in bottom third, buy 0%
                    if rank > top_two_thirds:
                        return 0.0

                    # For top two-thirds, use inverted wave function
                    x = (rank - 1) / (top_two_thirds - 1)
                    wave = 0.02 * np.sin(2 * np.pi * x)  # Smaller oscillation
                    linear = 0.48 - 0.48 * x  # Linear decrease from 0.48 to 0.0
                    return max(0.0, min(0.5, linear + wave))  # Clamp between 0.0 and 0.5

                buy_percentage = calculate_buy_percentage(current_rank, total_assets)
                amount_to_use = cash * buy_percentage
            else:
                # Fallback to the original fixed percentage
                amount_to_use = cash * 0.95

            # Calculate shares to buy
            shares_to_buy = amount_to_use / price if price > 0 else 0

            # Crypto can be fractional
            shares_to_buy = round(shares_to_buy, 8)

            # Update position
            position += shares_to_buy
            cash -= shares_to_buy * price

        # Process sell signal    
        elif signal == -1 and position > 0:
            if current_rank is not None:
                # Use the ranking-based sell percentage from backtest_individual.py
                total_assets = len(prices_dataset)

                # Calculate sell percentage using the same formula as in backtest_individual.py
                def calculate_sell_percentage(rank, total_assets):
                    # Calculate cutoff points
                    bottom_third = int(total_assets * (1 / 3))
                    top_two_thirds = total_assets - bottom_third

                    # If in bottom third, sell 100%
                    if rank > top_two_thirds:
                        return 1.0

                    # For top two-thirds, use a wave function
                    x = (rank - 1) / (top_two_thirds - 1)
                    wave = 0.1 * np.sin(2 * np.pi * x)  # Oscillation between -0.1 and 0.1
                    linear = 0.3 + 0.7 * x  # Linear increase from 0.3 to 1.0
                    return max(0.1, min(1.0, linear + wave))  # Clamp between 0.1 and 1.0

                sell_percentage = calculate_sell_percentage(current_rank, total_assets)
                shares_to_sell = position * sell_percentage
                cash += shares_to_sell * price
                position -= shares_to_sell
            else:
                # Fallback to selling all shares
                cash += position * price
                position = 0

        # Calculate portfolio value
        current_value = cash + (position * price)
        portfolio_value.append(current_value)
        shares_owned.append(position)
        trade_ranks.append(current_rank)  # Add rank information

    # Create a DataFrame for the portfolio history
    portfolio_df = pd.DataFrame({
        'portfolio_value': portfolio_value,
        'shares_owned': shares_owned,
        'rank': trade_ranks  # Include the rank information
    }, index=signals_df.index)

    return portfolio_df

# Create placeholders for charts
price_chart_placeholder = st.empty()
indicators_placeholder = st.empty()
signals_placeholder = st.empty()
portfolio_placeholder = st.empty()
performance_placeholder = st.empty()  # Add performance chart placeholder

# Calculate performance ranking for assets (similar to the one in backtest_individual.py)
def backtest_calculate_ranking(prices_dataset, current_time, lookback_days_param):
    """
    Calculate performance ranking of all symbols over the last N days,
    using the same logic as in backtest_individual.py

    Args:
        prices_dataset: Dictionary of price dataframes keyed by symbol
        current_time: Current time to use as the reference point
        lookback_days_param: Number of days to look back

    Returns:
        DataFrame with performance and rank columns
    """
    performance_dict = {}
    lookback_days = lookback_days_param

    # Convert current_time to pandas Timestamp if it's not already
    if not isinstance(current_time, pd.Timestamp):
        current_time = pd.Timestamp(current_time)

    lookback_time = current_time - pd.Timedelta(days=lookback_days)

    # Try to load best_params.json for strategy performance data
    best_params_data = {}
    try:
        import json
        from datetime import datetime, timedelta
        with open("best_params.json", "r") as f:
            best_params_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Silently continue
        pass

    for symbol, data in prices_dataset.items():
        try:
            # Get data up until current time and within lookback period
            mask = (data.index <= current_time) & (data.index >= lookback_time)
            symbol_data = data[mask]

            if len(symbol_data) >= 2:  # Need at least 2 points to calculate performance
                # Make column names lowercase if they aren't already
                if not all(col.islower() for col in symbol_data.columns):
                    symbol_data.columns = symbol_data.columns.str.lower()

                if 'close' not in symbol_data.columns:
                    continue

                # Calculate standard performance from price data
                start_price = symbol_data['close'].iloc[0]
                end_price = symbol_data['close'].iloc[-1]
                performance = ((end_price - start_price) / start_price) * 100

                # Check if we should use strategy performance instead
                if symbol in best_params_data:
                    symbol_entry = best_params_data[symbol]

                    # Check if entry is recent (less than a week old)
                    if 'date' in symbol_entry:
                        entry_date = datetime.strptime(symbol_entry['date'], "%Y-%m-%d")
                        is_recent = (datetime.now() - entry_date) < timedelta(weeks=1)

                        # Use strategy performance if entry is recent and has performance data
                        if is_recent and 'metrics' in symbol_entry and 'performance' in symbol_entry['metrics']:
                            performance = symbol_entry['metrics']['performance']

                # Store the final performance value
                performance_dict[symbol] = performance

                # Debug performance prints removed
        except Exception as e:
            pass

    # Convert to DataFrame and calculate rankings
    if performance_dict:
        perf_df = pd.DataFrame.from_dict(performance_dict, 
                                        orient='index', 
                                        columns=['performance'])
        perf_df['rank'] = perf_df['performance'].rank(pct=True)  # Percentile ranking

        return perf_df
    return None

# Original calculation for performance ranking display
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

    # Use the backtest ranking function to calculate rankings
    if prices_dataset:
        return backtest_calculate_ranking(prices_dataset, pd.Timestamp.now(), lookback_days)

    return None

# Function to update the charts
def update_charts():
    # Load optimal parameters if enabled
    params = None
    if use_best_params:
        params = load_best_params(selected_symbol)

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

    # Simulate portfolio early to get rank information for signals
    portfolio_df = None
    if enable_simulation:
        portfolio_df = simulate_portfolio(signals_df, df, initial_capital=initial_capital)

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
            # Prepare hover text with rank information if available
            hover_texts = []
            for idx in buy_signals.index:
                price = df.loc[idx, 'close']
                rank_info = ""

                # Add rank information if portfolio simulation is enabled and we have the data
                if portfolio_df is not None and not portfolio_df.empty and idx in portfolio_df.index:
                    rank = portfolio_df.loc[idx, 'rank']
                    if rank is not None:
                        rank_info = f"<br>Rank: {int(rank)}"

                hover_texts.append(f"BUY<br>Price: ${price:.2f}{rank_info}")

            fig1.add_trace(
                go.Scatter(
                    x=buy_signals.index, 
                    y=df.loc[buy_signals.index, 'high'] * 1.01,  # Slightly above the candle
                    mode='markers+text',
                    marker=dict(symbol='triangle-up', size=15, color='green'),
                    name="Buy Signal",
                    text=[f"R:{int(portfolio_df.loc[idx, 'rank'])}" if portfolio_df is not None and idx in portfolio_df.index and portfolio_df.loc[idx, 'rank'] is not None else "" for idx in buy_signals.index],
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=hover_texts
                )
            )

        # Get points where signal is -1 (sell)
        sell_signals = signals_df[signals_df['signal'] == -1]
        if not sell_signals.empty:
            # Prepare hover text with rank information if available
            hover_texts = []
            for idx in sell_signals.index:
                price = df.loc[idx, 'close']
                rank_info = ""

                # Add rank information if portfolio simulation is enabled and we have the data
                if portfolio_df is not None and not portfolio_df.empty and idx in portfolio_df.index:
                    rank = portfolio_df.loc[idx, 'rank']
                    if rank is not None:
                        rank_info = f"<br>Rank: {int(rank)}"

                hover_texts.append(f"SELL<br>Price: ${price:.2f}{rank_info}")

            fig1.add_trace(
                go.Scatter(
                    x=sell_signals.index, 
                    y=df.loc[sell_signals.index, 'low'] * 0.99,  # Slightly below the candle
                    mode='markers+text',
                    marker=dict(symbol='triangle-down', size=15, color='red'),
                    name="Sell Signal",
                    text=[f"R:{int(portfolio_df.loc[idx, 'rank'])}" if portfolio_df is not None and idx in portfolio_df.index and portfolio_df.loc[idx, 'rank'] is not None else "" for idx in sell_signals.index],
                    textposition="bottom center",
                    hoverinfo="text",
                    hovertext=hover_texts
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

                # Add rank information if available
                rank_display = "N/A"
                if portfolio_df is not None and not portfolio_df.empty and idx in portfolio_df.index:
                    rank = portfolio_df.loc[idx, 'rank']
                    if rank is not None:
                        rank_display = f"{int(rank)}"

                signal_display.append({
                    "Time": idx.strftime("%Y-%m-%d %H:%M"),
                    "Signal": signal_type,
                    "Price": f"${price:.2f}",
                    "Rank": rank_display,
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
    with performance_placeholder.container():
        st.subheader("Asset Performance Comparison")
        performance_df = calculate_performance_ranking(lookback_days=lookback_days)

        if performance_df is not None and not performance_df.empty:
            # Create a bar chart for the performance ranking
            fig_perf = go.Figure()

            # Sort by performance descending
            performance_df = performance_df.sort_values('performance', ascending=False)

            # Add bars with rank information
            fig_perf.add_trace(
                go.Bar(
                    x=performance_df.index,
                    y=performance_df['performance'],
                    marker_color=['green' if x > 0 else 'red' for x in performance_df['performance']],
                    text=[f"{x:.2f}%" for x in performance_df['performance']],
                    textposition='outside',
                    hovertext=[f"Rank: {r:.2f}" for r in performance_df['rank']],
                    hoverinfo='text'
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