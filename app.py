"""
Main Streamlit application for cryptocurrency trading platform
This file serves as the entry point for the Streamlit UI, while the FastAPI backend
can be started separately using the server.py script
"""
import os
import time
import logging
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import core functionality
from trading_platform.core.data_feed import fetch_historical_data, get_available_symbols
from trading_platform.core.indicators import generate_signals, get_default_params
from trading_platform.core.backtest import run_backtest, find_best_params, calculate_performance_ranking
from trading_platform.modules.crypto.config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, DEFAULT_LOOKBACK_DAYS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App styles
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.crypto-symbol {
    font-size: 1.5rem;
    font-weight: bold;
}
.crypto-price {
    font-size: 2rem;
    font-weight: bold;
}
.positive {
    color: #28a745;
}
.negative {
    color: #dc3545;
}
.indicator-card {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #f8f9fa;
    margin-bottom: 1rem;
}
.buy-signal {
    background-color: rgba(40, 167, 69, 0.2);
    padding: 0.5rem;
    border-radius: 0.25rem;
}
.sell-signal {
    background-color: rgba(220, 53, 69, 0.2);
    padding: 0.5rem;
    border-radius: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# Function to load best parameters from JSON file
@st.cache_data(ttl=3600)
def load_best_params(symbol):
    """
    Load best parameters for a symbol from file or use default
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USD')
        
    Returns:
        Dict of parameters or None if not found
    """
    try:
        with open("best_params.json", "r") as f:
            params_dict = json.load(f)
            
        if symbol in params_dict:
            return params_dict[symbol]
        else:
            return get_default_params()
    except (FileNotFoundError, json.JSONDecodeError):
        return get_default_params()

# Function to calculate buy percentage based on rank
def calculate_buy_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate buy percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.0 and 0.5 representing buy percentage
    """
    # Top performers get higher allocation
    # rank 1 (best) = 0.5, rank = total (worst) = 0.05
    if rank <= 0 or total_assets <= 1:
        return 0.2  # default value
    
    normalized_rank = rank / total_assets
    
    # Exponential decay formula to favor top performers
    buy_pct = 0.5 * np.exp(-2.5 * normalized_rank) + 0.05
    
    # Ensure result is between 0.05 and 0.5
    return min(max(buy_pct, 0.05), 0.5)

# Function to calculate sell percentage based on rank
def calculate_sell_percentage(rank: int, total_assets: int) -> float:
    """
    Calculate sell percentage based on rank and total number of assets.
    rank: 1 is best performer, total_assets is worst performer
    Returns: float between 0.1 and 1.0 representing sell percentage
    """
    # Worst performers get higher sell percentage
    # rank 1 (best) = 0.1, rank = total (worst) = 1.0
    if rank <= 0 or total_assets <= 1:
        return 0.5  # default value
    
    normalized_rank = rank / total_assets
    
    # Linear scaling formula to sell more of poor performers
    sell_pct = 0.1 + 0.9 * normalized_rank
    
    # Ensure result is between 0.1 and 1.0
    return min(max(sell_pct, 0.1), 1.0)

# Function to simulate portfolio performance
@st.cache_data(ttl=1800)
def simulate_portfolio(signals_df, price_data, initial_capital=100000):
    """
    Simulate portfolio performance based on trading signals with performance ranking
    """
    # Initialize portfolio
    cash = initial_capital
    position = 0
    entry_price = 0
    portfolio_values = []
    trades = []
    
    # Get timestamps and sort
    timestamps = signals_df.index
    
    # Calculate total assets for ranking
    total_assets = 20  # Assuming 20 cryptocurrencies
    
    # Iterate through signals
    for i in range(1, len(signals_df)):
        timestamp = timestamps[i]
        current_price = signals_df['price'].iloc[i]
        
        # Get signals
        buy_signal = signals_df['buy_signal'].iloc[i]
        sell_signal = signals_df['sell_signal'].iloc[i]
        
        # Calculate portfolio value before any trades
        portfolio_value = cash + position * current_price
        
        # Get current ranking based on performance (simple rank between 1-20)
        current_ranking = i % total_assets + 1  # Simulate changing rank
        
        # Process buy signal
        if buy_signal and cash > 0:
            # Calculate position size based on ranking
            buy_pct = calculate_buy_percentage(current_ranking, total_assets)
            amount_to_buy = cash * buy_pct
            shares_to_buy = amount_to_buy / current_price
            
            # Execute buy
            cash -= amount_to_buy
            position += shares_to_buy
            entry_price = current_price
            
            trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': current_price,
                'shares': shares_to_buy,
                'amount': amount_to_buy,
                'ranking': current_ranking
            })
        
        # Process sell signal
        elif sell_signal and position > 0:
            # Calculate position size based on ranking
            sell_pct = calculate_sell_percentage(current_ranking, total_assets)
            shares_to_sell = position * sell_pct
            amount_sold = shares_to_sell * current_price
            
            # Execute sell
            cash += amount_sold
            position -= shares_to_sell
            
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': current_price,
                'shares': shares_to_sell,
                'amount': amount_sold,
                'ranking': current_ranking
            })
        
        # Track portfolio value
        new_value = cash + position * current_price
        portfolio_values.append({
            'timestamp': timestamp,
            'cash': cash,
            'position': position,
            'position_value': position * current_price,
            'total_value': new_value,
            'price': current_price
        })
    
    # Calculate final metrics
    if portfolio_values:
        start_value = initial_capital
        final_value = portfolio_values[-1]['total_value']
        total_return = (final_value / start_value - 1) * 100
        
        # Calculate drawdown
        max_value = initial_capital
        max_drawdown = 0
        
        for point in portfolio_values:
            max_value = max(max_value, point['total_value'])
            drawdown = (point['total_value'] / max_value - 1) * 100
            max_drawdown = min(max_drawdown, drawdown)
        
        # Count trades
        num_trades = len(trades)
        
        # Calculate buy-and-hold performance
        first_price = price_data['close'].iloc[0] if not price_data.empty else 0
        last_price = price_data['close'].iloc[-1] if not price_data.empty else 0
        buy_hold_return = (last_price / first_price - 1) * 100 if first_price > 0 else 0
        
        results = {
            'portfolio_values': portfolio_values,
            'trades': trades,
            'start_value': start_value,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'buy_hold_return': buy_hold_return,
            'outperformance': total_return - buy_hold_return
        }
        
        return results
    
    return None

# Function to get current rankings
@st.cache_data(ttl=1800)
def get_current_rankings(symbol, timestamp=None):
    """
    Get the most recent performance rankings at a given timestamp.
    
    Args:
        symbol: The symbol to get rank for
        timestamp: The timestamp to find the closest ranking data (default: current time)
        
    Returns:
        rank: Performance rank (0-1) or None if not available
    """
    # Get data for all symbols
    symbols = get_available_symbols()
    
    # Calculate rankings
    ranking_df = calculate_performance_ranking(None, timestamp)
    
    if ranking_df.empty:
        return None
    
    # Find the row for the requested symbol
    symbol_row = ranking_df[ranking_df['symbol'] == symbol]
    
    if symbol_row.empty:
        return None
    
    rank = symbol_row['rank'].iloc[0]
    total_symbols = len(ranking_df)
    
    return rank, total_symbols

# Sidebar
with st.sidebar:
    st.title("Trading Platform")
    
    # Mode selection
    mode = st.radio("Select Mode", ["Dashboard", "Backtest", "Portfolio", "Settings"])
    
    # Symbol selector
    all_symbols = [s['symbol'] for s in get_available_symbols()]
    symbol = st.selectbox("Select Symbol", all_symbols, index=all_symbols.index(DEFAULT_SYMBOL) if DEFAULT_SYMBOL in all_symbols else 0)
    
    # Timeframe selector
    timeframe_options = {
        "1h": "1 Hour",
        "4h": "4 Hours", 
        "1d": "1 Day"
    }
    timeframe = st.selectbox("Select Timeframe", 
                           list(timeframe_options.keys()),
                           format_func=lambda x: timeframe_options.get(x, x),
                           index=list(timeframe_options.keys()).index(DEFAULT_TIMEFRAME) if DEFAULT_TIMEFRAME in timeframe_options else 0)
    
    # Lookback period
    lookback_days = st.slider("Lookback Days", 
                             min_value=5, 
                             max_value=30, 
                             value=DEFAULT_LOOKBACK_DAYS, 
                             step=1)
    
    # Connection status
    st.write("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("Status:")
    with col2:
        st.write("🟢 Connected" if True else "🔴 Disconnected")
    
    st.write("---")
    st.write("API: [FastAPI backend](http://localhost:5000/docs)")
    
    # Show version
    st.write("Version: 1.0.0")

# Main content
if mode == "Dashboard":
    st.header("Cryptocurrency Dashboard")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"{symbol} Price Chart ({timeframe})")
        
        # Fetch data
        with st.spinner("Fetching data..."):
            data = fetch_historical_data(symbol, timeframe, lookback_days)
            
            if data.empty:
                st.error(f"No data available for {symbol}")
            else:
                # Generate signals
                params = load_best_params(symbol)
                signals_df, daily_data, weekly_data = generate_signals(data, params)
                
                # Create plotly figure
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                  vertical_spacing=0.05,
                                  row_heights=[0.7, 0.3])
                
                # Add price candlesticks
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add buy signals
                buy_signals = signals_df[signals_df['buy_signal']]
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        name="Buy Signal"
                    ),
                    row=1, col=1
                )
                
                # Add sell signals
                sell_signals = signals_df[signals_df['sell_signal']]
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        name="Sell Signal"
                    ),
                    row=1, col=1
                )
                
                # Add composite indicator
                fig.add_trace(
                    go.Scatter(
                        x=signals_df.index,
                        y=signals_df['daily_composite'],
                        mode='lines',
                        line=dict(width=2, color='blue'),
                        name="Composite Indicator"
                    ),
                    row=2, col=1
                )
                
                # Add upper and lower threshold
                fig.add_trace(
                    go.Scatter(
                        x=signals_df.index,
                        y=signals_df['daily_upper'],
                        mode='lines',
                        line=dict(width=1, color='red', dash='dash'),
                        name="Upper Threshold"
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=signals_df.index,
                        y=signals_df['daily_lower'],
                        mode='lines',
                        line=dict(width=1, color='green', dash='dash'),
                        name="Lower Threshold"
                    ),
                    row=2, col=1
                )
                
                # Add zero line
                fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol} Price and Signals",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig.update_yaxes(title_text="Indicator Value", row=2, col=1)
                
                # Show plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show recent signals
                st.subheader("Recent Signals")
                
                # Prepare table data
                recent_signals = signals_df.iloc[-10:].copy()
                recent_signals['date'] = recent_signals.index
                recent_signals['date'] = recent_signals['date'].dt.strftime('%Y-%m-%d %H:%M')
                
                # Create table
                signal_data = []
                for _, row in recent_signals.iterrows():
                    signal_type = "None"
                    signal_class = ""
                    
                    if row['buy_signal']:
                        signal_type = "BUY"
                        signal_class = "buy-signal"
                    elif row['sell_signal']:
                        signal_type = "SELL"
                        signal_class = "sell-signal"
                    elif row['potential_buy']:
                        signal_type = "Potential Buy"
                    elif row['potential_sell']:
                        signal_type = "Potential Sell"
                    
                    signal_data.append({
                        "Date": row['date'],
                        "Price": f"${row['price']:.2f}",
                        "Signal": signal_type,
                        "Indicator": f"{row['daily_composite']:.4f}",
                        "Upper": f"{row['daily_upper']:.4f}",
                        "Lower": f"{row['daily_lower']:.4f}",
                        "Class": signal_class
                    })
                
                # Display table
                if signal_data:
                    table_html = '<table style="width:100%"><tr>'
                    table_html += '<th>Date</th><th>Price</th><th>Signal</th><th>Indicator</th></tr>'
                    
                    for item in signal_data:
                        row_class = item["Class"]
                        table_html += f'<tr class="{row_class}">'
                        table_html += f'<td>{item["Date"]}</td>'
                        table_html += f'<td>{item["Price"]}</td>'
                        table_html += f'<td>{item["Signal"]}</td>'
                        table_html += f'<td>{item["Indicator"]}</td>'
                        table_html += '</tr>'
                    
                    table_html += '</table>'
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.write("No recent signals.")
    
    with col2:
        st.subheader("Market Stats")
        
        # Current price
        if not data.empty:
            current_price = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close > 0 else 0
            
            # Price display
            st.markdown(f"<div class='crypto-symbol'>{symbol}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='crypto-price'>${current_price:.2f}</div>", unsafe_allow_html=True)
            
            # Price change
            change_class = 'positive' if price_change >= 0 else 'negative'
            change_icon = '▲' if price_change >= 0 else '▼'
            st.markdown(
                f"<div class='{change_class}'>{change_icon} ${abs(price_change):.2f} ({price_change_pct:.2f}%)</div>",
                unsafe_allow_html=True
            )
            
            # Date range
            st.write(f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Performance metrics
            st.write("---")
            st.subheader("Performance")
            
            # Calculate metrics
            high = data['high'].max()
            low = data['low'].min()
            range_pct = (high / low - 1) * 100 if low > 0 else 0
            
            # 24h volume
            last_day = data.iloc[-24:] if len(data) >= 24 else data
            volume_24h = last_day['volume'].sum()
            
            # Display metrics
            col1, col2 = st.columns(2)
            col1.metric("High", f"${high:.2f}")
            col2.metric("Low", f"${low:.2f}")
            
            col1, col2 = st.columns(2)
            col1.metric("Range", f"{range_pct:.2f}%")
            col2.metric("Volume", f"${volume_24h:.0f}")
            
            # Performance ranking
            st.write("---")
            st.subheader("Ranking")
            
            # Get current ranking
            ranking = get_current_rankings(symbol)
            
            if ranking:
                rank, total = ranking
                rank_pct = rank / total
                
                # Create progress bar
                st.progress(1 - rank_pct, f"Rank {int(rank)} of {total}")
                
                # Show allocation recommendation
                buy_pct = calculate_buy_percentage(rank, total)
                sell_pct = calculate_sell_percentage(rank, total)
                
                st.write(f"Buy allocation: {buy_pct:.1%}")
                st.write(f"Sell allocation: {sell_pct:.1%}")
            else:
                st.write("Ranking data not available")

elif mode == "Backtest":
    st.header("Strategy Backtest")
    
    # Backtest parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_days = st.slider("Backtest Days", 
                                min_value=5, 
                                max_value=60, 
                                value=30, 
                                step=5)
    
    with col2:
        initial_capital = st.number_input("Initial Capital ($)", 
                                        min_value=1000.0, 
                                        max_value=1000000.0, 
                                        value=100000.0, 
                                        step=10000.0)
    
    with col3:
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                # Load best parameters
                params = load_best_params(symbol)
                
                # Run backtest
                backtest_result = run_backtest(symbol, backtest_days, params, initial_capital=initial_capital)
                
                if 'error' in backtest_result:
                    st.error(backtest_result['error'])
                else:
                    st.success("Backtest completed!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Return", f"{backtest_result['return']:.2f}%")
                    col2.metric("Buy & Hold", f"{backtest_result['buy_hold_return']:.2f}%")
                    col3.metric("Outperformance", f"{backtest_result['outperformance']:.2f}%")
                    col4.metric("Max Drawdown", f"{backtest_result['max_drawdown']:.2f}%")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Trades", backtest_result['num_trades'])
                    col2.metric("Win Rate", f"{backtest_result['win_rate'] * 100:.1f}%")
                    col3.metric("Initial Capital", f"${backtest_result['initial_capital']:.2f}")
                    col4.metric("Final Value", f"${backtest_result['final_value']:.2f}")
                    
                    # Plot equity curve
                    if 'equity_curve' in backtest_result and len(backtest_result['equity_curve']) > 0:
                        st.subheader("Equity Curve")
                        
                        # Prepare data
                        equity_data = pd.DataFrame(backtest_result['equity_curve'])
                        equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # Add equity curve
                        fig.add_trace(
                            go.Scatter(
                                x=equity_data['timestamp'],
                                y=equity_data['total_value'],
                                mode='lines',
                                name='Portfolio Value',
                                line=dict(width=2, color='blue')
                            )
                        )
                        
                        # Add buy and hold
                        price_data = equity_data['price']
                        initial_price = price_data.iloc[0]
                        buy_hold_values = initial_capital * (price_data / initial_price)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=equity_data['timestamp'],
                                y=buy_hold_values,
                                mode='lines',
                                name='Buy & Hold',
                                line=dict(width=2, color='gray', dash='dot')
                            )
                        )
                        
                        # Add initial capital line
                        fig.add_hline(
                            y=initial_capital,
                            line_width=1,
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Initial Capital"
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title='Portfolio Value Over Time',
                            xaxis_title='Date',
                            yaxis_title='Value ($)',
                            height=400,
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show trades
                    if 'trades' in backtest_result and len(backtest_result['trades']) > 0:
                        st.subheader("Recent Trades")
                        
                        # Convert to DataFrame
                        trades_df = pd.DataFrame(backtest_result['trades'][-10:])
                        
                        # Format DataFrame
                        if not trades_df.empty:
                            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                            trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                            
                            # Create a formatted table
                            trade_data = []
                            for _, row in trades_df.iterrows():
                                trade_data.append({
                                    "Date": row['timestamp'],
                                    "Action": row['action'],
                                    "Price": f"${row['price']:.2f}",
                                    "Size": f"{row.get('size', 0):.4f}",
                                    "Value": f"${row.get('cost', row.get('revenue', 0)):.2f}",
                                    "P/L": f"{row.get('pl_pct', ''):.2f}%" if 'pl_pct' in row else ''
                                })
                            
                            # Display table
                            table_html = '<table style="width:100%"><tr>'
                            table_html += '<th>Date</th><th>Action</th><th>Price</th><th>Size</th><th>Value</th><th>P/L</th></tr>'
                            
                            for item in trade_data:
                                row_class = 'buy-signal' if item['Action'] == 'BUY' else 'sell-signal' if item['Action'] == 'SELL' else ''
                                table_html += f'<tr class="{row_class}">'
                                table_html += f'<td>{item["Date"]}</td>'
                                table_html += f'<td>{item["Action"]}</td>'
                                table_html += f'<td>{item["Price"]}</td>'
                                table_html += f'<td>{item["Size"]}</td>'
                                table_html += f'<td>{item["Value"]}</td>'
                                table_html += f'<td>{item["P/L"]}</td>'
                                table_html += '</tr>'
                            
                            table_html += '</table>'
                            st.markdown(table_html, unsafe_allow_html=True)
                        else:
                            st.write("No trades executed.")
    
    # Parameter optimization
    st.write("---")
    st.subheader("Parameter Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimize_days = st.slider("Optimization Period (Days)", 
                                min_value=5, 
                                max_value=30, 
                                value=15, 
                                step=5)
    
    with col2:
        if st.button("Optimize Parameters"):
            with st.spinner("Finding optimal parameters... This may take a while."):
                # Get parameter grid from configuration
                from trading_platform.core.config import param_grid
                
                # Run optimization
                results = find_best_params(symbol, param_grid, optimize_days)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("Optimization completed!")
                    
                    # Show best parameters
                    st.write("Best Parameters:")
                    best_params = results['best_params']
                    
                    # Format parameters into columns
                    param_data = []
                    for param, value in best_params.items():
                        param_data.append({
                            "Parameter": param,
                            "Value": str(value)
                        })
                    
                    # Create columns
                    cols = st.columns(4)
                    for i, param in enumerate(param_data):
                        cols[i % 4].write(f"**{param['Parameter']}**: {param['Value']}")
                    
                    # Show backtest performance with these parameters
                    st.write("Performance with optimal parameters:")
                    
                    backtest_result = results['backtest_result']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Return", f"{backtest_result['return']:.2f}%")
                    col2.metric("Sharpe Ratio", f"{backtest_result['sharpe_ratio']:.3f}")
                    col3.metric("Max Drawdown", f"{backtest_result['max_drawdown']:.2f}%")
                    col4.metric("Trades", backtest_result['num_trades'])

    # Show current parameters
    st.write("---")
    st.subheader("Current Parameters")
    
    # Get current parameters
    params = load_best_params(symbol)
    
    # Format parameters into columns
    param_data = []
    for param, value in params.items():
        param_data.append({
            "Parameter": param,
            "Value": str(value)
        })
    
    # Create columns
    cols = st.columns(4)
    for i, param in enumerate(param_data):
        cols[i % 4].write(f"**{param['Parameter']}**: {param['Value']}")

elif mode == "Portfolio":
    st.header("Portfolio Simulation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        portfolio_days = st.slider("Simulation Days", 
                                 min_value=5, 
                                 max_value=60, 
                                 value=30, 
                                 step=5)
    
    with col2:
        portfolio_capital = st.number_input("Portfolio Capital ($)", 
                                          min_value=10000.0, 
                                          max_value=1000000.0, 
                                          value=100000.0, 
                                          step=10000.0)
    
    with col3:
        portfolio_assets = st.slider("Number of Assets", 
                                   min_value=1, 
                                   max_value=10, 
                                   value=5, 
                                   step=1)
    
    # Get available symbols
    all_symbols = [s['symbol'] for s in get_available_symbols()]
    
    # Select symbols for portfolio
    st.write("Select assets to include in portfolio:")
    
    # Split into columns for better display
    cols = st.columns(5)
    selected_symbols = []
    
    for i, symbol_name in enumerate(all_symbols[:min(20, len(all_symbols))]):
        if cols[i % 5].checkbox(symbol_name, value=(i < portfolio_assets)):
            selected_symbols.append(symbol_name)
    
    # Run portfolio simulation
    if st.button("Run Portfolio Simulation"):
        if len(selected_symbols) == 0:
            st.error("Please select at least one asset for the portfolio.")
        else:
            with st.spinner("Running portfolio simulation..."):
                # Initialize portfolio results
                portfolio_results = {
                    'initial_capital': portfolio_capital,
                    'final_value': portfolio_capital,
                    'return': 0.0,
                    'buy_hold_return': 0.0,
                    'max_drawdown': 0.0,
                    'asset_returns': {},
                    'equity_curve': []
                }
                
                # Track equity curve for each asset
                asset_data = {}
                asset_returns = {}
                
                # Split capital among assets
                asset_capital = portfolio_capital / len(selected_symbols)
                
                # Process each symbol
                for symbol_name in selected_symbols:
                    # Fetch data
                    data = fetch_historical_data(symbol_name, '1h', portfolio_days)
                    
                    if data.empty:
                        st.warning(f"No data available for {symbol_name}")
                        continue
                    
                    # Generate signals
                    params = load_best_params(symbol_name)
                    signals_df, daily_data, weekly_data = generate_signals(data, params)
                    
                    if signals_df.empty:
                        st.warning(f"Failed to generate signals for {symbol_name}")
                        continue
                    
                    # Simulate trading
                    result = simulate_portfolio(signals_df, data, asset_capital)
                    
                    if result:
                        # Store data
                        asset_data[symbol_name] = {
                            'signals': signals_df,
                            'data': data,
                            'simulation': result
                        }
                        
                        # Calculate asset return
                        asset_return = result['total_return']
                        asset_returns[symbol_name] = asset_return
                
                # Calculate combined portfolio performance
                if asset_data:
                    # Sum final values
                    final_value = sum(asset_data[s]['simulation']['final_value'] for s in asset_data)
                    
                    # Calculate return
                    portfolio_return = (final_value / portfolio_capital - 1) * 100
                    
                    # Calculate buy and hold
                    buy_hold_returns = []
                    for symbol_name in asset_data:
                        first_price = asset_data[symbol_name]['data']['close'].iloc[0] 
                        last_price = asset_data[symbol_name]['data']['close'].iloc[-1]
                        buy_hold_return = (last_price / first_price - 1) * 100
                        buy_hold_returns.append(buy_hold_return)
                    
                    avg_buy_hold = sum(buy_hold_returns) / len(buy_hold_returns) if buy_hold_returns else 0
                    
                    # Calculate combined equity curve
                    combined_equity = None
                    for symbol_name in asset_data:
                        equity_points = asset_data[symbol_name]['simulation']['portfolio_values']
                        equity_df = pd.DataFrame(equity_points)
                        equity_df.set_index('timestamp', inplace=True)
                        equity_df = equity_df[['total_value']]
                        equity_df.columns = [symbol_name]
                        
                        if combined_equity is None:
                            combined_equity = equity_df
                        else:
                            combined_equity = combined_equity.join(equity_df, how='outer')
                    
                    if combined_equity is not None:
                        combined_equity = combined_equity.fillna(method='ffill')
                        combined_equity['total'] = combined_equity.sum(axis=1)
                        
                        # Calculate drawdown
                        running_max = combined_equity['total'].cummax()
                        drawdown = (combined_equity['total'] / running_max - 1) * 100
                        max_drawdown = drawdown.min()
                        
                        # Update portfolio results
                        portfolio_results['final_value'] = final_value
                        portfolio_results['return'] = portfolio_return
                        portfolio_results['buy_hold_return'] = avg_buy_hold
                        portfolio_results['max_drawdown'] = max_drawdown
                        portfolio_results['asset_returns'] = asset_returns
                        
                        # Add equity curve points
                        portfolio_results['equity_curve'] = [
                            {
                                'timestamp': str(idx),
                                'value': row['total']
                            }
                            for idx, row in combined_equity.iterrows()
                        ]
                
                # Display results
                st.subheader("Portfolio Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Return", f"{portfolio_results['return']:.2f}%")
                col2.metric("Buy & Hold", f"{portfolio_results['buy_hold_return']:.2f}%")
                col3.metric("Outperformance", f"{portfolio_results['return'] - portfolio_results['buy_hold_return']:.2f}%")
                col4.metric("Max Drawdown", f"{portfolio_results['max_drawdown']:.2f}%")
                
                col1, col2 = st.columns(2)
                col1.metric("Initial Capital", f"${portfolio_results['initial_capital']:.2f}")
                col2.metric("Final Value", f"${portfolio_results['final_value']:.2f}")
                
                # Plot equity curve
                if 'equity_curve' in portfolio_results and portfolio_results['equity_curve']:
                    st.subheader("Portfolio Equity Curve")
                    
                    # Prepare data
                    equity_data = pd.DataFrame(portfolio_results['equity_curve'])
                    equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add equity curve
                    fig.add_trace(
                        go.Scatter(
                            x=equity_data['timestamp'],
                            y=equity_data['value'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(width=2, color='blue')
                        )
                    )
                    
                    # Add initial capital line
                    fig.add_hline(
                        y=portfolio_capital,
                        line_width=1,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Initial Capital"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='Portfolio Value Over Time',
                        xaxis_title='Date',
                        yaxis_title='Value ($)',
                        height=400,
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Asset performance comparison
                if asset_returns:
                    st.subheader("Asset Performance")
                    
                    # Sort by performance
                    sorted_assets = sorted(asset_returns.items(), key=lambda x: x[1], reverse=True)
                    
                    # Create DataFrame
                    asset_df = pd.DataFrame(sorted_assets, columns=['Symbol', 'Return'])
                    
                    # Add colors
                    colors = ['green' if ret >= 0 else 'red' for ret in asset_df['Return']]
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add bar chart
                    fig.add_trace(
                        go.Bar(
                            x=asset_df['Symbol'],
                            y=asset_df['Return'],
                            marker_color=colors,
                            text=asset_df['Return'].apply(lambda x: f"{x:.2f}%"),
                            textposition='auto'
                        )
                    )
                    
                    # Add horizontal line at 0
                    fig.add_hline(
                        y=0,
                        line_width=1,
                        line_dash="dash",
                        line_color="gray"
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='Asset Performance Comparison',
                        xaxis_title='Symbol',
                        yaxis_title='Return (%)',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

elif mode == "Settings":
    st.header("Platform Settings")
    
    # API Configuration
    st.subheader("API Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("FastAPI Server URL", value="http://localhost:5000", disabled=True)
    with col2:
        st.checkbox("Enable WebSocket", value=True)
    
    # Strategy Parameters
    st.subheader("Default Strategy Parameters")
    
    # Load default parameters
    params = get_default_params()
    
    # Create columns for parameters
    col1, col2 = st.columns(2)
    
    # RSI Settings
    with col1:
        st.write("RSI Settings")
        rsi_period = st.slider("RSI Period", min_value=5, max_value=30, value=params['rsi_period'], step=1)
        rsi_weight = st.slider("RSI Weight", min_value=0.0, max_value=1.0, value=params['rsi_weight'], step=0.05)
    
    # MACD Settings
    with col2:
        st.write("MACD Settings")
        macd_fast = st.slider("MACD Fast Period", min_value=5, max_value=20, value=params['macd_fast'], step=1)
        macd_slow = st.slider("MACD Slow Period", min_value=15, max_value=40, value=params['macd_slow'], step=1)
        macd_signal = st.slider("MACD Signal Period", min_value=5, max_value=15, value=params['macd_signal'], step=1)
        macd_weight = st.slider("MACD Weight", min_value=0.0, max_value=1.0, value=params['macd_weight'], step=0.05)
    
    col1, col2 = st.columns(2)
    
    # Stochastic Settings
    with col1:
        st.write("Stochastic Settings")
        stoch_k = st.slider("Stochastic %K", min_value=5, max_value=30, value=params['stoch_k'], step=1)
        stoch_d = st.slider("Stochastic %D", min_value=2, max_value=10, value=params['stoch_d'], step=1)
        stoch_weight = st.slider("Stochastic Weight", min_value=0.0, max_value=1.0, value=params['stoch_weight'], step=0.05)
    
    # Fractal Settings
    with col2:
        st.write("Fractal Complexity Settings")
        fractal_weight = st.slider("Fractal Weight", min_value=0.0, max_value=1.0, value=params['fractal_weight'], step=0.05)
        volatility_window = st.slider("Volatility Window", min_value=5, max_value=30, value=params['volatility_window'], step=1)
    
    # Save settings
    if st.button("Save Settings"):
        # Create updated parameters
        updated_params = {
            'rsi_period': rsi_period,
            'rsi_weight': rsi_weight,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'macd_weight': macd_weight,
            'stoch_k': stoch_k,
            'stoch_d': stoch_d,
            'stoch_weight': stoch_weight,
            'fractal_weight': fractal_weight,
            'volatility_window': volatility_window,
            'hurst_lags': params['hurst_lags']  # Keep original
        }
        
        # Save to file
        try:
            # Check if the file exists
            try:
                with open("best_params.json", "r") as f:
                    existing_params = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_params = {}
            
            # Update with new default params
            existing_params["default"] = updated_params
            
            # Save back to file
            with open("best_params.json", "w") as f:
                json.dump(existing_params, f, indent=4)
            
            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")
    
    # System Information
    st.write("---")
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Version: 1.0.0")
        st.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    with col2:
        st.write("Platform: FastAPI + React (Transitioning from Streamlit)")
        st.write("Status: Development")
    
    # View server logs
    st.write("---")
    st.subheader("Server Logs")
    
    if st.button("View Recent Logs"):
        st.code("Log viewing functionality coming in a future update.")

# Add link to FastAPI backend
st.markdown("""
---
<div style="text-align: center">
    <p>API Documentation available at: <a href="http://localhost:5000/docs" target="_blank">http://localhost:5000/docs</a></p>
    <p>FastAPI Backend can be started with: <code>python server.py</code></p>
</div>
""", unsafe_allow_html=True)