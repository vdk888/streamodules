"""
Visualization module for cryptocurrency trading application.
Creates interactive charts for price data, indicators, and trading signals.
Based on Plotly for modern, interactive visualization.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import logging
from typing import Dict, List, Optional, Tuple, Union
import json

# Configure logger
logger = logging.getLogger(__name__)

def is_market_hours(timestamp: datetime.datetime, market_config: Dict) -> bool:
    """
    Check if timestamp is during market hours for the specific market
    
    Args:
        timestamp: The timestamp to check
        market_config: Market configuration with timezone and hours
        
    Returns:
        Boolean indicating if the timestamp is during market hours
    """
    market_tz = pytz.timezone(market_config['timezone'])
    ts_market = timestamp.astimezone(market_tz)
    
    # Parse market hours
    start_time = datetime.datetime.strptime(market_config['start'], '%H:%M').time()
    end_time = datetime.datetime.strptime(market_config['end'], '%H:%M').time()
    
    # For 24/7 markets like cryptocurrencies
    if start_time == datetime.datetime.strptime('00:00', '%H:%M').time() and \
       end_time == datetime.datetime.strptime('23:59', '%H:%M').time():
        return True
    
    # Check if it's a weekday
    if ts_market.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Create datetime objects for comparison
    market_start = ts_market.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=0,
        microsecond=0
    ).replace(tzinfo=ts_market.tzinfo)
    
    market_end = ts_market.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=0,
        microsecond=0
    ).replace(tzinfo=ts_market.tzinfo)
    
    return market_start <= ts_market <= market_end

def split_into_sessions(data: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split data into continuous market sessions
    
    Args:
        data: DataFrame with time-indexed price data
        
    Returns:
        List of DataFrames, each representing a continuous session
    """
    if len(data) == 0:
        return []
    
    # Ensure data index is timezone-aware
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    
    # Calculate typical time interval between data points
    if len(data) > 1:
        typical_interval = (data.index[1] - data.index[0]).total_seconds() / 60
    else:
        typical_interval = 1  # Default to 1 minute if there's only one data point
    
    sessions = []
    current_session = []
    last_timestamp = None
    
    for timestamp, row in data.iterrows():
        if last_timestamp is not None:
            time_diff = (timestamp - last_timestamp).total_seconds() / 60
            if time_diff > typical_interval * 10:  # More lenient gap threshold
                if current_session:
                    sessions.append(pd.DataFrame(current_session))
                current_session = []
        
        current_session.append(row)
        last_timestamp = timestamp
    
    if current_session:
        sessions.append(pd.DataFrame(current_session))
    
    return sessions

def create_price_chart(
    price_data: pd.DataFrame, 
    signals_df: Optional[pd.DataFrame] = None,
    daily_composite: Optional[pd.DataFrame] = None, 
    weekly_composite: Optional[pd.DataFrame] = None,
    portfolio_df: Optional[pd.DataFrame] = None,
    performance_df: Optional[pd.DataFrame] = None,
    show_macd: bool = True,
    show_rsi: bool = True,
    show_stochastic: bool = True,
    show_fractal: bool = True,
    show_signals: bool = True
) -> go.Figure:
    """
    Create a comprehensive price chart with indicators and signals

    Args:
        price_data: DataFrame with price data
        signals_df: DataFrame with trading signals
        daily_composite: DataFrame with daily composite indicator
        weekly_composite: DataFrame with weekly composite indicator
        portfolio_df: DataFrame with portfolio performance
        performance_df: DataFrame with asset performance ranking
        show_macd: Whether to show MACD indicator
        show_rsi: Whether to show RSI indicator
        show_stochastic: Whether to show Stochastic indicator
        show_fractal: Whether to show Fractal Complexity indicator
        show_signals: Whether to show buy/sell signals

    Returns:
        Plotly figure with the chart
    """
    # Count how many rows we need based on active indicators
    active_indicators = [
        True,  # Price chart is always present
        daily_composite is not None,
        weekly_composite is not None,
        show_macd and signals_df is not None and 'macd' in signals_df.columns,
        show_rsi and signals_df is not None and 'rsi' in signals_df.columns,
        show_stochastic and signals_df is not None and 'stoch_k' in signals_df.columns,
        show_fractal and signals_df is not None and 'fractal' in signals_df.columns,
        portfolio_df is not None,
        performance_df is not None
    ]
    
    rows = sum(active_indicators)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4 if i == 0 else 0.2 for i in range(rows)]  # Larger first plot for price
    )
    
    # Flag to check if we have daily and weekly composite indicators
    has_daily_composite = daily_composite is not None and not daily_composite.empty
    has_weekly_composite = weekly_composite is not None and not weekly_composite.empty
    
    # Add price chart (always first)
    # Candlestick chart for price data
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name="Price"
    ), row=1, col=1)
    
    # Add buy signals if available
    if show_signals and signals_df is not None:
        buy_signals = signals_df[signals_df['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name="Buy Signal"
            ), row=1, col=1)
        
        # Add sell signals if available
        sell_signals = signals_df[signals_df['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name="Sell Signal"
            ), row=1, col=1)
    
    # Current row counter - start at row 2 for additional charts
    current_row = 2
    
    # Add portfolio equity curve if available (in price chart as secondary y-axis)
    if portfolio_df is not None:
        # Add portfolio value line to price chart
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            line=dict(color='black', width=1.5),
            name="Portfolio Value",
            yaxis="y2"
        ), row=1, col=1)
        
        # Add number of shares line to price chart
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['position'],
            mode='lines',
            line=dict(color='purple', width=1.5, dash='dot'),
            name="Shares Held",
            yaxis="y3"
        ), row=1, col=1)
        
        # Update layout to include multiple y-axes
        fig.update_layout(
            yaxis2=dict(
                title=dict(text="Portfolio Value", font=dict(color="black")),
                tickfont=dict(color="black"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            yaxis3=dict(
                title=dict(text="Shares", font=dict(color="purple")),
                tickfont=dict(color="purple"),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.95
            )
        )
    
    # Add Daily Composite indicator if available
    if has_daily_composite and current_row <= rows:
        # Add daily composite line
        fig.add_trace(go.Scatter(
            x=daily_composite.index,
            y=daily_composite['daily_composite'],
            mode='lines',
            line=dict(color='blue', width=1.5),
            name="Daily Composite"
        ), row=current_row, col=1)
        
        # Add upper threshold
        fig.add_trace(go.Scatter(
            x=daily_composite.index,
            y=daily_composite['upper_threshold'],
            mode='lines',
            line=dict(color='green', width=1, dash='dot'),
            name="Upper Threshold",
            opacity=0.7
        ), row=current_row, col=1)
        
        # Add lower threshold
        fig.add_trace(go.Scatter(
            x=daily_composite.index,
            y=daily_composite['lower_threshold'],
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            name="Lower Threshold",
            opacity=0.7
        ), row=current_row, col=1)
        
        # Add title for the subplot
        fig.update_yaxes(
            title=dict(text="Daily Composite", font=dict(color="blue")), 
            row=current_row, 
            col=1
        )
        current_row += 1
    
    # Add Weekly Composite indicator if available
    if has_weekly_composite and current_row <= rows:
        # Add weekly composite line
        fig.add_trace(go.Scatter(
            x=weekly_composite.index,
            y=weekly_composite['weekly_composite'],
            mode='lines',
            line=dict(color='purple', width=1.5),
            name="Weekly Composite"
        ), row=current_row, col=1)
        
        # Add upper threshold if available
        if 'upper_threshold' in weekly_composite.columns:
            fig.add_trace(go.Scatter(
                x=weekly_composite.index,
                y=weekly_composite['upper_threshold'],
                mode='lines',
                line=dict(color='green', width=1, dash='dot'),
                name="Upper Threshold (Weekly)",
                opacity=0.7
            ), row=current_row, col=1)
        
        # Add lower threshold if available
        if 'lower_threshold' in weekly_composite.columns:
            fig.add_trace(go.Scatter(
                x=weekly_composite.index,
                y=weekly_composite['lower_threshold'],
                mode='lines',
                line=dict(color='red', width=1, dash='dot'),
                name="Lower Threshold (Weekly)",
                opacity=0.7
            ), row=current_row, col=1)
        
        # Add title for the subplot
        fig.update_yaxes(
            title=dict(text="Weekly Composite", font=dict(color="blue")),
            row=current_row,
            col=1
        )
        current_row += 1
    
    # Add MACD indicator if enabled
    if show_macd and current_row <= rows and signals_df is not None:
        # Ensure we have the necessary columns
        if 'macd' in signals_df.columns and 'macd_signal' in signals_df.columns and 'macd_hist' in signals_df.columns:
            # Create histogram colors based on values
            colors = ['green' if val >= 0 else 'red' for val in signals_df['macd_hist']]
            
            # Add Histogram
            fig.add_trace(go.Bar(
                x=signals_df.index,
                y=signals_df['macd_hist'],
                marker_color=colors,
                name="MACD Histogram",
                showlegend=True
            ), row=current_row, col=1)
            
            # Add MACD line
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['macd'],
                mode='lines',
                line=dict(color='blue', width=1.5),
                name="MACD",
                showlegend=True
            ), row=current_row, col=1)
            
            # Add Signal line
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['macd_signal'],
                mode='lines',
                line=dict(color='red', width=1.5),
                name="Signal",
                showlegend=True
            ), row=current_row, col=1)
            
            # Add title for the subplot
            fig.update_yaxes(
                title=dict(text="MACD", font=dict(color="blue")),
                row=current_row,
                col=1
            )
            current_row += 1
    
    # Add RSI indicator if enabled
    if show_rsi and current_row <= rows and signals_df is not None:
        # Ensure we have the necessary columns
        if 'rsi' in signals_df.columns:
            # Add RSI line
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['rsi'],
                mode='lines',
                line=dict(color='purple', width=1.5),
                name="RSI"
            ), row=current_row, col=1)
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=signals_df.index[0],
                x1=signals_df.index[-1],
                y0=70,
                y1=70,
                line=dict(color="red", width=1, dash="dash"),
                row=current_row,
                col=1
            )
            
            fig.add_shape(
                type="line",
                x0=signals_df.index[0],
                x1=signals_df.index[-1],
                y0=30,
                y1=30,
                line=dict(color="green", width=1, dash="dash"),
                row=current_row,
                col=1
            )
            
            # Add title for the subplot
            fig.update_yaxes(
                title=dict(text="RSI", font=dict(color="blue")),
                row=current_row,
                col=1
            )
            current_row += 1
    
    # Add Stochastic indicator if enabled
    if show_stochastic and current_row <= rows and signals_df is not None:
        # Ensure we have the necessary columns
        if 'stoch_k' in signals_df.columns:
            # Add Stochastic %K line
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['stoch_k'],
                mode='lines',
                line=dict(color='blue', width=1.5),
                name="Stoch %K"
            ), row=current_row, col=1)
            
            # Add Stochastic %D line if available
            if 'stoch_d' in signals_df.columns:
                fig.add_trace(go.Scatter(
                    x=signals_df.index,
                    y=signals_df['stoch_d'],
                    mode='lines',
                    line=dict(color='red', width=1.5),
                    name="Stoch %D"
                ), row=current_row, col=1)
            
            # Add overbought/oversold lines
            fig.add_shape(
                type="line",
                x0=signals_df.index[0],
                x1=signals_df.index[-1],
                y0=80,
                y1=80,
                line=dict(color="red", width=1, dash="dash"),
                row=current_row,
                col=1
            )
            
            fig.add_shape(
                type="line",
                x0=signals_df.index[0],
                x1=signals_df.index[-1],
                y0=20,
                y1=20,
                line=dict(color="green", width=1, dash="dash"),
                row=current_row,
                col=1
            )
            
            # Add title for the subplot
            fig.update_yaxes(
                title=dict(text="Stochastic", font=dict(color="blue")),
                row=current_row,
                col=1
            )
            current_row += 1
    
    # Add Fractal Complexity indicator if enabled
    if show_fractal and current_row <= rows and signals_df is not None:
        # Ensure we have the necessary columns
        if 'fractal' in signals_df.columns:
            # Add Fractal line
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['fractal'],
                mode='lines',
                line=dict(color='orange', width=1.5),
                name="Fractal"
            ), row=current_row, col=1)
            
            # Add reference line at 0.5 (random walk)
            fig.add_shape(
                type="line",
                x0=signals_df.index[0],
                x1=signals_df.index[-1],
                y0=0.5,
                y1=0.5,
                line=dict(color="black", width=1, dash="dash"),
                row=current_row,
                col=1
            )
            
            # Add title for the subplot
            fig.update_yaxes(
                title=dict(text="Fractal Complexity", font=dict(color="blue")),
                row=current_row,
                col=1
            )
            current_row += 1
    
    # Add Portfolio Position and Asset Ranking if available
    if portfolio_df is not None and current_row <= rows:
        # Create figure with secondary y-axis for asset rank
        # Add position size (number of shares)
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['position'],
                mode='lines',
                line=dict(color='green', width=2),
                name="Shares Held"
            ),
            row=current_row, col=1
        )
    
        # Add asset ranking at signal points
        # Filter for signal points only
        buy_signals = portfolio_df[portfolio_df['signal'] == 1]
        sell_signals = portfolio_df[portfolio_df['signal'] == -1]
    
        # Add buy signal ranks
        if not buy_signals.empty and 'asset_rank' in buy_signals.columns:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['asset_rank'],
                    mode='markers',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    name="Rank at Buy",
                    text=[f"Rank: {int(r)}" for r in buy_signals['asset_rank']],
                    hoverinfo='text'
                ),
                row=current_row, col=1
            )
    
        # Add sell signal ranks
        if not sell_signals.empty and 'asset_rank' in sell_signals.columns:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['asset_rank'],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='circle',
                        line=dict(color='black', width=1)
                    ),
                    name="Rank at Sell",
                    text=[f"Rank: {int(r)}" for r in sell_signals['asset_rank']],
                    hoverinfo='text'
                ),
                row=current_row, col=1
            )
        
        # Add title for the subplot
        fig.update_yaxes(
            title=dict(text="Position & Asset Ranking", font=dict(color="green")),
            row=current_row,
            col=1
        )
        current_row += 1
    
    # Add Performance Ranking chart if available
    if performance_df is not None and current_row <= rows:
        # Make sure performance data is sorted
        sorted_perf = performance_df.sort_values('performance', ascending=False)
        
        # Create bar colors based on performance
        colors = ['green' if val >= 0 else 'red' for val in sorted_perf['performance']]
        
        # Add Performance Bars
        fig.add_trace(
            go.Bar(
                x=sorted_perf.index,
                y=sorted_perf['performance'] * 100,  # Convert to percentage
                marker_color=colors,
                text=[f"{val:.2f}%" for val in sorted_perf['performance'] * 100],
                textposition='auto',
                name="Performance"
            ),
            row=current_row, col=1
        )
        
        # Add title for the subplot
        fig.update_yaxes(
            title=dict(text="Performance (%)", font=dict(color="blue")),
            row=current_row,
            col=1
        )
    
    # Update layout for the entire figure
    symbol_name = price_data.iloc[-1]['symbol'] if 'symbol' in price_data.columns else ''
    fig.update_layout(
        title=f"{symbol_name} Price Chart with Indicators",
        xaxis_title="Date",
        height=200 * rows,  # Adjust height based on number of rows
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(t=50, l=50, r=50, b=30)
    )
    
    # Customize candlestick colors
    fig.update_traces(
        increasing_line_color='green',
        decreasing_line_color='red',
        selector=dict(type='candlestick')
    )
    
    return fig

def create_performance_ranking_chart(performance_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing performance ranking of assets

    Args:
        performance_df: DataFrame with performance metrics

    Returns:
        Plotly figure with the chart
    """
    if performance_df is None or performance_df.empty:
        return None
    
    # Sort by performance (descending)
    df_sorted = performance_df.sort_values('performance', ascending=False)
    
    # Create bar colors based on performance (green for positive, red for negative)
    colors = ['green' if val >= 0 else 'red' for val in df_sorted['performance']]
    
    # Create the bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_sorted.index,
        y=df_sorted['performance'] * 100,  # Convert to percentage
        marker_color=colors,
        text=[f"{val:.2f}%" for val in df_sorted['performance'] * 100],
        textposition='auto',
        name="Performance"
    ))
    
    # Update layout
    fig.update_layout(
        title="Asset Performance Ranking",
        xaxis_title="Asset",
        yaxis_title="Performance (%)",
        height=400,
        margin=dict(t=30, l=50, r=50, b=50)
    )
    
    return fig

def create_portfolio_performance_chart(portfolio_df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing portfolio performance metrics

    Args:
        portfolio_df: DataFrame with portfolio performance

    Returns:
        Plotly figure with the chart
    """
    if portfolio_df is None or portfolio_df.empty:
        return None
    
    # Create figures for different aspects of portfolio performance
    # First figure shows portfolio value and drawdown
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add portfolio value
    fig1.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            line=dict(color='blue', width=2),
            name="Portfolio Value"
        ),
        secondary_y=False
    )
    
    # Add drawdown
    fig1.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['drawdown'] * 100,  # Convert to percentage
            mode='lines',
            line=dict(color='red', width=1.5),
            name="Drawdown (%)",
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ),
        secondary_y=True
    )
    
    # Update y-axes titles
    fig1.update_yaxes(
        title=dict(text="Portfolio Value", font=dict(color="blue")), 
        secondary_y=False
    )
    fig1.update_yaxes(
        title=dict(text="Drawdown (%)", font=dict(color="red")), 
        secondary_y=True
    )
    
    # Second figure for position size and asset ranking
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add position size (number of shares)
    fig2.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['position'],
            mode='lines',
            line=dict(color='green', width=2),
            name="Shares Held"
        ),
        secondary_y=False
    )
    
    # Add asset ranking at signal points
    # Filter for signal points only
    buy_signals = portfolio_df[portfolio_df['signal'] == 1]
    sell_signals = portfolio_df[portfolio_df['signal'] == -1]
    
    # Add buy signal ranks
    if not buy_signals.empty and 'asset_rank' in buy_signals.columns:
        fig2.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['asset_rank'],
                mode='markers',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                name="Rank at Buy",
                text=[f"Rank: {int(r)}" for r in buy_signals['asset_rank']],
                hoverinfo='text'
            ),
            secondary_y=True
        )
    
    # Add sell signal ranks
    if not sell_signals.empty and 'asset_rank' in sell_signals.columns:
        fig2.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['asset_rank'],
                mode='markers',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                name="Rank at Sell",
                text=[f"Rank: {int(r)}" for r in sell_signals['asset_rank']],
                hoverinfo='text'
            ),
            secondary_y=True
        )
    
    # Update y-axes titles
    fig2.update_yaxes(
        title=dict(text="Shares Held", font=dict(color="green")), 
        secondary_y=False
    )
    fig2.update_yaxes(
        title=dict(text="Asset Rank", font=dict(color="purple")), 
        secondary_y=True,
        # Lower rank is better (rank 1 = best performer), so invert the axis
        autorange='reversed'
    )
    
    # Combine plots into a single figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.1,
                      subplot_titles=("Portfolio Value & Drawdown", "Position Size & Asset Ranking"))
    
    # Copy traces from fig1 to the combined figure (first row)
    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)
    
    # Copy traces from fig2 to the combined figure (second row)
    for trace in fig2.data:
        fig.add_trace(trace, row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance & Position Analysis",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,  # Increase height for both charts
        margin=dict(t=50, l=50, r=50, b=30)
    )
    
    return fig

def create_multi_asset_chart(price_data_dict: Dict[str, pd.DataFrame], 
                          days: int = 5,
                          sort_by_performance: bool = True) -> go.Figure:
    """
    Create a chart showing multiple assets for comparison
    
    Args:
        price_data_dict: Dictionary mapping symbols to price DataFrames
        days: Number of days to display
        sort_by_performance: Whether to sort assets by performance
        
    Returns:
        Plotly figure with the chart
    """
    if not price_data_dict:
        return None
    
    # Calculate performance for each asset
    performance_data = {}
    for symbol, df in price_data_dict.items():
        if len(df) < 2:
            continue
        
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        performance = (end_price - start_price) / start_price
        performance_data[symbol] = {
            'performance': performance,
            'start_price': start_price,
            'end_price': end_price,
            'color': 'green' if performance >= 0 else 'red'
        }
    
    # Sort by performance if requested
    if sort_by_performance:
        performance_data = dict(sorted(
            performance_data.items(), 
            key=lambda x: x[1]['performance'], 
            reverse=True
        ))
    
    # Create figure
    fig = go.Figure()
    
    # Add a line for each asset
    for symbol, data in performance_data.items():
        price_df = price_data_dict[symbol]
        
        # Normalize the price to start at 1.0
        normalized_price = price_df['close'] / price_df['close'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=price_df.index,
            y=normalized_price,
            mode='lines',
            name=f"{symbol} ({data['performance']*100:.1f}%)",
            line=dict(
                color=data['color'],
                width=2
            )
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Comparative Asset Performance (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Starting at 1.0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(t=50, l=50, r=50, b=50),
        hovermode="x unified"
    )
    
    # Add a horizontal line at y=1 for reference
    fig.add_shape(
        type="line",
        x0=min([df.index[0] for df in price_data_dict.values()]),
        x1=max([df.index[-1] for df in price_data_dict.values()]),
        y0=1,
        y1=1,
        line=dict(color="black", width=1, dash="dash")
    )
    
    return fig

def load_best_params(symbol: str) -> Optional[Dict]:
    """
    Load best parameters for a given symbol from a local JSON file
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with parameters or None if not found
    """
    try:
        with open("best_params.json", "r") as f:
            best_params_data = json.load(f)
            if symbol in best_params_data:
                params = best_params_data[symbol]['best_params']
                logger.info(f"Using best parameters for {symbol}: {params}")
                return params
            else:
                logger.info(f"No best parameters found for {symbol}. Using default parameters.")
                return None
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Best parameters file not found or invalid. Using default parameters.")
        return None