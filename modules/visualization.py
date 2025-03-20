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

# Configure logger
logger = logging.getLogger(__name__)

def create_price_chart(
    price_data: pd.DataFrame, 
    signals_df: Optional[pd.DataFrame] = None,
    daily_composite: Optional[pd.DataFrame] = None, 
    weekly_composite: Optional[pd.DataFrame] = None,
    portfolio_df: Optional[pd.DataFrame] = None,
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
        show_fractal and signals_df is not None and 'fractal' in signals_df.columns
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
    
    # Current row counter (define it before we use it)
    current_row = 1
    
    # Add portfolio equity curve if available
    if portfolio_df is not None:
        # Create a secondary axis for portfolio value on the price chart (row 1)
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            line=dict(color='black', width=1.5),
            name="Portfolio Value",
            yaxis="y2"
        ), row=1, col=1)
        
        # Add number of shares if available
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['position'],
            mode='lines',
            line=dict(color='purple', width=1.5, dash='dot'),
            name="Shares Held",
            yaxis="y3"
        ), row=1, col=1)
        
        # Update layout to include the secondary y-axes
        fig.update_layout(
            yaxis2=dict(
                title=dict(
                    text="Portfolio Value",
                    font=dict(color="black")
                ),
                tickfont=dict(color="black"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            yaxis3=dict(
                title=dict(
                    text="Shares",
                    font=dict(color="purple")
                ),
                tickfont=dict(color="purple"),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.95
            )
        )
    
    # Set the current row for indicator panels
    current_row = 2
    
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
            title=dict(
                text="Daily Composite",
                font=dict(color="blue")
            ), 
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
            title=dict(
                text="Weekly Composite",
                font=dict(color="blue")
            ),
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
            title=dict(
                text="MACD",
                font=dict(color="blue")
            ),
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
            title=dict(
                text="RSI",
                font=dict(color="blue")
            ),
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
            title=dict(
                text="Stochastic",
                font=dict(color="blue")
            ),
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
            title=dict(
                text="Fractal Complexity",
                font=dict(color="blue")
            ),
            row=current_row,
            col=1
        )
    
    # Update layout
    symbol_name = price_data.iloc[-1]['symbol'] if 'symbol' in price_data.columns else ''
    fig.update_layout(
        title=f"{symbol_name} Price Chart with Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        height=rows * 200,  # Height per row
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(t=30, l=50, r=50, b=30)  # Add some margin
    )
    
    # Set y-axis range for price chart (row 1) to fit all data plus some margin
    price_range = price_data['high'].max() - price_data['low'].min()
    y_min = price_data['low'].min() - price_range * 0.05
    y_max = price_data['high'].max() + price_range * 0.05
    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    
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