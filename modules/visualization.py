import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Tuple, Optional, List

def create_price_chart(price_data: pd.DataFrame, 
                      signals_df: Optional[pd.DataFrame] = None,
                      daily_composite: Optional[pd.DataFrame] = None, 
                      weekly_composite: Optional[pd.DataFrame] = None,
                      portfolio_df: Optional[pd.DataFrame] = None,
                      show_macd: bool = True,
                      show_rsi: bool = True,
                      show_stochastic: bool = True,
                      show_fractal: bool = True,
                      show_signals: bool = True) -> go.Figure:
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
    # Calculate how many rows we need for our subplot grid
    rows = 1  # Price chart always shown
    if show_macd:
        rows += 1
    if show_rsi:
        rows += 1
    if show_stochastic:
        rows += 1
    if show_fractal:
        rows += 1
        
    # Create subplots with appropriate row heights
    row_heights = [0.5]  # First row is the price chart (50% of height)
    height_per_indicator = 0.5 / (rows - 1) if rows > 1 else 0.5  # Remaining rows share 50% of height equally
    for _ in range(rows - 1):
        row_heights.append(height_per_indicator)
        
    # Create figure
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.02, row_heights=row_heights)
    
    # Add price candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name="Price"
    ), row=1, col=1)
    
    # Add daily and weekly composite bands if available
    if daily_composite is not None and 'daily_composite' in daily_composite.columns:
        # Add daily composite line
        fig.add_trace(go.Scatter(
            x=daily_composite.index,
            y=daily_composite['daily_composite'],
            mode='lines',
            line=dict(color='blue', width=1.5),
            name="Daily Composite"
        ), row=1, col=1)
        
        # Add upper threshold
        fig.add_trace(go.Scatter(
            x=daily_composite.index,
            y=daily_composite['upper_threshold'],
            mode='lines',
            line=dict(color='green', width=1, dash='dot'),
            name="Upper Threshold",
            opacity=0.7
        ), row=1, col=1)
        
        # Add lower threshold
        fig.add_trace(go.Scatter(
            x=daily_composite.index,
            y=daily_composite['lower_threshold'],
            mode='lines',
            line=dict(color='red', width=1, dash='dot'),
            name="Lower Threshold",
            opacity=0.7
        ), row=1, col=1)
    
    if weekly_composite is not None and 'weekly_composite' in weekly_composite.columns:
        # Add weekly composite line
        fig.add_trace(go.Scatter(
            x=weekly_composite.index,
            y=weekly_composite['weekly_composite'],
            mode='lines',
            line=dict(color='purple', width=2),
            name="Weekly Composite"
        ), row=1, col=1)
    
    # Add buy/sell signals if available
    if signals_df is not None and show_signals:
        # Make sure signals_df has the close column
        if 'close' not in signals_df.columns and 'close' in price_data.columns:
            # Add close price from price_data to signals_df
            common_dates = signals_df.index.intersection(price_data.index)
            signals_df = signals_df.copy()  # Avoid modifying the original DataFrame
            signals_df.loc[common_dates, 'close'] = price_data.loc[common_dates, 'close']
            
        # Buy signals (only if 'close' is now available)
        if 'close' in signals_df.columns:
            buy_signals = signals_df[signals_df['signal'] == 1]
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'] * 0.99,  # Place markers slightly below the close price
                    mode='markers',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    name="Buy Signal"
                ), row=1, col=1)
            
            # Sell signals
            sell_signals = signals_df[signals_df['signal'] == -1]
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['close'] * 1.01,  # Place markers slightly above the close price
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    name="Sell Signal"
                ), row=1, col=1)
    
    # Add portfolio equity curve if available
    if portfolio_df is not None:
        # Create a secondary axis for portfolio value
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            line=dict(color='black', width=1.5),
            name="Portfolio Value",
            yaxis="y2"
        ), row=1, col=1)
        
        # Add a secondary y-axis for portfolio value
        fig.update_layout(
            yaxis2=dict(
                title="Portfolio Value ($)",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
    
    # Current row counter
    current_row = 2
    
    # Add MACD indicator if enabled
    if show_macd and current_row <= rows:
        # Ensure we have the necessary columns
        if signals_df is not None and 'macd' in signals_df.columns:
            # Add MACD line
            fig.add_trace(go.Scatter(
                x=signals_df.index,
                y=signals_df['macd'],
                mode='lines',
                line=dict(color='blue', width=1.5),
                name="MACD"
            ), row=current_row, col=1)
            
            # Add Signal line
            if 'macd_signal' in signals_df.columns:
                fig.add_trace(go.Scatter(
                    x=signals_df.index,
                    y=signals_df['macd_signal'],
                    mode='lines',
                    line=dict(color='red', width=1.5),
                    name="Signal"
                ), row=current_row, col=1)
            
            # Add Histogram
            if 'macd_hist' in signals_df.columns:
                # Create histogram colors based on values
                colors = ['green' if val >= 0 else 'red' for val in signals_df['macd_hist']]
                
                fig.add_trace(go.Bar(
                    x=signals_df.index,
                    y=signals_df['macd_hist'],
                    marker_color=colors,
                    name="MACD Histogram"
                ), row=current_row, col=1)
            
            # Add title for the subplot
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
            current_row += 1
    
    # Add RSI indicator if enabled
    if show_rsi and current_row <= rows:
        # Ensure we have the necessary columns
        if signals_df is not None and 'rsi' in signals_df.columns:
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
            fig.update_yaxes(title_text="RSI", row=current_row, col=1)
            current_row += 1
    
    # Add Stochastic indicator if enabled
    if show_stochastic and current_row <= rows:
        # Ensure we have the necessary columns
        if signals_df is not None and 'stoch_k' in signals_df.columns:
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
            fig.update_yaxes(title_text="Stochastic", row=current_row, col=1)
            current_row += 1
    
    # Add Fractal Complexity indicator if enabled
    if show_fractal and current_row <= rows:
        # Ensure we have the necessary columns
        if signals_df is not None and 'fractal' in signals_df.columns:
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
            fig.update_yaxes(title_text="Fractal Complexity", row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{price_data.iloc[-1]['symbol'] if 'symbol' in price_data.columns else ''} Price Chart with Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        height=rows * 200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False
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
    
    # Sort by performance
    sorted_df = performance_df.sort_values(by='performance', ascending=False)
    
    # Create colors based on performance (green for positive, red for negative)
    colors = ['green' if perf >= 0 else 'red' for perf in sorted_df['performance']]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for performance
    fig.add_trace(go.Bar(
        x=sorted_df.index,
        y=sorted_df['performance'] * 100,  # Convert to percentage
        marker_color=colors,
        text=[f"{perf:.2f}%" for perf in sorted_df['performance'] * 100],
        textposition='auto',
        name="Performance"
    ))
    
    # Update layout
    fig.update_layout(
        title="Asset Performance Ranking",
        xaxis_title="Symbol",
        yaxis_title="Performance (%)",
        height=400,
        template="plotly_white"
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
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add portfolio value trace
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            name="Portfolio Value",
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    # Add drawdown trace on secondary axis
    if 'drawdown' in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['drawdown'] * 100,  # Convert to percentage
                mode='lines',
                name="Drawdown",
                line=dict(color='red', width=1.5)
            ),
            secondary_y=True
        )
    
    # Add portfolio cash trace
    if 'cash' in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['cash'],
                mode='lines',
                name="Cash",
                line=dict(color='green', width=1.5, dash='dot')
            ),
            secondary_y=False
        )
    
    # Add position value trace
    if 'position_value' in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['position_value'],
                mode='lines',
                name="Position Value",
                line=dict(color='orange', width=1.5, dash='dot')
            ),
            secondary_y=False
        )
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Value ($)", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown (%)", secondary_y=True)
    
    return fig