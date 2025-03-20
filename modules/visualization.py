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
    # Determine number of rows needed for subplot
    num_indicator_rows = sum([show_macd, show_rsi, show_stochastic, show_fractal])
    if portfolio_df is not None:
        num_indicator_rows += 1
    
    # Create subplot with price chart and indicators
    fig = make_subplots(
        rows=1 + num_indicator_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5] + [0.5 / num_indicator_rows] * num_indicator_rows if num_indicator_rows > 0 else [1]
    )
    
    # Add price candles to the chart
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume as bar chart on price chart
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['volume'],
            name="Volume",
            marker_color='rgba(0,0,0,0.2)',
            opacity=0.3
        ),
        row=1, col=1
    )
    
    # Add daily composite indicator to price chart if available
    if daily_composite is not None:
        # Plot the daily composite indicator
        fig.add_trace(
            go.Scatter(
                x=daily_composite.index,
                y=daily_composite['daily_composite'],
                name="Daily Composite",
                line=dict(color='purple', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add threshold bands
        fig.add_trace(
            go.Scatter(
                x=daily_composite.index,
                y=daily_composite['upper_threshold'],
                name="Upper Threshold",
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_composite.index,
                y=daily_composite['lower_threshold'],
                name="Lower Threshold",
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.5
            ),
            row=1, col=1
        )
    
    # Add weekly composite indicator if available
    if weekly_composite is not None:
        fig.add_trace(
            go.Scatter(
                x=weekly_composite.index,
                y=weekly_composite['weekly_composite'],
                name="Weekly Composite",
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add threshold bands
        fig.add_trace(
            go.Scatter(
                x=weekly_composite.index,
                y=weekly_composite['upper_threshold'],
                name="Weekly Upper",
                line=dict(color='green', width=1.5, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=weekly_composite.index,
                y=weekly_composite['lower_threshold'],
                name="Weekly Lower",
                line=dict(color='red', width=1.5, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add buy/sell signals if requested and available
    if show_signals and signals_df is not None:
        # Buy signals (green triangles)
        buy_signals = signals_df[signals_df['signal'] == 1]
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=price_data.loc[buy_signals.index, 'low'] * 0.99,  # Slightly below price
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name="Buy Signal"
                ),
                row=1, col=1
            )
        
        # Sell signals (red triangles)
        sell_signals = signals_df[signals_df['signal'] == -1]
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=price_data.loc[sell_signals.index, 'high'] * 1.01,  # Slightly above price
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name="Sell Signal"
                ),
                row=1, col=1
            )
    
    # Current row index for adding indicators
    current_row = 2
    
    # Add MACD indicator if requested
    if show_macd and 'macd' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['macd'],
                name="MACD",
                line=dict(color='blue', width=1.5)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['macd_signal'],
                name="MACD Signal",
                line=dict(color='red', width=1)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=price_data.index,
                y=price_data['macd_hist'],
                name="MACD Histogram",
                marker_color=np.where(price_data['macd_hist'] >= 0, 'green', 'red')
            ),
            row=current_row, col=1
        )
        
        # Update y-axis title for MACD
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1
    
    # Add RSI indicator if requested
    if show_rsi and 'rsi' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['rsi'],
                name="RSI",
                line=dict(color='purple', width=1.5)
            ),
            row=current_row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=[70] * len(price_data),
                name="Overbought",
                line=dict(color='red', width=1, dash='dash')
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=[30] * len(price_data),
                name="Oversold",
                line=dict(color='green', width=1, dash='dash')
            ),
            row=current_row, col=1
        )
        
        # Update y-axis title for RSI
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1
    
    # Add Stochastic indicator if requested
    if show_stochastic and 'stoch_k' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['stoch_k'],
                name="%K",
                line=dict(color='blue', width=1.5)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['stoch_d'],
                name="%D",
                line=dict(color='red', width=1)
            ),
            row=current_row, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=[80] * len(price_data),
                name="Overbought",
                line=dict(color='red', width=1, dash='dash')
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=[20] * len(price_data),
                name="Oversold",
                line=dict(color='green', width=1, dash='dash')
            ),
            row=current_row, col=1
        )
        
        # Update y-axis title for Stochastic
        fig.update_yaxes(title_text="Stochastic", row=current_row, col=1)
        current_row += 1
    
    # Add Fractal Complexity indicator if requested
    if show_fractal and 'fractal_complexity' in price_data.columns:
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data['fractal_complexity'],
                name="Fractal Complexity",
                line=dict(color='orange', width=1.5)
            ),
            row=current_row, col=1
        )
        
        # Update y-axis title for Fractal Complexity
        fig.update_yaxes(title_text="Fractal", row=current_row, col=1)
        current_row += 1
    
    # Add portfolio performance if available
    if portfolio_df is not None:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                name="Portfolio Value",
                line=dict(color='green', width=1.5)
            ),
            row=current_row, col=1
        )
        
        # Update y-axis title for Portfolio
        fig.update_yaxes(title_text="Portfolio $", row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        title="Price Chart with Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        height=900,  # Adjust height based on number of indicators
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Hide duplicate legend entries
    fig.update_layout(showlegend=True)
    
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
    sorted_df = performance_df.sort_values('performance', ascending=False)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars for each asset
    fig.add_trace(
        go.Bar(
            y=sorted_df.index,
            x=sorted_df['performance'] * 100,  # Convert to percentage
            orientation='h',
            marker_color=['green' if x >= 0 else 'red' for x in sorted_df['performance']],
            name="Performance"
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Asset Performance Ranking (%)",
        xaxis_title="Performance (%)",
        yaxis_title="Asset",
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
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
    
    # Create subplot with price, portfolio value, and drawdown
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.3, 0.2]
    )
    
    # Add price chart
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['price'],
            name="Asset Price",
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            name="Portfolio Value",
            line=dict(color='green', width=1.5)
        ),
        row=2, col=1
    )
    
    # Initial capital reference line
    if 'portfolio_value' in portfolio_df.columns and not portfolio_df.empty:
        initial_capital = portfolio_df['portfolio_value'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=[initial_capital] * len(portfolio_df),
                name="Initial Capital",
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=2, col=1
        )
    
    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['drawdown'] * 100,  # Convert to percentage
            name="Drawdown",
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    return fig