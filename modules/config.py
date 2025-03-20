import datetime
from typing import Dict, Any

# App configuration
APP_CONFIG = {
    "title": "Crypto Price Monitor with Technical Indicators",
    "description": "Real-time price data with signal alerts based on technical indicators",
    "version": "1.0.0",
    "author": "Replit"
}

# Trading settings
TRADING_SETTINGS = {
    "default_timeframe": "1h",
    "default_lookback_days": 15,
    "default_update_interval": 15,  # seconds
    "default_initial_capital": 100000
}

# Chart settings
CHART_SETTINGS = {
    "price_chart_height": 600,
    "indicator_chart_height": 200,
    "portfolio_chart_height": 400,
    "ranking_chart_height": 400
}

# Parameter grids for optimization
PARAMETER_GRIDS = {
    "default": {
        "macd_weight": [0.25, 0.5, 0.75, 1.0],
        "rsi_weight": [0.25, 0.5, 0.75, 1.0],
        "stoch_weight": [0.25, 0.5, 0.75, 1.0],
        "fractal_weight": [0.1, 0.25, 0.5],
        "reactivity": [0.5, 1.0, 1.5, 2.0]
    }
}