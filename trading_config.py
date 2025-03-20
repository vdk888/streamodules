"""
Configuration settings for the trading application
"""

# Define trading symbols with their configuration
TRADING_SYMBOLS = {
    "BTC/USD": {
        "yfinance": "BTC-USD",
        "description": "Bitcoin to US Dollar",
        "type": "crypto",
        "decimal_places": 8
    },
    "ETH/USD": {
        "yfinance": "ETH-USD",
        "description": "Ethereum to US Dollar",
        "type": "crypto",
        "decimal_places": 8
    },
    "AAPL": {
        "yfinance": "AAPL",
        "description": "Apple Inc.",
        "type": "stock",
        "decimal_places": 2
    },
    "MSFT": {
        "yfinance": "MSFT",
        "description": "Microsoft Corporation",
        "type": "stock",
        "decimal_places": 2
    },
    "AMZN": {
        "yfinance": "AMZN",
        "description": "Amazon.com Inc.",
        "type": "stock",
        "decimal_places": 2
    }
}

# Trading costs (simplified)
TRADING_COSTS = {
    "crypto": {
        "maker_fee": 0.001,  # 0.1%
        "taker_fee": 0.002,  # 0.2%
        "min_fee": 0.0      # No minimum fee
    },
    "stock": {
        "maker_fee": 0.0,    # No fee for limit orders
        "taker_fee": 0.0005, # 0.05% for market orders
        "min_fee": 1.0       # Minimum $1 fee
    }
}

# Default risk percentage for position sizing
DEFAULT_RISK_PERCENT = 0.02  # 2% risk per trade

# Other constants
TRADE_WINDOW_LOOKBACK = 30   # Days to look back for trade window
MIN_DATA_POINTS = 20         # Minimum number of data points needed for analysis