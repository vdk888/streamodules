"""
Cryptocurrency-specific configuration and symbol definitions
"""

# Default settings
DEFAULT_SYMBOL = "BTC/USD"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_LOOKBACK_DAYS = 15
DEFAULT_BACKTEST_DAYS = 30

# Available crypto symbols
AVAILABLE_SYMBOLS = [
    "BTC/USD",    # Bitcoin
    "ETH/USD",    # Ethereum
    "SOL/USD",    # Solana
    "AVAX/USD",   # Avalanche
    "DOT/USD",    # Polkadot
    "LINK/USD",   # Chainlink
    "DOGE/USD",   # Dogecoin
    "AAVE/USD",   # Aave
    "UNI/USD",    # Uniswap
    "LTC/USD",    # Litecoin
    "XRP/USD",    # Ripple
    "ADA/USD",    # Cardano
    "MATIC/USD",  # Polygon
    "ATOM/USD",   # Cosmos
    "ALGO/USD",   # Algorand
    "XLM/USD",    # Stellar
    "FIL/USD",    # Filecoin
    "NEAR/USD",   # NEAR Protocol
    "SAND/USD",   # The Sandbox
    "MANA/USD",   # Decentraland
]

# Exchange profiles
EXCHANGE_PROFILES = {
    "coinbase": {
        "name": "Coinbase",
        "url": "https://api.coinbase.com",
        "fee_rate": 0.5,  # percentage
        "trading_hours": {
            "always_open": True
        }
    },
    "binance": {
        "name": "Binance",
        "url": "https://api.binance.com",
        "fee_rate": 0.1,  # percentage
        "trading_hours": {
            "always_open": True
        }
    }
}

# Map symbols to specific exchanges (if needed)
SYMBOL_EXCHANGE_MAP = {}  # Default to first exchange

# Time intervals for data retrieval
TIME_INTERVALS = {
    "1m": {
        "name": "1 Minute",
        "seconds": 60,
        "max_lookback_days": 1
    },
    "5m": {
        "name": "5 Minutes",
        "seconds": 300,
        "max_lookback_days": 5
    },
    "15m": {
        "name": "15 Minutes",
        "seconds": 900,
        "max_lookback_days": 7
    },
    "30m": {
        "name": "30 Minutes",
        "seconds": 1800,
        "max_lookback_days": 14
    },
    "1h": {
        "name": "1 Hour",
        "seconds": 3600,
        "max_lookback_days": 30
    },
    "4h": {
        "name": "4 Hours",
        "seconds": 14400,
        "max_lookback_days": 60
    },
    "1d": {
        "name": "1 Day",
        "seconds": 86400,
        "max_lookback_days": 365
    },
    "1w": {
        "name": "1 Week",
        "seconds": 604800,
        "max_lookback_days": 730
    }
}

# Strategy parameters
STRATEGY_PARAMS = {
    "default": {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_k": 14,
        "stoch_d": 3,
        "hurst_lags": [10, 20, 40, 80],
        "volatility_window": 20,
        "rsi_weight": 0.3,
        "macd_weight": 0.3,
        "stoch_weight": 0.2,
        "fractal_weight": 0.2
    },
    "aggressive": {
        "rsi_period": 10,
        "macd_fast": 8,
        "macd_slow": 20,
        "macd_signal": 7,
        "stoch_k": 10,
        "stoch_d": 3,
        "hurst_lags": [5, 10, 20, 40],
        "volatility_window": 15,
        "rsi_weight": 0.25,
        "macd_weight": 0.35,
        "stoch_weight": 0.15,
        "fractal_weight": 0.25
    },
    "conservative": {
        "rsi_period": 21,
        "macd_fast": 16,
        "macd_slow": 32,
        "macd_signal": 11,
        "stoch_k": 21,
        "stoch_d": 7,
        "hurst_lags": [20, 40, 80, 160],
        "volatility_window": 30,
        "rsi_weight": 0.35,
        "macd_weight": 0.25,
        "stoch_weight": 0.25,
        "fractal_weight": 0.15
    }
}

# Portfolio settings
PORTFOLIO_SETTINGS = {
    "max_assets": 10,        # Maximum number of assets to hold at once
    "position_sizing": {
        "method": "ranking",  # Can be 'equal', 'ranking', 'volatility'
        "min_position": 0.02, # Minimum position size (as fraction of portfolio)
        "max_position": 0.2   # Maximum position size (as fraction of portfolio)
    },
    "rebalancing": {
        "frequency": "daily",  # How often to rebalance portfolio
        "threshold": 0.05      # Minimum deviation to trigger rebalance
    }
}