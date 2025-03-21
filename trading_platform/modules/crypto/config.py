"""
Cryptocurrency-specific configuration and symbol definitions
"""

# Trading symbols configuration for crypto assets
TRADING_SYMBOLS = {
    # Cryptocurrencies
    'BTC/USD': {
        'name': 'Bitcoin',
        'market': 'CRYPTO',
        'yfinance': 'BTC-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'ETH/USD': {
        'name': 'Ethereum',
        'market': 'CRYPTO',
        'yfinance': 'ETH-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SOL/USD': {
        'name': 'Solana',
        'market': 'CRYPTO',
        'yfinance': 'SOL-USD',
        'interval': '1h',       
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    }, 
    'AVAX/USD': {
        'name': 'Avalanche',
        'market': 'CRYPTO',
        'yfinance': 'AVAX-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'DOT/USD': {
        'name': 'Polkadot',
        'market': 'CRYPTO',
        'yfinance': 'DOT-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'LINK/USD': {
        'name': 'Chainlink',
        'market': 'CRYPTO',
        'yfinance': 'LINK-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'DOGE/USD': {
        'name': 'Dogecoin',
        'market': 'CRYPTO',
        'yfinance': 'DOGE-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'AAVE/USD': {
        'name': 'Aave',
        'market': 'CRYPTO',
        'yfinance': 'AAVE-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'UNI/USD': {
        'name': 'Uniswap',
        'market': 'CRYPTO',
        'yfinance': 'UNI7083-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'LTC/USD': {
        'name': 'Litecoin',
        'market': 'CRYPTO',
        'yfinance': 'LTC-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SHIB/USD': {
        'name': 'Shiba Inu',
        'market': 'CRYPTO',
        'yfinance': 'SHIB-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'BAT/USD': {
        'name': 'Basic Attention Token',
        'market': 'CRYPTO',
        'yfinance': 'BAT-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'BCH/USD': {
        'name': 'Bitcoin Cash',
        'market': 'CRYPTO',
        'yfinance': 'BCH-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'CRV/USD': {
        'name': 'Curve DAO Token',
        'market': 'CRYPTO',
        'yfinance': 'CRV-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'GRT/USD': {
        'name': 'The Graph',
        'market': 'CRYPTO',
        'yfinance': 'GRT6719-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'MKR/USD': {
        'name': 'Maker',
        'market': 'CRYPTO',
        'yfinance': 'MKR-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'SUSHI/USD': {
        'name': 'SushiSwap',
        'market': 'CRYPTO',
        'yfinance': 'SUSHI-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'XTZ/USD': {
        'name': 'Tezos',
        'market': 'CRYPTO',
        'yfinance': 'XTZ-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'YFI/USD': {
        'name': 'yearn.finance',
        'market': 'CRYPTO',
        'yfinance': 'YFI-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    },
    'XRP/USD': {
        'name': 'Ripple',
        'market': 'CRYPTO',
        'yfinance': 'XRP-USD',
        'interval': '1h',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        }
    }
}

# Trading costs specific to crypto
TRADING_COSTS = {
    'trading_fee': 0.003,  # 0.3% taker fee
    'spread': 0.002,  # 0.2% maker fee
}

# Default settings
DEFAULT_SYMBOL = 'BTC/USD'
DEFAULT_TIMEFRAME = '1h'
DEFAULT_LOOKBACK_DAYS = 15