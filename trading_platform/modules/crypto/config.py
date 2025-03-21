"""
Cryptocurrency-specific configuration and symbol definitions
"""

# Default parameters
DEFAULT_SYMBOL = 'BTC/USD'
DEFAULT_TIMEFRAME = '1h'
DEFAULT_LOOKBACK_DAYS = 15

# Trading costs for crypto
TRADING_COSTS = {
    'trading_fee': 0.001,  # 0.1% per trade
    'spread': 0.001,       # 0.1% spread (bid-ask)
    'slippage': 0.001      # 0.1% slippage
}

# Available trading symbols with metadata
TRADING_SYMBOLS = {
    'BTC/USD': {
        'name': 'Bitcoin',
        'description': 'Bitcoin to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.001,  # Minimum order size in BTC
        'price_precision': 2,     # Price precision in decimal places
        'amount_precision': 6,    # Amount precision in decimal places
        'exchange': 'Coinbase'
    },
    'ETH/USD': {
        'name': 'Ethereum',
        'description': 'Ethereum to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.01,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'SOL/USD': {
        'name': 'Solana',
        'description': 'Solana to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'AVAX/USD': {
        'name': 'Avalanche',
        'description': 'Avalanche to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'DOT/USD': {
        'name': 'Polkadot',
        'description': 'Polkadot to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'LINK/USD': {
        'name': 'Chainlink',
        'description': 'Chainlink to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'DOGE/USD': {
        'name': 'Dogecoin',
        'description': 'Dogecoin to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 10,
        'price_precision': 6,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    },
    'AAVE/USD': {
        'name': 'Aave',
        'description': 'Aave to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.01,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'UNI/USD': {
        'name': 'Uniswap',
        'description': 'Uniswap to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'LTC/USD': {
        'name': 'Litecoin',
        'description': 'Litecoin to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.01,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'SHIB/USD': {
        'name': 'Shiba Inu',
        'description': 'Shiba Inu to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 100000,
        'price_precision': 8,
        'amount_precision': 0,
        'exchange': 'Coinbase'
    },
    'BAT/USD': {
        'name': 'Basic Attention Token',
        'description': 'Basic Attention Token to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 1,
        'price_precision': 4,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    },
    'BCH/USD': {
        'name': 'Bitcoin Cash',
        'description': 'Bitcoin Cash to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.01,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'CRV/USD': {
        'name': 'Curve DAO Token',
        'description': 'Curve DAO Token to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 1,
        'price_precision': 4,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    },
    'GRT/USD': {
        'name': 'The Graph',
        'description': 'The Graph to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 1,
        'price_precision': 4,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    },
    'MKR/USD': {
        'name': 'Maker',
        'description': 'Maker to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.001,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'SUSHI/USD': {
        'name': 'SushiSwap',
        'description': 'SushiSwap to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 4,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    },
    'XTZ/USD': {
        'name': 'Tezos',
        'description': 'Tezos to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.1,
        'price_precision': 4,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    },
    'YFI/USD': {
        'name': 'yearn.finance',
        'description': 'yearn.finance to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 0.0001,
        'price_precision': 2,
        'amount_precision': 6,
        'exchange': 'Coinbase'
    },
    'XRP/USD': {
        'name': 'Ripple',
        'description': 'Ripple to US Dollar',
        'market_hours': {
            'start': '00:00',
            'end': '23:59',
            'timezone': 'UTC'
        },
        'min_order_size': 1,
        'price_precision': 4,
        'amount_precision': 2,
        'exchange': 'Coinbase'
    }
}