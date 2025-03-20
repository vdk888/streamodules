# Import configuration from attached_assets
from attached_assets.config import (
    TRADING_SYMBOLS,
    param_grid as DEFAULT_PARAM_GRID,  # Rename imported param_grid
    calculate_capital_multiplier,
    get_max_days,
    get_update_interval
)

# Define MAX_LOOKBACK_DAYS
MAX_LOOKBACK_DAYS = 30