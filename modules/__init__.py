# modules/__init__.py
# This file marks the modules directory as a Python package

# Import key components for easier access
from modules.data import get_price_data, get_multi_asset_data, generate_trading_signals
from modules.visualization import create_price_chart, create_performance_ranking_chart, create_portfolio_performance_chart
from modules.storage import load_best_params, save_best_params, check_storage_connection
from modules.backtest import run_backtest, find_best_params, calculate_performance_ranking
from modules.simulation import simulate_portfolio
from modules.ui import (
    create_sidebar, 
    display_performance_metrics, 
    create_header, 
    setup_auto_refresh, 
    update_progress_bar, 
    handle_backtest_button, 
    handle_optimization_button
)