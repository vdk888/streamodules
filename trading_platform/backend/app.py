"""
FastAPI backend for the trading platform
"""
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import logging
import asyncio
import time
from datetime import datetime

# Import crypto services
from ..modules.crypto.service import CryptoService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trading Platform API",
    description="API for cryptocurrency trading platform",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)
        logger.info(f"WebSocket client connected for {symbol}. Total clients: {len(self.active_connections[symbol])}")
    
    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active_connections:
            if websocket in self.active_connections[symbol]:
                self.active_connections[symbol].remove(websocket)
            logger.info(f"WebSocket client disconnected from {symbol}. Remaining clients: {len(self.active_connections[symbol])}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, symbol: str):
        if symbol in self.active_connections:
            for connection in self.active_connections[symbol]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to client: {str(e)}")

# Initialize connection manager
manager = ConnectionManager()

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Crypto endpoints
@app.get("/api/crypto/symbols")
async def get_crypto_symbols():
    try:
        symbols = CryptoService.get_available_symbols()
        return {"success": True, "symbols": symbols}
    except Exception as e:
        logger.error(f"Error getting crypto symbols: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/crypto/data/{symbol}")
async def get_crypto_data(
    symbol: str,
    timeframe: str = Query("1h", description="Data timeframe (1h, 4h, 1d)"),
    lookback_days: int = Query(15, description="Number of days to look back")
):
    try:
        data = CryptoService.get_symbol_data(symbol, timeframe, lookback_days)
        return {
            "success": True,
            "price_data": data["price_data"].reset_index().to_dict(orient="records") if "price_data" in data else [],
            "signals_data": data["signals_data"].reset_index().to_dict(orient="records") if "signals_data" in data else []
        }
    except Exception as e:
        logger.error(f"Error getting crypto data: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/crypto/ranking")
async def get_crypto_ranking(
    lookback_days: int = Query(15, description="Number of days to look back")
):
    try:
        ranking_data = CryptoService.get_performance_ranking(lookback_days)
        return {"success": True, "ranking": ranking_data}
    except Exception as e:
        logger.error(f"Error getting crypto ranking: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/crypto/backtest/{symbol}")
async def run_crypto_backtest(
    symbol: str,
    days: int = 15,
    params: Optional[Dict[str, Any]] = None
):
    try:
        backtest_result = CryptoService.run_backtest(symbol, days, params)
        return {"success": True, "backtest_result": backtest_result}
    except Exception as e:
        logger.error(f"Error running crypto backtest: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/crypto/optimize/{symbol}")
async def optimize_crypto_parameters(
    symbol: str,
    days: int = 15
):
    try:
        optimization_result = CryptoService.optimize_parameters(symbol, days)
        return {"success": True, "optimization_result": optimization_result}
    except Exception as e:
        logger.error(f"Error optimizing crypto parameters: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/crypto/portfolio/simulate")
async def simulate_crypto_portfolio(
    symbols: Optional[List[str]] = None,
    days: int = 15,
    initial_capital: float = 100000.0
):
    try:
        simulation_result = CryptoService.simulate_portfolio(symbols, days, initial_capital)
        return {"success": True, "simulation_result": simulation_result}
    except Exception as e:
        logger.error(f"Error simulating crypto portfolio: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# WebSocket endpoint for real-time updates
@app.websocket("/ws/crypto/{symbol}")
async def websocket_crypto_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket, symbol)
    
    # Create update task for real-time data
    update_task = None
    
    try:
        # Start background task for updates
        update_task = asyncio.create_task(send_updates(websocket, symbol))
        
        # Keep connection open and handle disconnection
        while True:
            data = await websocket.receive_text()
            
            # You can handle client messages here if needed
            logger.info(f"Received message from client: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Cancel update task when disconnected
        if update_task:
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass

async def send_updates(websocket: WebSocket, symbol: str):
    """Background task to send real-time updates to WebSocket clients"""
    try:
        # Get update interval based on symbol/asset class
        # For simplicity, use 5 seconds for all
        update_interval = 5  
        
        while True:
            try:
                # Fetch latest data
                data = CryptoService.get_symbol_data(symbol, "1h", 15)
                
                # Prepare response data
                response_data = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "price_data": data["price_data"].reset_index().to_dict(orient="records") if "price_data" in data else [],
                    "signals_data": data["signals_data"].reset_index().to_dict(orient="records") if "signals_data" in data else []
                }
                
                # Send update to client
                await websocket.send_text(json.dumps(response_data))
                
            except Exception as e:
                # Send error to client
                error_data = {
                    "success": False,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                await websocket.send_text(json.dumps(error_data))
            
            # Wait for next update
            await asyncio.sleep(update_interval)
            
    except asyncio.CancelledError:
        # Task was cancelled, clean up
        logger.info(f"Update task for {symbol} cancelled")
        raise