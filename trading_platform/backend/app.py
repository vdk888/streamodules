"""
FastAPI backend for the trading platform
"""
import os
import time
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

from trading_platform.modules.crypto.service import CryptoService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Trading Platform API",
    description="API for cryptocurrency trading platform with real-time data and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
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

    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active_connections:
            if websocket in self.active_connections[symbol]:
                self.active_connections[symbol].remove(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, symbol: str):
        if symbol in self.active_connections:
            for connection in self.active_connections[symbol]:
                await connection.send_text(message)

# Initialize connection manager
manager = ConnectionManager()

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

# Cryptocurrency endpoints
@app.get("/api/crypto/symbols")
async def get_crypto_symbols():
    try:
        symbols = CryptoService.get_available_symbols()
        return {
            "success": True,
            "symbols": symbols
        }
    except Exception as e:
        logger.error(f"Error getting crypto symbols: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/crypto/data/{symbol}")
async def get_crypto_data(
    symbol: str,
    timeframe: str = Query("1h", description="Data timeframe (1h, 4h, 1d)"),
    lookback_days: int = Query(15, description="Number of days to look back")
):
    try:
        result = CryptoService.get_symbol_data(symbol, timeframe, lookback_days)
        return result
    except Exception as e:
        logger.error(f"Error getting crypto data: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/crypto/ranking")
async def get_crypto_ranking(
    lookback_days: int = Query(15, description="Number of days to look back")
):
    try:
        result = CryptoService.get_performance_ranking(lookback_days)
        return result
    except Exception as e:
        logger.error(f"Error getting performance ranking: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Pydantic models for request data
class BacktestRequest(BaseModel):
    days: int = 15
    params: Optional[Dict[str, Any]] = None

@app.post("/api/crypto/backtest/{symbol}")
async def run_crypto_backtest(
    symbol: str,
    request: BacktestRequest
):
    try:
        result = CryptoService.run_backtest(symbol, request.days, request.params)
        return result
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

class OptimizeRequest(BaseModel):
    days: int = 15

@app.post("/api/crypto/optimize/{symbol}")
async def optimize_crypto_parameters(
    symbol: str,
    request: OptimizeRequest
):
    try:
        result = CryptoService.optimize_parameters(symbol, request.days)
        return result
    except Exception as e:
        logger.error(f"Error optimizing parameters: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

class PortfolioRequest(BaseModel):
    symbols: Optional[List[str]] = None
    days: int = 15
    initial_capital: float = 100000.0

@app.post("/api/crypto/portfolio/simulate")
async def simulate_crypto_portfolio(
    request: PortfolioRequest
):
    try:
        result = CryptoService.simulate_portfolio(
            request.symbols, request.days, request.initial_capital
        )
        return result
    except Exception as e:
        logger.error(f"Error simulating portfolio: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# WebSocket endpoint for real-time data
@app.websocket("/ws/crypto/{symbol}")
async def websocket_crypto_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket, symbol)
    try:
        # Start background task for sending updates
        task = asyncio.create_task(send_updates(websocket, symbol))
        
        # Wait for client messages (e.g., configuration changes)
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages (e.g., change timeframe)
                if "timeframe" in message:
                    # Can implement logic to change update frequency
                    await manager.send_personal_message(
                        json.dumps({
                            "success": True,
                            "message": f"Timeframe changed to {message['timeframe']}"
                        }),
                        websocket
                    )
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "success": False,
                        "error": "Invalid JSON"
                    }),
                    websocket
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
        # Cancel the background task
        task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket, symbol)
        # Cancel the background task
        task.cancel()

async def send_updates(websocket: WebSocket, symbol: str):
    """Background task to send real-time updates to WebSocket clients"""
    try:
        timeframe = "1h"  # Default timeframe
        lookback_days = 15  # Default lookback days
        
        while True:
            # Get the latest data
            result = CryptoService.get_symbol_data(symbol, timeframe, lookback_days)
            
            # Send updated data to the client
            await websocket.send_text(json.dumps(result))
            
            # Wait before next update (adjust based on timeframe)
            if timeframe == "1m":
                await asyncio.sleep(10)  # Every 10 seconds for 1-minute data
            elif timeframe == "5m":
                await asyncio.sleep(60)  # Every minute for 5-minute data
            elif timeframe == "15m":
                await asyncio.sleep(180)  # Every 3 minutes for 15-minute data
            elif timeframe == "30m":
                await asyncio.sleep(300)  # Every 5 minutes for 30-minute data
            elif timeframe == "1h":
                await asyncio.sleep(600)  # Every 10 minutes for 1-hour data
            elif timeframe == "4h":
                await asyncio.sleep(1800)  # Every 30 minutes for 4-hour data
            else:
                await asyncio.sleep(3600)  # Every hour for daily data
                
    except asyncio.CancelledError:
        # Task was cancelled due to client disconnect
        logger.info(f"Update task for {symbol} cancelled")
    except Exception as e:
        logger.error(f"Error in send_updates for {symbol}: {str(e)}")

# Create the FastAPI app
app = FastAPI()