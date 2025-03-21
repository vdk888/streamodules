"""
FastAPI backend for the trading platform
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any
import asyncio
import json
import logging
from datetime import datetime

# Import modules
from ..modules.crypto import CryptoService

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
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# --- Crypto Endpoints ---

# Get available crypto symbols
@app.get("/api/crypto/symbols")
async def get_crypto_symbols():
    try:
        symbols = CryptoService.get_available_symbols()
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Error getting crypto symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get data for a specific crypto symbol
@app.get("/api/crypto/data/{symbol}")
async def get_crypto_data(symbol: str, timeframe: str = "1h", lookback_days: int = 15):
    try:
        result = CryptoService.get_symbol_data(symbol, timeframe, lookback_days)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to fetch data"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get performance ranking
@app.get("/api/crypto/ranking")
async def get_crypto_ranking(lookback_days: int = 15):
    try:
        result = CryptoService.get_performance_ranking(lookback_days)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to calculate ranking"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance ranking: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run backtest for a crypto symbol
@app.post("/api/crypto/backtest/{symbol}")
async def run_crypto_backtest(symbol: str, days: int = 15, params: Optional[Dict[str, Any]] = None):
    try:
        result = CryptoService.run_backtest(symbol, days, params)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to run backtest"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Optimize parameters for a crypto symbol
@app.post("/api/crypto/optimize/{symbol}")
async def optimize_crypto_parameters(symbol: str, days: int = 15):
    try:
        result = CryptoService.optimize_parameters(symbol, days)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to optimize parameters"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing parameters for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Simulate portfolio
@app.post("/api/crypto/portfolio/simulate")
async def simulate_crypto_portfolio(symbols: List[str] = None, days: int = 15, initial_capital: float = 100000.0):
    try:
        result = CryptoService.simulate_portfolio(symbols, days, initial_capital)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to simulate portfolio"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error simulating portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws/crypto/{symbol}")
async def websocket_crypto_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket)
    try:
        # Send initial data
        data = CryptoService.get_symbol_data(symbol)
        await manager.send_personal_message(json.dumps(data), websocket)
        
        # Keep connection alive and send updates
        while True:
            # Wait for 15 seconds between updates
            await asyncio.sleep(15)
            
            # Get updated data
            data = CryptoService.get_symbol_data(symbol)
            
            # Send update
            await manager.send_personal_message(json.dumps(data), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)