"""
Main server file to run the FastAPI application
"""
import os
import uvicorn
import argparse
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Run the FastAPI server"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the trading platform API server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    args = parser.parse_args()
    
    # Log startup information
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "trading_platform.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()