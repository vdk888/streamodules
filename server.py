"""
Main server file to run the FastAPI application
"""
import os
import logging
import uvicorn
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI server"""
    parser = argparse.ArgumentParser(description='Start the trading platform server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import the app here to avoid circular imports
    from trading_platform.backend.app import app
    
    # Log startup
    logger.info(f"Starting server at http://{args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "trading_platform.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.debug else "debug"
    )

if __name__ == "__main__":
    main()