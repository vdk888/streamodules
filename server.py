"""
Main server file to run the FastAPI application
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Run the FastAPI server"""
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the server
    uvicorn.run(
        "trading_platform.backend.app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

if __name__ == "__main__":
    main()