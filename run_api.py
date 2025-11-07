"""
Main entry point for EchoMimic V3 FastAPI Server
Run this file to start the API server
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from api.config import APIConfig

if __name__ == "__main__":
    config = APIConfig()
    
    print("🚀 Starting EchoMimic V3 API Server...")
    print(f"📡 Server will run on http://{config.host}:{config.port}")
    print(f"📖 API docs will be available at http://{config.host}:{config.port}/docs")
    print("\n⏳ Please wait while models are being checked and loaded...\n")
    
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
