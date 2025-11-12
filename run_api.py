"""
Main entry point for EchoMimic V3 FastAPI Server
Run this file to start the API server

Usage:
    python run_api.py
"""

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting EchoMimic V3 API Server...")
    print("📡 Server will run on http://0.0.0.0:8000")
    print("📖 API docs will be available at http://localhost:8000/docs")
    print("\n⏳ Please wait while models are being checked and loaded...\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
