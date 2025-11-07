"""
FastAPI server for EchoMimic V3 video generation.
Automatically downloads and loads models on startup.
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from api.model_manager import ModelManager
from api.inference_service import InferenceService
from api.config import APIConfig
from api.utils import cleanup_temp_files, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize configuration
config = APIConfig()

# Global model manager
model_manager = None
inference_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events for FastAPI.
    Downloads models if missing and loads them into memory.
    """
    global model_manager, inference_service
    
    logger.info("🚀 Starting EchoMimic V3 FastAPI Server...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager(config)
        
        # Check and download models if needed
        await model_manager.ensure_models_available()
        
        # Load models into memory
        logger.info("🔄 Loading models into GPU/CPU memory...")
        await model_manager.load_models()
        
        # Initialize inference service
        inference_service = InferenceService(model_manager, config)
        
        logger.info("✅ Server ready to accept requests!")
        logger.info(f"📖 API documentation: http://{config.host}:{config.port}/docs")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        logger.error("🔧 Please check:")
        logger.error("  1. Internet connection (for model downloads)")
        logger.error("  2. Disk space (~10 GB required)")
        logger.error("  3. GPU availability (if using CUDA)")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("🔄 Shutting down server...")
        if model_manager:
            await model_manager.cleanup()
        logger.info("✅ Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="EchoMimic V3 API",
    description="Audio-driven portrait video generation API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "EchoMimic V3 API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "generate": "/generate-video",
            "health": "/health",
            "models": "/models/status",
            "limits": "/limits"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with model status."""
    if model_manager is None or not model_manager.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "initializing",
                "ready": False,
                "message": "Models are still loading"
            }
        )
    
    return {
        "status": "healthy",
        "ready": True,
        "models_loaded": model_manager.loaded_models,
        "device": str(model_manager.device),
        "gpu_available": model_manager.gpu_available
    }


@app.get("/models/status")
async def models_status():
    """Get detailed model status."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    return await model_manager.get_status()


@app.post("/generate-video")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Reference portrait image (PNG/JPG)"),
    audio: UploadFile = File(..., description="Audio file (WAV/MP3)"),
    prompt: str = Form(
        default="A person speaking naturally with expressive facial movements and appropriate hand gestures, maintaining eye contact with the camera.",
        description="Text prompt describing the desired video"
    ),
    guidance_scale: float = Form(default=4.0, ge=1.0, le=10.0, description="Guidance scale (1.0-10.0)"),
    audio_guidance_scale: float = Form(default=2.9, ge=1.0, le=5.0, description="Audio guidance scale (1.0-5.0)"),
    fps: int = Form(default=25, ge=15, le=30, description="Frames per second (15-30)"),
    seed: int = Form(default=43, description="Random seed for reproducibility"),
):
    """
    Generate a video from a reference image and audio.
    
    The video length automatically matches the audio duration.
    
    Parameters:
    - image: Reference portrait image (max 10MB)
    - audio: Audio file that drives the video (max 60 seconds)
    - prompt: Text description of desired video style
    - guidance_scale: Control adherence to prompt (default: 4.0)
    - audio_guidance_scale: Control audio-video synchronization (default: 2.9)
    - fps: Video frame rate (default: 25)
    - seed: Random seed for reproducibility (default: 43)
    
    Returns:
    - Generated video file with synchronized audio
    """
    
    if inference_service is None or not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Please wait and try again."
        )
    
    try:
        # Validate file sizes
        image_size = 0
        audio_size = 0
        
        # Read and validate image
        image_content = await image.read()
        image_size = len(image_content)
        
        if image_size > config.max_image_size:
            raise HTTPException(
                status_code=413,
                detail=f"Image file too large. Max size: {config.max_image_size // 1024 // 1024}MB"
            )
        
        # Read and validate audio
        audio_content = await audio.read()
        audio_size = len(audio_content)
        
        if audio_size > config.max_audio_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Max size: {config.max_audio_size // 1024 // 1024}MB"
            )
        
        # Validate file formats
        allowed_image_extensions = ['.png', '.jpg', '.jpeg']
        allowed_audio_extensions = ['.wav', '.mp3']
        
        image_ext = os.path.splitext(image.filename)[1].lower()
        audio_ext = os.path.splitext(audio.filename)[1].lower()
        
        if image_ext not in allowed_image_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Allowed: {allowed_image_extensions}"
            )
        
        if audio_ext not in allowed_audio_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio format. Allowed: {allowed_audio_extensions}"
            )
        
        logger.info(f"📥 Received request - Image: {image.filename}, Audio: {audio.filename}")
        
        # Run inference
        video_path, temp_dir = await inference_service.generate_video(
            image_content=image_content,
            image_filename=image.filename,
            audio_content=audio_content,
            audio_filename=audio.filename,
            prompt=prompt,
            guidance_scale=guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            fps=fps,
            seed=seed,
        )
        
        logger.info(f"✅ Video generated successfully: {video_path}")
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup_temp_files, temp_dir)
        
        # Return video file
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"generated_{os.path.basename(video_path)}",
            background=background_tasks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")


@app.get("/limits")
async def get_limits():
    """Get API limits and constraints."""
    return {
        "max_image_size_mb": config.max_image_size // 1024 // 1024,
        "max_audio_size_mb": config.max_audio_size // 1024 // 1024,
        "max_audio_duration_seconds": config.max_audio_duration,
        "supported_image_formats": [".png", ".jpg", ".jpeg"],
        "supported_audio_formats": [".wav", ".mp3"],
        "fps_range": [15, 30],
        "guidance_scale_range": [1.0, 10.0],
        "audio_guidance_scale_range": [1.0, 5.0],
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
