"""
API Configuration
"""

import os


class APIConfig:
    """API Configuration settings."""
    
    def __init__(self):
        # Server settings
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.port = int(os.getenv("API_PORT", "8000"))
        self.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # File limits
        self.max_image_size = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))  # 10MB
        self.max_audio_size = int(os.getenv("MAX_AUDIO_SIZE", 50 * 1024 * 1024))  # 50MB
        self.max_audio_duration = int(os.getenv("MAX_AUDIO_DURATION", 60))  # 60 seconds
        
        # Temp directory
        self.temp_dir = os.getenv("TEMP_DIR", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Inference settings
        self.sample_size = [768, 768]
        self.partial_video_length = 113
        self.overlap_video_length = 8
        self.num_inference_steps = 25
        self.audio_scale = 1.0
        self.neg_scale = 1.5
        self.neg_steps = 2
        
        # Default prompt
        self.negative_prompt = (
            "Gesture is bad. Gesture is unclear. Strange and twisted hands. "
            "Bad hands. Bad fingers. Unclear and blurry hands. "
            "手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作."
        )
