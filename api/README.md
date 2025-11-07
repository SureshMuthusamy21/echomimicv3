# EchoMimic V3 FastAPI Server

## Features

- ✅ **Auto-downloads models** on first startup
- ✅ **GPU acceleration** (CUDA) with CPU fallback
- ✅ **Audio-driven video generation** matching audio length
- ✅ **RESTful API** with FastAPI
- ✅ **Automatic cleanup** of temporary files
- ✅ **Health checks** and model status endpoints

## Quick Start

### 1. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install API requirements
pip install -r api/requirements.txt
```

### 2. Start Server

```bash
# Windows PowerShell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using Python directly
cd echomimic_v3
python -m api.main
```

The server will automatically:
- Check for required models
- Download missing models (~10 GB, takes 10-30 min)
- Load models into GPU memory
- Start accepting requests

### 3. Test the API

```powershell
# Health check
curl http://localhost:8000/health

# Generate video (PowerShell)
curl -X POST "http://localhost:8000/generate-video" `
  -F "image=@path/to/portrait.jpg" `
  -F "audio=@path/to/speech.wav" `
  -F "prompt=A person speaking naturally" `
  -F "guidance_scale=4.0" `
  -F "audio_guidance_scale=2.9" `
  -o generated_video.mp4
```

Or use Python to test:

```python
import requests

url = "http://localhost:8000/generate-video"

files = {
    'image': open('datasets/echomimicv3_demos/imgs/demo_ch_man_01.png', 'rb'),
    'audio': open('datasets/echomimicv3_demos/audios/test_audio.wav', 'rb')
}

data = {
    'prompt': 'A person speaking naturally with expressive facial movements',
    'guidance_scale': 4.0,
    'audio_guidance_scale': 2.9,
    'fps': 25,
    'seed': 43
}

response = requests.post(url, files=files, data=data)

with open('output.mp4', 'wb') as f:
    f.write(response.content)

print("Video saved as output.mp4")
```

## API Endpoints

### `POST /generate-video`

Generate video from image and audio.

**Parameters:**
- `image` (file): Portrait image (PNG/JPG, max 10MB)
- `audio` (file): Audio file (WAV/MP3, max 60s)
- `prompt` (string, optional): Description of desired video
- `guidance_scale` (float, optional): 1.0-10.0, default 4.0
- `audio_guidance_scale` (float, optional): 1.0-5.0, default 2.9
- `fps` (int, optional): 15-30, default 25
- `seed` (int, optional): Random seed, default 43

**Returns:** MP4 video file

### `GET /health`

Check server health and model status.

### `GET /models/status`

Get detailed model loading status.

### `GET /limits`

Get API limits and constraints.

## Configuration

Environment variables:

```powershell
$env:API_HOST="0.0.0.0"          # Server host
$env:API_PORT="8000"             # Server port
$env:MAX_IMAGE_SIZE="10485760"   # Max image size (bytes)
$env:MAX_AUDIO_SIZE="52428800"   # Max audio size (bytes)
$env:MAX_AUDIO_DURATION="60"     # Max audio duration (seconds)
$env:TEMP_DIR="temp"             # Temporary files directory
```

## Performance

- **First startup**: 10-30 minutes (model download)
- **Subsequent startups**: 30-60 seconds (model loading)
- **Video generation**: 
  - 5s audio: ~30-60 seconds
  - 30s audio: ~3-5 minutes
  - (RTX 3090/4090 recommended)

## Troubleshooting

**Models not downloading:**
- Check internet connection
- Ensure ~10 GB free disk space

**Out of memory:**
- Reduce `max_audio_duration`
- Use smaller `sample_size`
- Enable CPU mode (slower)

**Slow inference:**
- Ensure CUDA is available
- Check GPU utilization

## Example Client Code

```python
import requests
from pathlib import Path

def generate_video(image_path, audio_path, output_path="output.mp4"):
    """Generate video using the API."""
    
    url = "http://localhost:8000/generate-video"
    
    with open(image_path, 'rb') as img, open(audio_path, 'rb') as aud:
        files = {
            'image': img,
            'audio': aud
        }
        
        data = {
            'prompt': 'A person speaking naturally with expressive facial movements',
            'guidance_scale': 4.0,
            'audio_guidance_scale': 2.9,
            'fps': 25,
            'seed': 43
        }
        
        print("🚀 Sending request to API...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Video saved as {output_path}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.json())

# Usage
generate_video(
    "datasets/echomimicv3_demos/imgs/demo_ch_man_01.png",
    "datasets/echomimicv3_demos/audios/test_audio.wav"
)
```

## Architecture

```
Server Startup
  ↓
Check Models (5 sec)
  ↓
Download if Missing (0-30 min)
  ↓
Load into Memory (30-60 sec)
  ↓
Store in app.state
  ↓
Server Ready ✅
  ↓
All requests use cached models (fast!)
```

## File Structure

```
api/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application entry point
├── model_manager.py     # Model downloading and loading
├── inference_service.py # Video generation logic
├── config.py            # Configuration management
├── utils.py             # Helper functions
├── requirements.txt     # API dependencies
└── README.md            # This file
```
