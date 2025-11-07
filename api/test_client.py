"""
Test client for EchoMimic V3 API
"""

import requests
import time
from pathlib import Path


class EchoMimicClient:
    """Client for interacting with EchoMimic V3 API."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if the server is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e), "ready": False}
    
    def get_limits(self):
        """Get API limits."""
        response = requests.get(f"{self.base_url}/limits")
        return response.json()
    
    def get_model_status(self):
        """Get model loading status."""
        response = requests.get(f"{self.base_url}/models/status")
        return response.json()
    
    def wait_for_ready(self, timeout=600, check_interval=5):
        """Wait for the server to be ready."""
        print("⏳ Waiting for server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            health = self.health_check()
            
            if health.get("ready"):
                print("✅ Server is ready!")
                return True
            
            status = health.get("status", "unknown")
            print(f"   Status: {status} (waiting {int(time.time() - start_time)}s / {timeout}s)")
            time.sleep(check_interval)
        
        print("❌ Timeout waiting for server")
        return False
    
    def generate_video(
        self,
        image_path,
        audio_path,
        output_path="output.mp4",
        prompt="A person speaking naturally with expressive facial movements and appropriate hand gestures, maintaining eye contact with the camera.",
        guidance_scale=4.0,
        audio_guidance_scale=2.9,
        fps=25,
        seed=43,
    ):
        """
        Generate video from image and audio.
        
        Args:
            image_path: Path to input image
            audio_path: Path to input audio
            output_path: Path to save output video
            prompt: Text prompt
            guidance_scale: Guidance scale (1.0-10.0)
            audio_guidance_scale: Audio guidance scale (1.0-5.0)
            fps: Frames per second (15-30)
            seed: Random seed
        
        Returns:
            True if successful, False otherwise
        """
        
        # Check if files exist
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return False
        
        if not Path(audio_path).exists():
            print(f"❌ Audio not found: {audio_path}")
            return False
        
        print(f"📤 Uploading files...")
        print(f"   Image: {image_path}")
        print(f"   Audio: {audio_path}")
        
        try:
            with open(image_path, 'rb') as img_file, open(audio_path, 'rb') as aud_file:
                files = {
                    'image': (Path(image_path).name, img_file),
                    'audio': (Path(audio_path).name, aud_file),
                }
                
                data = {
                    'prompt': prompt,
                    'guidance_scale': guidance_scale,
                    'audio_guidance_scale': audio_guidance_scale,
                    'fps': fps,
                    'seed': seed,
                }
                
                print(f"🎬 Generating video...")
                print(f"   Prompt: {prompt}")
                print(f"   Guidance Scale: {guidance_scale}")
                print(f"   Audio Guidance Scale: {audio_guidance_scale}")
                print(f"   FPS: {fps}")
                
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/generate-video",
                    files=files,
                    data=data,
                    timeout=600  # 10 minute timeout
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    # Save video
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"✅ Video generated successfully!")
                    print(f"   Output: {output_path}")
                    print(f"   Time: {elapsed_time:.1f} seconds")
                    return True
                else:
                    print(f"❌ Error: {response.status_code}")
                    print(f"   Message: {response.text}")
                    return False
                    
        except requests.exceptions.Timeout:
            print("❌ Request timed out (>10 minutes)")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    """Example usage of the client."""
    
    # Initialize client
    client = EchoMimicClient("http://localhost:8000")
    
    # Check server health
    print("🔍 Checking server health...")
    health = client.health_check()
    print(f"   Status: {health}")
    
    if not health.get("ready"):
        print("\n⚠️  Server not ready yet. Waiting...")
        if not client.wait_for_ready():
            print("❌ Server failed to start. Exiting.")
            return
    
    # Get limits
    print("\n📊 API Limits:")
    limits = client.get_limits()
    for key, value in limits.items():
        print(f"   {key}: {value}")
    
    # Get model status
    print("\n🤖 Model Status:")
    status = client.get_model_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Example: Generate video
    print("\n" + "="*50)
    print("🎬 Generating example video...")
    print("="*50)
    
    # Update these paths to your actual test files
    image_path = "datasets/echomimicv3_demos/imgs/demo_ch_man_01.png"
    audio_path = "datasets/echomimicv3_demos/audios/test_audio.wav"
    
    # Check if example files exist
    if not Path(image_path).exists() or not Path(audio_path).exists():
        print(f"\n⚠️  Example files not found. Please update paths in test_client.py")
        print(f"   Expected image: {image_path}")
        print(f"   Expected audio: {audio_path}")
        return
    
    success = client.generate_video(
        image_path=image_path,
        audio_path=audio_path,
        output_path="test_output.mp4",
        prompt="A person speaking naturally with expressive facial movements",
        guidance_scale=4.0,
        audio_guidance_scale=2.9,
        fps=25,
        seed=43,
    )
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed")


if __name__ == "__main__":
    main()
