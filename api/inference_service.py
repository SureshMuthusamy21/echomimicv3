"""
Inference Service: Handles video generation logic.
"""

import os
import uuid
import logging
import math
import asyncio
from typing import Tuple
from PIL import Image
import torch
import numpy as np
import librosa
from moviepy import VideoFileClip, AudioFileClip

from src.utils import get_image_to_video_latent3, save_videos_grid
from src.face_detect import get_mask_coord

logger = logging.getLogger(__name__)


class InferenceService:
    """Handles video generation inference."""
    
    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
    
    async def generate_video(
        self,
        image_content: bytes,
        image_filename: str,
        audio_content: bytes,
        audio_filename: str,
        prompt: str,
        guidance_scale: float,
        audio_guidance_scale: float,
        fps: int,
        seed: int,
    ) -> Tuple[str, str]:
        """
        Generate video from image and audio.
        Returns: (video_path, temp_directory)
        """
        
        # Create temporary directory for this request
        request_id = str(uuid.uuid4())
        temp_dir = os.path.join(self.config.temp_dir, request_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save uploaded files
            image_path = os.path.join(temp_dir, f"input{os.path.splitext(image_filename)[1]}")
            audio_path = os.path.join(temp_dir, f"input{os.path.splitext(audio_filename)[1]}")
            
            with open(image_path, "wb") as f:
                f.write(image_content)
            
            with open(audio_path, "wb") as f:
                f.write(audio_content)
            
            # Convert MP3 to WAV if needed
            if audio_path.endswith('.mp3'):
                wav_path = os.path.join(temp_dir, "input.wav")
                await asyncio.to_thread(self._convert_to_wav, audio_path, wav_path)
                audio_path = wav_path
            
            # Run inference in thread pool to avoid blocking
            video_path = await asyncio.to_thread(
                self._run_inference,
                image_path,
                audio_path,
                prompt,
                guidance_scale,
                audio_guidance_scale,
                fps,
                seed,
                temp_dir
            )
            
            return video_path, temp_dir
            
        except Exception as e:
            # Cleanup on error
            logger.error(f"Error in inference: {e}", exc_info=True)
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    
    def _convert_to_wav(self, input_path: str, output_path: str):
        """Convert audio file to WAV format."""
        audio_clip = AudioFileClip(input_path)
        audio_clip.write_audiofile(output_path, fps=16000, codec='pcm_s16le', logger=None)
        audio_clip.close()
    
    def _run_inference(
        self,
        image_path: str,
        audio_path: str,
        prompt: str,
        guidance_scale: float,
        audio_guidance_scale: float,
        fps: int,
        seed: int,
        temp_dir: str
    ) -> str:
        """Run the actual inference (blocking, called in thread)."""
        
        models = self.model_manager.models
        pipeline = models["pipeline"]
        vae = models["vae"]
        wav2vec_processor = models["wav2vec_processor"]
        wav2vec_model = models["wav2vec_model"]
        device = models["device"]
        weight_dtype = models["weight_dtype"]
        
        # Load and validate image
        ref_img = Image.open(image_path).convert("RGB")
        
        # Detect face and get mask
        logger.info("🔍 Detecting face in image...")
        try:
            y1, y2, x1, x2, h_, w_ = get_mask_coord(image_path)
        except Exception as e:
            raise ValueError(f"No face detected in image: {e}")
        
        # Extract audio features
        logger.info("🎵 Extracting audio features...")
        audio_features = self._extract_audio_features(audio_path, wav2vec_processor, wav2vec_model)
        audio_embeds = audio_features.unsqueeze(0).to(device=device, dtype=weight_dtype)
        
        # Get audio duration and calculate video length
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        
        if audio_duration > self.config.max_audio_duration:
            raise ValueError(
                f"Audio duration ({audio_duration:.1f}s) exceeds maximum allowed "
                f"({self.config.max_audio_duration}s)"
            )
        
        logger.info(f"🎬 Audio duration: {audio_duration:.2f}s, generating {audio_duration * fps:.0f} frames...")
        
        video_length = int(audio_duration * fps)
        video_length = (
            int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            if video_length != 1 else 1
        )
        
        # Adjust sample size
        sample_height, sample_width = self._get_sample_size(ref_img, self.config.sample_size)
        
        # Create IP mask
        downratio = math.sqrt(sample_height * sample_width / h_ / w_)
        coords = (
            int(y1 * downratio // 16), int(y2 * downratio // 16),
            int(x1 * downratio // 16), int(x2 * downratio // 16),
            sample_height // 16, sample_width // 16,
        )
        ip_mask = self._get_ip_mask(coords).unsqueeze(0)
        ip_mask = torch.cat([ip_mask] * 3).to(device=device, dtype=weight_dtype)
        
        # Setup generator
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate video
        logger.info("🎬 Generating video...")
        
        partial_video_length = int(
            (self.config.partial_video_length - 1) // vae.config.temporal_compression_ratio 
            * vae.config.temporal_compression_ratio
        ) + 1 if video_length != 1 else 1
        
        # Get clip image
        _, _, clip_image = get_image_to_video_latent3(
            ref_img, None, 
            video_length=partial_video_length, 
            sample_size=[sample_height, sample_width]
        )
        
        # Generate in chunks
        init_frames = 0
        last_frames = init_frames + partial_video_length
        new_sample = None
        mix_ratio = torch.linspace(0, 1, steps=self.config.overlap_video_length).view(1, 1, -1, 1, 1)
        
        negative_prompt = self.config.negative_prompt
        
        while init_frames < video_length:
            if last_frames >= video_length:
                partial_video_length = video_length - init_frames
                partial_video_length = (
                    int((partial_video_length - 1) // vae.config.temporal_compression_ratio 
                        * vae.config.temporal_compression_ratio) + 1
                    if video_length != 1 else 1
                )
                
                if partial_video_length <= 0:
                    break
            
            input_video, input_video_mask, _ = get_image_to_video_latent3(
                ref_img, None, 
                video_length=partial_video_length, 
                sample_size=[sample_height, sample_width]
            )
            
            partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + partial_video_length) * 2]
            
            logger.info(f"  Processing frames {init_frames} to {init_frames + partial_video_length}...")
            
            sample = pipeline(
                prompt,
                num_frames=partial_video_length,
                negative_prompt=negative_prompt,
                audio_embeds=partial_audio_embeds,
                audio_scale=self.config.audio_scale,
                ip_mask=ip_mask,
                use_un_ip_mask=False,
                height=sample_height,
                width=sample_width,
                generator=generator,
                neg_scale=self.config.neg_scale,
                neg_steps=self.config.neg_steps,
                use_dynamic_cfg=True,
                use_dynamic_acfg=True,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_image=clip_image,
                cfg_skip_ratio=0,
                shift=5.0,
                use_longvideo_cfg=False,
                overlap_video_length=self.config.overlap_video_length,
                partial_video_length=partial_video_length,
            ).videos
            
            # Blend overlapping frames
            if init_frames != 0:
                new_sample[:, :, -self.config.overlap_video_length:] = (
                    new_sample[:, :, -self.config.overlap_video_length:] * (1 - mix_ratio) +
                    sample[:, :, :self.config.overlap_video_length] * mix_ratio
                )
                new_sample = torch.cat([new_sample, sample[:, :, self.config.overlap_video_length:]], dim=2)
                sample = new_sample
            else:
                new_sample = sample
            
            if last_frames >= video_length:
                break
            
            ref_img = [
                Image.fromarray(
                    (sample[0, :, i].transpose(0, 1).transpose(1, 2) * 255).cpu().numpy().astype(np.uint8)
                ) for i in range(-self.config.overlap_video_length, 0)
            ]
            
            init_frames += partial_video_length - self.config.overlap_video_length
            last_frames = init_frames + partial_video_length
        
        # Save video
        logger.info("💾 Saving video...")
        video_path = os.path.join(temp_dir, "generated.mp4")
        video_audio_path = os.path.join(temp_dir, "output.mp4")
        
        save_videos_grid(sample[:, :, :video_length], video_path, fps=fps)
        
        # Add audio to video
        logger.info("🎵 Adding audio to video...")
        video_clip = VideoFileClip(video_path)
        audio_clip_trimmed = audio_clip.subclipped(0, video_length / fps)
        video_clip = video_clip.with_audio(audio_clip_trimmed)
        video_clip.write_videofile(video_audio_path, codec="libx264", audio_codec="aac", threads=2, logger=None)
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        os.remove(video_path)
        
        logger.info("✅ Video generation complete!")
        
        return video_audio_path
    
    def _extract_audio_features(self, audio_path: str, processor, model):
        """Extract audio features using Wav2Vec."""
        sr = 16000
        audio_segment, sample_rate = librosa.load(audio_path, sr=sr)
        input_values = processor(audio_segment, sampling_rate=sample_rate, return_tensors="pt").input_values
        with torch.no_grad():
            features = model(input_values).last_hidden_state
        return features.squeeze(0)
    
    def _get_sample_size(self, image, default_size):
        """Calculate the sample size based on the input image dimensions."""
        width, height = image.size
        original_area = width * height
        default_area = default_size[0] * default_size[1]
        
        if default_area < original_area:
            ratio = math.sqrt(original_area / default_area)
            width = width / ratio // 16 * 16
            height = height / ratio // 16 * 16
        else:
            width = width // 16 * 16
            height = height // 16 * 16
        
        return int(height), int(width)
    
    def _get_ip_mask(self, coords):
        """Create IP mask from coordinates."""
        y1, y2, x1, x2, h, w = coords
        Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
        mask = mask.reshape(-1)
        return mask.float()
