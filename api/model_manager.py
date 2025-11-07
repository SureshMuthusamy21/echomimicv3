"""
Model Manager: Handles model downloading, loading, and lifecycle.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from diffusers import FlowMatchEulerDiscreteScheduler

from download import safe_download
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import CLIPModel
from src.wan_text_encoder import WanT5EncoderModel
from src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
from src.utils import filter_kwargs

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloading, loading, and lifecycle."""
    
    def __init__(self, config):
        self.config = config
        self.is_loaded = False
        self.models = {}
        self.device = None
        self.gpu_available = torch.cuda.is_available()
        self.loaded_models = []
        
    async def ensure_models_available(self):
        """Check if all required models exist, download if missing."""
        logger.info("🔍 Checking model availability...")
        
        missing = self._check_required_models()
        
        if missing:
            logger.warning(f"⚠️  Missing models: {missing}")
            logger.info("📥 Downloading missing models (this may take 10-30 minutes)...")
            await self._download_missing_models(missing)
        else:
            logger.info("✅ All models found locally")
    
    def _check_required_models(self) -> List[str]:
        """
        Check which model components are missing.
        Returns list of missing model names.
        """
        required_paths = {
            "transformer": [
                "models/transformer/diffusion_pytorch_model.safetensors",
                "models/transformer/config.json"
            ],
            "vae": [
                "models/Wan2.1-Fun-V1.1-1.3B-InP/vae/config.json",
                "models/Wan2.1-Fun-V1.1-1.3B-InP/vae/diffusion_pytorch_model.safetensors"
            ],
            "text_encoder": [
                "models/Wan2.1-Fun-V1.1-1.3B-InP/text_encoder/config.json",
            ],
            "image_encoder": [
                "models/Wan2.1-Fun-V1.1-1.3B-InP/image_encoder/config.json",
            ],
            "tokenizer": [
                "models/Wan2.1-Fun-V1.1-1.3B-InP/tokenizer/tokenizer_config.json",
            ],
            "wav2vec2": [
                "models/wav2vec2-base-960h/config.json",
                "models/wav2vec2-base-960h/pytorch_model.bin"
            ],
            "config": [
                "config/config.yaml"
            ]
        }
        
        missing = []
        for model_name, paths in required_paths.items():
            for path in paths:
                if not os.path.exists(path):
                    logger.debug(f"❌ Missing: {path}")
                    missing.append(model_name)
                    break  # Only need to know component is missing
        
        return list(set(missing))  # Remove duplicates
    
    async def _download_missing_models(self, missing: List[str]):
        """Download only the missing model components."""
        
        download_map = {
            "transformer": ("antgroup/EchoMimicV3-preview", "./models/transformer"),
            "vae": ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "./models/Wan2.1-Fun-V1.1-1.3B-InP"),
            "text_encoder": ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "./models/Wan2.1-Fun-V1.1-1.3B-InP"),
            "image_encoder": ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "./models/Wan2.1-Fun-V1.1-1.3B-InP"),
            "tokenizer": ("alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", "./models/Wan2.1-Fun-V1.1-1.3B-InP"),
            "wav2vec2": ("facebook/wav2vec2-base-960h", "./models/wav2vec2-base-960h"),
        }
        
        # Get unique repos to download
        unique_repos = {}
        for model in missing:
            if model == "config":
                raise FileNotFoundError(
                    "config/config.yaml not found. Please ensure it exists in the repository."
                )
            if model in download_map:
                repo, path = download_map[model]
                unique_repos[repo] = path
        
        # Download each unique repository
        for repo_id, local_dir in unique_repos.items():
            logger.info(f"📥 Downloading {repo_id}...")
            # Use asyncio.to_thread to avoid blocking
            try:
                await asyncio.to_thread(safe_download, repo_id, local_dir)
            except Exception as e:
                # Try fallback for transformer
                if repo_id == "antgroup/EchoMimicV3-preview":
                    logger.warning(f"⚠️  Failed to download from {repo_id}, trying fallback...")
                    await asyncio.to_thread(safe_download, "BadToBest/EchoMimicV3", local_dir)
                else:
                    raise
        
        logger.info("✅ All models downloaded successfully!")
    
    async def load_models(self):
        """Load all models into memory."""
        try:
            # Determine device
            self.device = torch.device("cuda" if self.gpu_available else "cpu")
            weight_dtype = torch.bfloat16 if self.gpu_available else torch.float32
            
            logger.info(f"🔧 Using device: {self.device}")
            logger.info(f"🔧 Using dtype: {weight_dtype}")
            
            # Load config
            cfg = OmegaConf.load("config/config.yaml")
            
            # Load Transformer
            logger.info("📦 Loading transformer...")
            transformer = WanTransformerAudioMask3DModel.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
                transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
                torch_dtype=weight_dtype,
            )
            
            # Load transformer checkpoint
            transformer_path = "models/transformer/diffusion_pytorch_model.safetensors"
            if os.path.exists(transformer_path):
                from safetensors.torch import load_file
                state_dict = load_file(transformer_path)
                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
                missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
                logger.info(f"Transformer loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            self.loaded_models.append("transformer")
            
            # Load VAE
            logger.info("📦 Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
            ).to(weight_dtype)
            self.loaded_models.append("vae")
            
            # Load Tokenizer
            logger.info("📦 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )
            self.loaded_models.append("tokenizer")
            
            # Load Text Encoder
            logger.info("📦 Loading text encoder...")
            text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
                torch_dtype=weight_dtype,
            ).eval()
            self.loaded_models.append("text_encoder")
            
            # Load Image Encoder
            logger.info("📦 Loading image encoder...")
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            ).to(weight_dtype).eval()
            self.loaded_models.append("clip_image_encoder")
            
            # Load Scheduler
            logger.info("📦 Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg['scheduler_kwargs']))
            )
            self.loaded_models.append("scheduler")
            
            # Create Pipeline
            logger.info("📦 Creating pipeline...")
            pipeline = WanFunInpaintAudioPipeline(
                transformer=transformer,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
            pipeline.to(device=self.device)
            self.loaded_models.append("pipeline")
            
            # Load Wav2Vec2
            logger.info("📦 Loading Wav2Vec2...")
            wav2vec_processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base-960h")
            wav2vec_model = Wav2Vec2Model.from_pretrained("models/wav2vec2-base-960h").eval()
            wav2vec_model.requires_grad_(False)
            self.loaded_models.append("wav2vec2")
            
            # Store all models
            self.models = {
                "pipeline": pipeline,
                "vae": vae,
                "transformer": transformer,
                "wav2vec_processor": wav2vec_processor,
                "wav2vec_model": wav2vec_model,
                "device": self.device,
                "weight_dtype": weight_dtype,
                "config": cfg,
            }
            
            self.is_loaded = True
            logger.info("✅ All models loaded successfully into memory!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}", exc_info=True)
            raise
    
    async def get_status(self) -> Dict:
        """Get detailed model status."""
        return {
            "loaded": self.is_loaded,
            "device": str(self.device) if self.device else None,
            "gpu_available": self.gpu_available,
            "loaded_components": self.loaded_models,
            "missing_components": self._check_required_models() if not self.is_loaded else [],
        }
    
    async def cleanup(self):
        """Cleanup models and free memory."""
        logger.info("🧹 Cleaning up models...")
        if self.models:
            for key in list(self.models.keys()):
                self.models[key] = None
            self.models.clear()
        
        if self.gpu_available:
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("✅ Cleanup complete")
