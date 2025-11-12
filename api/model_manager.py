"""
Model Manager: Handles model downloading, loading, and lifecycle.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional
import torch
import gc
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
    """Manages model downloading, loading, and lifecycle with memory optimization."""
    
    def __init__(self, config):
        self.config = config
        self.is_loaded = False
        self.models = {}
        self.device = None
        self.gpu_available = torch.cuda.is_available()
        self.loaded_models = []
        
        # Memory optimization settings
        self.use_cpu_offload = getattr(config, 'use_cpu_offload', False)
        self.enable_memory_efficient_attention = getattr(config, 'enable_memory_efficient_attention', True)
        self.low_vram_mode = getattr(config, 'low_vram_mode', False)
        
    def _clear_memory(self):
        """Aggressively clear GPU memory."""
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def _log_memory_usage(self, stage: str):
        """Log current GPU memory usage."""
        if self.gpu_available:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"💾 [{stage}] GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        
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
        """Load all models into memory with memory optimization."""
        try:
            # Determine device
            self.device = torch.device("cuda" if self.gpu_available else "cpu")
            weight_dtype = torch.bfloat16 if self.gpu_available else torch.float32
            
            # Configure memory optimization
            if self.gpu_available:
                # Enable memory fragmentation reduction
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                
                # Clear memory before loading
                self._clear_memory()
                self._log_memory_usage("Initial")
                
                # Set conservative memory allocation
                torch.cuda.set_per_process_memory_fraction(0.95, 0)
            
            logger.info(f"🔧 Using device: {self.device}")
            logger.info(f"🔧 Using dtype: {weight_dtype}")
            logger.info(f"🔧 CPU Offload: {self.use_cpu_offload}")
            logger.info(f"🔧 Low VRAM Mode: {self.low_vram_mode}")
            
            # Load config
            cfg = OmegaConf.load("config/config.yaml")
            
            # === Load Transformer ===
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
                logger.info("   Loading transformer checkpoint...")
                from safetensors.torch import load_file
                state_dict = load_file(transformer_path)
                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
                missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
                logger.info(f"   Transformer loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # Enable gradient checkpointing to save memory
            if hasattr(transformer, 'enable_gradient_checkpointing'):
                transformer.enable_gradient_checkpointing()
                logger.info("   ✓ Gradient checkpointing enabled")
            
            # Keep transformer on CPU if using offload mode
            if not self.use_cpu_offload:
                transformer = transformer.to(device=self.device)
            
            logger.info("   Transformer ready!")
            self.loaded_models.append("transformer")
            self._clear_memory()
            self._log_memory_usage("After Transformer")
            
            # === Load VAE ===
            logger.info("📦 Loading VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['vae_kwargs'].get('vae_subpath', 'vae')),
                additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
            ).to(weight_dtype)
            
            # Enable slicing and tiling for VAE to reduce memory
            if hasattr(vae, 'enable_slicing'):
                vae.enable_slicing()
                logger.info("   ✓ VAE slicing enabled")
            if hasattr(vae, 'enable_tiling'):
                vae.enable_tiling()
                logger.info("   ✓ VAE tiling enabled")
            
            # Keep VAE on CPU if using offload mode  
            if not self.use_cpu_offload:
                vae = vae.to(device=self.device)
                
            self.loaded_models.append("vae")
            self._clear_memory()
            self._log_memory_usage("After VAE")
            
            # === Load Tokenizer (CPU only, minimal memory) ===
            logger.info("📦 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
            )
            self.loaded_models.append("tokenizer")
            
            # === Load Text Encoder (HIGH MEMORY - use optimization) ===
            logger.info("📦 Loading text encoder...")
            self._clear_memory()  # Clear before loading heavy model
            
            # Load with lower precision if in low VRAM mode
            text_encoder_dtype = torch.float16 if self.low_vram_mode else weight_dtype
            
            text_encoder = WanT5EncoderModel.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
                additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
                torch_dtype=text_encoder_dtype,
            )
            
            # Enable gradient checkpointing
            if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                text_encoder.enable_gradient_checkpointing()
                logger.info("   ✓ Text encoder gradient checkpointing enabled")
            
            # Move to device in eval mode
            logger.info("   Moving text encoder to device...")
            if not self.use_cpu_offload:
                text_encoder = text_encoder.to(device=self.device)
            text_encoder.eval()
            
            # Disable gradients to save memory
            for param in text_encoder.parameters():
                param.requires_grad = False
                
            logger.info("   Text encoder loaded successfully!")
            self.loaded_models.append("text_encoder")
            self._clear_memory()
            self._log_memory_usage("After Text Encoder")
            
            # === Load Image Encoder ===
            logger.info("📦 Loading image encoder...")
            self._clear_memory()
            
            clip_image_encoder = CLIPModel.from_pretrained(
                os.path.join("models/Wan2.1-Fun-V1.1-1.3B-InP", 
                           cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
            )
            
            logger.info("   Moving image encoder to device...")
            if not self.use_cpu_offload:
                clip_image_encoder = clip_image_encoder.to(device=self.device, dtype=weight_dtype)
            else:
                clip_image_encoder = clip_image_encoder.to(dtype=weight_dtype)
            clip_image_encoder.eval()
            
            # Disable gradients
            for param in clip_image_encoder.parameters():
                param.requires_grad = False
                
            logger.info("   Image encoder loaded successfully!")
            self.loaded_models.append("clip_image_encoder")
            self._clear_memory()
            self._log_memory_usage("After Image Encoder")
            
            # === Load Scheduler (CPU only, minimal memory) ===
            logger.info("📦 Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(cfg['scheduler_kwargs']))
            )
            self.loaded_models.append("scheduler")
            
            # === Create Pipeline ===
            logger.info("📦 Creating pipeline...")
            pipeline = WanFunInpaintAudioPipeline(
                transformer=transformer,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                scheduler=scheduler,
                clip_image_encoder=clip_image_encoder,
            )
            
            # Move pipeline to device if not using offload
            if not self.use_cpu_offload:
                pipeline.to(device=self.device)
            else:
                # Enable model CPU offload for memory efficiency
                if hasattr(pipeline, 'enable_model_cpu_offload'):
                    pipeline.enable_model_cpu_offload()
                    logger.info("   ✓ Model CPU offload enabled")
                elif hasattr(pipeline, 'enable_sequential_cpu_offload'):
                    pipeline.enable_sequential_cpu_offload()
                    logger.info("   ✓ Sequential CPU offload enabled")
            
            # Enable memory efficient attention if available
            if self.enable_memory_efficient_attention:
                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("   ✓ xFormers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Could not enable xFormers: {e}")
                elif hasattr(pipeline, 'enable_attention_slicing'):
                    pipeline.enable_attention_slicing(1)
                    logger.info("   ✓ Attention slicing enabled")
                    
            self.loaded_models.append("pipeline")
            self._clear_memory()
            self._log_memory_usage("After Pipeline")
            
            # === Load Wav2Vec2 ===
            logger.info("📦 Loading Wav2Vec2...")
            self._clear_memory()
            
            wav2vec_processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base-960h")
            logger.info("   Loading Wav2Vec2 model...")
            wav2vec_model = Wav2Vec2Model.from_pretrained("models/wav2vec2-base-960h")
            
            if not self.use_cpu_offload:
                wav2vec_model = wav2vec_model.to(device=self.device)
            wav2vec_model.eval()
            wav2vec_model.requires_grad_(False)
            logger.info("   Wav2Vec2 loaded successfully!")
            self.loaded_models.append("wav2vec2")
            self._clear_memory()
            self._log_memory_usage("After Wav2Vec2")
            
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
            self._log_memory_usage("Final")
            logger.info("✅ All models loaded successfully into memory!")
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"❌ CUDA Out of Memory Error: {e}")
            logger.error("💡 Suggestions to fix:")
            logger.error("   1. Set use_cpu_offload=True in config")
            logger.error("   2. Set low_vram_mode=True in config")
            logger.error("   3. Close other GPU applications")
            logger.error("   4. Use a GPU with more VRAM")
            logger.error("   5. Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            
            # Cleanup on OOM
            await self.cleanup()
            raise
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}", exc_info=True)
            await self.cleanup()
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
        
        # Delete model references
        if self.models:
            for key in list(self.models.keys()):
                self.models[key] = None
            self.models.clear()
        
        # Force garbage collection
        self._clear_memory()
        
        self.is_loaded = False
        self.loaded_models.clear()
        logger.info("✅ Cleanup complete")
