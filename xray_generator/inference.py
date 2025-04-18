# xray_generator/inference.py
import os
import torch
import numpy as np
from PIL import Image
import logging
from typing import Union, List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from pathlib import Path

from .models.diffusion import DiffusionModel
from .utils.processing import get_device, apply_clahe

logger = logging.getLogger(__name__)

class XrayGenerator:
    """
    Wrapper class for chest X-ray generation from text prompts.
    """
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1",
    ):
        """
        Initialize the X-ray generator.
        
        Args:
            model_path: Path to the saved model weights
            device: Device to run the model on (defaults to CUDA if available)
            tokenizer_name: Name of the HuggingFace tokenizer
        """
        self.device = device if device is not None else get_device()
        self.model_path = Path(model_path)
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        # Load model
        self.model = self._load_model()
        
        # Set model to evaluation mode
        self.model.vae.eval()
        self.model.text_encoder.eval()
        self.model.unet.eval()
        
        logger.info("XrayGenerator initialized successfully")
    
    def _load_model(self) -> DiffusionModel:
        """Load the diffusion model from saved weights."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Import model components here to avoid circular imports
            from .models.vae import MedicalVAE
            from .models.text_encoder import MedicalTextEncoder
            from .models.unet import DiffusionUNet
            
            # Get model configuration
            config = checkpoint.get('config', {})
            latent_channels = config.get('latent_channels', 8) 
            model_channels = config.get('model_channels', 48)
            
            # Initialize model components
            vae = MedicalVAE(
                in_channels=1,
                out_channels=1, 
                latent_channels=latent_channels,
                hidden_dims=[model_channels, model_channels*2, model_channels*4, model_channels*8]
            ).to(self.device)
            
            text_encoder = MedicalTextEncoder(
                model_name=config.get('text_model', "dmis-lab/biobert-base-cased-v1.1"),
                projection_dim=768,
                freeze_base=True
            ).to(self.device)
            
            unet = DiffusionUNet(
                in_channels=latent_channels,
                model_channels=model_channels,
                out_channels=latent_channels,
                num_res_blocks=2,
                attention_resolutions=(8, 16, 32),
                dropout=0.1,
                channel_mult=(1, 2, 4, 8),
                context_dim=768
            ).to(self.device)
            
            # Load state dictionaries
            if 'vae_state_dict' in checkpoint:
                vae.load_state_dict(checkpoint['vae_state_dict'])
                logger.info("Loaded VAE weights")
            
            if 'text_encoder_state_dict' in checkpoint:
                text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
                logger.info("Loaded text encoder weights")
                
            if 'unet_state_dict' in checkpoint:
                unet.load_state_dict(checkpoint['unet_state_dict'])
                logger.info("Loaded UNet weights")
            
            # Create diffusion model
            model = DiffusionModel(
                vae=vae,
                unet=unet,
                text_encoder=text_encoder,
                scheduler_type=config.get('scheduler_type', "ddim"),
                num_train_timesteps=config.get('num_train_timesteps', 1000),
                beta_schedule=config.get('beta_schedule', "linear"),
                prediction_type=config.get('prediction_type', "epsilon"),
                guidance_scale=config.get('guidance_scale', 7.5),
                device=self.device
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {e}")
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 10.0,
        eta: float = 0.0,
        output_type: str = "pil",
        return_dict: bool = True,
        seed: Optional[int] = None,
    ) -> Union[Dict, List[Image.Image]]:
        """
        Generate chest X-rays from text prompts.
        
        Args:
            prompt: Text prompt(s) describing the X-ray
            height: Output image height
            width: Output image width
            num_inference_steps: Number of denoising steps (more = higher quality, slower)
            guidance_scale: Controls adherence to the text prompt (higher = more faithful)
            eta: Controls randomness in sampling (0 = deterministic, 1 = stochastic)
            output_type: Output format, one of ["pil", "np", "tensor"]
            return_dict: Whether to return a dictionary with additional metadata
            seed: Random seed for reproducible generation
            
        Returns:
            Images and optionally metadata
        """
        # Set seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Generate images
        try:
            results = self.model.sample(
                text=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                tokenizer=self.tokenizer
            )
            
            # Get images
            images_tensor = results['images']
            
            # Convert to desired output format
            if output_type == "tensor":
                images = images_tensor
            elif output_type == "np":
                images = [img.cpu().numpy().transpose(1, 2, 0) for img in images_tensor]
            elif output_type == "pil":
                images = []
                for img in images_tensor:
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    if img_np.shape[-1] == 1:  # Remove channel dimension for grayscale
                        img_np = img_np.squeeze(-1)
                    images.append(Image.fromarray(img_np))
            else:
                raise ValueError(f"Unknown output type: {output_type}")
            
            # Return results
            if return_dict:
                return {
                    'images': images,
                    'latents': results['latents'].cpu(),
                    'prompt': prompt,
                    'parameters': {
                        'height': height,
                        'width': width,
                        'num_inference_steps': num_inference_steps,
                        'guidance_scale': guidance_scale,
                        'eta': eta,
                        'seed': seed
                    }
                }
            else:
                return images
                
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def save_images(self, images, output_dir, base_filename="generated", add_prompt=True, prompts=None):
        """
        Save generated images to disk.
        
        Args:
            images: List of images (PIL, numpy, or tensor)
            output_dir: Directory to save images
            base_filename: Base name for saved files
            add_prompt: Whether to include prompt in filename
            prompts: List of prompts corresponding to images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to PIL if needed
        if isinstance(images[0], torch.Tensor):
            images_pil = []
            for img in images:
                img_np = img.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)
                images_pil.append(Image.fromarray(img_np))
            images = images_pil
        elif isinstance(images[0], np.ndarray):
            images_pil = []
            for img in images:
                img_np = (img * 255).astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)
                images_pil.append(Image.fromarray(img_np))
            images = images_pil
        
        # Save each image
        for i, img in enumerate(images):
            # Create filename
            if add_prompt and prompts is not None:
                # Clean prompt for filename
                prompt_str = prompts[i] if isinstance(prompts, list) else prompts
                prompt_str = prompt_str.replace(" ", "_").replace(".", "").lower()
                prompt_str = ''.join(c for c in prompt_str if c.isalnum() or c == '_')
                prompt_str = prompt_str[:50]  # Limit length
                filename = f"{base_filename}_{i+1}_{prompt_str}.png"
            else:
                filename = f"{base_filename}_{i+1}.png"
            
            # Save image
            file_path = output_dir / filename
            img.save(file_path)
            logger.info(f"Saved image to {file_path}")