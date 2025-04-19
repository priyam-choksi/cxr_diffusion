#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive X-ray Diffusion Model Evaluation Script
Evaluates checkpoint_epoch_480.pt and extracts all possible metrics

Usage:
python evaluate_model.py
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
import cv2
import logging
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import warnings
from transformers import AutoTokenizer

# Configure paths
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "outputs" / "diffusion_checkpoints"
VAE_CHECKPOINTS_DIR = BASE_DIR / "outputs" / "vae_checkpoints"
DEFAULT_MODEL_PATH = str(CHECKPOINTS_DIR / "best_model.pt")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "dmis-lab/biobert-base-cased-v1.1")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BASE_DIR / "outputs" / "generated"))
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
DATASET_PATH = os.environ.get("DATASET_PATH", str(BASE_DIR / "dataset"))
IMAGES_PATH = os.environ.get("IMAGES_PATH", str(Path(DATASET_PATH) / "images" / "images_normalized"))

# Import project modules
from xray_generator.models.diffusion import DiffusionModel
from xray_generator.models.vae import MedicalVAE
from xray_generator.models.text_encoder import MedicalTextEncoder
from xray_generator.models.unet import DiffusionUNet
from xray_generator.utils.processing import get_device, apply_clahe, create_transforms
from xray_generator.utils.dataset import ChestXrayDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "visualizations"), exist_ok=True)

# Configure device
device = get_device()
logger.info(f"Using device: {device}")

def load_diffusion_model(checkpoint_path):
    """Load a diffusion model from checkpoint"""
    logger.info(f"Loading diffusion model from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
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
        ).to(device)
        
        text_encoder = MedicalTextEncoder(
            model_name=config.get('text_model', "dmis-lab/biobert-base-cased-v1.1"),
            projection_dim=768,
            freeze_base=True
        ).to(device)
        
        unet = DiffusionUNet(
            in_channels=latent_channels,
            model_channels=model_channels,
            out_channels=latent_channels,
            num_res_blocks=2,
            attention_resolutions=(8, 16, 32),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            context_dim=768
        ).to(device)
        
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
            device=device
        )
        
        return model, checkpoint
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load model: {e}")

def load_tokenizer():
    """Load tokenizer for text conditioning"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        logger.info(f"Loaded tokenizer: {TOKENIZER_NAME}")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        return None

def load_dataset(split_ratio=0.1):
    """Load a small subset of the dataset for evaluation"""
    
    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path {DATASET_PATH} does not exist.")
        return None
    
    # Try to find the reports and projections CSV files
    reports_csv = None
    projections_csv = None
    
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.csv'):
                if 'report' in file.lower():
                    reports_csv = os.path.join(root, file)
                elif 'projection' in file.lower():
                    projections_csv = os.path.join(root, file)
    
    if not reports_csv or not projections_csv:
        logger.error(f"Could not find reports or projections CSV files.")
        logger.info("Creating dummy dataset for evaluation...")
        
        # Create a dummy dataset with random noise
        class DummyDataset:
            def __init__(self, size=50):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Create random image
                img = torch.randn(1, 256, 256)
                
                # Normalize to [-1, 1]
                img = torch.clamp(img, -1, 1)
                
                # Create dummy text
                report = "Normal chest X-ray with no significant findings."
                
                # Create dummy encoding
                input_ids = torch.ones(256, dtype=torch.long)
                attention_mask = torch.ones(256, dtype=torch.long)
                
                return {
                    'image': img,
                    'report': report,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'uid': f'dummy_{idx}',
                    'filename': f'dummy_{idx}.png'
                }
        
        dataset = DummyDataset()
        logger.info(f"Created dummy dataset with {len(dataset)} samples")
        
        # Create dataloader
        from torch.utils.data import DataLoader
        from xray_generator.utils.processing import custom_collate_fn
        
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        return dataloader
    
    # Load the actual dataset
    logger.info(f"Loading dataset from {DATASET_PATH}")
    logger.info(f"Reports CSV: {reports_csv}")
    logger.info(f"Projections CSV: {projections_csv}")
    
    try:
        # Create transforms
        _, val_transform = create_transforms(256)
        
        # Create dataset
        dataset = ChestXrayDataset(
            reports_csv=reports_csv,
            projections_csv=projections_csv,
            image_folder=IMAGES_PATH,  # Use the images subfolder path
            transform=val_transform,
            target_size=(256, 256),
            filter_frontal=True,
            tokenizer_name=TOKENIZER_NAME,
            max_length=256,
            use_clahe=True
        )
        # Take a small subset for evaluation
        from torch.utils.data import Subset
        import random
        
        # Set seed for reproducibility
        random.seed(42)
        
        # Select random subset of indices
        indices = random.sample(range(len(dataset)), max(1, int(len(dataset) * split_ratio)))
        subset = Subset(dataset, indices)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        from xray_generator.utils.processing import custom_collate_fn
        
        dataloader = DataLoader(
            subset,
            batch_size=8,
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        logger.info(f"Created dataloader with {len(subset)} samples")
        return dataloader
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

class ModelMetrics:
    """Class to extract and calculate metrics from the model"""
    
    def __init__(self, model, checkpoint):
        self.model = model
        self.checkpoint = checkpoint
        self.metrics = {}
        
    def extract_checkpoint_metadata(self):
        """Extract metadata from the checkpoint"""
        metadata = {}
        
        # Extract epoch number if available
        if 'epoch' in self.checkpoint:
            metadata['epoch'] = self.checkpoint['epoch']
            
        # Extract loss values if available
        if 'best_metrics' in self.checkpoint:
            metadata['best_metrics'] = self.checkpoint['best_metrics']
            
        # Extract optimizer state if available
        if 'optimizer_state_dict' in self.checkpoint:
            optimizer = self.checkpoint['optimizer_state_dict']
            if 'param_groups' in optimizer:
                metadata['optimizer_param_groups'] = len(optimizer['param_groups'])
                if len(optimizer['param_groups']) > 0:
                    metadata['learning_rate'] = optimizer['param_groups'][0].get('lr', None)
                    
        # Extract model config if available
        if 'config' in self.checkpoint:
            metadata['config'] = self.checkpoint['config']
            
        # Extract scheduler state if available
        if 'scheduler_state_dict' in self.checkpoint:
            metadata['scheduler_state_present'] = True
            
        # Extract global step if available
        if 'global_step' in self.checkpoint:
            metadata['global_step'] = self.checkpoint['global_step']
            
        self.metrics['checkpoint_metadata'] = metadata
        return metadata
    
    def extract_model_architecture(self):
        """Extract model architecture information"""
        architecture = {}
        
        # VAE architecture
        vae_info = {
            'in_channels': self.model.vae.encoder.conv_in.in_channels,
            'out_channels': self.model.vae.decoder.final[-1].out_channels,
            'latent_channels': self.model.vae.latent_channels,
            'encoder_blocks': len(self.model.vae.encoder.down_blocks),
            'decoder_blocks': len(self.model.vae.decoder.up_blocks),
        }
        
        # UNet architecture
        unet_info = {
            'in_channels': self.model.unet.in_channels,
            'out_channels': self.model.unet.out_channels,
            'model_channels': self.model.unet.model_channels,
            'attention_resolutions': self.model.unet.attention_resolutions,
            'channel_mult': self.model.unet.channel_mult,
            'context_dim': self.model.unet.context_dim,
            'input_blocks': len(self.model.unet.input_blocks),
            'output_blocks': len(self.model.unet.output_blocks),
        }
        
        # Text encoder architecture
        text_encoder_info = {
            'model_name': self.model.text_encoder.model_name,
            'hidden_dim': self.model.text_encoder.hidden_dim,
            'projection_dim': self.model.text_encoder.projection_dim,
        }
        
        # Diffusion process parameters
        diffusion_info = {
            'scheduler_type': self.model.scheduler_type,
            'num_train_timesteps': self.model.num_train_timesteps,
            'beta_schedule': self.model.beta_schedule,
            'prediction_type': self.model.prediction_type,
            'guidance_scale': self.model.guidance_scale,
        }
        
        architecture['vae'] = vae_info
        architecture['unet'] = unet_info
        architecture['text_encoder'] = text_encoder_info
        architecture['diffusion'] = diffusion_info
        
        self.metrics['architecture'] = architecture
        return architecture
    
    def count_parameters(self):
        """Count model parameters"""
        param_counts = {}
        
        def count_params(model):
            return sum(p.numel() for p in model.parameters())
        
        def count_trainable_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # VAE parameters
        param_counts['vae_total'] = count_params(self.model.vae)
        param_counts['vae_trainable'] = count_trainable_params(self.model.vae)
        
        # UNet parameters
        param_counts['unet_total'] = count_params(self.model.unet)
        param_counts['unet_trainable'] = count_trainable_params(self.model.unet)
        
        # Text encoder parameters
        param_counts['text_encoder_total'] = count_params(self.model.text_encoder)
        param_counts['text_encoder_trainable'] = count_trainable_params(self.model.text_encoder)
        
        # Total parameters
        param_counts['total'] = param_counts['vae_total'] + param_counts['unet_total'] + param_counts['text_encoder_total']
        param_counts['trainable'] = param_counts['vae_trainable'] + param_counts['unet_trainable'] + param_counts['text_encoder_trainable']
        
        # Memory footprint (in MB)
        param_memory = 0
        buffer_memory = 0
        
        for module in [self.model.vae, self.model.unet, self.model.text_encoder]:
            param_memory += sum(p.nelement() * p.element_size() for p in module.parameters())
            buffer_memory += sum(b.nelement() * b.element_size() for b in module.buffers())
        
        param_counts['memory_footprint_mb'] = (param_memory + buffer_memory) / (1024 * 1024)
        
        self.metrics['parameters'] = param_counts
        return param_counts
    
    def analyze_beta_schedule(self):
        """Analyze the beta schedule used in the diffusion model"""
        beta_info = {}
        
        # Get beta schedule info
        betas = self.model.betas.cpu().numpy()
        beta_info['min'] = float(betas.min())
        beta_info['max'] = float(betas.max())
        beta_info['mean'] = float(betas.mean())
        beta_info['std'] = float(betas.std())
        
        # Get alphas info
        alphas_cumprod = self.model.alphas_cumprod.cpu().numpy()
        beta_info['alphas_cumprod_min'] = float(alphas_cumprod.min())
        beta_info['alphas_cumprod_max'] = float(alphas_cumprod.max())
        
        # Plot beta schedule
        plt.figure(figsize=(10, 6))
        plt.plot(betas, label='Beta Schedule')
        plt.xlabel('Timestep')
        plt.ylabel('Beta Value')
        plt.title(f'Beta Schedule ({self.model.beta_schedule})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'beta_schedule.png'))
        plt.close()
        
        # Plot alphas_cumprod
        plt.figure(figsize=(10, 6))
        plt.plot(alphas_cumprod, label='Cumulative Product of Alphas')
        plt.xlabel('Timestep')
        plt.ylabel('Alpha Cumprod Value')
        plt.title('Alphas Cumulative Product')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'alphas_cumprod.png'))
        plt.close()
        
        self.metrics['beta_schedule'] = beta_info
        return beta_info
    
    def analyze_vae_latent_space(self, dataloader):
        """Analyze the VAE latent space"""
        logger.info("Analyzing VAE latent space...")
        
        latent_info = {}
        latent_vectors = []
        orig_images = []
        recon_images = []
        
        # Set model to eval mode
        self.model.vae.eval()
        
        with torch.no_grad():
            # Process a few batches
            for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                if i >= 5:  # Limit to 5 batches for efficiency
                    break
                
                # Get images
                images = batch['image'].to(device)
                
                # Get latent vectors
                mu, logvar = self.model.vae.encode(images)
                
                # Store latent vectors
                latent_vectors.append(mu.cpu().numpy())
                
                # Store original images (first batch only)
                if i == 0:
                    orig_images = images[:8].cpu()  # Store up to 8 images
                    
                    # Generate reconstructions
                    recon, _, _ = self.model.vae(images[:8])
                    recon_images = recon.cpu()
        
        # Concatenate latent vectors
        latent_vectors = np.concatenate(latent_vectors, axis=0)
        
        # Calculate latent space statistics
        latent_info['mean'] = float(np.mean(latent_vectors))
        latent_info['std'] = float(np.std(latent_vectors))
        latent_info['min'] = float(np.min(latent_vectors))
        latent_info['max'] = float(np.max(latent_vectors))
        latent_info['dimensions'] = latent_vectors.shape[1]
        
        # Calculate active dimensions (standard deviation > 0.1)
        active_dims = np.sum(np.std(latent_vectors, axis=0) > 0.1)
        latent_info['active_dimensions'] = int(active_dims)
        latent_info['active_dimensions_ratio'] = float(active_dims / latent_vectors.shape[1])
        
        # Save visualization of latent space (t-SNE)
        if len(latent_vectors) > 10:
            try:
                # Subsample for efficiency
                sample_indices = np.random.choice(len(latent_vectors), min(500, len(latent_vectors)), replace=False)
                sampled_vectors = latent_vectors[sample_indices]
                
                # Apply t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                latent_2d = tsne.fit_transform(sampled_vectors.reshape(sampled_vectors.shape[0], -1))
                
                # Plot t-SNE
                plt.figure(figsize=(10, 10))
                plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.5)
                plt.title("t-SNE Visualization of VAE Latent Space")
                plt.colorbar()
                plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'vae_latent_tsne.png'))
                plt.close()
            except Exception as e:
                logger.error(f"Error creating t-SNE visualization: {e}")
        
        # Save original and reconstructed images
        if len(orig_images) > 0 and len(recon_images) > 0:
            # Combine into grid
            from torchvision.utils import make_grid
            
            # Denormalize from [-1, 1] to [0, 1]
            orig_images = (orig_images + 1) / 2
            recon_images = (recon_images + 1) / 2
            
            # Create comparison grid
            comparison = torch.cat([make_grid(orig_images, nrow=4, padding=2),
                                   make_grid(recon_images, nrow=4, padding=2)], dim=2)
            
            # Save grid
            from torchvision.utils import save_image
            save_image(comparison, os.path.join(OUTPUT_DIR, 'visualizations', 'vae_reconstruction.png'))
            
            # Calculate reconstruction error
            mse = torch.mean((orig_images - recon_images) ** 2).item()
            latent_info['reconstruction_mse'] = mse
        
        self.metrics['vae_latent'] = latent_info
        return latent_info
    
    def generate_samples(self, tokenizer, num_samples=4):
        """Generate samples from the diffusion model"""
        logger.info("Generating samples from diffusion model...")
        
        # Set model to eval mode
        self.model.vae.eval()
        self.model.unet.eval()
        self.model.text_encoder.eval()
        
        # Sample prompts
        prompts = [
            "Normal chest X-ray with clear lungs and no abnormalities.",
            "Right lower lobe pneumonia with focal consolidation.",
            "Mild cardiomegaly with pulmonary edema.",
            "Left pleural effusion with adjacent atelectasis."
        ]
        
        # Create folder for samples
        samples_dir = os.path.join(OUTPUT_DIR, 'samples')
        os.makedirs(samples_dir, exist_ok=True)
        
        generated_samples = []
        
        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts[:num_samples], desc="Generating samples")):
                try:
                    # Generate sample
                    results = self.model.sample(
                        prompt,
                        height=256,
                        width=256,
                        num_inference_steps=50,
                        tokenizer=tokenizer
                    )
                    
                    # Get image
                    img = results['images'][0]
                    
                    # Convert to numpy and save
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Remove channel dimension for grayscale
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                        
                    # Save image
                    img_path = os.path.join(samples_dir, f"sample_{i+1}.png")
                    Image.fromarray(img_np).save(img_path)
                    
                    # Save prompt
                    prompt_path = os.path.join(samples_dir, f"prompt_{i+1}.txt")
                    with open(prompt_path, "w") as f:
                        f.write(prompt)
                        
                    # Store generated sample
                    generated_samples.append({
                        'prompt': prompt,
                        'image_path': img_path
                    })
                    
                except Exception as e:
                    logger.error(f"Error generating sample {i+1}: {e}")
                    continue
        
        # Create a grid of all samples
        try:
            # Read all samples
            sample_images = []
            for i in range(num_samples):
                img_path = os.path.join(samples_dir, f"sample_{i+1}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_tensor = torch.tensor(np.array(img) / 255.0).unsqueeze(0)
                    if len(img_tensor.shape) == 3:  # Add channel dimension if needed
                        img_tensor = img_tensor.unsqueeze(0)
                    else:
                        img_tensor = img_tensor.permute(0, 3, 1, 2)
                    sample_images.append(img_tensor)
            
            if sample_images:
                # Create grid
                from torchvision.utils import make_grid
                grid = make_grid(torch.cat(sample_images, dim=0), nrow=2, padding=2)
                
                # Save grid
                from torchvision.utils import save_image
                save_image(grid, os.path.join(OUTPUT_DIR, 'visualizations', 'generated_samples_grid.png'))
        except Exception as e:
            logger.error(f"Error creating sample grid: {e}")
        
        self.metrics['generated_samples'] = generated_samples
        return generated_samples
    
    def measure_inference_speed(self, tokenizer, num_runs=10):
        """Measure inference speed"""
        logger.info("Measuring inference speed...")
        
        # Set model to eval mode
        self.model.vae.eval()
        self.model.unet.eval()
        self.model.text_encoder.eval()
        
        # Sample prompt
        prompt = "Normal chest X-ray with clear lungs and no abnormalities."
        
        # Warm-up run
        logger.info("Performing warm-up run...")
        with torch.no_grad():
            _ = self.model.sample(
                prompt,
                height=256,
                width=256,
                num_inference_steps=20,  # Use fewer steps for speed
                tokenizer=tokenizer
            )
        
        # Measure inference time
        logger.info(f"Measuring inference time over {num_runs} runs...")
        inference_times = []
        
        for i in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            start.record()
            
            with torch.no_grad():
                _ = self.model.sample(
                    prompt,
                    height=256,
                    width=256,
                    num_inference_steps=20,  # Use fewer steps for speed
                    tokenizer=tokenizer
                )
            
            end.record()
            torch.cuda.synchronize()
            
            # Calculate elapsed time in milliseconds
            inference_time = start.elapsed_time(end)
            inference_times.append(inference_time)
            
            logger.info(f"Run {i+1}/{num_runs}: {inference_time:.2f} ms")
        
        # Calculate statistics
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        inference_speed = {
            'avg_inference_time_ms': float(avg_time),
            'std_inference_time_ms': float(std_time),
            'min_inference_time_ms': float(np.min(inference_times)),
            'max_inference_time_ms': float(np.max(inference_times)),
            'num_runs': num_runs,
            'num_inference_steps': 20
        }
        
        # Plot inference times
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, num_runs + 1), inference_times)
        plt.axhline(avg_time, color='r', linestyle='--', label=f'Avg: {avg_time:.2f} ms')
        plt.xlabel('Run #')
        plt.ylabel('Inference Time (ms)')
        plt.title('Diffusion Model Inference Time')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'inference_time.png'))
        plt.close()
        
        self.metrics['inference_speed'] = inference_speed
        return inference_speed
    
    def visualize_unet_attention(self, tokenizer):
        """Visualize UNet attention maps"""
        logger.info("Visualizing UNet attention maps...")
        
        # This is a complex task and might need model code modification
        # Here we'll just create a placeholder for this analysis
        
        self.metrics['unet_attention'] = {
            'note': 'UNet attention visualization requires model modifications to extract attention maps'
        }
        
        return self.metrics['unet_attention']
    
    def visualize_noise_levels(self):
        """Visualize noise levels at different timesteps"""
        logger.info("Visualizing noise levels...")
        
        # Create a random image
        x_0 = torch.randn(1, 1, 256, 256).to(device)
        
        # Sample timesteps
        timesteps = torch.linspace(0, self.model.num_train_timesteps - 1, 10).long().to(device)
        
        # Create folder for noise visualizations
        noise_dir = os.path.join(OUTPUT_DIR, 'visualizations', 'noise_levels')
        os.makedirs(noise_dir, exist_ok=True)
        
        # Generate noisy samples at different timesteps
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # Add noise
                noisy_x = self.model.q_sample(x_0, t.unsqueeze(0))
                
                # Convert to image
                img = noisy_x[0].cpu()
                
                # Normalize to [0, 1]
                img = (img - img.min()) / (img.max() - img.min())
                
                # Save image
                from torchvision.utils import save_image
                save_image(img, os.path.join(noise_dir, f"noise_t{t.item()}.png"))
        
        # Create a grid of noise levels
        try:
            # Read all noise images
            noise_images = []
            for i, t in enumerate(timesteps):
                img_path = os.path.join(noise_dir, f"noise_t{t.item()}.png")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_tensor = torch.tensor(np.array(img) / 255.0)
                    if len(img_tensor.shape) == 2:  # Add channel dimension if needed
                        img_tensor = img_tensor.unsqueeze(0)
                    else:
                        img_tensor = img_tensor.permute(2, 0, 1)
                    noise_images.append(img_tensor)
            
            if noise_images:
                # Create grid
                from torchvision.utils import make_grid
                grid = make_grid(torch.stack(noise_images), nrow=5, padding=2)
                
                # Save grid
                from torchvision.utils import save_image
                save_image(grid, os.path.join(OUTPUT_DIR, 'visualizations', 'noise_levels_grid.png'))
        except Exception as e:
            logger.error(f"Error creating noise levels grid: {e}")
        
        self.metrics['noise_levels'] = {
            'timesteps': timesteps.cpu().numpy().tolist(),
            'visualization_path': noise_dir
        }
        
        return self.metrics['noise_levels']
    
    def plot_learning_curves(self):
        """Plot learning curves if available in checkpoint"""
        logger.info("Plotting learning curves...")
        
        # Check if loss values are available
        if 'best_metrics' not in self.checkpoint:
            logger.info("No loss values found in checkpoint")
            return None
        
        # Extract metrics
        metrics = self.checkpoint['best_metrics']
        
        if 'train_loss' in metrics and 'val_loss' in metrics:
            # Plot training and validation loss
            plt.figure(figsize=(10, 6))
            plt.bar(['Training Loss', 'Validation Loss'], 
                   [metrics['train_loss'], metrics['val_loss']])
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'loss_comparison.png'))
            plt.close()
        
        if 'train_diffusion_loss' in metrics and 'val_diffusion_loss' in metrics:
            # Plot diffusion loss
            plt.figure(figsize=(10, 6))
            plt.bar(['Training Diffusion Loss', 'Validation Diffusion Loss'], 
                   [metrics['train_diffusion_loss'], metrics['val_diffusion_loss']])
            plt.ylabel('Diffusion Loss')
            plt.title('Diffusion Loss')
            plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'diffusion_loss.png'))
            plt.close()
        
        return metrics
    
    def create_parameter_distribution_plots(self):
        """Plot parameter distributions"""
        logger.info("Creating parameter distribution plots...")
        
        # Collect parameters from different components
        vae_params = torch.cat([p.detach().cpu().flatten() for p in self.model.vae.parameters()])
        unet_params = torch.cat([p.detach().cpu().flatten() for p in self.model.unet.parameters()])
        text_encoder_params = torch.cat([p.detach().cpu().flatten() for p in self.model.text_encoder.parameters()])
        
        # Plot parameter distributions
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(vae_params.numpy(), bins=50, alpha=0.7)
        plt.title('VAE Parameters')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 2)
        plt.hist(unet_params.numpy(), bins=50, alpha=0.7)
        plt.title('UNet Parameters')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 3)
        plt.hist(text_encoder_params.numpy(), bins=50, alpha=0.7)
        plt.title('Text Encoder Parameters')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'visualizations', 'parameter_distributions.png'))
        plt.close()
        
        # Calculate statistics
        param_stats = {
            'vae': {
                'mean': float(vae_params.mean()),
                'std': float(vae_params.std()),
                'min': float(vae_params.min()),
                'max': float(vae_params.max())
            },
            'unet': {
                'mean': float(unet_params.mean()),
                'std': float(unet_params.std()),
                'min': float(unet_params.min()),
                'max': float(unet_params.max())
            },
            'text_encoder': {
                'mean': float(text_encoder_params.mean()),
                'std': float(text_encoder_params.std()),
                'min': float(text_encoder_params.min()),
                'max': float(text_encoder_params.max())
            }
        }
        
        self.metrics['parameter_stats'] = param_stats
        return param_stats
    
    def generate_text_conditioning_analysis(self, tokenizer):
        """Analyze the effect of text conditioning on generation"""
        logger.info("Generating text conditioning analysis...")
        
        if tokenizer is None:
            logger.error("Tokenizer is required for text conditioning analysis")
            return None
        
        # Create a test case with multiple prompts
        test_prompts = [
            "Normal chest X-ray with no abnormalities.",
            "Severe pneumonia with bilateral infiltrates.",
            "Cardiomegaly with pulmonary edema.",
            "Pneumothorax with collapsed left lung."
        ]
        
        # Create folder for text conditioning visualizations
        text_dir = os.path.join(OUTPUT_DIR, 'visualizations', 'text_conditioning')
        os.makedirs(text_dir, exist_ok=True)
        
        # Generate samples for each prompt
        generated_images = []
        
        with torch.no_grad():
            # Generate one sample with fixed seed for each prompt
            for i, prompt in enumerate(tqdm(test_prompts, desc="Generating conditioned samples")):
                try:
                    # Set seed for reproducibility
                    torch.manual_seed(42)
                    
                    # Generate sample
                    results = self.model.sample(
                        prompt,
                        height=256,
                        width=256,
                        num_inference_steps=50,
                        tokenizer=tokenizer
                    )
                    
                    # Get image
                    img = results['images'][0]
                    
                    # Save image
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                    
                    img_path = os.path.join(text_dir, f"prompt_{i+1}.png")
                    Image.fromarray(img_np).save(img_path)
                    
                    # Save prompt
                    prompt_path = os.path.join(text_dir, f"prompt_{i+1}.txt")
                    with open(prompt_path, "w") as f:
                        f.write(prompt)
                    
                    # Store generated image
                    generated_images.append(img.cpu())
                    
                except Exception as e:
                    logger.error(f"Error generating sample for prompt {i+1}: {e}")
                    continue
        
        # Create a grid of all samples
        if generated_images:
            try:
                # Create grid
                from torchvision.utils import make_grid
                grid = make_grid(torch.stack(generated_images), nrow=2, padding=2)
                
                # Save grid
                from torchvision.utils import save_image
                save_image(grid, os.path.join(OUTPUT_DIR, 'visualizations', 'text_conditioning_grid.png'))
            except Exception as e:
                logger.error(f"Error creating text conditioning grid: {e}")
        
        # Test different guidance scales on a single prompt
        guidance_scales = [1.0, 3.0, 7.5, 10.0, 15.0]
        guidance_images = []
        
        with torch.no_grad():
            # Generate samples with different guidance scales
            for i, scale in enumerate(tqdm(guidance_scales, desc="Testing guidance scales")):
                try:
                    # Set seed for reproducibility
                    torch.manual_seed(42)
                    
                    # Generate sample
                    results = self.model.sample(
                        test_prompts[0],  # Use the first prompt
                        height=256,
                        width=256,
                        num_inference_steps=50,
                        guidance_scale=scale,
                        tokenizer=tokenizer,
                        seed=42  # Fixed seed
                    )
                    
                    # Get image
                    img = results['images'][0]
                    
                    # Save image
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                    
                    img_path = os.path.join(text_dir, f"guidance_{scale}.png")
                    Image.fromarray(img_np).save(img_path)
                    
                    # Store generated image
                    guidance_images.append(img.cpu())
                    
                except Exception as e:
                    logger.error(f"Error generating sample for guidance scale {scale}: {e}")
                    continue
        
        # Create a grid of guidance scale samples
        if guidance_images:
            try:
                # Create grid
                from torchvision.utils import make_grid
                grid = make_grid(torch.stack(guidance_images), nrow=len(guidance_scales), padding=2)
                
                # Save grid
                from torchvision.utils import save_image
                save_image(grid, os.path.join(OUTPUT_DIR, 'visualizations', 'guidance_scale_grid.png'))
            except Exception as e:
                logger.error(f"Error creating guidance scale grid: {e}")
        
        self.metrics['text_conditioning'] = {
            'test_prompts': test_prompts,
            'guidance_scales': guidance_scales,
            'visualization_path': text_dir
        }
        
        return self.metrics['text_conditioning']
    
    def analyze_all(self, dataloader, tokenizer):
        """Run all analysis methods and collect metrics"""
        
        # Extract checkpoint metadata
        self.extract_checkpoint_metadata()
        
        # Extract model architecture information
        self.extract_model_architecture()
        
        # Count parameters
        self.count_parameters()
        
        # Analyze beta schedule
        self.analyze_beta_schedule()
        
        # Analyze VAE latent space
        if dataloader is not None:
            self.analyze_vae_latent_space(dataloader)
        
        # Generate samples
        if tokenizer is not None:
            self.generate_samples(tokenizer)
        
        # Measure inference speed
        if tokenizer is not None:
            self.measure_inference_speed(tokenizer, num_runs=5)
        
        # Visualize UNet attention
        if tokenizer is not None:
            self.visualize_unet_attention(tokenizer)
        
        # Visualize noise levels
        self.visualize_noise_levels()
        
        # Plot learning curves
        self.plot_learning_curves()
        
        # Create parameter distribution plots
        self.create_parameter_distribution_plots()
        
        # Generate text conditioning analysis
        if tokenizer is not None:
            self.generate_text_conditioning_analysis(tokenizer)
        
        # Save all metrics to file
        with open(os.path.join(METRICS_DIR, 'diffusion_metrics.json'), 'w') as f:
            # Convert non-serializable values to strings or lists
            serializable_metrics = json.loads(
                json.dumps(self.metrics, default=lambda o: str(o) if not isinstance(o, (int, float, str, bool, list, dict, type(None))) else o)
            )
            json.dump(serializable_metrics, f, indent=2)
        
        return self.metrics

def create_model_summary(metrics):
    """Create a human-readable summary of model metrics"""
    logger.info("Creating model summary...")
    
    summary = []
    
    # Add header
    summary.append("# X-ray Diffusion Model Evaluation Summary")
    summary.append("\n## Model Information")
    
    # Add model architecture
    if 'architecture' in metrics:
        arch = metrics['architecture']
        
        summary.append("\n### Diffusion Model")
        summary.append(f"- Scheduler Type: {arch['diffusion']['scheduler_type']}")
        summary.append(f"- Timesteps: {arch['diffusion']['num_train_timesteps']}")
        summary.append(f"- Beta Schedule: {arch['diffusion']['beta_schedule']}")
        summary.append(f"- Prediction Type: {arch['diffusion']['prediction_type']}")
        summary.append(f"- Guidance Scale: {arch['diffusion']['guidance_scale']}")
        
        summary.append("\n### VAE")
        summary.append(f"- Latent Channels: {arch['vae']['latent_channels']}")
        summary.append(f"- Encoder Blocks: {arch['vae']['encoder_blocks']}")
        summary.append(f"- Decoder Blocks: {arch['vae']['decoder_blocks']}")
        
        summary.append("\n### UNet")
        summary.append(f"- Model Channels: {arch['unet']['model_channels']}")
        summary.append(f"- Attention Resolutions: {arch['unet']['attention_resolutions']}")
        summary.append(f"- Channel Multipliers: {arch['unet']['channel_mult']}")
        
        summary.append("\n### Text Encoder")
        summary.append(f"- Model: {arch['text_encoder']['model_name']}")
        summary.append(f"- Hidden Dimension: {arch['text_encoder']['hidden_dim']}")
        summary.append(f"- Projection Dimension: {arch['text_encoder']['projection_dim']}")
    
    # Add parameter counts
    if 'parameters' in metrics:
        params = metrics['parameters']
        
        summary.append("\n## Parameter Counts")
        summary.append(f"- Total Parameters: {params['total']:,}")
        summary.append(f"- Trainable Parameters: {params['trainable']:,}")
        summary.append(f"- Memory Footprint: {params['memory_footprint_mb']:.2f} MB")
        
        summary.append("\n### Component Breakdown")
        summary.append(f"- VAE: {params['vae_total']:,} parameters ({params['vae_trainable']:,} trainable)")
        summary.append(f"- UNet: {params['unet_total']:,} parameters ({params['unet_trainable']:,} trainable)")
        summary.append(f"- Text Encoder: {params['text_encoder_total']:,} parameters ({params['text_encoder_trainable']:,} trainable)")
    
    # Add training information
    if 'checkpoint_metadata' in metrics:
        meta = metrics['checkpoint_metadata']
        
        summary.append("\n## Training Information")
        if 'epoch' in meta:
            summary.append(f"- Trained for {meta['epoch']} epochs")
        
        if 'global_step' in meta:
            summary.append(f"- Global steps: {meta['global_step']}")
        
        if 'best_metrics' in meta:
            summary.append("\n### Best Metrics")
            best = meta['best_metrics']
            for key, value in best.items():
                summary.append(f"- {key}: {value}")
    
    # Add VAE latent information
    if 'vae_latent' in metrics:
        latent = metrics['vae_latent']
        
        summary.append("\n## VAE Latent Space Analysis")
        summary.append(f"- Latent Dimensions: {latent.get('dimensions', 'N/A')}")
        summary.append(f"- Active Dimensions: {latent.get('active_dimensions', 'N/A')} ({latent.get('active_dimensions_ratio', 'N/A'):.2%})")
        
        if 'reconstruction_mse' in latent:
            summary.append(f"- Reconstruction MSE: {latent['reconstruction_mse']:.6f}")
    
    # Add inference speed
    if 'inference_speed' in metrics:
        speed = metrics['inference_speed']
        
        summary.append("\n## Inference Performance")
        summary.append(f"- Average Inference Time: {speed['avg_inference_time_ms']:.2f} ms")
        summary.append(f"- Standard Deviation: {speed['std_inference_time_ms']:.2f} ms")
        summary.append(f"- Range: {speed['min_inference_time_ms']:.2f} - {speed['max_inference_time_ms']:.2f} ms")
    
    # Add visualization paths
    summary.append("\n## Visualizations")
    summary.append(f"- All visualizations saved to: {os.path.join(OUTPUT_DIR, 'visualizations')}")
    
    if 'generated_samples' in metrics:
        summary.append(f"- Generated samples saved to: {os.path.join(OUTPUT_DIR, 'samples')}")
    
    # Save summary to file
    summary_text = "\n".join(summary)
    with open(os.path.join(METRICS_DIR, 'model_summary.md'), 'w') as f:
        f.write(summary_text)
    
    logger.info(f"Model summary saved to {os.path.join(METRICS_DIR, 'model_summary.md')}")
    
    return summary_text

def main():
    """Main function to run all analyses"""
    logger.info("Starting model evaluation script")
    
    # Load diffusion model from checkpoint
    diffusion_model, checkpoint = load_diffusion_model(
        os.path.join(CHECKPOINTS_DIR, "checkpoint_epoch_480.pt")
    )
    
    # Load tokenizer
    tokenizer = load_tokenizer()
    
    # Load dataset
    dataloader = load_dataset()
    
    # Create metrics calculator
    metrics_calculator = ModelMetrics(diffusion_model, checkpoint)
    
    # Run all analyses
    metrics = metrics_calculator.analyze_all(dataloader, tokenizer)
    
    # Create human-readable summary
    summary = create_model_summary(metrics)
    
    logger.info("Model evaluation complete")
    logger.info(f"Results saved to {METRICS_DIR}")
    logger.info(f"Visualizations saved to {os.path.join(OUTPUT_DIR, 'visualizations')}")

if __name__ == "__main__":
    main()