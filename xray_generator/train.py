# xray_generator/train.py
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import random
import math
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import Subset

from .models.vae import MedicalVAE
from .models.unet import DiffusionUNet
from .models.text_encoder import MedicalTextEncoder
from .models.diffusion import DiffusionModel
from .utils.processing import set_seed, get_device, log_gpu_memory, create_transforms
from .utils.dataset import ChestXrayDataset
from transformers import AutoTokenizer
from torch.utils.data import random_split

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping implementation."""
    def __init__(self, patience=7, verbose=True, delta=0, path='checkpoint.pt'):
        """Initialize early stopping."""
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model=None):
        """Call early stopping logic."""
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        return False
    
    def save_checkpoint(self, val_loss, model):
        """Save model checkpoint."""
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        if model is not None:
            torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def create_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Create learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_metrics, checkpoint_dir, is_best=False):
    """Save checkpoint every checkpoint_freq epochs plus best model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare checkpoint data
    if isinstance(model, dict):
        # For VAE-only training
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model['vae'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metrics': best_metrics,
            'global_step': global_step
        }
    else:
        # For diffusion model
        checkpoint = {
            'epoch': epoch,
            'vae_state_dict': model.vae.state_dict(),
            'unet_state_dict': model.unet.state_dict(),
            'text_encoder_state_dict': model.text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metrics': best_metrics,
            'global_step': global_step,
            'config': {
                'latent_channels': model.vae.latent_channels,
                'model_channels': model.unet.model_channels,
                'scheduler_type': model.scheduler_type,
                'beta_schedule': model.beta_schedule,
                'prediction_type': model.prediction_type,
                'guidance_scale': model.guidance_scale,
                'num_train_timesteps': model.num_train_timesteps
            }
        }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save path
    if not is_best:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Cleanup old checkpoints
    if not is_best:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n=5)

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n):
    """Remove old checkpoints, keeping only the most recent n checkpoints"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    
    if len(checkpoints) <= keep_last_n:
        return
        
    # Sort by epoch number
    checkpoints.sort(key=lambda x: int(x.split("_epoch_")[1].split(".")[0]))
    
    # Remove older checkpoints
    for old_ckpt in checkpoints[:-keep_last_n]:
        old_path = os.path.join(checkpoint_dir, old_ckpt)
        try:
            os.remove(old_path)
            logger.info(f"Removed old checkpoint: {old_path}")
        except Exception as e:
            logger.error(f"Failed to remove old checkpoint {old_path}: {e}")

def load_checkpoint(model, optimizer, scheduler, path):
    """Load checkpoint and resume training"""
    if not os.path.exists(path):
        logger.info(f"No checkpoint found at {path}")
        return 0, 0, {'val_loss': float('inf')}
        
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model states
    if isinstance(model, dict):
        # For VAE-only training
        model['vae'].load_state_dict(checkpoint['model_state_dict'])
    else:
        # For diffusion model
        model.vae.load_state_dict(checkpoint['vae_state_dict'])
        model.unet.load_state_dict(checkpoint['unet_state_dict'])
        model.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
    
    # Load optimizer and scheduler
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get training state
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    best_metrics = checkpoint.get('best_metrics', {'val_loss': float('inf')})
    
    logger.info(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch, global_step, best_metrics

def visualize_epoch_results(epoch, model, tokenizer, val_loader, output_dir):
    """Generate and save visualization samples after each epoch."""
    # Create output directory
    samples_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch+1}")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Visualization types
    # 1. Real samples from dataset with VAE reconstruction
    try:
        # Get a batch from validation set
        val_batch = next(iter(val_loader))
        
        # Take 2 random samples from the batch
        batch_size = min(2, len(val_batch['image']))
        indices = random.sample(range(len(val_batch['image'])), batch_size)
        
        for i, idx in enumerate(indices):
            # Save real image
            img = val_batch['image'][idx].unsqueeze(0)
            if isinstance(model, dict):
                device = next(model['vae'].parameters()).device
                img = img.to(device)
                vae = model['vae']
            else:
                img = img.to(model.device)
                vae = model.vae
                
            report = val_batch['report'][idx]
            
            # Save original image
            img_np = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 0.5 + 0.5) * 255  # Denormalize
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
            img_path = os.path.join(samples_dir, f"real_{i+1}.png")
            from PIL import Image
            Image.fromarray(img_np.astype(np.uint8)).save(img_path)
            
            # Generate reconstruction
            with torch.no_grad():
                recon, _, _ = vae(img)
                
                # Save reconstruction
                recon_np = recon.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                recon_np = (recon_np * 0.5 + 0.5) * 255  # Denormalize
                if recon_np.shape[-1] == 1:
                    recon_np = recon_np.squeeze(-1)
                recon_path = os.path.join(samples_dir, f"recon_{i+1}.png")
                Image.fromarray(recon_np.astype(np.uint8)).save(recon_path)
                
            # Save report
            report_path = os.path.join(samples_dir, f"report_{i+1}.txt")
            with open(report_path, "w") as f:
                f.write(report)
    except Exception as e:
        logger.error(f"Error generating real samples: {e}")
    
    # 2. Generated samples from prompts
    if not isinstance(model, dict) and tokenizer is not None:  # Only for full model, not VAE-only
        try:
            # Sample prompts
            sample_prompts = [
                "Normal chest X-ray with clear lungs and no abnormalities.",
                "Right lower lobe pneumonia with focal consolidation."
            ]
            
            # Generate samples
            model.vae.eval()
            model.text_encoder.eval()
            model.unet.eval()
            
            with torch.no_grad():
                for i, prompt in enumerate(sample_prompts):
                    results = model.sample(
                        prompt,
                        height=256,
                        width=256,
                        num_inference_steps=30,
                        tokenizer=tokenizer
                    )
                    
                    # Save generated image
                    img = results['images'][0]
                    img_np = img.cpu().numpy().transpose(1, 2, 0)
                    img_np = img_np * 255
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                    img_path = os.path.join(samples_dir, f"gen_{i+1}.png")
                    from PIL import Image
                    Image.fromarray(img_np.astype(np.uint8)).save(img_path)
                    
                    # Save prompt
                    prompt_path = os.path.join(samples_dir, f"prompt_{i+1}.txt")
                    with open(prompt_path, "w") as f:
                        f.write(prompt)
        except Exception as e:
            logger.error(f"Error generating samples from prompts: {e}")
    
    logger.info(f"Saved visualization for epoch {epoch+1} to {samples_dir}")

def create_quick_test_dataset(dataset, percentage=0.01):
    """Create a small subset of a dataset for quick testing."""
    from torch.utils.data import Dataset
    
    class SmallDatasetWrapper(Dataset):
        def __init__(self, dataset, percentage=0.01):
            self.dataset = dataset
            indices = random.sample(range(len(dataset)), int(len(dataset) * percentage))
            logger.info(f"Using {len(indices)} samples out of {len(dataset)} ({percentage*100:.1f}%)")
            self.indices = indices
            
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
            
        def __len__(self):
            return len(self.indices)
    
    return SmallDatasetWrapper(dataset, percentage)

def train(
    config: Dict,
    dataset_path: str,
    reports_csv: str,
    projections_csv: str,
    output_dir: str = "./outputs",
    resume_from: Optional[str] = None,
    train_vae_only: bool = False,
    seed: int = 42,
    quick_test: bool = False  # Added quick test parameter
):
    """
    Train the chest X-ray diffusion model.
    
    Args:
        config: Configuration dictionary with model and training parameters
        dataset_path: Path to the X-ray image directory
        reports_csv: Path to the reports CSV file
        projections_csv: Path to the projections CSV file
        output_dir: Path to save outputs
        resume_from: Path to resume training from checkpoint
        train_vae_only: Whether to train only the VAE component
        seed: Random seed for reproducibility
        quick_test: Whether to run a quick test with reduced settings
    """
    # If quick test, override settings
    if quick_test:
        logger.warning("⚠️ RUNNING IN TEST MODE - QUICK TEST WITH 1% OF DATA AND REDUCED SETTINGS ⚠️")
        # Modify config for quick test
        quick_config = config.copy()
        quick_config["batch_size"] = min(config.get("batch_size", 4), 2)
        quick_config["epochs"] = min(config.get("epochs", 100), 2)
        quick_config["num_workers"] = 0
        config = quick_config
    
    # Extract configuration parameters
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 0)
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    latent_channels = config.get('latent_channels', 8)
    model_channels = config.get('model_channels', 48)
    image_size = config.get('image_size', 256)
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    use_amp = config.get('use_amp', True)
    checkpoint_freq = config.get('checkpoint_freq', 5)
    tokenizer_name = config.get('tokenizer_name', "dmis-lab/biobert-base-cased-v1.1")
    
    # Set up logging and seed
    set_seed(seed)
    device = get_device()
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Add this code to create separate directories for VAE and diffusion
    if train_vae_only:
        checkpoint_dir = os.path.join(output_dir, "checkpoints", "vae")
    else:
        checkpoint_dir = os.path.join(output_dir, "checkpoints", "diffusion")

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up dataset
    transforms = create_transforms(image_size)
    logger.info(f"Creating dataset from {dataset_path}")
    
    # Create dataset
    dataset = ChestXrayDataset(
        reports_csv=reports_csv,
        projections_csv=projections_csv,
        image_folder=dataset_path,
        transform=None,  # Will set per split
        target_size=(image_size, image_size),
        filter_frontal=True,
        tokenizer_name=tokenizer_name,
        max_length=256,
        use_clahe=True
    )
    
    # If quick test, use a smaller subset of the dataset
    if quick_test:
        dataset = create_quick_test_dataset(dataset, percentage=0.01)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(0.1 * dataset_size)
    test_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Create splits
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Set transforms for each split
    train_transform, val_transform = transforms
    
    # Apply transforms to splits
    def set_dataset_transform(dataset, transform):
        """Set transform for a specific dataset split."""
        dataset.transform = transform
        
        # Monkey patch the __getitem__ method to apply our transform
        original_getitem = dataset.__getitem__
        
        def new_getitem(idx):
            item = original_getitem(idx)
            if dataset.transform and 'image' in item and item['image'] is not None:
                item['image'] = dataset.transform(item['image'])
            return item
            
        dataset.__getitem__ = new_getitem
    
    set_dataset_transform(train_dataset, train_transform)
    set_dataset_transform(val_dataset, val_transform)
    set_dataset_transform(test_dataset, val_transform)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    from .utils.processing import custom_collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
    # Initialize models
    logger.info("Initializing models")
    
    # VAE
    vae = MedicalVAE(
        in_channels=1,
        out_channels=1, 
        latent_channels=latent_channels,
        hidden_dims=[model_channels, model_channels*2, model_channels*4, model_channels*8]
    ).to(device)
    
    # For VAE-only training
    if train_vae_only:
        optimizer = AdamW(vae.parameters(), lr=learning_rate, weight_decay=1e-6)
        
        # Training state tracking
        start_epoch = 0
        global_step = 0
        best_metrics = {'val_loss': float('inf')}
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch, global_step, best_metrics = load_checkpoint(
                {'vae': vae}, optimizer, None, resume_from
            )
            logger.info(f"Resumed VAE training from epoch {start_epoch}")
        
        # Create learning rate scheduler
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
        
        # Train the VAE
        vae_trainer = VAETrainer(
            model=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config
        )
        
        best_model = vae_trainer.train(
            num_epochs=epochs,
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch,
            global_step=global_step,
            best_metrics=best_metrics
        )
        
        logger.info("VAE training complete")
        return best_model
    
    # Full diffusion model training
    else:
        # Text encoder
        text_encoder = MedicalTextEncoder(
            model_name=tokenizer_name,
            projection_dim=768,
            freeze_base=True
        ).to(device)
        
        # UNet
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
        
        # Diffusion model
        diffusion_model = DiffusionModel(
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
        
        # Create optimizer - train UNet only by default
        train_unet_only = config.get('train_unet_only', True)
        
        if train_unet_only:
            optimizer = AdamW(unet.parameters(), lr=learning_rate, weight_decay=1e-6)
        else:
            parameters = list(unet.parameters())
            parameters.extend(vae.parameters())
            parameters.extend(text_encoder.parameters())
            optimizer = AdamW(parameters, lr=learning_rate, weight_decay=1e-6)
        
        # Training state tracking
        start_epoch = 0
        global_step = 0
        best_metrics = {'val_loss': float('inf')}
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch, global_step, best_metrics = load_checkpoint(
                diffusion_model, optimizer, None, resume_from
            )
            logger.info(f"Resumed diffusion training from epoch {start_epoch}")
        
        # Create tokenizer for sampling
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Loaded tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            logger.warning("Will not generate samples during training")
            tokenizer = None
        
        # Create learning rate scheduler
        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
        
        # Train the diffusion model
        diffusion_trainer = DiffusionTrainer(
            model=diffusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            device=device,
            config=config
        )
        
        trained_model = diffusion_trainer.train(
            num_epochs=epochs,
            checkpoint_dir=checkpoint_dir,
            train_unet_only=train_unet_only,
            start_epoch=start_epoch,
            global_step=global_step,
            best_metrics=best_metrics
        )
        
        logger.info("Diffusion model training complete")
        return trained_model

class VAETrainer:
    """Trainer for VAE model."""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device=None,
        config=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else {}
        
        # Extract config parameters
        self.use_amp = self.config.get('use_amp', True)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        self.checkpoint_freq = self.config.get('checkpoint_freq', 5)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
    
    def vae_loss_fn(self, recon_x, x, mu, logvar, kld_weight=1e-4):
        """VAE loss function."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + kld_weight * kld_loss
        
        return loss, recon_loss, kld_loss
    
    def train(
        self, 
        num_epochs, 
        checkpoint_dir, 
        start_epoch=0, 
        global_step=0, 
        best_metrics=None
    ):
        """Train the VAE model."""
        logger.info("Starting VAE training")
        
        # Best model tracking
        best_loss = best_metrics.get('val_loss', float('inf')) if best_metrics else float('inf')
        best_model_state = None
        
        # Set up early stopping
        early_stopping_path = os.path.join(checkpoint_dir, "best_vae.pt")
        early_stopping = EarlyStopping(
            patience=5,
            verbose=True,
            path=early_stopping_path
        )
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Starting VAE epoch {epoch+1}/{num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kld_loss = 0.0
            
            # Initialize gradient accumulation
            self.optimizer.zero_grad()
            
            # Train loop with progress bar
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (VAE Training)")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Get images
                    images = batch['image'].to(self.device)
                    
                    # Skip problematic batches
                    if images.shape[0] < 2:  # Need at least 2 samples for batch norm
                        logger.warning(f"Skipping batch with only {images.shape[0]} samples")
                        continue
                    
                    # Forward pass with mixed precision
                    if self.use_amp and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            recon, mu, logvar = self.model(images)
                            loss, recon_loss, kld_loss = self.vae_loss_fn(recon, images, mu, logvar)
                            # Scale loss for gradient accumulation
                            loss = loss / self.gradient_accumulation_steps
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Step with gradient accumulation
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx + 1 == len(self.train_loader):
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            
                            # Update scheduler
                            if self.scheduler:
                                self.scheduler.step()
                            global_step += 1
                    else:
                        recon, mu, logvar = self.model(images)
                        loss, recon_loss, kld_loss = self.vae_loss_fn(recon, images, mu, logvar)
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                        
                        loss.backward()
                        
                        # Step with gradient accumulation
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx + 1 == len(self.train_loader):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            
                            # Update scheduler
                            if self.scheduler:
                                self.scheduler.step()
                            global_step += 1
                    
                    # Update metrics (using original loss)
                    train_loss += loss.item() * self.gradient_accumulation_steps
                    train_recon_loss += recon_loss.item()
                    train_kld_loss += kld_loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'kld': f"{kld_loss.item():.4f}"
                    })
                        
                except Exception as e:
                    logger.error(f"Error in VAE training batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Calculate average training losses
            train_loss /= max(1, len(self.train_loader))
            train_recon_loss /= max(1, len(self.train_loader))
            train_kld_loss /= max(1, len(self.train_loader))
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kld_loss = 0.0
            
            with torch.no_grad():
                # Validation loop with progress bar
                val_progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (VAE Validation)")
                for batch_idx, batch in enumerate(val_progress):
                    try:
                        # Get images
                        images = batch['image'].to(self.device)
                        
                        # Skip problematic batches
                        if images.shape[0] < 2:
                            continue
                        
                        # Forward pass
                        recon, mu, logvar = self.model(images)
                        loss, recon_loss, kld_loss = self.vae_loss_fn(recon, images, mu, logvar)
                        
                        # Update metrics
                        val_loss += loss.item()
                        val_recon_loss += recon_loss.item()
                        val_kld_loss += kld_loss.item()
                    
                    except Exception as e:
                        logger.error(f"Error in VAE validation: {e}")
                        continue
            
            # Calculate average validation losses
            val_loss /= max(1, len(self.val_loader))
            val_recon_loss /= max(1, len(self.val_loader))
            val_kld_loss /= max(1, len(self.val_loader))
            
            # Log metrics
            logger.info(f"VAE Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KLD: {train_kld_loss:.4f}) | "
                      f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KLD: {val_kld_loss:.4f})")
            
            # Check if this is the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                
                # Save best checkpoint
                save_checkpoint(
                    {'vae': self.model},
                    self.optimizer,
                    self.scheduler,
                    epoch+1,
                    global_step,
                    {'val_loss': val_loss},
                    checkpoint_dir,
                    is_best=True
                )
            
            # Save regular checkpoint
            if (epoch + 1) % self.checkpoint_freq == 0:
                save_checkpoint(
                    {'vae': self.model},
                    self.optimizer,
                    self.scheduler,
                    epoch+1,
                    global_step,
                    {'val_loss': val_loss},
                    checkpoint_dir,
                    is_best=False
                )
            
            # Check early stopping
            if early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Visualize results after each epoch
            if Path(checkpoint_dir).exists():
                from PIL import Image
                visualize_epoch_results(
                    epoch,
                    {"vae": self.model},
                    None,
                    self.val_loader,
                    checkpoint_dir
                )
        
        # Return best model state
        if best_model_state is not None:
            logger.info(f"VAE training complete. Best validation loss: {best_loss:.4f}")
            return best_model_state
        else:
            logger.warning("VAE training complete, but no best model state was saved.")
            return self.model.state_dict()

class DiffusionTrainer:
    """Trainer for diffusion model."""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        tokenizer=None,
        device=None,
        config=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config if config is not None else {}
        
        # Extract config parameters
        self.use_amp = self.config.get('use_amp', True)
        self.checkpoint_freq = self.config.get('checkpoint_freq', 5)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
    
    def train(
        self, 
        num_epochs, 
        checkpoint_dir, 
        train_unet_only=True,
        start_epoch=0,
        global_step=0,
        best_metrics=None
    ):
        """Train the diffusion model."""
        logger.info("Starting diffusion model training")
        logger.info(f"Training {'UNet only' if train_unet_only else 'all components'}")
        
        # Test dataloader by extracting first batch
        logger.info("Testing diffusion dataloader by extracting first batch...")
        
        # Try to get the first batch
        try:
            first_batch = next(iter(self.train_loader))
            logger.info(f"First batch loaded successfully")
            
            # Debug: Try a forward pass
            with torch.no_grad():
                loss, metrics = self.model.training_step(first_batch, train_unet_only)
                logger.info(f"Forward pass successful. Loss: {loss.item()}")
            
            # Free memory
            del first_batch
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in diffusion dataloader test: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError("Failed to test diffusion dataloader - check configuration")
        
        # Early stopping setup
        early_stopping_path = os.path.join(checkpoint_dir, "best_diffusion.pt")
        early_stopping = EarlyStopping(
            patience=8,
            verbose=True,
            path=early_stopping_path
        )
        
        # Best model tracking
        best_loss = best_metrics.get('val_loss', float('inf')) if best_metrics else float('inf')
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Starting diffusion epoch {epoch+1}/{num_epochs}")
            
            # Training
            if train_unet_only:
                self.model.vae.eval()
                self.model.text_encoder.eval()
                self.model.unet.train()
            else:
                self.model.vae.train()
                self.model.text_encoder.train()
                self.model.unet.train()
                
            train_loss = 0.0
            train_diffusion_loss = 0.0
            train_vae_loss = 0.0
            
            # Debug counter for batch tracking
            processed_batches = 0
            
            # Train loop with progress bar
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Clear gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    if self.use_amp and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            loss, metrics = self.model.training_step(batch, train_unet_only)
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        if train_unet_only:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=1.0)
                        else:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                list(self.model.vae.parameters()) +
                                list(self.model.text_encoder.parameters()) +
                                list(self.model.unet.parameters()),
                                max_norm=1.0
                            )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss, metrics = self.model.training_step(batch, train_unet_only)
                        
                        loss.backward()
                        
                        # Gradient clipping
                        if train_unet_only:
                            torch.nn.utils.clip_grad_norm_(self.model.unet.parameters(), max_norm=1.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                list(self.model.vae.parameters()) +
                                list(self.model.text_encoder.parameters()) +
                                list(self.model.unet.parameters()),
                                max_norm=1.0
                            )
                        
                        self.optimizer.step()
                    
                    # Update learning rate
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Update global step
                    global_step += 1
                    
                    # Update metrics
                    train_loss += metrics['total_loss']
                    train_diffusion_loss += metrics['diffusion_loss']
                    if 'vae_loss' in metrics:
                        train_vae_loss += metrics['vae_loss']
                    
                    # Update processed batches counter
                    processed_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{metrics['total_loss']:.4f}",
                        'diff': f"{metrics['diffusion_loss']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.6f}" if self.scheduler else "N/A"
                    })
                    
                except Exception as e:
                    logger.error(f"Error in diffusion training batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Calculate average training losses
            train_loss /= max(1, len(self.train_loader))
            train_diffusion_loss /= max(1, len(self.train_loader))
            if not train_unet_only:
                train_vae_loss /= max(1, len(self.train_loader))
            
            # Validation
            self.model.vae.eval()
            self.model.text_encoder.eval()
            self.model.unet.eval()
            
            val_loss = 0.0
            val_diffusion_loss = 0.0
            val_vae_loss = 0.0
            
            with torch.no_grad():
                # Validation loop with progress bar
                val_progress = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
                for batch_idx, batch in enumerate(val_progress):
                    try:
                        # Compute validation metrics
                        metrics = self.model.validation_step(batch)
                        
                        # Update metrics
                        val_loss += metrics['val_loss']
                        val_diffusion_loss += metrics['val_diffusion_loss']
                        val_vae_loss += metrics['val_vae_loss']
                        
                    except Exception as e:
                        logger.error(f"Error in diffusion validation batch {batch_idx}: {e}")
                        continue
            
            # Calculate average validation losses
            val_loss /= max(1, len(self.val_loader))
            val_diffusion_loss /= max(1, len(self.val_loader))
            val_vae_loss /= max(1, len(self.val_loader))
            
            # All these post-validation actions should be indented at the same level
            # as the validation code - INSIDE the epoch loop
            # Visualize results
            if Path(checkpoint_dir).exists() and self.tokenizer:
                from PIL import Image
                visualize_epoch_results(
                    epoch,
                    self.model,
                    self.tokenizer,
                    self.val_loader,
                    checkpoint_dir
                )
                
            # Log metrics
            vae_loss_str = f", VAE: {train_vae_loss:.4f}/{val_vae_loss:.4f}" if not train_unet_only else ""
            logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train/Val Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"Diff: {train_diffusion_loss:.4f}/{val_diffusion_loss:.4f}"
                      f"{vae_loss_str}")
            
            # Save checkpoint if enabled
            # Regular checkpoint
            if (epoch + 1) % self.checkpoint_freq == 0 or epoch == num_epochs - 1:
                metrics = {
                    'train_loss': train_loss,
                    'train_diffusion_loss': train_diffusion_loss,
                    'val_loss': val_loss,
                    'val_diffusion_loss': val_diffusion_loss
                }
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    global_step,
                    metrics,
                    checkpoint_dir,
                    is_best=False
                )
            
            # Save if best model
            if val_loss < best_loss:
                best_loss = val_loss
                
                metrics = {
                    'train_loss': train_loss,
                    'train_diffusion_loss': train_diffusion_loss,
                    'val_loss': val_loss,
                    'val_diffusion_loss': val_diffusion_loss
                }
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch + 1,
                    global_step,
                    metrics,
                    checkpoint_dir,
                    is_best=True
                )
                logger.info(f"New best model saved with val_loss={val_loss:.4f}")
            
            # Generate samples every 10 epochs if tokenizer is available
            if self.tokenizer is not None and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
                try:
                    # Sample prompts
                    sample_prompts = [
                        "Normal chest X-ray with clear lungs and no abnormalities.",
                        "Right lower lobe pneumonia with focal consolidation."
                    ]
                    
                    # Generate and save samples
                    logger.info("Generating sample images...")
                    
                    self.model.vae.eval()
                    self.model.text_encoder.eval()
                    self.model.unet.eval()
                    samples_dir = os.path.join(checkpoint_dir, "samples")
                    os.makedirs(samples_dir, exist_ok=True)
                    
                    with torch.no_grad():
                        for i, prompt in enumerate(sample_prompts):
                            results = self.model.sample(
                                prompt,
                                height=256,
                                width=256,
                                num_inference_steps=30,
                                tokenizer=self.tokenizer
                            )
                            
                            # Save image
                            img = results['images'][0]
                            img_np = img.cpu().numpy().transpose(1, 2, 0)
                            img_np = (img_np * 255).astype(np.uint8)
                            if img_np.shape[-1] == 1:
                                img_np = img_np.squeeze(-1)
                                
                            from PIL import Image
                            img_path = os.path.join(samples_dir, f"sample_epoch{epoch+1}_{i}.png")
                            Image.fromarray(img_np).save(img_path)
                            
                    logger.info(f"Saved sample images to {samples_dir}")
                    
                except Exception as e:
                    logger.error(f"Error generating samples: {e}")
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            _, _, _ = load_checkpoint(self.model, None, None, best_path)
            logger.info("Loaded best model from saved checkpoint")
        
        logger.info("Diffusion model training complete")
        
        return self.model