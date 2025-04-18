# xray_generator/models/diffusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def extract_into_tensor(a, t, shape):
    """Extract specific timestep values and broadcast to target shape."""
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    a = a.to(t.device)
    
    b, *_ = t.shape
    out = a.gather(-1, t)
    while len(out.shape) < len(shape):
        out = out[..., None]
    
    return out.expand(shape)

def get_named_beta_schedule(schedule_type, num_diffusion_steps):
    """
    Get a pre-defined beta schedule for the given name.
    
    Available schedules:
    - linear: linear schedule from Ho et al
    - cosine: cosine schedule from Improved DDPM
    """
    if schedule_type == "linear":
        # Linear schedule from Ho et al.
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_steps, dtype=torch.float32)
    
    elif schedule_type == "cosine":
        # Cosine schedule from Improved DDPM
        steps = num_diffusion_steps + 1
        x = torch.linspace(0, num_diffusion_steps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / num_diffusion_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    elif schedule_type == "scaled_linear":
        # Scaled linear schedule
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_steps, dtype=torch.float32) ** 2
    
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_type}")

class DiffusionModel:
    """
    Diffusion model for medical image generation.
    Combines VAE, UNet, and text encoder with diffusion process.
    """
    def __init__(
        self,
        vae,
        unet,
        text_encoder,
        scheduler_type="ddpm",
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        guidance_scale=7.5,
        device=None
    ):
        """Initialize diffusion model."""
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.guidance_scale = guidance_scale
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize diffusion parameters
        self._initialize_diffusion_parameters()

        logger.info(f"Initialized diffusion model with {scheduler_type} scheduler, {beta_schedule} beta schedule")

    def _initialize_diffusion_parameters(self):
        """Initialize diffusion parameters."""
        # Get beta schedule
        self.betas = get_named_beta_schedule(
            self.beta_schedule, self.num_train_timesteps
        ).to(self.device)
        
        # Calculate alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]])
        
        # Calculate diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        
        # Calculate posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from noise."""
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        sqrt_recip_alphas_cumprod_t = extract_into_tensor(sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract_into_tensor(sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute posterior mean and variance: q(x_{t-1} | x_t, x_0)."""
        posterior_mean_coef1_t = extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape)
        posterior_mean_coef2_t = extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape)
        
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        posterior_variance_t = extract_into_tensor(self.posterior_variance, t, x_start.shape)
        posterior_log_variance_t = extract_into_tensor(self.posterior_log_variance_clipped, t, x_start.shape)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_t
    
    def p_mean_variance(self, x_t, t, context):
        """Predict mean and variance for the denoising process."""
        # Predict noise using UNet
        noise_pred = self.unet(x_t, t, context)
        
        # Predict x_0
        x_0 = self.predict_start_from_noise(x_t, t, noise_pred)
        
        # Clip prediction
        x_0 = torch.clamp(x_0, -1.0, 1.0)
        
        # Get posterior parameters
        mean, var, log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        
        return mean, var, log_var
    
    def p_sample(self, x_t, t, context):
        """Sample from p(x_{t-1} | x_t)."""
        # Get mean and variance
        mean, _, log_var = self.p_mean_variance(x_t, t, context)
        
        # Sample
        noise = torch.randn_like(x_t)
        mask = (t > 0).float().reshape(-1, *([1] * (len(x_t.shape) - 1)))
        
        return mean + mask * torch.exp(0.5 * log_var) * noise
    
    def ddim_sample(self, x_t, t, prev_t, context, eta=0.0):
        """DDIM sampling step."""
        # Get alphas
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t]
        
        # Predict noise
        noise_pred = self.unet(x_t, t, context)
        
        # Predict x_0
        x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
        
        # Clip prediction
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # DDIM formula
        variance = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        
        # Mean component
        mean = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev - variance**2) * noise_pred
        
        # Add noise if eta > 0
        noise = torch.randn_like(x_t)
        x_prev = mean
        
        if eta > 0:
            x_prev = x_prev + variance * noise
            
        return x_prev
    
    def training_step(self, batch, train_unet_only=True):
        """Training step for diffusion model."""
        # Extract data
        images = batch['image'].to(self.device)
        input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
        attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
        
        if input_ids is None or attention_mask is None:
            raise ValueError("Batch must contain tokenized text")
        
        # Metrics dictionary
        metrics = {}
        
        try:
            # Encode images to latent space
            with torch.set_grad_enabled(not train_unet_only):
                # Get latent distribution
                mu, logvar = self.vae.encode(images)
                
                # Use latent mean for stability in early training
                latents = mu
                
                # Scale latents 
                latents = latents * 0.18215
                
                # Compute VAE loss if not training UNet only
                if not train_unet_only:
                    recon, mu, logvar = self.vae(images)
                    
                    # Reconstruction loss
                    recon_loss = F.mse_loss(recon, images)
                    
                    # KL divergence
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # Total VAE loss
                    vae_loss_val = recon_loss + 1e-4 * kl_loss
                    
                    metrics['vae_loss'] = vae_loss_val.item()
                    metrics['recon_loss'] = recon_loss.item()
                    metrics['kl_loss'] = kl_loss.item()
                    
            # Encode text
            with torch.set_grad_enabled(not train_unet_only):
                context = self.text_encoder(input_ids, attention_mask)
                
            # Sample timestep
            batch_size = images.shape[0]
            t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device).long()
            
            # Generate noise
            noise = torch.randn_like(latents)
            
            # Add noise to latents (forward diffusion)
            noisy_latents = self.q_sample(latents, t, noise=noise)
            
            # Sometimes train with empty context (10% of the time)
            import random
            if random.random() < 0.1:
                context = torch.zeros_like(context)
                
            # Predict noise
            noise_pred = self.unet(noisy_latents, t, context)
            
            # Compute loss based on prediction type
            if self.prediction_type == "epsilon":
                # Predict noise (ε)
                diffusion_loss = F.mse_loss(noise_pred, noise)
                
            elif self.prediction_type == "v_prediction":
                # Predict velocity (v)
                velocity = self.sqrt_alphas_cumprod[t] * noise - self.sqrt_one_minus_alphas_cumprod[t] * latents
                diffusion_loss = F.mse_loss(noise_pred, velocity)
                
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")
                
            metrics['diffusion_loss'] = diffusion_loss.item()
            
            # Total loss
            if train_unet_only:
                total_loss = diffusion_loss
            else:
                total_loss = diffusion_loss + vae_loss_val
                
            metrics['total_loss'] = total_loss.item()
            
            return total_loss, metrics
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return dummy values to avoid breaking training loop
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss, {'total_loss': 0.0, 'diffusion_loss': 0.0}
    
    def validation_step(self, batch):
        """Validation step for diffusion model."""
        with torch.no_grad():
            # Extract data
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device) if 'input_ids' in batch else None
            attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
            
            if input_ids is None or attention_mask is None:
                raise ValueError("Batch must contain tokenized text")
            
            try:
                # Encode images to latent space
                mu, logvar = self.vae.encode(images)
                latents = mu  # Use mean for validation
                
                # Scale latents
                latents = latents * 0.18215
                
                # Compute VAE loss
                recon, mu, logvar = self.vae(images)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(recon, images)
                
                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                # Total VAE loss
                vae_loss_val = recon_loss + 1e-4 * kl_loss
                
                # Encode text
                context = self.text_encoder(input_ids, attention_mask)
                
                # Sample timestep
                batch_size = images.shape[0]
                t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device).long()
                
                # Generate noise
                noise = torch.randn_like(latents)
                
                # Add noise to latents
                noisy_latents = self.q_sample(latents, t, noise=noise)
                
                # Predict noise
                noise_pred = self.unet(noisy_latents, t, context)
                
                # Compute diffusion loss
                if self.prediction_type == "epsilon":
                    diffusion_loss = F.mse_loss(noise_pred, noise)
                elif self.prediction_type == "v_prediction":
                    velocity = self.sqrt_alphas_cumprod[t] * noise - self.sqrt_one_minus_alphas_cumprod[t] * latents
                    diffusion_loss = F.mse_loss(noise_pred, velocity)
                
                # Total loss
                total_loss = diffusion_loss + vae_loss_val
                
                # Return metrics
                return {
                    'val_loss': total_loss.item(),
                    'val_diffusion_loss': diffusion_loss.item(),
                    'val_vae_loss': vae_loss_val.item(),
                    'val_recon_loss': recon_loss.item(),
                    'val_kl_loss': kl_loss.item()
                }
                
            except Exception as e:
                logger.error(f"Error in validation step: {e}")
                
                # Return dummy metrics
                return {
                    'val_loss': 0.0,
                    'val_diffusion_loss': 0.0,
                    'val_vae_loss': 0.0
                }
    
    @torch.no_grad()
    def sample(
        self,
        text,
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=None,
        eta=0.0,
        tokenizer=None,
        latents=None,
        return_all_latents=False
    ):
        """Sample from diffusion model given text prompt."""
        # Default guidance scale
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        
        # Check if tokenizer is provided
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for sampling")
        
        # Encode text
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=256,  # Replace with your max token length
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        context = self.text_encoder(tokens.input_ids, tokens.attention_mask)
        
        # Calculate latent size
        latent_height = height // 8  # VAE downsampling factor
        latent_width = width // 8
        
        # Generate random latents if not provided
        if latents is None:
            latents = torch.randn(
                (batch_size, self.vae.latent_channels, latent_height, latent_width),
                device=self.device
            )
            latents = latents * 0.18215  # Scale factor
        
        # Store all latents if requested
        if return_all_latents:
            all_latents = [latents.clone()]
        
        # Prepare scheduler timesteps
        if self.scheduler_type == "ddim":
            # DDIM timesteps
            timesteps = torch.linspace(
                self.num_train_timesteps - 1,
                0,
                num_inference_steps,
                dtype=torch.long,
                device=self.device
            )
        else:
            # DDPM timesteps
            step_indices = list(range(0, self.num_train_timesteps, self.num_train_timesteps // num_inference_steps))
            timesteps = torch.tensor(sorted(step_indices, reverse=True), dtype=torch.long, device=self.device)
        
        # Text embeddings for classifier-free guidance
        uncond_context = torch.zeros_like(context)
        
        # Sampling loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating image")):
            # Expand for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            t_input = torch.cat([t.unsqueeze(0)] * 2 * batch_size)
            
            # Get text conditioning
            text_embeddings = torch.cat([uncond_context, context])
            
            # Predict noise
            noise_pred = self.unet(latent_model_input, t_input, text_embeddings)
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Sampling step
            if self.scheduler_type == "ddim":
                # DDIM step
                prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor([0], device=self.device)
                latents = self.ddim_sample(latents, t.repeat(batch_size), prev_t.repeat(batch_size), context, eta)
            else:
                # DDPM step
                latents = self.p_sample(latents, t.repeat(batch_size), context)
            
            # Store latent if requested
            if return_all_latents:
                all_latents.append(latents.clone())
        
        # Scale latents
        latents = 1 / 0.18215 * latents
        
        # Decode latents
        images = self.vae.decode(latents)
        
        # Normalize to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        result = {
            'images': images,
            'latents': latents
        }
        
        if return_all_latents:
            result['all_latents'] = all_latents
            
        return result