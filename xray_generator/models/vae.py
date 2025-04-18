# xray_generator/models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import SelfAttention

class VAEEncoder(nn.Module):
    """Encoder for VAE with attention mechanisms."""
    def __init__(
        self,
        in_channels=1,
        latent_channels=4,
        hidden_dims=[64, 128, 256, 512],
        attention_resolutions=[32, 16]
    ):
        """Initialize VAE encoder."""
        super().__init__()
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        
        # Create downsampling blocks
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            # Determine resolution
            resolution = 256 // (2 ** i)
            use_attention = resolution in attention_resolutions
            
            block = []
            
            # Add attention if needed
            if use_attention:
                block.append(SelfAttention(in_dim))
            
            # Convolution with GroupNorm and activation
            block.append(nn.Sequential(
                nn.GroupNorm(8, in_dim),
                nn.SiLU(),
                nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)
            ))
            
            self.down_blocks.append(nn.Sequential(*block))
        
        # Final layers
        self.final = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], latent_channels * 2, 3, padding=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights with Kaiming normal."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through encoder."""
        # Initial convolution
        x = self.conv_in(x)
        
        # Downsampling
        for block in self.down_blocks:
            x = block(x)
        
        # Final layers
        x = self.final(x)
        
        # Split into mu and logvar
        mu, logvar = torch.chunk(x, 2, dim=1)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """Decoder for VAE with attention mechanisms."""
    def __init__(
        self,
        latent_channels=4,
        out_channels=1,
        hidden_dims=[512, 256, 128, 64],
        attention_resolutions=[16, 32]
    ):
        """Initialize VAE decoder."""
        super().__init__()
        
        # Input convolution
        self.conv_in = nn.Conv2d(latent_channels, hidden_dims[0], 3, padding=1)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        # Create upsampling blocks
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            
            # Determine resolution
            resolution = 16 * (2 ** i)  # Starting at 16x16 for latent space
            use_attention = resolution in attention_resolutions
            
            block = []
            
            # Add attention if needed
            if use_attention:
                block.append(SelfAttention(in_dim))
            
            # Add upsampling
            block.append(nn.Sequential(
                nn.GroupNorm(8, in_dim),
                nn.SiLU(),
                nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1)
            ))
            
            self.up_blocks.append(nn.Sequential(*block))
        
        # Final layers
        self.final = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights with Kaiming normal."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through decoder."""
        # Initial convolution
        x = self.conv_in(x)
        
        # Upsampling
        for block in self.up_blocks:
            x = block(x)
        
        # Final layers
        x = self.final(x)
        
        return x

class MedicalVAE(nn.Module):
    """Complete VAE model for medical images."""
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        hidden_dims=[64, 128, 256, 512],
        attention_resolutions=[16, 32]
    ):
        """Initialize VAE."""
        super().__init__()
        
        # Create encoder and decoder
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            hidden_dims=hidden_dims,
            attention_resolutions=attention_resolutions
        )
        
        self.decoder = VAEDecoder(
            latent_channels=latent_channels,
            out_channels=out_channels,
            hidden_dims=list(reversed(hidden_dims)),
            attention_resolutions=attention_resolutions
        )
        
        # Save parameters
        self.latent_channels = latent_channels
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """Forward pass through the VAE."""
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z)
        
        return recon, mu, logvar