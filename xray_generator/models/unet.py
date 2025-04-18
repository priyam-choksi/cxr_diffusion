# xray_generator/models/unet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimeEmbedding(nn.Module):
    """Time embedding module for diffusion models."""
    def __init__(self, dim, dim_out=None):
        """Initialize time embedding."""
        super().__init__()
        if dim_out is None:
            dim_out = dim
            
        self.dim = dim
        
        # Linear layers for time embedding
        self.main = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim_out)
        )
    
    def forward(self, time):
        """Forward pass through time embedding."""
        time_emb = timestep_embedding(time, self.dim)
        return self.main(time_emb)

class SelfAttention(nn.Module):
    """Self-attention module for VAE and UNet."""
    def __init__(self, channels, num_heads=8):
        """Initialize self-attention module."""
        super().__init__()
        assert channels % num_heads == 0, f"Channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)
        
        # Normalization
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x):
        """Forward pass through self-attention."""
        b, c, h, w = x.shape
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Get QKV
        qkv = self.to_qkv(x_norm).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.num_heads), qkv)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Combine
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        # Project to output
        out = self.to_out(out)
        
        # Add residual
        return out + x

class CrossAttention(nn.Module):
    """Cross-attention module for conditioning on text."""
    def __init__(self, channels, text_dim, num_heads=8):
        """Initialize cross-attention module."""
        super().__init__()
        assert channels % num_heads == 0, f"Channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from image features
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        # Key and value from text
        self.to_k = nn.Linear(text_dim, channels, bias=False)
        self.to_v = nn.Linear(text_dim, channels, bias=False)
        
        self.to_out = nn.Conv2d(channels, channels, 1)
        
        # Normalization
        self.norm = nn.GroupNorm(8, channels)
    
    def forward(self, x, context):
        """Forward pass through cross-attention."""
        b, c, h, w = x.shape
        
        # Apply normalization
        x_norm = self.norm(x)
        
        # Get query from image features
        q = self.to_q(x_norm)
        q = rearrange(q, 'b c h w -> b (h w) c')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Get key and value from text context
        k = self.to_k(context)
        v = self.to_v(context)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Combine
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        # Project to output
        out = self.to_out(out)
        
        # Add residual
        return out + x

class ResnetBlock(nn.Module):
    """Residual block with time embedding and optional attention."""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        time_channels, 
        dropout=0.0,
        use_attention=False,
        attention_type="self",
        text_dim=None
    ):
        """Initialize residual block."""
        super().__init__()
        
        # First convolution block
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # Time embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # Second convolution block
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # Attention
        self.use_attention = use_attention
        if use_attention:
            if attention_type == "self":
                self.attention = SelfAttention(out_channels)
            elif attention_type == "cross":
                assert text_dim is not None, "Text dimension required for cross-attention"
                self.attention = CrossAttention(out_channels, text_dim)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb, context=None):
        """Forward pass through residual block."""
        # Shortcut
        shortcut = self.shortcut(x)
        
        # Block 1
        h = self.block1(x)
        
        # Add time embedding
        h += self.time_emb(time_emb)[:, :, None, None]
        
        # Block 2
        h = self.block2(h)
        
        # Apply attention
        if self.use_attention:
            if isinstance(self.attention, CrossAttention) and context is not None:
                h = self.attention(h, context)
            else:
                h = self.attention(h)
        
        # Add shortcut
        return h + shortcut

class Downsample(nn.Module):
    """Downsampling layer for UNet."""
    def __init__(self, channels, use_conv=True):
        """Initialize downsampling layer."""
        super().__init__()
        if use_conv:
            self.downsample = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.downsample = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        """Forward pass through downsampling layer."""
        return self.downsample(x)

class Upsample(nn.Module):
    """Upsampling layer for UNet."""
    def __init__(self, channels, use_conv=True):
        """Initialize upsampling layer."""
        super().__init__()
        self.upsample = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        """Forward pass through upsampling layer."""
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x

class DiffusionUNet(nn.Module):
    """UNet model for diffusion process with cross-attention for text conditioning."""
    def __init__(
        self,
        in_channels=4,
        model_channels=64,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=(8, 16, 32),
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        context_dim=768
    ):
        """Initialize UNet model."""
        super().__init__()
        
        # Parameters
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.context_dim = context_dim
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = TimeEmbedding(model_channels, time_embed_dim)
        
        # Input block
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # Keep track of channels for skip connections
        input_block_channels = [model_channels]
        ch = model_channels
        ds = 1  # Downsampling factor
        
        # Downsampling blocks
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # Use cross-attention if at an attention resolution
                use_attention = ds in attention_resolutions
                
                # Create block
                block = ResnetBlock(
                    ch,
                    model_channels * mult,
                    time_embed_dim,
                    dropout,
                    use_attention,
                    "cross" if use_attention else None,
                    context_dim if use_attention else None
                )
                
                # Add to input blocks
                self.input_blocks.append(block)
                
                # Update channels
                ch = model_channels * mult
                input_block_channels.append(ch)
            
            # Add downsampling except for last level
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_channels.append(ch)
                ds *= 2
        
        # Middle blocks (bottleneck) with cross-attention
        self.middle_block = nn.ModuleList([
            ResnetBlock(
                ch, ch, time_embed_dim, dropout, True, "cross", context_dim
            ),
            ResnetBlock(
                ch, ch, time_embed_dim, dropout, False
            )
        ])
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Combine with skip connection
                skip_ch = input_block_channels.pop()
                
                # Use cross-attention if at an attention resolution
                use_attention = ds in attention_resolutions
                
                # Create block
                block = ResnetBlock(
                    ch + skip_ch,
                    model_channels * mult,
                    time_embed_dim,
                    dropout,
                    use_attention,
                    "cross" if use_attention else None,
                    context_dim if use_attention else None
                )
                
                # Add to output blocks
                self.output_blocks.append(block)
                
                # Update channels
                ch = model_channels * mult
                
                # Add upsampling except for last block of last level
                if level != 0 and i == num_res_blocks:
                    self.output_blocks.append(Upsample(ch))
                    ds //= 2
        
        # Final layers
        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, timesteps, context=None):
        """Forward pass through UNet."""
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Input blocks (downsampling)
        h = x
        hs = [h]  # Store intermediate activations for skip connections
        
        for module in self.input_blocks:
            if isinstance(module, ResnetBlock):
                h = module(h, t_emb, context)
            else:
                h = module(h)
            hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            h = module(h, t_emb, context) if isinstance(module, ResnetBlock) else module(h)
        
        # Output blocks (upsampling)
        for module in self.output_blocks:
            if isinstance(module, ResnetBlock):
                # Add skip connection
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, t_emb, context)
            else:
                h = module(h)
        
        # Final output
        return self.out(h)