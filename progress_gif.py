import os
import glob
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np
from pathlib import Path
import re

def create_epoch_progress_gif(
    visualizations_dir,
    output_path,
    image_pattern,  # Pattern to match - gen_1.png, gen_2.png, recon_1.png, or recon_2.png
    duration=0.75,  # Duration for each frame in seconds
    prefix='epoch_'  # Prefix for epoch folders
):
    """
    Create a GIF showing training progress across epochs for a specific image type.
    
    Parameters:
    - visualizations_dir: Directory containing epoch folders
    - output_path: Path to save the output GIF
    - image_pattern: Specific image file to use from each epoch folder
    - duration: Duration for each frame in seconds
    - prefix: Prefix for epoch folders
    """
    print(f"Creating progress GIF for {image_pattern} in {visualizations_dir}")
    
    # Find all epoch folders
    epoch_folders = [f for f in os.listdir(visualizations_dir) 
                    if os.path.isdir(os.path.join(visualizations_dir, f)) and f.startswith(prefix)]
    
    if not epoch_folders:
        print(f"No epoch folders found in {visualizations_dir}")
        return
    
    print(f"Found {len(epoch_folders)} epoch folders")
    
    # Sort folders by epoch number
    def extract_epoch_number(folder_name):
        match = re.search(rf'{prefix}(\d+)', folder_name)
        if match:
            return int(match.group(1))
        return float('inf')  # For sorting purposes
    
    epoch_folders.sort(key=extract_epoch_number)
    
    # Load images
    images = []
    for folder in epoch_folders:
        folder_path = os.path.join(visualizations_dir, folder)
        image_path = os.path.join(folder_path, image_pattern)
        
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                
                # Add epoch information as text overlay
                draw = ImageDraw.Draw(img)
                
                # Extract epoch number for overlay
                epoch_num = extract_epoch_number(folder)
                
                # Try to get a font, use default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Add text overlay with epoch number
                draw.text((10, 10), f"Epoch {epoch_num}", fill="white", font=font)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                images.append(np.array(img))
                print(f"Added image from {folder}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Image {image_pattern} not found in {folder_path}")
    
    if not images:
        print(f"No valid images found to create GIF for {image_pattern}")
        return
    
    print(f"Creating GIF with {len(images)} frames")
    
    # Create GIF
    imageio.mimsave(output_path, images, duration=duration, loop=0)
    print(f"GIF saved to {output_path}")

# Example usage 
if __name__ == "__main__":
    # For diffusion model progress - with faster playback (0.2s per frame)
    diffusion_dir = 'outputs/diffusion_checkpoints/visualizations'
    
    # Create GIF for the first diffusion prompt (gen_1.png)
    create_epoch_progress_gif(
        visualizations_dir=diffusion_dir,
        output_path='diffusion_progress_gen1.gif',
        image_pattern='gen_1.png',
        duration=0.2  # Faster playback for diffusion (was 0.75)
    )
    
    # Create GIF for the second diffusion prompt (gen_2.png)
    create_epoch_progress_gif(
        visualizations_dir=diffusion_dir,
        output_path='diffusion_progress_gen2.gif',
        image_pattern='gen_2.png',
        duration=0.2  # Faster playback for diffusion (was 0.75)
    )
    
    # For VAE reconstruction progress - keep the original speed
    vae_dir = 'outputs/vae_checkpoints/visualizations'
    
    # Create GIF for the first VAE reconstruction (recon_1.png)
    create_epoch_progress_gif(
        visualizations_dir=vae_dir,
        output_path='vae_progress_recon1.gif',
        image_pattern='recon_1.png',
        duration=0.75  # Original speed for VAE
    )
    
    # Create GIF for the second VAE reconstruction (recon_2.png)
    create_epoch_progress_gif(
        visualizations_dir=vae_dir,
        output_path='vae_progress_recon2.gif',
        image_pattern='recon_2.png',
        duration=0.75  # Original speed for VAE
    )