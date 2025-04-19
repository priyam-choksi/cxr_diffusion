# post_process.py
import os
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter, ImageEnhance

from xray_generator.inference import XrayGenerator

# Set up paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "outputs" / "diffusion_checkpoints" / "checkpoint_epoch_480.pt"
OUTPUT_DIR = BASE_DIR / "outputs" / "enhanced_xrays"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test prompt
TEST_PROMPTS = [
    "Normal chest X-ray with clear lungs and no abnormalities.",
    "Right lower lobe pneumonia with focal consolidation.",
    "Bilateral pleural effusions, greater on the right."
]

def apply_windowing(image, window_center=0.5, window_width=0.8):
    """
    Apply window/level adjustment (similar to radiological windowing).
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply windowing formula
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    img_array = np.clip((img_array - min_val) / (max_val - min_val), 0, 1)
    
    return Image.fromarray((img_array * 255).astype(np.uint8))

def apply_edge_enhancement(image, amount=1.5):
    """Apply edge enhancement using unsharp mask."""
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # Create sharpen filter
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(amount)

def apply_median_filter(image, size=3):
    """Apply median filter to reduce noise."""
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Ensure size is valid (odd number)
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
        
    # Apply median filter using numpy instead of PIL for more reliability
    img_array = np.array(image)
    filtered = cv2.medianBlur(img_array, size)
    
    return Image.fromarray(filtered)

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE to enhance contrast."""
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(img_array)
    
    return Image.fromarray(enhanced)

def apply_histogram_equalization(image):
    """Apply histogram equalization to enhance contrast."""
    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    return ImageOps.equalize(image)

def apply_vignette(image, amount=0.85):
    """Apply vignette effect (darker edges) to mimic X-ray effect."""
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32)
    
    # Create vignette mask
    height, width = img_array.shape
    center_x, center_y = width // 2, height // 2
    radius = np.sqrt(width**2 + height**2) / 2
    
    # Create coordinate grid
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create vignette mask
    mask = 1 - amount * (dist_from_center / radius)
    mask = np.clip(mask, 0, 1)
    
    # Apply mask
    img_array = img_array * mask
    
    return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

def enhance_xray(image, params=None):
    """
    Apply a sequence of enhancements to make the image look more like an authentic X-ray.
    """
    # Default parameters
    if params is None:
        params = {
            'window_center': 0.5,
            'window_width': 0.8,
            'edge_amount': 1.3,
            'median_size': 3,
            'clahe_clip': 2.5,
            'clahe_grid': (8, 8),
            'vignette_amount': 0.25,
            'apply_hist_eq': True
        }
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    # 1. Apply windowing for better contrast
    image = apply_windowing(image, params['window_center'], params['window_width'])
    
    # 2. Apply CLAHE for adaptive contrast
    image_np = np.array(image)
    image = apply_clahe(image_np, params['clahe_clip'], params['clahe_grid'])
    
    # 3. Apply median filter to reduce noise
    image = apply_median_filter(image, params['median_size'])
    
    # 4. Apply edge enhancement to highlight lung markings
    image = apply_edge_enhancement(image, params['edge_amount'])
    
    # 5. Apply histogram equalization for better grayscale distribution (optional)
    if params['apply_hist_eq']:
        image = apply_histogram_equalization(image)
    
    # 6. Apply vignette effect for authentic X-ray look
    image = apply_vignette(image, params['vignette_amount'])
    
    return image

def generate_and_enhance(generator, prompt, params_list=None):
    """
    Generate an X-ray and apply different enhancement parameter sets.
    """
    # Generate the raw X-ray
    results = generator.generate(prompt=prompt, num_inference_steps=100, guidance_scale=10.0)
    raw_image = results['images'][0]
    
    # Create default parameters if none provided
    if params_list is None:
        params_list = [{
            'window_center': 0.5,
            'window_width': 0.8,
            'edge_amount': 1.3,
            'median_size': 3,
            'clahe_clip': 2.5,
            'clahe_grid': (8, 8),
            'vignette_amount': 0.25,
            'apply_hist_eq': True
        }]
    
    # Apply different enhancement parameters
    enhanced_images = []
    for i, params in enumerate(params_list):
        enhanced = enhance_xray(raw_image, params)
        enhanced_images.append({
            'image': enhanced,
            'params': params,
            'index': i+1
        })
    
    return {
        'raw_image': raw_image,
        'enhanced_images': enhanced_images,
        'prompt': prompt
    }

def save_results(results, output_dir):
    """Save all generated and enhanced images."""
    prompt_clean = results['prompt'].replace(" ", "_").replace(".", "").lower()[:30]
    
    # Save raw image
    raw_path = Path(output_dir) / f"raw_{prompt_clean}.png"
    results['raw_image'].save(raw_path)
    
    # Save enhanced images
    for item in results['enhanced_images']:
        enhanced_path = Path(output_dir) / f"enhanced_{item['index']}_{prompt_clean}.png"
        item['image'].save(enhanced_path)
        
        # Save parameters as json
        params_path = Path(output_dir) / f"params_{item['index']}_{prompt_clean}.txt"
        with open(params_path, 'w') as f:
            for key, value in item['params'].items():
                f.write(f"{key}: {value}\n")
    
    return raw_path

def display_results(results):
    """Display the raw and enhanced images for comparison."""
    n_enhanced = len(results['enhanced_images'])
    fig, axes = plt.subplots(1, n_enhanced+1, figsize=(4*(n_enhanced+1), 4))
    
    # Plot raw image
    axes[0].imshow(results['raw_image'], cmap='gray')
    axes[0].set_title("Original (Raw)")
    axes[0].axis('off')
    
    # Plot enhanced images
    for i, item in enumerate(results['enhanced_images']):
        axes[i+1].imshow(item['image'], cmap='gray')
        axes[i+1].set_title(f"Enhanced {item['index']}")
        axes[i+1].axis('off')
    
    plt.suptitle(f"Prompt: {results['prompt']}")
    plt.tight_layout()
    return fig

def main():
    """Main function to load model and generate enhanced X-rays."""
    # Initialize generator with the epoch 480 model
    print(f"Loading model from: {MODEL_PATH}")
    generator = XrayGenerator(
        model_path=str(MODEL_PATH),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Different parameter sets to try
    params_sets = [
        # Parameter Set 1: Balanced enhancement
        {
            'window_center': 0.5,
            'window_width': 0.8,
            'edge_amount': 1.3, 
            'median_size': 3,
            'clahe_clip': 2.5,
            'clahe_grid': (8, 8),
            'vignette_amount': 0.25,
            'apply_hist_eq': True
        },
        # Parameter Set 2: More contrast
        {
            'window_center': 0.45,
            'window_width': 0.7,
            'edge_amount': 1.5,
            'median_size': 3,
            'clahe_clip': 3.0,
            'clahe_grid': (8, 8),
            'vignette_amount': 0.3,
            'apply_hist_eq': True
        },
        # Parameter Set 3: Sharper lung markings
        {
            'window_center': 0.55,
            'window_width': 0.85,
            'edge_amount': 1.8,
            'median_size': 3,
            'clahe_clip': 2.0,
            'clahe_grid': (6, 6),
            'vignette_amount': 0.2,
            'apply_hist_eq': False
        }
    ]
    
    # Process each prompt
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"Processing prompt {i+1}/{len(TEST_PROMPTS)}: {prompt}")
        
        # Generate and enhance images
        results = generate_and_enhance(generator, prompt, params_sets)
        
        # Save results
        output_path = save_results(results, OUTPUT_DIR)
        print(f"Saved results to {output_path.parent}")
        
        # Display results (save figure)
        fig = display_results(results)
        fig_path = Path(OUTPUT_DIR) / f"comparison_{i+1}.png"
        fig.savefig(fig_path)
        plt.close(fig)

if __name__ == "__main__":
    main()