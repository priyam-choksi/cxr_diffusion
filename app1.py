import os
import json
import torch
import numpy as np
import streamlit as st
import pandas as pd
import time
import random
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import cv2
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance
import seaborn as sns

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# App configuration
st.set_page_config(
    page_title="Advanced X-Ray Research Console",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure paths
BASE_DIR = Path(__file__).parent
CHECKPOINTS_DIR = BASE_DIR / "outputs" / "diffusion_checkpoints" 
VAE_CHECKPOINTS_DIR = BASE_DIR / "outputs" / "vae_checkpoints"
DEFAULT_MODEL_PATH = str(CHECKPOINTS_DIR / "best_model.pt")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BASE_DIR / "outputs" / "generated"))
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
DATASET_PATH = os.environ.get("DATASET_PATH", str(BASE_DIR / "dataset"))

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Path to saved metrics from evaluate_model.py
DIFFUSION_METRICS_PATH = os.path.join(METRICS_DIR, 'diffusion_metrics.json')
MODEL_SUMMARY_PATH = os.path.join(METRICS_DIR, 'model_summary.md')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

# =============================================================================
# METRICS LOADING FUNCTIONS
# =============================================================================

def load_saved_metrics():
    """Load metrics saved by the evaluation script"""
    metrics = {}
    
    # Check if diffusion metrics file exists
    if os.path.exists(DIFFUSION_METRICS_PATH):
        try:
            with open(DIFFUSION_METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            st.success(f"Loaded pre-computed metrics from {DIFFUSION_METRICS_PATH}")
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    else:
        st.warning(f"No pre-computed metrics found at {DIFFUSION_METRICS_PATH}")
        st.info("Please run 'evaluate_model.py' first to generate metrics.")
    
    return metrics

def load_model_summary():
    """Load the human-readable model summary"""
    if os.path.exists(MODEL_SUMMARY_PATH):
        try:
            with open(MODEL_SUMMARY_PATH, 'r') as f:
                summary = f.read()
            return summary
        except Exception as e:
            st.error(f"Error loading model summary: {e}")
    
    return None

def get_available_visualizations():
    """Get all available visualizations saved by the evaluation script"""
    visualizations = {}
    
    if os.path.exists(VISUALIZATIONS_DIR):
        # Get all image files
        for file in os.listdir(VISUALIZATIONS_DIR):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                vis_path = os.path.join(VISUALIZATIONS_DIR, file)
                vis_name = file.replace('.png', '').replace('_', ' ').title()
                visualizations[vis_name] = vis_path
                
        # Also check subdirectories
        for subdir in ['noise_levels', 'text_conditioning']:
            subdir_path = os.path.join(VISUALIZATIONS_DIR, subdir)
            if os.path.exists(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        vis_path = os.path.join(subdir_path, file)
                        vis_name = f"{subdir.replace('_', ' ').title()} - {file.replace('.png', '').replace('_', ' ').title()}"
                        visualizations[vis_name] = vis_path
    
    return visualizations

def load_samples():
    """Load generated samples from the evaluation script"""
    samples = []
    samples_dir = os.path.join(OUTPUT_DIR, 'samples')
    
    if os.path.exists(samples_dir):
        # Get all image files
        for i in range(1, 10):  # Check up to 10 samples
            img_path = os.path.join(samples_dir, f"sample_{i}.png")
            prompt_path = os.path.join(samples_dir, f"prompt_{i}.txt")
            
            if os.path.exists(img_path) and os.path.exists(prompt_path):
                # Load prompt
                with open(prompt_path, 'r') as f:
                    prompt = f.read()
                
                samples.append({
                    'image_path': img_path,
                    'prompt': prompt
                })
    
    return samples

# =============================================================================
# METRICS VISUALIZATION FUNCTIONS
# =============================================================================

def plot_parameter_counts(metrics):
    """Plot parameter counts by component"""
    if 'parameters' not in metrics:
        return None
    
    params = metrics['parameters']
    
    # Extract parameter counts
    components = ['VAE', 'UNet', 'Text Encoder']
    total_params = [
        params.get('vae_total', 0),
        params.get('unet_total', 0),
        params.get('text_encoder_total', 0)
    ]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(components, total_params, color=['lightpink', 'lightgreen', 'lightblue'])
    
    # Add parameter counts as labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height/1e6:.1f}M',
                ha='center', va='bottom')
    
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Model Parameter Distribution')
    
    return fig

def plot_beta_schedule(metrics):
    """Plot beta schedule from metrics"""
    if 'beta_schedule' not in metrics:
        return None
    
    # Check if visualization exists
    vis_path = os.path.join(VISUALIZATIONS_DIR, 'beta_schedule.png')
    if os.path.exists(vis_path):
        img = Image.open(vis_path)
        return img
    
    # Otherwise create a simple plot of key values
    beta_info = metrics['beta_schedule']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot min, mean, and max as horizontal lines
    x = np.arange(3)
    values = [beta_info.get('min', 0), beta_info.get('mean', 0), beta_info.get('max', 0)]
    
    ax.bar(x, values, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Min', 'Mean', 'Max'])
    ax.set_ylabel('Beta Value')
    ax.set_title('Beta Schedule Summary')
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.6f}', ha='center', va='bottom')
    
    return fig

def plot_inference_speed(metrics):
    """Plot inference speed metrics"""
    if 'inference_speed' not in metrics:
        return None
    
    # Check if visualization exists
    vis_path = os.path.join(VISUALIZATIONS_DIR, 'inference_time.png')
    if os.path.exists(vis_path):
        img = Image.open(vis_path)
        return img
    
    # Otherwise create a simple summary plot
    speed = metrics['inference_speed']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot average, min, and max
    x = np.arange(3)
    values = [
        speed.get('avg_inference_time_ms', 0),
        speed.get('min_inference_time_ms', 0),
        speed.get('max_inference_time_ms', 0)
    ]
    
    ax.bar(x, values, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Average', 'Min', 'Max'])
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Speed Summary')
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.2f} ms', ha='center', va='bottom')
    
    return fig

def plot_vae_latent_stats(metrics):
    """Plot VAE latent space statistics"""
    if 'vae_latent' not in metrics:
        return None
    
    latent = metrics['vae_latent']
    
    # Create a plot with key statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract statistics
    keys = ['mean', 'std', 'min', 'max']
    values = [latent.get(k, 0) for k in keys]
    
    ax.bar(keys, values, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('VAE Latent Space Statistics')
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    return fig

def display_architecture_info(metrics):
    """Display model architecture information"""
    if 'architecture' not in metrics:
        return
    
    arch = metrics['architecture']
    
    # Create separate tables for each component
    col1, col2 = st.columns(2)
    
    with col1:
        # VAE architecture
        st.subheader("VAE Architecture")
        vae_data = pd.DataFrame({
            "Property": arch['vae'].keys(),
            "Value": arch['vae'].values()
        })
        st.table(vae_data)
        
        # UNet architecture
        st.subheader("UNet Architecture")
        unet_data = pd.DataFrame({
            "Property": arch['unet'].keys(),
            "Value": arch['unet'].values()
        })
        st.table(unet_data)
    
    with col2:
        # Text encoder architecture
        st.subheader("Text Encoder")
        text_data = pd.DataFrame({
            "Property": arch['text_encoder'].keys(),
            "Value": arch['text_encoder'].values()
        })
        st.table(text_data)
        
        # Diffusion process parameters
        st.subheader("Diffusion Process")
        diff_data = pd.DataFrame({
            "Property": arch['diffusion'].keys(),
            "Value": arch['diffusion'].values()
        })
        st.table(diff_data)

def display_parameter_counts(metrics):
    """Display model parameter counts"""
    if 'parameters' not in metrics:
        return
    
    params = metrics['parameters']
    
    # Display total parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Parameters", f"{params['total']:,}")
    
    with col2:
        st.metric("Trainable Parameters", f"{params['trainable']:,}")
    
    with col3:
        st.metric("Memory Footprint", f"{params['memory_footprint_mb']:.2f} MB")
    
    # Display parameter distribution chart
    fig = plot_parameter_counts(metrics)
    if fig:
        st.pyplot(fig)
    
    # Component breakdown
    st.subheader("Component Breakdown")
    
    component_data = pd.DataFrame({
        "Component": ["VAE", "UNet", "Text Encoder"],
        "Total Parameters": [
            f"{params['vae_total']:,}",
            f"{params['unet_total']:,}",
            f"{params['text_encoder_total']:,}"
        ],
        "Trainable Parameters": [
            f"{params['vae_trainable']:,}",
            f"{params['unet_trainable']:,}",
            f"{params['text_encoder_trainable']:,}"
        ],
        "Percentage of Total": [
            f"{params['vae_total'] / params['total']:.2%}",
            f"{params['unet_total'] / params['total']:.2%}",
            f"{params['text_encoder_total'] / params['total']:.2%}"
        ]
    })
    
    st.table(component_data)

def display_parameter_statistics(metrics):
    """Display parameter statistics by component"""
    if 'parameter_stats' not in metrics:
        return
    
    stats = metrics['parameter_stats']
    
    # Create a table for each component
    for component, comp_stats in stats.items():
        st.subheader(f"{component.replace('_', ' ').title()} Parameters")
        
        stats_data = pd.DataFrame({
            "Statistic": comp_stats.keys(),
            "Value": comp_stats.values()
        })
        
        st.table(stats_data)

def display_checkpoint_metadata(metrics):
    """Display checkpoint metadata"""
    if 'checkpoint_metadata' not in metrics:
        return
    
    meta = metrics['checkpoint_metadata']
    
    # Display basic training information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'epoch' in meta:
            st.metric("Training Epochs", meta['epoch'])
    
    with col2:
        if 'global_step' in meta:
            st.metric("Global Steps", meta['global_step'])
    
    with col3:
        if 'learning_rate' in meta:
            st.metric("Learning Rate", meta['learning_rate'])
    
    # Display best metrics if available
    if 'best_metrics' in meta:
        st.subheader("Best Metrics")
        
        best = meta['best_metrics']
        best_data = pd.DataFrame({
            "Metric": best.keys(),
            "Value": best.values()
        })
        
        st.table(best_data)
    
    # Display config if available
    if 'config' in meta:
        with st.expander("Training Configuration"):
            config = meta['config']
            config_data = pd.DataFrame({
                "Parameter": config.keys(),
                "Value": config.values()
            })
            
            st.table(config_data)

def display_inference_performance(metrics):
    """Display inference performance metrics"""
    if 'inference_speed' not in metrics:
        return
    
    speed = metrics['inference_speed']
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Inference Time", f"{speed['avg_inference_time_ms']:.2f} ms")
    
    with col2:
        st.metric("Min Inference Time", f"{speed['min_inference_time_ms']:.2f} ms")
    
    with col3:
        st.metric("Max Inference Time", f"{speed['max_inference_time_ms']:.2f} ms")
    
    # Display chart
    fig = plot_inference_speed(metrics)
    if fig:
        st.image(fig)
    
    # Additional details
    st.info(f"Metrics based on {speed['num_runs']} runs with {speed['num_inference_steps']} diffusion steps.")

def display_vae_analysis(metrics):
    """Display VAE latent space analysis"""
    if 'vae_latent' not in metrics:
        return
    
    latent = metrics['vae_latent']
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Latent Dimensions", latent.get('dimensions', 'N/A'))
    
    with col2:
        active_dims = latent.get('active_dimensions', 'N/A')
        active_ratio = latent.get('active_dimensions_ratio', 'N/A')
        st.metric("Active Dimensions", f"{active_dims} ({active_ratio:.2%})")
    
    with col3:
        if 'reconstruction_mse' in latent:
            st.metric("Reconstruction MSE", f"{latent['reconstruction_mse']:.6f}")
    
    # Display latent space statistics
    fig = plot_vae_latent_stats(metrics)
    if fig:
        st.pyplot(fig)
    
    # Check for t-SNE visualization
    tsne_path = os.path.join(VISUALIZATIONS_DIR, 'vae_latent_tsne.png')
    if os.path.exists(tsne_path):
        st.subheader("t-SNE Visualization of VAE Latent Space")
        st.image(Image.open(tsne_path))
    
    # Check for reconstruction visualization
    recon_path = os.path.join(VISUALIZATIONS_DIR, 'vae_reconstruction.png')
    if os.path.exists(recon_path):
        st.subheader("VAE Reconstruction Examples")
        st.image(Image.open(recon_path))

def display_beta_schedule_analysis(metrics):
    """Display beta schedule analysis"""
    if 'beta_schedule' not in metrics:
        return
    
    beta_info = metrics['beta_schedule']
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Min Beta", f"{beta_info['min']:.6f}")
    
    with col2:
        st.metric("Mean Beta", f"{beta_info['mean']:.6f}")
    
    with col3:
        st.metric("Max Beta", f"{beta_info['max']:.6f}")
    
    # Display alphas cumprod metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Min Alpha Cumprod", f"{beta_info['alphas_cumprod_min']:.6f}")
    
    with col2:
        st.metric("Max Alpha Cumprod", f"{beta_info['alphas_cumprod_max']:.6f}")
    
    # Check for beta schedule visualization
    beta_path = os.path.join(VISUALIZATIONS_DIR, 'beta_schedule.png')
    if os.path.exists(beta_path):
        st.subheader("Beta Schedule")
        st.image(Image.open(beta_path))
    
    # Check for alphas cumprod visualization
    alphas_path = os.path.join(VISUALIZATIONS_DIR, 'alphas_cumprod.png')
    if os.path.exists(alphas_path):
        st.subheader("Alphas Cumulative Product")
        st.image(Image.open(alphas_path))

def display_noise_levels(metrics):
    """Display noise levels visualization"""
    # Check for noise levels grid
    grid_path = os.path.join(VISUALIZATIONS_DIR, 'noise_levels_grid.png')
    if os.path.exists(grid_path):
        st.subheader("Noise Levels at Different Timesteps")
        st.image(Image.open(grid_path))
        st.caption("Visualization of noise levels across different diffusion timesteps")
    else:
        # Check individual noise level images
        noise_dir = os.path.join(VISUALIZATIONS_DIR, 'noise_levels')
        if os.path.exists(noise_dir):
            images = []
            for file in sorted(os.listdir(noise_dir)):
                if file.endswith('.png'):
                    images.append(os.path.join(noise_dir, file))
            
            if images:
                st.subheader("Noise Levels at Different Timesteps")
                cols = st.columns(min(5, len(images)))
                for i, img_path in enumerate(images):
                    cols[i % len(cols)].image(Image.open(img_path), caption=f"t={os.path.basename(img_path).replace('noise_t', '').replace('.png', '')}")

def display_text_conditioning_analysis(metrics):
    """Display text conditioning analysis"""
    # Check for text conditioning grid
    grid_path = os.path.join(VISUALIZATIONS_DIR, 'text_conditioning_grid.png')
    if os.path.exists(grid_path):
        st.subheader("Text Conditioning Examples")
        st.image(Image.open(grid_path))
        
        # If we have the prompts, display them
        if 'text_conditioning' in metrics and 'test_prompts' in metrics['text_conditioning']:
            prompts = metrics['text_conditioning']['test_prompts']
            for i, prompt in enumerate(prompts[:4]):
                st.markdown(f"**Prompt {i+1}**: {prompt}")
    
    # Check for guidance scale grid
    guidance_path = os.path.join(VISUALIZATIONS_DIR, 'guidance_scale_grid.png')
    if os.path.exists(guidance_path):
        st.subheader("Effect of Guidance Scale")
        st.image(Image.open(guidance_path))
        
        # If we have the guidance scales, display them
        if 'text_conditioning' in metrics and 'guidance_scales' in metrics['text_conditioning']:
            scales = metrics['text_conditioning']['guidance_scales']
            st.markdown(f"**Guidance scales**: {', '.join([str(s) for s in scales])}")
            st.caption("Higher guidance scales increase the influence of the text prompt on generation")

def display_parameter_distributions(metrics):
    """Display parameter distribution visualizations"""
    # Check for parameter distributions visualization
    dist_path = os.path.join(VISUALIZATIONS_DIR, 'parameter_distributions.png')
    if os.path.exists(dist_path):
        st.subheader("Parameter Distributions")
        st.image(Image.open(dist_path))
        st.caption("Distribution of parameter values across different model components")

def display_learning_curves(metrics):
    """Display learning curves if available"""
    # Check for loss comparison visualization
    loss_path = os.path.join(VISUALIZATIONS_DIR, 'loss_comparison.png')
    if os.path.exists(loss_path):
        st.subheader("Training and Validation Loss")
        st.image(Image.open(loss_path))
    
    # Check for diffusion loss visualization
    diff_loss_path = os.path.join(VISUALIZATIONS_DIR, 'diffusion_loss.png')
    if os.path.exists(diff_loss_path):
        st.subheader("Diffusion Loss")
        st.image(Image.open(diff_loss_path))

def display_generated_samples(metrics):
    """Display generated samples"""
    # Check for samples grid
    grid_path = os.path.join(VISUALIZATIONS_DIR, 'generated_samples_grid.png')
    if os.path.exists(grid_path):
        st.subheader("Generated Samples")
        st.image(Image.open(grid_path))
    
    # If grid doesn't exist, try to load individual samples
    samples = load_samples()
    if samples and not os.path.exists(grid_path):
        st.subheader("Generated Samples")
        
        # Display samples in columns
        cols = st.columns(min(4, len(samples)))
        for i, sample in enumerate(samples):
            with cols[i % len(cols)]:
                st.image(Image.open(sample['image_path']))
                st.markdown(f"**Prompt**: {sample['prompt']}")

# =============================================================================
# DASHBOARD FUNCTIONS
# =============================================================================

def run_model_metrics_dashboard():
    """Run the model metrics dashboard using pre-computed metrics"""
    st.header("Model Metrics Dashboard")
    
    # Load metrics
    metrics = load_saved_metrics()
    
    if not metrics:
        st.warning("No metrics available. Please run the evaluation script first.")
        
        # Show instructions for running the evaluation script
        with st.expander("How to run the evaluation script"):
            st.code("""
            # Run the evaluation script
            python evaluate_model.py
            """)
        
        return
    
    # Create tabs for different metrics categories
    tabs = st.tabs([
        "Model Summary", 
        "Architecture", 
        "Parameters",
        "Training Info",
        "Diffusion Analysis",
        "VAE Analysis",
        "Performance",
        "Samples & Visualization"
    ])
    
    with tabs[0]:
        st.subheader("Model Summary")
        
        # Try to load model summary
        summary = load_model_summary()
        if summary:
            st.markdown(summary)
        else:
            # Create a basic summary from metrics
            st.write("### X-ray Diffusion Model Summary")
            
            # Display architecture overview if available
            if 'architecture' in metrics:
                arch = metrics['architecture']
                st.write("#### Model Configuration")
                st.write(f"- **Diffusion Model**: {arch['diffusion']['scheduler_type']} scheduler with {arch['diffusion']['num_train_timesteps']} timesteps")
                st.write(f"- **VAE**: {arch['vae']['latent_channels']} latent channels")
                st.write(f"- **UNet**: {arch['unet']['model_channels']} model channels")
                st.write(f"- **Text Encoder**: {arch['text_encoder']['model_name']}")
            
            # Display parameter counts if available
            if 'parameters' in metrics:
                params = metrics['parameters']
                st.write("#### Model Size")
                st.write(f"- **Total Parameters**: {params['total']:,}")
                st.write(f"- **Memory Footprint**: {params['memory_footprint_mb']:.2f} MB")
            
            # Display inference speed if available
            if 'inference_speed' in metrics:
                speed = metrics['inference_speed']
                st.write("#### Inference Performance")
                st.write(f"- **Average Inference Time**: {speed['avg_inference_time_ms']:.2f} ms with {speed['num_inference_steps']} steps")
    
    with tabs[1]:
        st.subheader("Model Architecture")
        display_architecture_info(metrics)
    
    with tabs[2]:
        st.subheader("Model Parameters")
        display_parameter_counts(metrics)
        
        # Show parameter distribution plot
        display_parameter_distributions(metrics)
        
        # Show parameter statistics
        display_parameter_statistics(metrics)
    
    with tabs[3]:
        st.subheader("Training Information")
        display_checkpoint_metadata(metrics)
        
        # Show learning curves
        display_learning_curves(metrics)
    
    with tabs[4]:
        st.subheader("Diffusion Process Analysis")
        
        # Show beta schedule analysis
        display_beta_schedule_analysis(metrics)
        
        # Show noise levels visualization
        display_noise_levels(metrics)
        
        # Show text conditioning analysis
        display_text_conditioning_analysis(metrics)
    
    with tabs[5]:
        st.subheader("VAE Analysis")
        display_vae_analysis(metrics)
    
    with tabs[6]:
        st.subheader("Performance Analysis")
        display_inference_performance(metrics)
    
    with tabs[7]:
        st.subheader("Samples & Visualizations")
        
        # Show generated samples
        display_generated_samples(metrics)
        
        # Show all available visualizations
        visualizations = get_available_visualizations()
        if visualizations:
            st.subheader("All Available Visualizations")
            
            # Allow selecting visualization
            selected_vis = st.selectbox("Select Visualization", list(visualizations.keys()))
            if selected_vis:
                st.image(Image.open(visualizations[selected_vis]))
                st.caption(selected_vis)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application function."""
    # Header with app title
    st.title("🫁 Advanced X-Ray Diffusion Model Analysis Dashboard")
    
    # Run the model metrics dashboard
    run_model_metrics_dashboard()
    
    # Footer
    st.markdown("---")
    st.caption("X-Ray Diffusion Model Analysis Dashboard - For research purposes only. Not for clinical use.")

# Run the app
if __name__ == "__main__":
    main()