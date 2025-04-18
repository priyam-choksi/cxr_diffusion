# Medical Chest X-Ray Generator

A deep learning-based application that generates realistic chest X-ray images from text descriptions using latent diffusion models. This project provides an interactive interface for generating, analyzing, and enhancing synthetic chest X-rays for medical education, research, and model evaluation.

![X-ray Generation Example](https://placeholder.com/your-xray-example.png)

## Features

- **Text-to-Image Generation**: Generate realistic chest X-rays from textual descriptions of findings or conditions
- **Multiple Enhancement Presets**: Apply radiological post-processing techniques to make generated images look more authentic
- **Model Analysis Dashboard**: Examine model architecture, parameters, and performance metrics
- **Dataset Explorer**: Browse and analyze the training dataset samples
- **Advanced Metrics**: View comprehensive quality metrics and performance evaluations
- **Interactive Research Console**: Compare real vs. generated X-rays and analyze differences
- **Condition-specific Analysis**: Evaluate generation quality for different radiological conditions
- **Multiple Resolution Support**: Generate images at 256×256, 512×512, or 768×768 resolution

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8+ GB VRAM recommended for optimal performance
  - Can run on CPU, but generation will be significantly slower
- **RAM**: 16+ GB system memory
- **Storage**: At least 10 GB free space for model weights, dataset, and outputs

### Software Requirements
- **Python**: 3.8+
- **CUDA**: 11.3+ (if using GPU)
- **Operating System**: Linux, macOS, or Windows 10/11

## Installation

### 1. Set up Python Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch according to your CUDA version
# Visit https://pytorch.org/get-started/locally/ for specific install command

# Install other dependencies
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
torch==2.0.0
torchvision==0.15.0
transformers==4.28.1
streamlit==1.22.0
pillow==9.5.0
numpy==1.24.3
pandas==2.0.1
matplotlib==3.7.1
scikit-image==0.20.0
opencv-python==4.7.0.72
tqdm==4.65.0
einops==0.6.1
seaborn==0.12.2
```

### 3. Download Pre-trained Model and Dataset

To save time, you can download the pre-trained model and dataset directly from Google Drive:

```
https://drive.google.com/drive/folders/your-shared-folder-link
```

The Google Drive folder contains:
- Pre-trained model checkpoints (`outputs/` folder)
- Dataset files (`dataset/` folder)

Extract these folders to your project directory, maintaining the following structure:

```
chest-xray-generator/
├── outputs/
│   ├── diffusion_checkpoints/
│   │   ├── best_model.pt          # Main model checkpoint
│   │   ├── checkpoint_epoch_40.pt # Intermediate checkpoint
│   │   └── checkpoint_epoch_480.pt # Final checkpoint
│   └── metrics/                   # Pre-computed model metrics
│       ├── diffusion_metrics.json
│       └── model_summary.md
├── dataset/
│   ├── images/
│   │   └── images_normalized/
│   ├── indiana_reports.csv
│   └── indiana_projections.csv
```

## Project Structure

```
chest-xray-generator/
├── app.py                    # Main Streamlit application
├── app_new.py                # New version of the application with additional features
├── extract_metrics.py        # Script for extracting model metrics
├── post_process.py           # Enhancement pipeline implementation
├── quick_test.py             # Quick test script for model validation
├── xray_generator/           # Core model implementation
│   ├── __init__.py
│   ├── inference.py          # Inference wrapper
│   ├── train.py              # Training script
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── diffusion.py      # Diffusion model implementation
│   │   ├── text_encoder.py   # Text encoder model
│   │   ├── unet.py           # UNet noise prediction model
│   │   └── vae.py            # VAE model for latent space
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── dataset.py        # Dataset loading and processing
│       └── processing.py     # Image processing utilities
├── outputs/                  # Generated outputs
│   ├── diffusion_checkpoints/
│   ├── vae_checkpoints/
│   ├── generated/            # Generated images
│   ├── metrics/              # Saved metrics
│   └── visualizations/       # Visualization outputs
└── dataset/                  # Dataset storage (see Prepare Dataset)
```

## Quick Start Guide (Using Pre-trained Model)

If you want to quickly start using the application without training the model from scratch, follow these steps:

### 1. Download Pre-trained Model

Download the pre-trained model and necessary files from Google Drive:
```
https://drive.google.com/drive/folders/your-shared-folder-link
```

### 2. Setup Directory Structure

Extract the downloaded folders to maintain this structure:
```
chest-xray-generator/
├── outputs/             # Pre-trained model and metrics
├── dataset/             # Optional - only needed if exploring dataset
└── xray_generator/      # Core module code
```

### 3. Run the Application

Simply run the Streamlit application:
```bash
streamlit run app.py
```

Or for the enhanced version with additional features:
```bash
streamlit run app_new.py
```

### 4. Generate X-rays

1. Select "X-Ray Generator" mode from the dropdown
2. Enter a text prompt describing the X-ray you want to generate
3. Adjust parameters as needed (resolution, steps, guidance scale)
4. Select an enhancement preset
5. Click "Generate X-ray"

This approach lets you immediately use the model for inference without needing to train it first, saving significant time and computational resources.

## Usage

### Application Modes

After launching the application with `streamlit run app.py` or `streamlit run app_new.py`, you can select from several modes:

1. **X-Ray Generator**: Generate X-rays from text descriptions
2. **Model Analysis**: Analyze model architecture and parameters
3. **Dataset Explorer**: Explore the training dataset
4. **Enhancement Comparison**: Compare different enhancement presets side by side
5. **Static Metrics Dashboard**: View pre-computed model metrics
6. **Research Dashboard**: Compare generated and real X-rays
7. **Pre-computed Metrics Dashboard**: View comprehensive model evaluation metrics

#### X-Ray Generator Mode

1. Enter a text description of the X-ray in the input field
2. Adjust parameters:
   - **Resolution**: Higher resolution gives more detail but requires more memory
   - **Diffusion Steps**: More steps give higher quality but slower generation
   - **Guidance Scale**: Controls adherence to the text prompt
3. Select an enhancement preset or "None" for raw output
4. Click "Generate X-ray"

#### Enhancement Comparison Mode

1. Enter a text description
2. Configure generation parameters
3. Click "Generate & Compare"
4. View side-by-side comparison of different enhancement presets

### Running the Evaluation Script

To compute and save comprehensive model metrics (if not using the pre-computed metrics):

```bash
python extract_metrics.py
```

This will analyze the model and save metrics to `outputs/metrics/diffusion_metrics.json` and visualizations to `outputs/visualizations/`.

### Running the Post-processing Script

To test the enhancement pipeline independently:

```bash
python post_process.py
```

This will generate test X-rays, apply different enhancement presets, and save the results to `outputs/enhanced_xrays/`.

## Model Architecture

Our model is based on the Latent Diffusion Model architecture for text-to-image generation, specialized for chest X-rays.

### Components

1. **VAE (Variational Autoencoder)**
   - Parameters: 3.25M
   - Purpose: Encodes images into a compact latent space and decodes them back to pixel space
   - Architecture: Convolutional encoder/decoder with attention mechanisms at 16px and 32px resolutions
   - Latent Space: 8 channels at 32×32 for 256×256 images
   - Reconstruction MSE: 0.11

2. **UNet**
   - Parameters: 39.66M
   - Purpose: Performs the denoising diffusion process
   - Architecture: UNet with cross-attention to inject text conditioning
   - Features: Time embedding, residual blocks, self-attention, and cross-attention layers
   - Attention Resolutions: [8, 16, 32]

3. **Text Encoder**
   - Parameters: 108.9M (only 593K trainable)
   - Model: BioBERT (dmis-lab/biobert-base-cased-v1.1)
   - Purpose: Encodes medical text descriptions into embeddings for conditioning
   - Output: 768-dimensional context vectors for cross-attention

4. **Diffusion Process**
   - Timesteps: 1000
   - Sampling: DDIM scheduler for faster sampling
   - Beta Schedule: Linear (min: 0.0001, max: 0.02)
   - Prediction Type: Epsilon (noise prediction)
   - Classifier-Free Guidance: Variable scale (7.5 default)

### Total Model Size

- Combined Parameters: 151.81M (43.5M trainable)
- Memory Footprint: 579.11 MB

## Training Process

The model was trained using a two-stage process:

### Stage 1: VAE Training

First, only the VAE component was trained for reconstruction:

```bash
python -m xray_generator.train \
    --config config/vae_config.json \
    --dataset_path dataset/images/images_normalized \
    --reports_csv dataset/indiana_reports.csv \
    --projections_csv dataset/indiana_projections.csv \
    --output_dir outputs \
    --train_vae_only
```

Training Parameters:
- Batch Size: Configurable in config file
- Learning Rate: Started at 1e-4 with schedule
- VAE Loss: MSE reconstruction + KL divergence (weight 1e-4)
- Reconstruction MSE: 0.11064

### Stage 2: Diffusion Model Training

Once the VAE was trained, the full diffusion model was trained:

```bash
python -m xray_generator.train \
    --config config/diffusion_config.json \
    --dataset_path dataset/images/images_normalized \
    --reports_csv dataset/indiana_reports.csv \
    --projections_csv dataset/indiana_projections.csv \
    --output_dir outputs \
    --resume_from outputs/vae_checkpoints/best_model.pt
```

Training Parameters from checkpoint metadata:
- Epochs: 480 
- Final Learning Rate: 4.62e-05 (with scheduler)
- Train Loss: 0.0266
- Validation Loss: 0.0360
- Validation Diffusion Loss: 0.0350

## Enhancement Pipeline

The post-processing pipeline applies several radiological image enhancements to make the generated X-rays look more authentic:

1. **Windowing** (Level/Width Adjustment)
   - Adjusts the pixel intensity distribution
   - Parameters: Window center and width
   - Radiological significance: Focuses on specific tissue density ranges

2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
   - Enhances local contrast while limiting noise amplification
   - Parameters: Clip limit and grid size
   - Improves visibility of subtle structures like lung markings

3. **Median Filtering**
   - Reduces noise while preserving edges
   - Parameter: Kernel size
   - Simulates the grain reduction in processed radiographs

4. **Edge Enhancement**
   - Sharpens lung markings and structural boundaries
   - Parameter: Enhancement amount
   - Makes anatomical structures more distinct

5. **Histogram Equalization**
   - Improves overall contrast distribution
   - Optional step for some presets
   - Adjusts the global dynamic range

6. **Vignetting**
   - Adds subtle darkening toward the edges
   - Parameter: Amount
   - Simulates natural X-ray beam intensity falloff

### Enhancement Presets

The system includes four presets:
- **Balanced**: General-purpose enhancement for most X-rays
- **High Contrast**: Emphasizes density differences, suitable for lung pathologies
- **Sharp Detail**: Highlights fine structures like lung markings
- **Radiographic Film**: Mimics traditional film X-rays

## Evaluation

The model is evaluated using multiple metrics:

### Image Quality Metrics

- **SSIM (Structural Similarity Index)**: Measures structural similarity between real and generated images
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **Contrast Ratio**: Measures image contrast
- **Entropy**: Measures image information content
- **Sharpness**: Measures edge definition quality

### Performance Metrics

Based on actual measurements from the model evaluation:

- **Average Inference Time**: 663.15 ms (for 20 diffusion steps)
- **Range**: 627.94 - 684.32 ms
- **Steps per Second**: Approximately 30.16 steps/second
- **Memory Footprint**: 579.11 MB for model weights

### Generation Quality Metrics

Metrics from an example generation of a normal chest X-ray (256×256 resolution):

- **Generation Time**: 4.18 seconds (100 diffusion steps)
- **Mean Pixel Value**: 123.46
- **Standard Deviation**: 71.03
- **Contrast Ratio**: 1.0
- **Sharpness**: 349.05
- **Entropy**: 7.94

## License and Citation

This project is released under the MIT License.

If you use this project in your research, please cite it as:

```
@software{chest_xray_generator,
  author = {Your Name},
  title = {Medical Chest X-Ray Generator},
  year = {2023},
  url = {https://github.com/yourusername/chest-xray-generator}
}
```

## Acknowledgments

- The model was trained on the Indiana University Chest X-Ray Collection
- Architecture based on Latent Diffusion Models (Rombach et al., 2022)
- BioBERT model by DMIS Lab
