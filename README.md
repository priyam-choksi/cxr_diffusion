# Medical Chest X-Ray Generator

<div align="center">

![X-ray Generation Banner](https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/2.png)

*Generating realistic chest X-ray images from text descriptions using latent diffusion models*

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Dataset](https://img.shields.io/badge/dataset-Indiana%20University-yellow)]()

</div>

---

> ⚠️ **IMPORTANT**:  
> Before running this application, you **must download the model weights and dataset files** from the following sources: 
>
> - 🔗 [Google Drive - Dataset, Model Weights & Preprocessed Files](https://drive.google.com/drive/folders/1fNZavpgZ46zEHnimYAHWQ6-mwJhuXYTy?usp=sharing)  
> - 📦 [Kaggle - Dataset)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
>
> After downloading, **place the contents inside your local repository**, following the project structure described below.  
> ⚠️ The application **will not function** without these required files.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technical Details](#technical-details)
  - [Latent Diffusion Model](#latent-diffusion-model)
  - [Text Conditioning](#text-conditioning)
  - [Training Procedure](#training-procedure)
  - [Inference Pipeline](#inference-pipeline)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Launching the Application](#launching-the-application)
  - [X-Ray Generator Mode](#x-ray-generator-mode)
  - [Dataset Explorer Mode](#dataset-explorer-mode)
  - [Model Information Mode](#model-information-mode)
  - [Enhancement Comparison Mode](#enhancement-comparison-mode)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Enhancement Pipeline](#enhancement-pipeline)
- [Sample Results](#sample-results)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Future Work](#future-work)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## 🔍 Overview

This project presents a deep learning-based system designed to generate realistic chest X-ray images from textual clinical findings. By leveraging latent diffusion models and a domain-specific text encoder, the system can create synthetic medical images conditioned on natural language descriptions.

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/11.gif" width="350" />
<p><i>Visualization of the diffusion process from random noise to a synthetic X-ray</i></p>
</div>

This tool supports medical education, research, and data augmentation by enabling the generation of on-demand X-rays with specified pathological conditions. The interactive interface allows users to control generation parameters, compare enhancement presets, and analyze image quality metrics.

**Problem Statement**: Large and well-annotated datasets are rare in clinical imaging due to privacy concerns and logistical constraints around expert labeling. This project aims to overcome these limitations by enabling the generation of synthetic, realistic X-ray images from text descriptions.

## ✨ Features

- **Text-to-Image Generation**: Generate realistic chest X-rays from radiological descriptions
- **Multiple Enhancement Presets**: Apply radiological post-processing techniques to make X-rays more authentic
- **Model Analysis Dashboard**: Visualize model architecture and performance metrics
- **Dataset Explorer**: Browse and analyze the training dataset samples
- **Comprehensive Metrics**: Track and evaluate generation quality through multiple metrics
- **Interactive Console**: Compare real vs. generated X-rays
- **Pathology Highlighting**: Automatic annotation of detected pathological areas
- **Multiple Resolution Support**: Generate at 256×256, 512×512, or 768×768 resolution

## 🏗️ System Architecture

The architecture consists of three core modules working together to transform text descriptions into realistic X-ray images:

<div align="center">
  <div style="display: flex; justify-content: center;">
    <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/3.png" width="400" style="margin-right: 20px;"/>
    <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/2.png" width="400"/>
  </div>
</div>

### Components:

1. **Variational Autoencoder (VAE)**
   - Parameters: 26.2 million
   - Compresses X-ray images into a compact latent space
   - Latent Dimensions: 8 channels at 32×32 resolution
   - Reconstruction MSE: 0.11

   <div align="center">
   <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/13.gif" width="350" />
   <p><i>VAE encoding and decoding process</i></p>
   </div>

2. **Text Encoder (BioBERT)**
   - Parameters: 108.9 million (only 593K trainable)
   - Processes medical text descriptions into embedding vectors
   - Based on domain-specific biomedical language model
   - Output: 768-dimensional context vectors for cross-attention

3. **UNet with Cross-Attention**
   - Parameters: 39.66 million
   - Performs the denoising diffusion process
   - Incorporates text conditioning through cross-attention
   - Attention at multiple resolutions: 8×8, 16×16, and 32×32

**Total Parameters:** 151.81 million (43.5M trainable)

## 🧠 Technical Details

### Latent Diffusion Model

The generation process is based on latent diffusion models (LDMs), which operate in the compressed latent space of a VAE rather than in pixel space:

1. **Forward Diffusion Process**:
   - During training, Gaussian noise is gradually added to the latent representation following a fixed schedule
   - After T timesteps, the original latent is transformed into pure noise
   - Mathematically described by the forward process q(x_t|x_{t-1})

2. **Reverse Diffusion Process**:
   - The model learns to reverse this process, predicting the noise component at each step
   - The UNet is trained to estimate ε_θ(x_t, t, c) where c is the conditioning information
   - Sampling follows x_{t-1} = (x_t - σ_t·ε_θ(x_t, t, c)) / α_t + σ_t·z, where z ~ N(0, I)

3. **Latent Space Advantages**:
   - 8× spatial compression (32×32 vs 256×256) reduces computational requirements
   - Focusing on semantic content rather than pixel details
   - Enables faster training and inference while maintaining quality

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/1.png" width="500" />
<p><i>Forward and reverse diffusion processes</i></p>
</div>

### Text Conditioning

The text conditioning mechanism allows the generation to be guided by radiological descriptions:

1. **BioBERT Encoder**:
   - Pre-trained on biomedical literature (PubMed abstracts and PMC full-text articles)
   - Fine-tuned for understanding radiological terminology and findings
   - Input text is tokenized and processed through 12 transformer layers
   - Final layer embeddings are extracted as context vectors

2. **Cross-Attention Mechanism**:
   - Implements conditioning as: Attention(Q, K, V) = softmax(QK^T/√d)·V
   - Where Q comes from UNet features and K,V from text embeddings
   - Integrated at multiple levels in the UNet (8×8, 16×16, and 32×32)
   - This allows text information to influence the denoising at multiple scales

3. **Classifier-Free Guidance**:
   - During training, 10% of samples use an empty/null conditioning signal
   - During inference, both conditional and unconditional outputs are combined:
     - ε_CFG(x_t, t, c) = ε_θ(x_t, t, ∅) + s·(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
   - The guidance scale (s) controls adherence to the text prompt (default: 7.5)

### Training Procedure

The model is trained in two distinct stages:

1. **Stage 1: VAE Training**
   - Objective: Minimize reconstruction loss (MSE) + KL divergence
   - Duration: 200 epochs
   - Batch Size: 32
   - Learning Rate: 1e-4 with cosine schedule
   - Loss Weights: MSE (1.0), KL (1e-4)
   - Result: Compact latent representation with good reconstruction

2. **Stage 2: Diffusion Model Training**
   - Objective: Predict noise component added during forward process
   - Duration: 480 epochs
   - Batch Size: 16
   - Learning Rate: Started at 1e-4, ended at 4.62e-5
   - Loss: Mean squared error between predicted and actual noise
   - Classifier-Free Guidance: 10% of training with null conditioning

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/6.png" width="500" />
<p><i>Two-stage training process timeline</i></p>
</div>

### Inference Pipeline

The end-to-end inference process follows these steps:

1. **Text Input**: User provides a radiological description
2. **Text Encoding**: BioBERT processes the text into a 768-dimensional vector
3. **Latent Initialization**: Random noise sampled from N(0, I)
4. **Denoising Loop**: Iterative refinement through 20-100 timesteps (DDIM sampler)
5. **VAE Decoding**: Latent vector decoded to pixel space
6. **Enhancement**: Optional post-processing for visual quality improvement

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/7.png" width="500" />
<p><i>Inference pipeline flowchart</i></p>
</div>

## 📊 Dataset

The model is trained on the Indiana University Chest X-ray Collection:

- **Size**: ~7,000 frontal chest X-rays with paired radiological reports
- **Resolution**: All images normalized to 256×256 pixels
- **Format**: Grayscale images with 8-bit depth
- **Filtering**: Only frontal (PA/AP) views were used
- **Reports**: Contains both "Findings" and "Impression" sections
- **Split**: 90% training, 10% validation

**Data Processing Pipeline**:
1. Extract DICOM files and convert to PNG format
2. Normalize pixel values to [0, 1] range
3. Resize all images to 256×256 resolution
4. Extract and clean radiological reports (remove headers, patient IDs, etc.)
5. Parse reports to separate findings and impressions
6. Create text-image pairs for training

**Example Report**:
```
FINDINGS: The cardiomediastinal silhouette is normal in size and contour. 
The lungs are clear without evidence of infiltrate, effusion, or pneumothorax. 
No acute osseous abnormalities.

IMPRESSION: Normal chest radiograph.
```

## 🔧 Installation

### System Requirements

**Hardware Requirements**:
- GPU: NVIDIA with 8+ GB VRAM recommended (CPU mode available but slower)
- RAM: 16+ GB
- Storage: 10+ GB for model weights and datasets

**Software Requirements**:
- Python 3.8+
- CUDA 11.3+ (for GPU acceleration)
- Operating System: Linux, macOS, or Windows 10/11

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/priyam-choksi/cxr_diffusion.git
cd cxr_diffusion
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Download model and dataset files**:
   - Download from: https://drive.google.com/drive/folders/1fNZavpgZ46zEHnimYAHWQ6-mwJhuXYTy?usp=sharing
   - Extract the downloaded files to maintain the following structure:
     - Place `dataset` folder in the root directory
     - Place contents from `outputs` in your local `outputs` directory

5. Verify structure:
   - Ensure `outputs/diffusion_checkpoints/best_model.pt` exists
   - Ensure `outputs/vae_checkpoints/best_model.pt` exists
   - For dataset explorer, ensure `dataset/images` and `dataset/*.csv` exist

## 🚀 Usage

### Launching the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

This will start a local web server and open the application in your default web browser, typically at http://localhost:8501.

### X-Ray Generator Mode

This is the primary mode for generating synthetic X-rays from text descriptions:

1. **Enter radiological description**:
   - Type a detailed X-ray finding in the text area
   - Use medical terminology for best results
   - Click "Random Example" to use a pre-defined prompt

2. **Configure generation parameters**:
   - **Resolution**: Higher values give more detail but require more memory and time
   - **Quality (Steps)**: More steps produce higher quality but take longer
   - **Prompt Adherence**: Controls how closely the output follows the text description

3. **Set additional options**:
   - **Enhancement Preset**: Choose post-processing style (Balanced, High Contrast, etc.)
   - **Add Highlighting**: Toggle automatic pathology highlighting
   - **Random Seed**: Set for reproducible results or -1 for random generation

4. **Generate the X-ray**:
   - Click "Generate X-Ray" button
   - Wait for the generation and enhancement process to complete
   - Review the resulting images and metrics

5. **Analyze results**:
   - Compare enhanced and highlighted versions
   - Review comprehensive metrics (contrast, sharpness, entropy, etc.)
   - Check the quality progression chart for generation steps

6. **Download images**:
   - Use the download buttons to save your generated X-rays
   - Images are saved as PNG files with timestamp identifiers

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/9.png" width="700" />
<p><i>X-Ray Generator interface with enhanced results and metrics</i></p>
</div>

### Dataset Explorer Mode

This mode allows you to explore the training dataset:

1. **View dataset statistics**:
   - Check the total number of images
   - Review dataset characteristics and sources

2. **Load random samples**:
   - Click "Load Random Sample" to see real X-rays from the dataset
   - View the associated radiological report
   - Analyze image metrics for real X-rays

3. **Compare with generated images**:
   - Use as reference for evaluating your generated X-rays
   - Understand the style and quality of the training data

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/3.png" width="700" />
<p><i>Dataset Explorer interface showing sample X-ray and report</i></p>
</div>

### Model Information Mode

This mode provides detailed information about the model architecture and performance:

1. **Explore architectural components**:
   - View diagrams of the system architecture
   - Examine component details and parameter counts
   - Understand the latent diffusion process

2. **Review performance metrics**:
   - Check generation times across different resolutions
   - Examine quality metrics and their distributions
   - Understand hardware acceleration benefits

3. **Learn about enhancement pipeline**:
   - Explore the post-processing techniques
   - Compare different enhancement presets
   - Understand the effects of each processing step

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/2.png" width="700" />
<p><i>Model Information interface showing architecture diagrams</i></p>
</div>

### Enhancement Comparison Mode

This mode allows you to compare different post-processing techniques:

1. **Generate a single X-ray**:
   - Enter a description and set parameters
   - Generate a base X-ray image

2. **Compare enhancement presets**:
   - View the original unenhanced image
   - See how different presets affect the same image
   - Compare metrics between presets

3. **Analyze differences**:
   - Check percentage changes in metrics
   - Review visual differences in tissue contrast and detail
   - Identify optimal presets for different pathologies

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/7.png" width="700" />
<p><i>Enhancement Comparison interface showing multiple presets</i></p>
</div>

## 📂 Project Structure

The project is organized as follows:

```
cxr_diffusion/
├── app.py                    # Main Streamlit application
├── app_new.py                # New version with additional features
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
│   │   ├── best_model.pt     # Best diffusion model checkpoint
│   │   ├── checkpoint_epoch_40.pt
│   │   └── checkpoint_epoch_480.pt
│   ├── vae_checkpoints/
│   │   └── best_model.pt     # Best VAE model checkpoint
│   ├── generated/            # Generated images
│   │   └── generation_metrics.json  # Saved metrics history
│   ├── metrics/              # Saved metrics
│   │   ├── diffusion_metrics.json
│   │   └── model_summary.md
│   └── visualizations/       # Visualization outputs
├── dataset/                  # Dataset storage
│   ├── images/
│   │   └── images_normalized/
│   ├── indiana_reports.csv
│   └── indiana_projections.csv
├── examples/                 # Example generated images
│   ├── normal.png
│   ├── pneumonia.png
│   ├── cardiomegaly.png
│   └── pleural_effusion.png
├── images/                   # Documentation images
│   ├── 1.png                 # Forward/Reverse diffusion diagram
│   ├── 2.png                 # LDM overview
│   ├── 3.png                 # Simplified architecture
│   ├── 4.png                 # Component interaction
│   ├── 5.png                 # System components
│   ├── 6.png                 # Training timeline
│   ├── 7.png                 # Inference pipeline
│   ├── 8.png                 # Evaluation metrics
│   ├── 9.png                 # Enhancement pipeline
│   ├── 10.png                # Future enhancements
│   ├── 11.gif                # Diffusion process GIF 1
│   ├── 12.gif                # Diffusion process GIF 2
│   ├── 13.gif                # VAE process GIF 1
│   └── 14.gif                # VAE process GIF 2
├── scripts/
│   ├── download_model.py     # Script to download model weights
│   └── download_dataset.py   # Script to download dataset
├── config/
│   ├── vae_config.json       # VAE training configuration
│   └── diffusion_config.json # Diffusion training configuration
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT license file
└── README.md                 # This file
```

> **Note**: You must download the model weights and dataset from the provided Google Drive link and place them in the appropriate directories as shown above before running the application.

### Key Files Explained

- **app.py**: Main application entry point using Streamlit for the user interface
- **inference.py**: Wrapper for generating images using the trained model
- **diffusion.py**: Implements the core diffusion model logic
- **vae.py**: Implements the Variational Autoencoder for latent space compression
- **dataset.py**: Handles loading and processing of the X-ray dataset
- **post_process.py**: Contains the enhancement pipeline functionality
- **best_model.pt**: Pre-trained model weights (both diffusion and VAE)

### Configuration Files

The `config` directory contains JSON files that define model hyperparameters:

**vae_config.json** (excerpt):
```json
{
  "model_type": "vae",
  "latent_channels": 8,
  "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
  "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
  "block_out_channels": [128, 256, 512, 512],
  "layers_per_block": 2,
  "act_fn": "silu",
  "latent_dim": 8
}
```

**diffusion_config.json** (excerpt):
```json
{
  "model_type": "latent_diffusion",
  "timesteps": 1000,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "beta_end": 0.02,
  "clip_sample": false,
  "attention_resolutions": [8, 16, 32],
  "dropout": 0.1,
  "use_checkpoint": true,
  "use_scale_shift_norm": true
}
```

## 📈 Model Performance

### Generation Performance

| Resolution | Steps | Time (s) | Memory (GB) |
| ---------- | ----- | -------- | ----------- |
| 256×256    | 20    | 0.66     | 0.6         |
| 256×256    | 100   | 3.32     | 0.6         |
| 512×512    | 20    | 1.35     | 2.1         |
| 512×512    | 100   | 6.63     | 2.1         |
| 768×768    | 100   | 15.21    | 4.5         |

### Quality Metrics

| Metric              | Value       |
| ------------------- | ----------- |
| SSIM                | 0.82 ± 0.08 |
| PSNR                | 22.3 ± 2.1 dB |
| Contrast Ratio      | 0.76 ± 0.05 |
| Entropy             | 7.94        |
| Sharpness           | 349         |
| Prompt Consistency  | 85%         |

### Hardware Acceleration

- **CPU Mode**: Available but ~4.5× slower
- **CUDA Acceleration**: Supported on NVIDIA GPUs
- **Average Steps/Second**: 30.2 on mid-range GPU
- **Memory Consumption**: 579.1 MB for model weights

### Comparative Benchmarks

Comparison with other text-to-image methods on chest X-ray generation:

| Method       | SSIM  | PSNR (dB) | FID   | Time (s) |
|--------------|-------|-----------|-------|----------|
| Our Model    | 0.82  | 22.3      | 18.7  | 3.32     |
| GAN-based    | 0.73  | 19.8      | 24.2  | 1.05     |
| VQGAN+CLIP   | 0.77  | 20.5      | 22.8  | 5.27     |
| Pixel Diffusion | 0.80 | 21.9    | 19.3  | 12.42    |

## 🎨 Enhancement Pipeline

The post-processing pipeline improves the visual quality and authenticity of generated X-rays:

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/9.png" width="500" />
<p><i>Enhancement Pipeline Process</i></p>
</div>

### Processing Techniques

1. **Windowing**: Adjusts pixel intensity distribution to focus on relevant tissue densities
   - Parameters: window_center, window_width
   - Clips intensities to highlight specific tissue ranges
   - Similar to radiological windowing on PACS systems

2. **CLAHE**: Contrast-Limited Adaptive Histogram Equalization for local contrast enhancement
   - Parameters: clip_limit, grid_size
   - Enhances local contrast while limiting noise amplification
   - Preserves subtle tissue differences

3. **Median Filtering**: Reduces noise while preserving edges and structures
   - Parameters: kernel_size
   - Removes salt-and-pepper noise
   - Preserves sharp transitions between tissues

4. **Edge Enhancement**: Sharpens lung markings and anatomical boundaries
   - Parameters: amount
   - Uses unsharp masking technique
   - Accentuates fine lung markings and vascular structures

5. **Histogram Equalization**: Improves overall contrast distribution
   - Normalizes intensity histogram
   - Expands dynamic range
   - Optional step depending on preset

6. **Vignetting**: Adds subtle darkening toward edges to simulate X-ray beam intensity falloff
   - Parameters: amount
   - Creates more realistic peripheral appearance
   - Mimics natural characteristics of radiographic imaging

### Enhancement Presets

The system includes four carefully tuned presets:

#### Balanced Preset
```python
'balanced': {
    'window_center': 0.5,
    'window_width': 0.8,
    'edge_amount': 1.3, 
    'median_size': 3,
    'clahe_clip': 2.5,
    'clahe_grid': (8, 8),
    'vignette_amount': 0.25,
    'apply_hist_eq': True
}
```
- **Best for**: General purpose X-ray viewing
- **Highlights**: Balanced contrast and detail preservation
- **Description**: Provides a good all-around enhancement suitable for most X-rays and pathologies

#### High Contrast Preset
```python
'high_contrast': {
    'window_center': 0.45,
    'window_width': 0.7,
    'edge_amount': 1.5,
    'median_size': 3,
    'clahe_clip': 3.0,
    'clahe_grid': (8, 8),
    'vignette_amount': 0.3,
    'apply_hist_eq': True
}
```
- **Best for**: Subtle findings and dense tissues
- **Highlights**: Enhanced contrast between tissues
- **Description**: Emphasizes differences between tissues, making subtle findings more apparent

#### Sharp Detail Preset
```python
'sharp_detail': {
    'window_center': 0.55,
    'window_width': 0.85,
    'edge_amount': 1.8,
    'median_size': 3,
    'clahe_clip': 2.0,
    'clahe_grid': (6, 6),
    'vignette_amount': 0.2,
    'apply_hist_eq': False
}
```
- **Best for**: Fine structures and bone detail
- **Highlights**: Edge enhancement and structural clarity
- **Description**: Accentuates fine structures like lung markings and bone edges

#### Radiographic Film Preset
```python
'radiographic_film': {
    'window_center': 0.48,
    'window_width': 0.75,
    'edge_amount': 1.4,
    'median_size': 3,
    'clahe_clip': 2.8,
    'clahe_grid': (7, 7),
    'vignette_amount': 0.35,
    'apply_hist_eq': True
}
```
- **Best for**: Traditional film-like appearance
- **Highlights**: Authentic radiographic look with vignetting
- **Description**: Simulates the appearance of traditional film radiographs

## 📷 Sample Results

<div align="center">
  <div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <div style="margin: 10px; text-align: center;">
      <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/sample_1.png" width="250" />
      <p><i>Normal chest X-ray</i></p>
    </div>
    <div style="margin: 10px; text-align: center;">
      <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/sample_2.png" width="250" />
      <p><i>Right lower lobe pneumonia</i></p>
    </div>
    <div style="margin: 10px; text-align: center;">
      <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/images/sample_3.png" width="250" />
      <p><i>Cardiomegaly with pulmonary congestion</i></p>
    </div>
    <div style="margin: 10px; text-align: center;">
      <img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/sample_4.png" width="250" />
      <p><i>Bilateral pleural effusions</i></p>
    </div>
  </div>
</div>

### Typical Metrics for Generated Images

| Condition | Contrast | Sharpness | Entropy | SSIM | PSNR (dB) |
| --------- | -------- | --------- | ------- | ---- | --------- |
| Normal    | 1.00     | 206.0     | 7.88    | 0.83 | 22.5      |
| Pneumonia | 0.98     | 235.7     | 7.64    | 0.78 | 21.9      |
| Cardiomegaly | 0.95  | 189.3     | 7.74    | 0.81 | 22.1      |
| Pleural Effusion | 0.97 | 217.5  | 7.92    | 0.79 | 21.7      |

### Example Generated Pathologies

**Normal Chest X-ray**:
```
prompt = "Normal chest X-ray with clear lungs and no abnormalities."
```
- Characteristics: Clear lung fields, normal heart size, well-defined costophrenic angles
- Average contrast ratio: 1.00
- Average generation time: 3.45s at 100 steps

**Right Lower Lobe Pneumonia**:
```
prompt = "Right lower lobe pneumonia with focal consolidation."
```
- Characteristics: Right lower zone opacity, air bronchograms, preserved heart silhouette
- Average contrast ratio: 0.98
- Average generation time: 3.51s at 100 steps

**Cardiomegaly**:
```
prompt = "Cardiomegaly with pulmonary vascular congestion."
```
- Characteristics: Enlarged cardiac silhouette, prominent vascular markings, cephalization
- Average contrast ratio: 0.95
- Average generation time: 3.48s at 100 steps

**Pleural Effusion**:
```
prompt = "Bilateral pleural effusions, greater on the right."
```
- Characteristics: Blunted costophrenic angles, meniscus sign, right-sided fluid level
- Average contrast ratio: 0.97
- Average generation time: 3.57s at 100 steps

## ⚙️ Programmatic Usage

The model can be used programmatically in your Python code:

### Basic Usage

```python
from xray_generator.inference import XrayGenerator

# Initialize the generator
generator = XrayGenerator(
    model_path="outputs/diffusion_checkpoints/best_model.pt",
    device="cuda",
    tokenizer_name="dmis-lab/biobert-base-cased-v1.1"
)

# Generate an X-ray
result = generator.generate(
    prompt="Normal chest X-ray with clear lungs and no abnormalities.",
    height=256,
    width=256,
    num_inference_steps=100,
    guidance_scale=7.5
)

# Save the image
result["images"][0].save("generated_xray.png")
```

### Advanced Usage with Enhancement

```python
import numpy as np
from PIL import Image
from xray_generator.inference import XrayGenerator
from post_process import enhance_xray, ENHANCEMENT_PRESETS

# Initialize generator
generator = XrayGenerator(
    model_path="outputs/diffusion_checkpoints/best_model.pt",
    device="cuda",
    tokenizer_name="dmis-lab/biobert-base-cased-v1.1"
)

# Generate base image
result = generator.generate(
    prompt="Left lower lobe pneumonia with consolidation.",
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42  # For reproducibility
)

# Get the base image
base_image = result["images"][0]

# Apply different enhancement presets
images = {
    "Original": base_image,
    "Balanced": enhance_xray(base_image, ENHANCEMENT_PRESETS["Balanced"]),
    "High Contrast": enhance_xray(base_image, ENHANCEMENT_PRESETS["High Contrast"]),
    "Sharp Detail": enhance_xray(base_image, ENHANCEMENT_PRESETS["Sharp Detail"]),
    "Radiographic Film": enhance_xray(base_image, ENHANCEMENT_PRESETS["Radiographic Film"])
}

# Save all versions
for name, img in images.items():
    img.save(f"pneumonia_{name.lower().replace(' ', '_')}.png")
```

### Batch Generation

```python
# Define multiple prompts
prompts = [
    "Normal chest X-ray with clear lungs and no abnormalities.",
    "Right lower lobe pneumonia with focal consolidation.",
    "Cardiomegaly with pulmonary vascular congestion.",
    "Bilateral pleural effusions, greater on the right."
]

# Batch generate (one at a time since models are usually large)
results = []
for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}...")
    result = generator.generate(
        prompt=prompt,
        height=256,
        width=256,
        num_inference_steps=50
    )
    results.append((prompt, result["images"][0]))

# Save results
for i, (prompt, image) in enumerate(results):
    condition = prompt.split()[0].lower()
    image.save(f"batch_generation_{i}_{condition}.png")
```

## 🐛 Troubleshooting

### Common Issues

**GPU Out of Memory**
- **Symptom**: CUDA out of memory error during generation
- **Solution**: 
  - Reduce resolution (try 256×256 instead of higher)
  - Decrease batch size if doing batch generation
  - Use fewer diffusion steps (20-50 instead of 100)
  - If all else fails, add `device="cpu"` when initializing the generator

**Model Loading Fails**
- **Symptom**: Error loading model weights
- **Solution**:
  - Make sure you downloaded the model files from the Google Drive link
  - Check path to model checkpoint is correct
  - Ensure you have the right model version
  - Verify the model was downloaded completely
  - Make sure PyTorch version is compatible

**Slow Generation**
- **Symptom**: Generation takes a long time
- **Solution**:
  - Ensure GPU is being used (check device with `torch.cuda.is_available()`)
  - Reduce diffusion steps (try 20-50 steps)
  - Use DDIM sampler instead of DDPM (much faster)
  - Check for background processes consuming GPU resources

**Poor Image Quality**
- **Symptom**: Blurry or unrealistic images
- **Solution**:
  - Increase guidance scale (7.5-10.0 works well)
  - Use more diffusion steps (at least 50)
  - Try different enhancement presets
  - Make prompts more detailed and specific

### Debugging Tools

**Memory Profiling**:
```python
# Memory profiling function in quick_test.py
def profile_memory_usage(model_path, prompt, resolution=256):
    import torch
    initial_mem = torch.cuda.memory_allocated()
    
    # Load model
    generator = XrayGenerator(model_path=model_path, device="cuda")
    
    model_mem = torch.cuda.memory_allocated() - initial_mem
    
    # Generate image
    result = generator.generate(
        prompt=prompt,
        height=resolution,
        width=resolution,
        num_inference_steps=50
    )
    
    gen_mem = torch.cuda.memory_allocated() - model_mem - initial_mem
    
    return {
        "model_size_mb": model_mem / (1024**2),
        "generation_overhead_mb": gen_mem / (1024**2),
        "total_mb": (model_mem + gen_mem) / (1024**2)
    }
```

**Performance Testing**:
```bash
# Run quick performance test script
python quick_test.py --resolution=256 --steps=20,50,100 --metrics
```

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes and commit**:
   ```bash
   git commit -m "Add your feature description"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a pull request**

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add docstrings to all functions and classes
- Write unit tests for new functionality
- Update documentation to reflect your changes
- Keep pull requests focused on a single change
- Add your name to CONTRIBUTORS.md

## 🔮 Future Work

<div align="center">
<img src="https://raw.githubusercontent.com/priyam-choksi/cxr_diffusion/main/images/10.png" width="500" />
<p><i>Current Limitations vs Future Enhancements</i></p>
</div>

### Current Limitations

- **Latent Resolution**: 32×32 latent space limits fine-grained details
- **Prompt Sensitivity**: Output quality heavily depends on prompt clarity and specificity
- **Pathology Localization**: No explicit enforcement of anatomical correctness
- **Clinical Validation**: Limited expert validation of generated images
- **Resolution Constraints**: Higher resolutions require exponentially more memory

### Proposed Enhancements

- **Segmentation Integration**: Fine-tune with segmented datasets for better localization
- **Adversarial Refinement**: Add discriminator network to improve realism
- **Multi-modal Conditioning**: Combine text with other clinical data 
- **Hierarchical Diffusion**: Implement cascade models for higher resolution
- **Clinical Partnerships**: Establish expert evaluation pipelines
- **Distillation Techniques**: Create smaller, faster models for deployment

### Development Roadmap

- **Short-term (1-3 months)**:
  - Improve pathology highlighting accuracy
  - Add support for lateral X-ray generation
  - Implement basic model quantization for faster inference
  - Expand example prompt library

- **Mid-term (3-6 months)**:
  - Integrate anatomical segmentation guidance
  - Develop web API for remote inference
  - Support additional medical imaging modalities
  - Create Python package for easier installation

- **Long-term (6-12 months)**:
  - Implement hierarchical diffusion for 1024×1024 resolution
  - Develop clinical validation pipeline
  - Create mobile-friendly interface
  - Extend to 3D imaging (CT, MRI)

## 📝 Citation

If you use this project in your research, please cite it as:

```bibtex
@software{chest_xray_generator,
  author = {Priyam Choksi},
  title = {Medical Chest X-Ray Generator},
  year = {2025},
  url = {https://github.com/priyam-choksi/cxr_diffusion}
}
```

## 🙏 Acknowledgments

- Indiana University Chest X-ray Collection for the training dataset
- [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) (Rombach et al., 2022)
- [BioBERT](https://github.com/dmis-lab/biobert) (Lee et al., 2020)
- [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion) (Ho et al., 2020)
- Streamlit for the interactive UI framework

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2023 Priyam Choksi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">
<p>Made with ❤️ for medical AI research and education</p>
<p>© 2023 Priyam Choksi</p>
</div>
