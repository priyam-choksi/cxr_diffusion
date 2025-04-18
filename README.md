# Chest X-Ray Generator

Generate realistic chest X-ray images from text descriptions using a latent diffusion model.

## Overview

This project provides a state-of-the-art generative model for creating synthetic chest X-ray images conditioned on text descriptions. The model has been trained on real X-ray images with corresponding radiologist reports and can generate high-quality, realistic X-rays based on medical text prompts.

The model architecture includes:
- A VAE encoder/decoder specialized for chest X-rays
- A medical text encoder based on BioBERT
- A UNet with cross-attention for conditioning
- A diffusion model that ties everything together

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chest-xray-generator.git
cd chest-xray-generator