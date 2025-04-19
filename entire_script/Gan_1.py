"""
Medical Chest X-ray Latent Diffusion Model

A comprehensive latent diffusion model implementation for generating synthetic chest X-rays 
from radiology reports using the Indiana University (IU) dataset.

This implementation includes:
- Complete VAE with advanced medical-specific architecture
- UNet with cross-attention for text conditioning
- Medical text encoder based on BioBERT/ClinicalBERT
- Robust diffusion sampling process
- Full training pipeline with VAE pretraining and diffusion model training
- Production-ready error handling and monitoring
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import math
import time
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW, lr_scheduler
import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image

from transformers import AutoModel, AutoTokenizer

# ========================================================================
# Configuration Parameters
# ========================================================================

# Path configuration
DATASET_PATH = r'F:\DeepLearning Course\Project\Kaggle\chest-xrays-indiana-university'
IMAGES_PATH = os.path.abspath(os.path.join(DATASET_PATH, 'images/images_normalized/'))
REPORTS_CSV = os.path.join(DATASET_PATH, 'indiana_reports.csv')
PROJECTIONS_CSV = os.path.join(DATASET_PATH, 'indiana_projections.csv')
OUTPUT_DIR = os.path.join(DATASET_PATH, "outputs_gan_testing")

MODEL_CHANNELS = 64
IMAGE_SIZE = 256
TEXT_MODEL = "dmis-lab/biobert-base-cased-v1.1"
MAX_TOKEN_LENGTH = 256

# Training configuration
BATCH_SIZE = 4 
NUM_WORKERS = 0  
EPOCHS_VAE = 200
EPOCHS_DIFFUSION = 200
LR_VAE = 1e-4
LR_DIFFUSION = 5e-5
GUIDANCE_SCALE = 7.5
TRAIN_UNET_ONLY = True
USE_AMP = True
RANDOM_SEED = 42
DATALOADER_TIMEOUT = 0
CHECKPOINT_FREQ = 10  # Save every 10 epochs
KEEP_LAST_CHECKPOINTS = 2  # Only keep 2 most recent checkpoints
RESUME_FROM = None  # Path to checkpoint to resume from, None for fresh start
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * 4
LATENT_CHANNELS = 8  # Increase from 4 to 8 for better detail
USE_CLAHE = True     # Better contrast
SAVE_CHECKPOINTS = True
VERBOSE_LOGGING = False  # Set to False to disable detailed logging
RUN_QUICK_TEST = False  # Change to False for real training
# ========================================================================
# Logging and Utilities
# ========================================================================

def setup_logging():
    """Setup proper logging with file and console output."""
    log_dir = os.path.join(OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'xray_diffusion_{timestamp}.log')
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Configure logging with different levels for file vs console
    logger = logging.getLogger("xray-diffusion")
    logger.setLevel(logging.INFO)
    
    # File handler - keep everything at INFO level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
                                              datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Console handler - only show WARNING and above (errors, etc.)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Change to logging.ERROR for even less output
    console_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                                                  datefmt='%Y-%m-%d %H:%M:%S'))
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create logger
logger = setup_logging()

# Set seeds for reproducibility across all random state
def set_seed(seed=RANDOM_SEED):
    """Set seeds for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")

# Set the random seed
set_seed()

# Select device with error handling
def get_device():
    """Get the best available device with proper error handling."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU. This will be very slow.")
        return torch.device("cpu")
    
    try:
        # Try to initialize CUDA device
        device = torch.device("cuda")
        _ = torch.zeros(1).to(device)  # Test CUDA functionality
        
        # Log device info
        device_properties = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {device_properties.total_memory / 1e9:.2f} GB")
        logger.info(f"CUDA Capability: {device_properties.major}.{device_properties.minor}")
        
        return device
    except Exception as e:
        logger.error(f"Error initializing CUDA: {e}")
        logger.warning("Falling back to CPU")
        return torch.device("cpu")

# Get the device
device = get_device()

# Function to monitor GPU memory
def log_gpu_memory(message=""):
    """Log GPU memory usage."""
    if not VERBOSE_LOGGING:
        return  # Skip logging if verbose logging is disabled
        
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"GPU Memory {message}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        # Reset max stats
        torch.cuda.reset_peak_memory_stats()

# Function to verify dataset files
def verify_dataset_files(dataset_path, sample_size=100):
    """Verify that dataset files exist and are readable."""
    logger.info(f"Verifying dataset files in {dataset_path}")
    
    # Check if path exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return False
    
    # Get list of files
    try:
        all_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        logger.error(f"Error listing files in {dataset_path}: {e}")
        return False
    
    if not all_files:
        logger.error(f"No image files found in {dataset_path}")
        return False
        
    logger.info(f"Found {len(all_files)} image files")
    
    # Sample files
    sample_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    # Try to open each file
    errors = 0
    for file in sample_files:
        file_path = os.path.join(dataset_path, file)
        try:
            with Image.open(file_path) as img:
                # Try to access image properties to ensure it's valid
                _ = img.size
        except Exception as e:
            logger.error(f"Error opening {file_path}: {e}")
            errors += 1
    
    if errors > 0:
        logger.error(f"Found {errors} errors in {len(sample_files)} sample files")
        return False
    
    logger.info(f"Successfully verified {len(sample_files)} sample files")
    return True

def custom_collate_fn(batch):
    """Custom collate function to handle variable sized items."""
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    collated_batch = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'image':
            collated_batch[key] = torch.stack([item[key] for item in batch])
        elif key in ['input_ids', 'attention_mask']:
            collated_batch[key] = torch.stack([item[key] for item in batch])
        elif key in ['uid', 'medical_concepts', 'filename', 'report']:
            collated_batch[key] = [item[key] for item in batch]
        else:
            collated_batch[key] = [item[key] for item in batch]
    
    return collated_batch


# ========================================================================
# Dataset Implementation
# ========================================================================

class MedicalReport:
    """
    Class to handle medical report text processing and normalization.
    """
    # Common sections in radiology reports
    SECTIONS = ["findings", "impression", "indication", "comparison", "technique"]
    
    # Common medical imaging abbreviations and their expansions
    ABBREVIATIONS = {
        "w/": "with",
        "w/o": "without",
        "b/l": "bilateral",
        "AP": "anteroposterior",
        "PA": "posteroanterior",
        "lat": "lateral",
    }
    
    @staticmethod
    def normalize_text(text):
        """Normalize and clean text content."""
        if pd.isna(text) or text is None:
            return ""
            
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Replace multiple whitespace with single space
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def preprocess_report(findings, impression):
        """
        Combine findings and impression with proper section markers.
        """
        findings = MedicalReport.normalize_text(findings)
        impression = MedicalReport.normalize_text(impression)
        
        # Build report with section markers
        report_parts = []
        
        if findings:
            report_parts.append(f"FINDINGS: {findings}")
        
        if impression:
            report_parts.append(f"IMPRESSION: {impression}")
            
        # Join sections with double newline for clear separation
        return " ".join(report_parts)
    
    @staticmethod
    def extract_medical_concepts(text):
        """
        Extract key medical concepts from text.
        Simple keyword-based extraction.
        """
        # Simple keyword-based extraction
        key_findings = []
        
        # Common radiological findings
        findings_keywords = [
            "pneumonia", "effusion", "edema", "cardiomegaly", 
            "atelectasis", "consolidation", "pneumothorax", "mass",
            "nodule", "infiltrate", "fracture", "opacity"
        ]
        
        # Check for keywords
        for keyword in findings_keywords:
            if keyword in text.lower():
                key_findings.append(keyword)
                
        return key_findings

class ChestXrayDataset(Dataset):
    """
    Dataset for chest X-ray images and reports from the IU dataset.
    """
    def __init__(
        self,
        reports_csv,
        projections_csv,
        image_folder,
        transform=None,
        target_size=(256, 256),
        filter_frontal=True,
        tokenizer_name="dmis-lab/biobert-base-cased-v1.1",
        max_length=256,
        load_tokenizer=True
    ):
        """Initialize the chest X-ray dataset."""
        self.image_folder = image_folder
        self.transform = transform
        self.target_size = target_size
        self.max_length = max_length
        # Add in ChestXrayDataset.__init__:
        self.use_clahe = USE_CLAHE
        self.report_processor = MedicalReport()
        
        # Load data with proper error handling
        try:
            logger.info(f"Loading reports from {reports_csv}")
            reports_df = pd.read_csv(reports_csv)
            
            logger.info(f"Loading projections from {projections_csv}")
            projections_df = pd.read_csv(projections_csv)
            
            # Log initial data statistics
            logger.info(f"Loaded reports CSV with {len(reports_df)} entries")
            logger.info(f"Loaded projections CSV with {len(projections_df)} entries")
            
            # Merge datasets on uid
            merged_df = pd.merge(reports_df, projections_df, on='uid')
            logger.info(f"Merged dataframe has {len(merged_df)} entries")
            
            # Filter for frontal projections if requested
            if filter_frontal:
                frontal_df = merged_df[merged_df['projection'] == 'Frontal'].reset_index(drop=True)
                logger.info(f"Filtered for frontal projections: {len(frontal_df)}/{len(merged_df)} entries")
                merged_df = frontal_df
                
            # Filter for entries with both findings and impression
            valid_df = merged_df.dropna(subset=['findings', 'impression']).reset_index(drop=True)
            logger.info(f"Filtered for valid reports: {len(valid_df)}/{len(merged_df)} entries")
            
            # Verify image files exist
            self.data = self._filter_existing_images(valid_df)
            
            # Load tokenizer if requested
            self.tokenizer = None
            if load_tokenizer:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    logger.info(f"Loaded tokenizer: {tokenizer_name}")
                except Exception as e:
                    logger.error(f"Error loading tokenizer: {e}")
                    logger.warning("Proceeding without tokenizer")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise
    
    def _filter_existing_images(self, df):
        """Filter dataframe to only include entries with existing image files."""
        valid_entries = []
        missing_files = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying image files"):
            img_path = os.path.join(self.image_folder, row['filename'])
            if os.path.exists(img_path):
                valid_entries.append(idx)
            else:
                missing_files += 1
                
        if missing_files > 0:
            logger.warning(f"Found {missing_files} missing image files out of {len(df)}")
            
        # Keep only entries with existing files
        valid_df = df.iloc[valid_entries].reset_index(drop=True)
        logger.info(f"Final dataset size after filtering: {len(valid_df)} entries")
        
        return valid_df
    
    def __len__(self):
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get dataset item with proper error handling."""
        try:
            row = self.data.iloc[idx]
            
            # Process image
            img_path = os.path.join(self.image_folder, row['filename'])
            
            # Check file existence (safety check)
            if not os.path.exists(img_path):
                logger.error(f"Image file not found despite prior filtering: {img_path}")
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Load and convert to grayscale
            try:
                img = Image.open(img_path).convert('L')
            except Exception as e:
                logger.error(f"Error opening image {img_path}: {e}")
                raise ValueError(f"Cannot open image: {e}")
            
            # Apply preprocessing
            img = self._preprocess_image(img)
            
            # Process report text
            report = self.report_processor.preprocess_report(
                row['findings'], row['impression']
            )
            
            # Extract key medical concepts for metadata
            medical_concepts = self.report_processor.extract_medical_concepts(report)
            
            # Create return dictionary
            item = {
                'image': img,
                'report': report,
                'uid': row['uid'],
                'medical_concepts': medical_concepts,
                'filename': row['filename']
            }
            
            # Add tokenized text if tokenizer is available
            if self.tokenizer:
                encoding = self._tokenize_text(report)
                item.update(encoding)
                
            return item
            
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            
            # For debugging only - in production we would handle this more gracefully
            raise e
    
    def _preprocess_image(self, img):
        """Preprocess image with standardized steps for medical imaging."""
        # Resize with proper interpolation for medical images
        if img.size != self.target_size:
            img = img.resize(self.target_size, Image.BICUBIC)
            
        # Convert to tensor [0, 1]
        img_tensor = TF.to_tensor(img)
        
        # Apply CLAHE preprocessing if enabled
        if hasattr(self, 'use_clahe') and self.use_clahe:
            img_np = img_tensor.numpy().squeeze()
            
            # Normalize to 0-255 range
            img_np = (img_np * 255).astype(np.uint8)
            
            # Apply CLAHE
            import cv2
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
            
            # Convert back to tensor [0, 1]
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
        # Apply additional transforms if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor
    
    def _tokenize_text(self, text):
        """Tokenize text with proper padding and truncation."""
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

class MedicalDataModule:
    """
    Data module to handle dataset splitting, preparation, and loading.
    """
    def __init__(
        self,
        reports_csv,
        projections_csv,
        image_folder,
        batch_size=16,
        num_workers=4,
        val_split=0.1,
        test_split=0.1,
        image_size=256,
        max_token_length=256,
        tokenizer_name="dmis-lab/biobert-base-cased-v1.1",
        seed=42,
        timeout=120
    ):
        """Initialize the data module."""
        self.reports_csv = reports_csv
        self.projections_csv = projections_csv
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.image_size = image_size
        self.max_token_length = max_token_length
        self.tokenizer_name = tokenizer_name
        self.seed = seed
        self.timeout = timeout
        
        # Create transforms
        self.train_transform = T.Compose([
            # Normalize to [-1, 1] range for diffusion models
            T.Normalize([0.5], [0.5])
        ])
        
        self.val_transform = T.Compose([
            T.Normalize([0.5], [0.5])
        ])
        
        # Dataset and loaders will be set up later
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Prepare the dataset (download, verify, etc.)."""
        # Verify that files exist
        assert os.path.exists(self.reports_csv), f"Reports CSV not found: {self.reports_csv}"
        assert os.path.exists(self.projections_csv), f"Projections CSV not found: {self.projections_csv}"
        assert os.path.exists(self.image_folder), f"Image folder not found: {self.image_folder}"
        
        # Verify dataset files
        verify_dataset_files(self.image_folder, sample_size=50)
        
        # Log preparation
        logger.info("Data preparation complete")
        
    def setup(self):
        """Set up the datasets for training, validation, and testing."""
        # Create full dataset
        logger.info("Creating dataset...")
        try:
            self.dataset = ChestXrayDataset(
                reports_csv=self.reports_csv,
                projections_csv=self.projections_csv,
                image_folder=self.image_folder,
                transform=None,  # Will be set per split
                target_size=(self.image_size, self.image_size),
                filter_frontal=True,
                tokenizer_name=self.tokenizer_name,
                max_length=self.max_token_length
            )
            
            # Test loading a few items to ensure dataset works
            logger.info("Testing dataset by loading a few items...")
            for i in range(min(5, len(self.dataset))):
                try:
                    item = self.dataset[i]
                    logger.info(f"Successfully loaded item {i}: {item['filename']}")
                except Exception as e:
                    logger.error(f"Failed to load item {i}: {e}")
                    
            logger.info(f"Dataset created with {len(self.dataset)} items")
            
            # Calculate split sizes
            dataset_size = len(self.dataset)
            val_size = int(self.val_split * dataset_size)
            test_size = int(self.test_split * dataset_size)
            train_size = dataset_size - val_size - test_size
            
            # Create splits
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size], generator=generator
            )
            
            # Set transforms for each split
            self._set_dataset_transform(self.train_dataset, self.train_transform)
            self._set_dataset_transform(self.val_dataset, self.val_transform)
            self._set_dataset_transform(self.test_dataset, self.val_transform)
            
            # Log split sizes
            logger.info(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
        
        except Exception as e:
            logger.error(f"Error setting up dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
    def _set_dataset_transform(self, dataset, transform):
        """Set transform for a specific dataset split."""
        # Since dataset is a Subset, we need to set the transform on the underlying dataset
        # But we only want it to apply to this specific subset
        dataset.transform = transform
        
        # Monkey patch the __getitem__ method to apply our transform
        original_getitem = dataset.__getitem__
        
        def new_getitem(idx):
            item = original_getitem(idx)
            if dataset.transform and 'image' in item and item['image'] is not None:
                item['image'] = dataset.transform(item['image'])
            return item
            
        dataset.__getitem__ = new_getitem
        
        
        
        from torch.utils.data import DataLoader  # ensure this is imported

    def train_dataloader(self):
        loader_args = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True,
            'worker_init_fn': lambda worker_id: np.random.seed(self.seed + worker_id),
            'collate_fn': custom_collate_fn
        }
        if self.num_workers > 0:
            loader_args.update({
                'timeout': self.timeout,
                'persistent_workers': True,
                'prefetch_factor': 2
            })
        return DataLoader(self.train_dataset, **loader_args)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=self.timeout,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=custom_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            timeout=self.timeout,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=custom_collate_fn
        )


# ========================================================================
# Model Components
# ========================================================================

class MedicalTextEncoder(nn.Module):
    """
    Text encoder for medical reports using BioBERT or other biomedical models.
    """
    def __init__(
        self,
        model_name="dmis-lab/biobert-base-cased-v1.1",
        projection_dim=768,
        freeze_base=True
    ):
        """Initialize the text encoder."""
        super().__init__()
        
        # Load the model with proper error handling
        try:
            self.transformer = AutoModel.from_pretrained(model_name)
            self.model_name = model_name
            logger.info(f"Loaded text encoder: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}")
            logger.warning("Falling back to bert-base-uncased")
            self.transformer = AutoModel.from_pretrained("bert-base-uncased")
            self.model_name = "bert-base-uncased"
        
        # Get transformer hidden dimension
        self.hidden_dim = self.transformer.config.hidden_size
        self.projection_dim = projection_dim
        
        # Projection layer with layer normalization for stability
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        # Freeze base transformer if requested
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info(f"Froze base transformer parameters")
    
    def forward(self, input_ids, attention_mask):
        """Forward pass through the text encoder."""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # Apply projection
        return self.projection(hidden_states)

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
        device=device
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
        self.device = device

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
        latents=None
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
            max_length=MAX_TOKEN_LENGTH,
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
        for i, t in enumerate(tqdm(timesteps, desc="Generating image", leave=False)):
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
        
        # Scale latents
        latents = 1 / 0.18215 * latents
        
        # Decode latents
        images = self.vae.decode(latents)
        
        # Normalize to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        return {
            'images': images,
            'latents': latents
        }

# ========================================================================
# Training Utilities
# ========================================================================

def visualize_training_progress(epoch, model, val_batch, metrics=None):
    """Display current training progress in Jupyter"""
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize=(15, 7))
    
    # Plot metrics if available
    if metrics and len(metrics.get('train_loss', [])) > 0:
        plt.subplot(2, 3, 1)
        plt.plot(metrics['train_loss'], label='Train Loss')
        if 'val_loss' in metrics:
            plt.plot(metrics['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.legend()
    
    # Show original image
    plt.subplot(2, 3, 2)
    try:
        img = val_batch['image'][0].cpu().numpy().squeeze()
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
    except:
        plt.text(0.5, 0.5, "Image not available", ha='center', va='center')
    
    # Show VAE reconstruction
    plt.subplot(2, 3, 3)
    try:
        with torch.no_grad():
            if isinstance(model, dict):
                vae = model['vae']
            else:
                vae = model.vae
            vae.eval()
            recon, _, _ = vae(val_batch['image'].to(vae.encoder.conv_in.weight.device))
            recon_img = recon[0].cpu().numpy().squeeze()
        plt.imshow(recon_img, cmap='gray')
        plt.title('VAE Reconstruction')
        plt.axis('off')
    except Exception as e:
        plt.text(0.5, 0.5, f"Reconstruction failed: {str(e)}", ha='center', va='center', wrap=True)
    
    # Show report/prompt
    if 'report' in val_batch:
        plt.subplot(2, 3, 4)
        report = val_batch['report'][0]
        if len(report) > 200:
            report = report[:200] + "..."
        plt.text(0.5, 0.5, f"Report: {report}", ha='center', va='center', wrap=True)
        plt.axis('off')
    
    # Show generated image if diffusion model
    plt.subplot(2, 3, 5)
    try:
        if hasattr(model, 'sample') and not isinstance(model, dict):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
            
            # Get short prompt from report
            report = val_batch['report'][0]
            if 'FINDINGS:' in report:
                prompt = report.split('FINDINGS:')[1].split('IMPRESSION:')[0].strip()
                if len(prompt) > 100:
                    prompt = prompt[:100]
            else:
                prompt = report[:100]
                
            # Generate sample
            with torch.no_grad():
                results = model.sample(
                    prompt, 
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    num_inference_steps=20,  # Faster for preview
                    tokenizer=tokenizer
                )
                gen_img = results['images'][0].cpu().numpy().squeeze()
                plt.imshow(gen_img, cmap='gray')
                plt.title('Generated from Text')
                plt.axis('off')
    except Exception as e:
        plt.text(0.5, 0.5, f"Generation failed: {str(e)}", ha='center', va='center', wrap=True)
        plt.axis('off')
    
    # Add title with epoch info
    plt.suptitle(f'Training Progress - Epoch {epoch}', fontsize=16)
    plt.tight_layout()
    
    # Display in notebook
    clear_output(wait=True)
    display(plt.gcf())
    plt.close()


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
                img = img.to(next(model['vae'].parameters()).device)
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
            
            # Generate samples - FIX: Set individual components to eval mode instead of model.eval()
            model.vae.eval()
            model.text_encoder.eval()
            model.unet.eval()
            
            with torch.no_grad():
                for i, prompt in enumerate(sample_prompts):
                    results = model.sample(
                        prompt,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
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
                    Image.fromarray(img_np.astype(np.uint8)).save(img_path)
                    
                    # Save prompt
                    prompt_path = os.path.join(samples_dir, f"prompt_{i+1}.txt")
                    with open(prompt_path, "w") as f:
                        f.write(prompt)
        except Exception as e:
            logger.error(f"Error generating samples from prompts: {e}")
    
    logger.info(f"Saved visualization for epoch {epoch+1} to {samples_dir}")

def vae_loss_fn(recon_x, x, mu, logvar, kld_weight=1e-4):
    """VAE loss function."""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    loss = recon_loss + kld_weight * kld_loss
    
    return loss, recon_loss, kld_loss

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
    
    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_metrics, checkpoint_dir, is_best=False):
    """Save checkpoint every CHECKPOINT_FREQ epochs plus best model"""
    # Only save at specified frequency or if best model
    if not is_best and (epoch % CHECKPOINT_FREQ != 0):
        return
        
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
            'global_step': global_step
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
        cleanup_old_checkpoints(checkpoint_dir, KEEP_LAST_CHECKPOINTS)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load checkpoint and resume training"""
    if not os.path.exists(path):
        logger.info(f"No checkpoint found at {path}")
        return 0, 0, {'val_loss': float('inf')}
        
    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=model.device if hasattr(model, 'device') else 'cuda')
    
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
            

# ========================================================================
# Training Functions
# ========================================================================

def train_vae(vae_model, train_loader, val_loader, optimizer, device, num_epochs=50, checkpoint_dir=None, use_amp=True, save_checkpoints=True, resume_from=None):
    """Train VAE model with checkpoint and gradient accumulation support."""
    logger.info("Starting VAE training")
    
    # Debug - log values before first iteration
    logger.info("Debug - Before VAE training loop")
    logger.info(f"VAE model on device: {next(vae_model.parameters()).device}")
    logger.info(f"Optimizer state: {optimizer.state_dict()['param_groups'][0]['lr']}")
    
    # Test dataloader by extracting first batch
    logger.info("Testing dataloader by extracting first batch...")
    
    # Try to get the first batch
    try:
        logger.info("Attempting to get first batch from train_loader")
        first_batch = next(iter(train_loader))
        logger.info(f"First batch keys: {list(first_batch.keys())}")
        logger.info(f"First batch image shape: {first_batch['image'].shape}")
        logger.info(f"First batch loaded successfully")
        # Free memory
        del first_batch
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error loading first batch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError("Failed to load first batch - check dataset and dataloader configuration")
    
    # Set up gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * num_epochs // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Training state tracking
    start_epoch = 0
    global_step = 0
    best_metrics = {'val_loss': float('inf')}
    metrics = {'train_loss': [], 'val_loss': [], 'epochs': []}
    
    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        start_epoch, global_step, best_metrics = load_checkpoint(
            {'vae': vae_model}, optimizer, scheduler, resume_from
        )
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Early stopping
    if checkpoint_dir:
        early_stopping_path = os.path.join(checkpoint_dir, "best_vae.pt")
    else:
        early_stopping_path = "best_vae.pt"
        
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        path=early_stopping_path
    )
    
    # Best model tracking
    best_loss = best_metrics.get('val_loss', float('inf'))
    best_model_state = None
    
    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Starting VAE epoch {epoch+1}/{num_epochs}")
            
            # Training
            vae_model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kld_loss = 0.0
            
            # Initialize gradient accumulation
            optimizer.zero_grad()
            
            # Train loop with progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (VAE Training)")
            start_time = time.time()  
            for batch_idx, batch in enumerate(progress_bar):
                # Debug logging
                if VERBOSE_LOGGING and (batch_idx == 0 or batch_idx % 10 == 0):
                    elapsed = time.time() - start_time
                    logger.info(f"Processing batch {batch_idx} (Elapsed: {elapsed:.2f}s, Processed: {batch_idx})")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()  # Ensure CUDA operations are complete
                        log_gpu_memory(f"Batch {batch_idx}")
                
                try:
                    # Debug - log first batch
                    if epoch == 0 and batch_idx == 0:
                        logger.info(f"First batch keys: {list(batch.keys())}")
                        logger.info(f"Image shape: {batch['image'].shape}")
                        
                    # Get images
                    images = batch['image'].to(device)
                    
                    # Skip problematic batches
                    if images.shape[0] < 2:  # Need at least 2 samples for batch norm
                        logger.warning(f"Skipping batch with only {images.shape[0]} samples")
                        continue
                    
                    # Forward pass with mixed precision
                    if use_amp and torch.cuda.is_available():
                        with autocast():
                            recon, mu, logvar = vae_model(images)
                            loss, recon_loss, kld_loss = vae_loss_fn(recon, images, mu, logvar)
                            # Scale loss for gradient accumulation
                            loss = loss / GRADIENT_ACCUMULATION_STEPS
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Step with gradient accumulation
                        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or batch_idx + 1 == len(train_loader):
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            
                            # Update scheduler
                            scheduler.step()
                            global_step += 1
                    else:
                        recon, mu, logvar = vae_model(images)
                        loss, recon_loss, kld_loss = vae_loss_fn(recon, images, mu, logvar)
                        # Scale loss for gradient accumulation
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                        
                        loss.backward()
                        
                        # Step with gradient accumulation
                        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or batch_idx + 1 == len(train_loader):
                            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Update scheduler
                            scheduler.step()
                            global_step += 1
                    
                    # Update metrics (using original loss)
                    train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                    train_recon_loss += recon_loss.item()
                    train_kld_loss += kld_loss.item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'kld': f"{kld_loss.item():.4f}"
                    })
                        
                except Exception as e:
                    logger.error(f"Error in VAE training batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Calculate average training losses
            train_loss /= max(1, len(train_loader))
            train_recon_loss /= max(1, len(train_loader))
            train_kld_loss /= max(1, len(train_loader))
            
            # Update metrics tracking
            metrics['train_loss'].append(train_loss)
            metrics['epochs'].append(epoch)
            
            # Validation
            vae_model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kld_loss = 0.0
            
            with torch.no_grad():
                # Validation loop with progress bar
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (VAE Validation)")
                for batch_idx, batch in enumerate(val_progress):
                    try:
                        # Get images
                        images = batch['image'].to(device)
                        
                        # Skip problematic batches
                        if images.shape[0] < 2:
                            continue
                        
                        # Forward pass
                        recon, mu, logvar = vae_model(images)
                        loss, recon_loss, kld_loss = vae_loss_fn(recon, images, mu, logvar)
                        
                        # Update metrics
                        val_loss += loss.item()
                        val_recon_loss += recon_loss.item()
                        val_kld_loss += kld_loss.item()
                        
                        # Visualize in notebook if it's the first batch
                        if batch_idx == 0:
                            try:
                                # Check if in Jupyter environment
                                if 'get_ipython' in globals():
                                    visualize_training_progress(
                                        epoch+1, 
                                        {'vae': vae_model}, 
                                        batch, 
                                        metrics
                                    )
                            except Exception as e:
                                logger.error(f"Error in visualization: {e}")
                    
                    except Exception as e:
                        logger.error(f"Error in VAE validation: {e}")
                        continue
            
            # Calculate average validation losses
            val_loss /= max(1, len(val_loader))
            val_recon_loss /= max(1, len(val_loader))
            val_kld_loss /= max(1, len(val_loader))
            
            # Update metrics tracking
            metrics['val_loss'].append(val_loss)
            
            # Log metrics
            logger.info(f"VAE Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KLD: {train_kld_loss:.4f}) | "
                      f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, KLD: {val_kld_loss:.4f})")
            
            # Check if this is the best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = vae_model.state_dict().copy()
                
                # Save best checkpoint
                if save_checkpoints and checkpoint_dir:
                    save_checkpoint(
                        {'vae': vae_model},
                        optimizer,
                        scheduler,
                        epoch+1,
                        global_step,
                        {'val_loss': val_loss},
                        checkpoint_dir,
                        is_best=True
                    )
            
            # Save regular checkpoint
            if save_checkpoints and checkpoint_dir:
                save_checkpoint(
                    {'vae': vae_model},
                    optimizer,
                    scheduler,
                    epoch+1,
                    global_step,
                    {'val_loss': val_loss},
                    checkpoint_dir,
                    is_best=False
                )
            
            # Check early stopping
            if early_stopping(val_loss, vae_model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Visualize results after each epoch
            if checkpoint_dir:
                visualize_epoch_results(
                    epoch,
                    {"vae": vae_model},
                    None,
                    val_loader,
                    checkpoint_dir
                )
    
    except Exception as e:
        logger.error(f"Error during VAE training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Save emergency checkpoint
        if checkpoint_dir and save_checkpoints:
            emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_metrics': {'val_loss': best_loss},
                'global_step': global_step
            }, emergency_path)
            logger.info(f"Saved emergency checkpoint to {emergency_path}")
    
    # Return best model state
    if best_model_state is not None:
        logger.info(f"VAE training complete. Best validation loss: {best_loss:.4f}")
        return best_model_state
    else:
        logger.warning("VAE training complete, but no best model state was saved.")
        return vae_model.state_dict()

def train_diffusion(model, train_loader, val_loader, optimizer, device, num_epochs=50, checkpoint_dir=None, use_amp=True, train_unet_only=True, tokenizer=None, save_checkpoints=True):
    """Train diffusion model."""
    logger.info("Starting diffusion model training")
    logger.info(f"Training {'UNet only' if train_unet_only else 'all components'}")
    # Add near the beginning of train_diffusion function
    global_step = 0
    # Test dataloader by extracting first batch
    logger.info("Testing diffusion dataloader by extracting first batch...")
    
    # Try to get the first batch
    try:
        logger.info("Attempting to get first batch from train_loader")
        first_batch = next(iter(train_loader))
        logger.info(f"First batch keys: {list(first_batch.keys())}")
        logger.info(f"First batch image shape: {first_batch['image'].shape}")
        logger.info(f"First batch loaded successfully")
        
        # Debug: Try a forward pass
        logger.info("Testing forward pass with first batch...")
        with torch.no_grad():
            loss, metrics = model.training_step(first_batch, train_unet_only)
            logger.info(f"Forward pass successful. Loss: {loss.item()}, Metrics: {metrics}")
        
        # Free memory
        del first_batch
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error in diffusion dataloader test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError("Failed to test diffusion dataloader - check configuration")
    
    # Set up gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
        
    # Early stopping
    if checkpoint_dir:
        early_stopping_path = os.path.join(checkpoint_dir, "best_diffusion.pt")
    else:
        early_stopping_path = "best_diffusion.pt"
        
    early_stopping = EarlyStopping(
        patience=8,
        verbose=True,
        path=early_stopping_path
    )
    
    # Best model tracking
    best_loss = float('inf')
    
    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting diffusion epoch {epoch+1}/{num_epochs}")
            
            # Training
            if train_unet_only:
                model.vae.eval()
                model.text_encoder.eval()
                model.unet.train()
            else:
                model.vae.train()
                model.text_encoder.train()
                model.unet.train()
                
            train_loss = 0.0
            train_diffusion_loss = 0.0
            train_vae_loss = 0.0
            
            # Debug counter for batch tracking
            processed_batches = 0
            start_time = time.time()
            
            # Train loop with progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)")
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Debug logging
                    if VERBOSE_LOGGING and (batch_idx == 0 or batch_idx % 10 == 0):
                        elapsed = time.time() - start_time
                        logger.info(f"Processing batch {batch_idx} (Elapsed: {elapsed:.2f}s, Processed: {processed_batches})")
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()  # Ensure CUDA operations are complete
                            log_gpu_memory(f"Batch {batch_idx}")
                    
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    if use_amp and torch.cuda.is_available():
                        with autocast():
                            loss, metrics = model.training_step(batch, train_unet_only)
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Gradient clipping
                        if train_unet_only:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
                        else:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                list(model.vae.parameters()) +
                                list(model.text_encoder.parameters()) +
                                list(model.unet.parameters()),
                                max_norm=1.0
                            )
                        
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss, metrics = model.training_step(batch, train_unet_only)
                        
                        loss.backward()
                        
                        # Gradient clipping
                        if train_unet_only:
                            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), max_norm=1.0)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                list(model.vae.parameters()) +
                                list(model.text_encoder.parameters()) +
                                list(model.unet.parameters()),
                                max_norm=1.0
                            )
                        
                        optimizer.step()
                    
                    # Update learning rate
                    scheduler.step()
                    
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
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
                    
                    # Force progress bar update
                    progress_bar.update(0)
                    
                    # Explicit CUDA synchronization
                    if torch.cuda.is_available() and (batch_idx % 10 == 0):
                        torch.cuda.synchronize()
                    
                except Exception as e:
                    logger.error(f"Error in diffusion training batch {batch_idx}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Calculate average training losses
            train_loss /= max(1, len(train_loader))
            train_diffusion_loss /= max(1, len(train_loader))
            if not train_unet_only:
                train_vae_loss /= max(1, len(train_loader))
            
            # Validation
            model.vae.eval()
            model.text_encoder.eval()
            model.unet.eval()
            
            val_loss = 0.0
            val_diffusion_loss = 0.0
            val_vae_loss = 0.0
            
            with torch.no_grad():
                # Validation loop with progress bar
                val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)")
                for batch_idx, batch in enumerate(val_progress):
                    try:
                        # Compute validation metrics
                        metrics = model.validation_step(batch)
                        
                        # Update metrics
                        val_loss += metrics['val_loss']
                        val_diffusion_loss += metrics['val_diffusion_loss']
                        val_vae_loss += metrics['val_vae_loss']
                        
                    except Exception as e:
                        logger.error(f"Error in diffusion validation batch {batch_idx}: {e}")
                        continue
            
            # Calculate average validation losses
            val_loss /= max(1, len(val_loader))
            val_diffusion_loss /= max(1, len(val_loader))
            val_vae_loss /= max(1, len(val_loader))
            
            # All these post-validation actions should be indented at the same level
            # as the validation code - INSIDE the epoch loop
            if checkpoint_dir:
                visualize_epoch_results(
                    epoch,
                    model,
                    tokenizer,
                    val_loader,
                    checkpoint_dir
                )
                
                # Log metrics
                vae_loss_str = f", VAE: {train_vae_loss:.4f}/{val_vae_loss:.4f}" if not train_unet_only else ""
                logger.info(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Train/Val Loss: {train_loss:.4f}/{val_loss:.4f} | "
                          f"Diff: {train_diffusion_loss:.4f}/{val_diffusion_loss:.4f}"
                          f"{vae_loss_str}")
                
                # Save checkpoint if enabled
                if save_checkpoints and checkpoint_dir:
                    # Regular checkpoint
                    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                        metrics = {
                            'train_loss': train_loss,
                            'train_diffusion_loss': train_diffusion_loss,
                            'val_loss': val_loss,
                            'val_diffusion_loss': val_diffusion_loss
                        }
                        
                        checkpoint_path = os.path.join(checkpoint_dir, f"diffusion_epoch_{epoch+1}.pt")
                        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, metrics, checkpoint_dir, is_best=False)
                    
                    # Save if best model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_path = os.path.join(checkpoint_dir, "best_diffusion.pt")
                        
                        metrics = {
                            'train_loss': train_loss,
                            'train_diffusion_loss': train_diffusion_loss,
                            'val_loss': val_loss,
                            'val_diffusion_loss': val_diffusion_loss
                        }
                        
                        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, metrics, checkpoint_dir, is_best=True)
                        logger.info(f"New best model saved with val_loss={val_loss:.4f}")
                
                # Generate samples every 10 epochs if tokenizer is available
                if tokenizer is not None and ((epoch + 1) % 10 == 0 or epoch == num_epochs - 1):
                    try:
                        # Sample prompts
                        sample_prompts = [
                            "Normal chest X-ray with clear lungs and no abnormalities.",
                            "Right lower lobe pneumonia with focal consolidation."
                        ]
                        
                        # Generate and save samples
                        logger.info("Generating sample images...")
                        
                        model.vae.eval()
                        model.text_encoder.eval()
                        model.unet.eval()
                        samples_dir = os.path.join(checkpoint_dir if checkpoint_dir else ".", "samples")
                        os.makedirs(samples_dir, exist_ok=True)
                        
                        with torch.no_grad():
                            for i, prompt in enumerate(sample_prompts):
                                results = model.sample(
                                    prompt,
                                    height=IMAGE_SIZE,
                                    width=IMAGE_SIZE,
                                    num_inference_steps=30,
                                    tokenizer=tokenizer
                                )
                                
                                # Save image
                                img = results['images'][0]
                                img_np = img.cpu().numpy().transpose(1, 2, 0)
                                img_np = (img_np * 255).astype(np.uint8)
                                if img_np.shape[-1] == 1:
                                    img_np = img_np.squeeze(-1)
                                    
                                img_path = os.path.join(samples_dir, f"sample_epoch{epoch+1}_{i}.png")
                                Image.fromarray(img_np).save(img_path)
                                
                        logger.info(f"Saved sample images to {samples_dir}")
                        
                    except Exception as e:
                        logger.error(f"Error generating samples: {e}")
            
            # Early stopping - still inside the epoch loop
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # End of epoch for loop
    except Exception as e:
        logger.error(f"Error during diffusion training: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Outside the try-except block but still in the function
    # Load best model
    best_path = os.path.join(checkpoint_dir, "best_diffusion.pt") if checkpoint_dir else "best_diffusion.pt"
    if os.path.exists(best_path):
        _, _ = load_checkpoint(model, None, None, best_path)
        logger.info("Loaded best model from saved checkpoint")
    
    logger.info("Diffusion model training complete")
    
    return model


# ========================================================================
# Main Training Loop
# ========================================================================

def main():
    """Main function to train the model."""
    logger.info("Starting training pipeline")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check paths before proceeding
    logger.info(f"Checking dataset paths...")
    logger.info(f"Dataset path: {DATASET_PATH} - Exists: {os.path.exists(DATASET_PATH)}")
    logger.info(f"Images path: {IMAGES_PATH} - Exists: {os.path.exists(IMAGES_PATH)}")
    logger.info(f"Reports CSV: {REPORTS_CSV} - Exists: {os.path.exists(REPORTS_CSV)}")
    logger.info(f"Projections CSV: {PROJECTIONS_CSV} - Exists: {os.path.exists(PROJECTIONS_CSV)}")
    
    # Verify image files
    if not verify_dataset_files(IMAGES_PATH, sample_size=20):
        logger.error("Image file verification failed. Please check your dataset paths.")
        # Continue anyway, as we'll filter missing files later
    
    # Prepare data module
    logger.info("Initializing data module")
    data_module = MedicalDataModule(
        reports_csv=REPORTS_CSV,
        projections_csv=PROJECTIONS_CSV,
        image_folder=IMAGES_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_split=0.1,
        test_split=0.1,
        image_size=IMAGE_SIZE,
        max_token_length=MAX_TOKEN_LENGTH,
        tokenizer_name=TEXT_MODEL,
        seed=RANDOM_SEED,
        timeout=DATALOADER_TIMEOUT
    )
    
    # Set up dataset
    logger.info("Setting up dataset")
    data_module.prepare_data()
    data_module.setup()
    
    # Get data loaders
    logger.info("Creating dataloaders")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Initialize models
    logger.info("Initializing models")
    
    # VAE
    vae = MedicalVAE(
        in_channels=1,
        out_channels=1, 
        latent_channels=LATENT_CHANNELS,  # Using our new increased value
        hidden_dims=[MODEL_CHANNELS, MODEL_CHANNELS*2, MODEL_CHANNELS*4, MODEL_CHANNELS*8]
    ).to(device)
        
    # Text encoder
    text_encoder = MedicalTextEncoder(
        model_name=TEXT_MODEL,
        projection_dim=768,
        freeze_base=True
    ).to(device)
    
    # UNet
    unet = DiffusionUNet(
        in_channels=LATENT_CHANNELS,
        model_channels=MODEL_CHANNELS,
        out_channels=LATENT_CHANNELS,
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
        scheduler_type="ddim",  # More efficient sampling
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
        guidance_scale=GUIDANCE_SCALE,
        device=device
    )
    
    # Log model parameters
    total_params = sum(p.numel() for p in vae.parameters()) + \
                   sum(p.numel() for p in unet.parameters()) + \
                   sum(p.numel() for p in text_encoder.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    logger.info(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    logger.info(f"Text encoder parameters: {sum(p.numel() for p in text_encoder.parameters()):,}")
    
    # Train VAE first
    logger.info("Starting VAE training")
    vae_checkpoint_dir = os.path.join(OUTPUT_DIR, "vae_checkpoints")
    os.makedirs(vae_checkpoint_dir, exist_ok=True)
    
    # Create VAE optimizer
    vae_optimizer = AdamW(vae.parameters(), lr=LR_VAE, weight_decay=1e-6)
    
    # Train VAE
    try:
        vae_state_dict = train_vae(
            vae_model=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=vae_optimizer,
            device=device,
            num_epochs=EPOCHS_VAE,
            checkpoint_dir=vae_checkpoint_dir,
            use_amp=USE_AMP,
            save_checkpoints=SAVE_CHECKPOINTS,
            resume_from=RESUME_FROM  # Add resume option
        )
            
        # Load best VAE weights
        vae.load_state_dict(vae_state_dict)
        logger.info("VAE training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during VAE training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Continuing with diffusion training using current VAE weights")
    
    # Create tokenizer for sampling
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        logger.warning("Will not generate samples during training")
        tokenizer = None
    
    # Train diffusion model
    logger.info("Starting diffusion model training")
    diffusion_checkpoint_dir = os.path.join(OUTPUT_DIR, "diffusion_checkpoints")
    os.makedirs(diffusion_checkpoint_dir, exist_ok=True)
    
    # Create diffusion optimizer
    if TRAIN_UNET_ONLY:
        diffusion_optimizer = AdamW(unet.parameters(), lr=LR_DIFFUSION, weight_decay=1e-6)
    else:
        parameters = list(unet.parameters())
        parameters.extend(vae.parameters())
        parameters.extend(text_encoder.parameters())
        diffusion_optimizer = AdamW(parameters, lr=LR_DIFFUSION, weight_decay=1e-6)
    
    # Train diffusion model
    try:
        diffusion_model = train_diffusion(
            model=diffusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=diffusion_optimizer,
            device=device,
            num_epochs=EPOCHS_DIFFUSION,
            checkpoint_dir=diffusion_checkpoint_dir,
            use_amp=USE_AMP,
            train_unet_only=TRAIN_UNET_ONLY,
            tokenizer=tokenizer,
            save_checkpoints=SAVE_CHECKPOINTS
        )
        
        logger.info("Diffusion model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during diffusion training: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Generate samples
    try:
        if tokenizer is not None:
            logger.info("Generating final samples")
            
            samples_dir = os.path.join(OUTPUT_DIR, "final_samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            # Sample prompts
            sample_prompts = [
                "Normal chest X-ray with clear lungs and no abnormalities.",
                "Right lower lobe pneumonia with focal consolidation.",
                "Bilateral pleural effusions, greater on the right.",
                "Cardiomegaly with pulmonary vascular congestion."
            ]
            
            # Generate samples
            diffusion_model.vae.eval()
            diffusion_model.text_encoder.eval()
            diffusion_model.unet.eval()
            with torch.no_grad():
                for i, prompt in enumerate(sample_prompts):
                    logger.info(f"Generating sample for prompt: {prompt}")
                    
                    results = diffusion_model.sample(
                        prompt,
                        height=IMAGE_SIZE,
                        width=IMAGE_SIZE,
                        num_inference_steps=50,
                        tokenizer=tokenizer
                    )
                    
                    # Save image
                    img = results['images'][0].cpu()
                    
                    # Convert to numpy and save
                    img_np = img.numpy().transpose(1, 2, 0)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Remove extra dimension for grayscale
                    if img_np.shape[-1] == 1:
                        img_np = img_np.squeeze(-1)
                        
                    # Save image
                    Image.fromarray(img_np).save(os.path.join(samples_dir, f"sample_{i+1}.png"))
                    
                    # Save prompt
                    with open(os.path.join(samples_dir, f"prompt_{i+1}.txt"), "w") as f:
                        f.write(prompt)
            
            logger.info(f"Final samples saved to {samples_dir}")
    
    except Exception as e:
        logger.error(f"Error generating samples: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return diffusion_model, tokenizer

# Debug function to test dataset access without training
def debug_dataset_only():
    """Debug function to test dataset loading without starting training."""
    logger.info("DEBUG MODE: Testing dataset loading only")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check paths
    logger.info(f"Dataset path: {DATASET_PATH} - Exists: {os.path.exists(DATASET_PATH)}")
    logger.info(f"Images path: {IMAGES_PATH} - Exists: {os.path.exists(IMAGES_PATH)}")
    logger.info(f"Reports CSV: {REPORTS_CSV} - Exists: {os.path.exists(REPORTS_CSV)}")
    logger.info(f"Projections CSV: {PROJECTIONS_CSV} - Exists: {os.path.exists(PROJECTIONS_CSV)}")
    
    # Create a smaller data module for testing
    logger.info("Creating test data module with small batch size")
    test_data_module = MedicalDataModule(
        reports_csv=REPORTS_CSV,
        projections_csv=PROJECTIONS_CSV,
        image_folder=IMAGES_PATH,
        batch_size=2,  # Very small batch size
        num_workers=0,  # No parallel workers
        val_split=0.1,
        test_split=0.1,
        image_size=IMAGE_SIZE,
        max_token_length=MAX_TOKEN_LENGTH,
        tokenizer_name=TEXT_MODEL,
        seed=RANDOM_SEED,
        timeout=0
    )
    
    # Set up dataset
    logger.info("Setting up test dataset")
    test_data_module.prepare_data()
    test_data_module.setup()
    
    # Get a single batch
    logger.info("Testing batch loading")
    train_loader = test_data_module.train_dataloader()
    
    try:
        logger.info("Trying to load a single batch...")
        batch = next(iter(train_loader))
        logger.info(f"Successfully loaded batch with keys: {list(batch.keys())}")
        logger.info(f"Batch image shape: {batch['image'].shape}")
        logger.info(f"Batch report sample: {batch['report'][0][:100]}...")
        
        # Try to visualize first image
        img = batch['image'][0].numpy()
        logger.info(f"First image stats - Min: {img.min():.4f}, Max: {img.max():.4f}, Mean: {img.mean():.4f}")
        
        logger.info("Dataset debug completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading batch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if RUN_QUICK_TEST:
    # Override settings for quick test run
    BATCH_SIZE = 2 
    EPOCHS_VAE = 2
    EPOCHS_DIFFUSION = 2
    NUM_WORKERS = 0
    
    # Use 1% of the dataset
    class SmallDatasetWrapper(Dataset):
        def __init__(self, dataset, percentage=0.01):
            self.dataset = dataset
            self.indices = random.sample(range(len(dataset)), int(len(dataset) * percentage))
            print(f"Using {len(self.indices)} samples out of {len(dataset)} ({percentage*100:.1f}%)")
            
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
            
        def __len__(self):
            return len(self.indices)
    
    # Override the setup method to use a small subset of data
    original_setup = MedicalDataModule.setup
    
    def setup_with_small_dataset(self):
        original_setup(self)
        # Wrap the datasets with our small dataset wrapper
        self.dataset = SmallDatasetWrapper(self.dataset)
        # Re-create the splits with the small dataset
        train_size = int((1 - self.val_split - self.test_split) * len(self.dataset))
        val_size = int(self.val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Apply transforms
        self._set_dataset_transform(self.train_dataset, self.train_transform)
        self._set_dataset_transform(self.val_dataset, self.val_transform)
        self._set_dataset_transform(self.test_dataset, self.val_transform)
        
        print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
    
    # Replace the setup method
    MedicalDataModule.setup = setup_with_small_dataset
    
    print("⚠️ RUNNING IN TEST MODE - QUICK TEST WITH 1% OF DATA AND REDUCED SETTINGS ⚠️")


# Execute training
if __name__ == "__main__":
    # Uncomment this line to run debug mode only
    # success = debug_dataset_only()
    
    try:
        # Normal training
        model, tokenizer = main()
        logger.info("Training complete")
        
        # Make model and tokenizer available in notebook
        logger.info("Model and tokenizer are available for inference")
    except Exception as e:
        logger.error(f"Critical error in main training loop: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try debug mode
        logger.info("Attempting debug mode to diagnose the issue...")
        debug_dataset_only()
