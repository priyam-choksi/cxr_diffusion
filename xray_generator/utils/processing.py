# xray_generator/utils/processing.py
import os
import random
import torch
import numpy as np
import logging
import cv2
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

logger = logging.getLogger(__name__)

def set_seed(seed=42):
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

def log_gpu_memory(message=""):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"GPU Memory {message}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        # Reset max stats
        torch.cuda.reset_peak_memory_stats()

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

def create_transforms(image_size=256):
    """Create standardized image transforms."""
    # Train transform with normalization to [-1, 1] for diffusion models
    train_transform = T.Compose([
        T.Normalize([0.5], [0.5])
    ])
    
    # Validation/test transform (same as train for consistent evaluation)
    val_transform = T.Compose([
        T.Normalize([0.5], [0.5])
    ])
    
    return train_transform, val_transform

def apply_clahe(image_tensor, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE to a tensor image for better contrast."""
    # Convert tensor to numpy array
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.cpu().numpy().squeeze()
    else:
        img_np = np.array(image_tensor)
    
    # Ensure proper range for CLAHE (0-255, uint8)
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    img_np = clahe.apply(img_np)
    
    # Convert back to tensor [0, 1]
    if isinstance(image_tensor, torch.Tensor):
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        if len(image_tensor.shape) > 2:  # If original had channel dim
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
    else:  # Return PIL or numpy
        return img_np

def create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0, 
                      drop_last=False, seed=42, timeout=0):
    """Create a data loader with standard settings."""
    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': drop_last,
        'worker_init_fn': lambda worker_id: np.random.seed(seed + worker_id),
        'collate_fn': custom_collate_fn
    }
    
    if num_workers > 0:
        loader_args.update({
            'timeout': timeout,
            'persistent_workers': True,
            'prefetch_factor': 2
        })
        
    return DataLoader(dataset, **loader_args)

def create_quick_test_dataset(dataset, percentage=0.01):
    """Create a small subset of a dataset for quick testing."""
    from torch.utils.data import Dataset
    
    class SmallDatasetWrapper(Dataset):
        def __init__(self, dataset, percentage=0.01):
            self.dataset = dataset
            import random
            self.indices = random.sample(range(len(dataset)), int(len(dataset) * percentage))
            logger.info(f"Using {len(self.indices)} samples out of {len(dataset)} ({percentage*100:.1f}%)")
            
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
            
        def __len__(self):
            return len(self.indices)
    
    return SmallDatasetWrapper(dataset, percentage)