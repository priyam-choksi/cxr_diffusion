# xray_generator/utils/dataset.py
import os
import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
from transformers import AutoTokenizer
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

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
        load_tokenizer=True,
        use_clahe=True
    ):
        """Initialize the chest X-ray dataset."""
        self.image_folder = image_folder
        self.transform = transform
        self.target_size = target_size
        self.max_length = max_length
        self.use_clahe = use_clahe
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
        if self.use_clahe:
            img_np = img_tensor.numpy().squeeze()
            
            # Normalize to 0-255 range
            img_np = (img_np * 255).astype(np.uint8)
            
            # Apply CLAHE
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