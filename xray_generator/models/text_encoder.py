# xray_generator/models/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)

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