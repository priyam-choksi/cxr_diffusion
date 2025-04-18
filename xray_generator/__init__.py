# xray_generator/__init__.py
import logging
from pkg_resources import get_distribution, DistributionNotFound

# Set up package-wide logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Import main components
from .models import MedicalVAE, MedicalTextEncoder, DiffusionUNet, DiffusionModel
from .inference import XrayGenerator

# Version tracking
try:
    __version__ = get_distribution("xray_generator").version
except DistributionNotFound:
    # Package not installed
    __version__ = "0.1.0-dev"

__all__ = [
    'MedicalVAE', 
    'MedicalTextEncoder', 
    'DiffusionUNet', 
    'DiffusionModel',
    'XrayGenerator'
]