# quick_test.py
from pathlib import Path
import sys

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from xray_generator.train import train

# Set up paths
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "dataset" / "images" / "images_normalized"
REPORTS_CSV = BASE_DIR / "dataset" / "indiana_reports.csv"
PROJECTIONS_CSV = BASE_DIR / "dataset" / "indiana_projections.csv"

# Create a specific test output directory
TEST_OUTPUT_DIR = BASE_DIR / "outputs" / "test_runs"

# Configuration with minimal settings - exactly as in original script
config = {
    "batch_size": 2, 
    "epochs": 2,
    "learning_rate": 1e-4,
    "latent_channels": 8,
    "model_channels": 48,
    "image_size": 256,
    "use_amp": True,
    "checkpoint_freq": 1,
    "num_workers": 0
}

if __name__ == "__main__":
    print("Running quick test with minimal settings")
    print(f"Test outputs will be saved to: {TEST_OUTPUT_DIR}")
    
    # Run training with quick test flag
    train(
        config=config,
        dataset_path=str(DATASET_PATH),
        reports_csv=str(REPORTS_CSV),
        projections_csv=str(PROJECTIONS_CSV),
        output_dir=str(TEST_OUTPUT_DIR),  # Use the test output directory
        train_vae_only=True,
        quick_test=True
    )
    
    print("Quick test completed successfully!")