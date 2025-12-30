from pathlib import Path
import torch

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

STATIC_DIR = BASE_DIR / "static"
# MODEL_PATH = BASE_DIR / "model_unet.pt"
MODEL_PATH = BASE_DIR / "taco_unet_model_classes.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
