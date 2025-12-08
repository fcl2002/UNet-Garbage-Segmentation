from typing import Tuple, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.core.config import MODEL_PATH, DEVICE

# ============================
# UNet model configuration
# ============================

# Same normalization used during training
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 256

_inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# Load TorchScript model once on startup
try:
    _model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
    _model.eval()
    print("[INFO] UNet TorchScript model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load TorchScript model: {e}")
    _model = None


def is_model_loaded() -> bool:
    """Return True if the model was loaded successfully."""
    return _model is not None


def run_segmentation(image: Image.Image) -> Tuple[Image.Image, Dict]:
    """
    Run UNet segmentation on a PIL.Image and return:

      - annotated_image: original image with green contours over the mask
      - stats: dictionary with basic segmentation statistics
    """
    if _model is None:
        raise RuntimeError("Model is not loaded on the server.")

    # Convert to numpy RGB array (copy of original image)
    original_np = np.array(image.convert("RGB"))

    # Preprocess
    img_tensor = _inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(img_tensor)       # [1, 1, H, W]
        probs = torch.sigmoid(logits)     # probabilities in [0, 1]

    prob_map = probs[0, 0].cpu().numpy()           # [H, W]
    mask_small = (prob_map > 0.5).astype(np.uint8)

    mean_prob = float(prob_map.mean())
    mask_ratio = float(mask_small.mean())
    contains_object = mask_ratio > 0.01

    # Resize mask back to original size
    h, w, _ = original_np.shape
    mask_resized = cv2.resize(
        mask_small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Draw contours in green over the original image
    annotated = original_np.copy()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), thickness=3)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    annotated_pil = Image.fromarray(annotated)

    stats = {
        "contains_object": contains_object,
        "mean_probability": mean_prob,
        "mask_pixel_ratio": mask_ratio,
    }

    return annotated_pil, stats
