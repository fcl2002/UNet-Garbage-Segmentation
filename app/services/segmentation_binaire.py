# app/services/segmentation_binaire.py

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import MODEL_PATH, DEVICE

# ---------- Model / Input ----------
INPUT_SIZE_WH = (256, 256)  # (W, H)

# IMPORTANT: Must match your training transform (ImageNet normalization)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]  # (3,1,1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]   # (3,1,1)

# ---------- Thresholds ----------
# Pixel-level threshold used to convert probability map -> binary mask
PIXEL_PROB_THRESHOLD = 0.50

# Component filters to avoid boxing noise
MIN_AREA_RATIO = 0.0008
MIN_CONFIDENCE = 0.35

# ---------- Drawing ----------
BOX_THICKNESS = 2
FONT_SCALE = 0.4
FONT_THICKNESS = 1
OVERLAY_ALPHA = 0.35

# Single color for binary "trash" overlay/bbox (BGR)
TRASH_COLOR_BGR = (0, 224, 224)

_model: Optional[torch.jit.ScriptModule] = None


def _torch_device() -> torch.device:
    if isinstance(DEVICE, torch.device):
        return DEVICE
    return torch.device(str(DEVICE))


def _load_model() -> None:
    global _model
    if _model is not None:
        return
    dev = _torch_device()
    m = torch.jit.load(str(MODEL_PATH), map_location=dev)
    m.eval()
    _model = m


def is_model_loaded() -> bool:
    try:
        if _model is None:
            _load_model()
        return _model is not None
    except Exception:
        return False


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Preprocess must match training:
      1) resize to (256,256)
      2) convert to RGB
      3) scale to [0,1]
      4) normalize with ImageNet mean/std
      5) CHW + batch dimension
    """
    w, h = INPUT_SIZE_WH
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = rgb.astype(np.float32) / 255.0                 # (H,W,3) in [0,1]
    x = np.transpose(x, (2, 0, 1))                     # (3,H,W)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD             # normalize
    x = np.expand_dims(x, 0)                           # (1,3,H,W)

    return torch.from_numpy(x).to(_torch_device())


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Model output is not a torch.Tensor.")


@torch.inference_mode()
def _infer_binary(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports binary segmentation models that output:
      - (1,1,H,W): logits -> sigmoid
      - (1,2,H,W): logits -> softmax, class 1 is object

    Returns:
      pred_small: (H,W) uint8 {0,1}
      prob_small: (H,W) float32 object probability
    """
    if _model is None:
        _load_model()
    if _model is None:
        raise RuntimeError("Model not loaded.")

    x = _preprocess(image_bgr)
    logits = _as_tensor(_model(x))

    if logits.ndim != 4:
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

    c = int(logits.shape[1])

    if c == 1:
        obj_prob = torch.sigmoid(logits)[0, 0]  # (H,W)
    elif c == 2:
        probs = torch.softmax(logits, dim=1)[0]  # (2,H,W)
        obj_prob = probs[1]
    else:
        raise RuntimeError(f"This binary pipeline expects C=1 or C=2, got C={c}.")

    pred = (obj_prob >= PIXEL_PROB_THRESHOLD).to(torch.uint8)

    return pred.cpu().numpy(), obj_prob.to(torch.float32).cpu().numpy()


def _extract_components(
    bin_mask_01: np.ndarray,
    prob_map: Optional[np.ndarray],
    min_area: int
) -> List[Dict[str, Any]]:
    """
    bin_mask_01: (H,W) uint8 0/1
    prob_map:    (H,W) float object probability (optional)
    """
    mask_u8 = (bin_mask_01.astype(np.uint8) * 255)

    # Light cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), connectivity=8)

    comps: List[Dict[str, Any]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        comp_mask = (labels == i)

        score = None
        if prob_map is not None:
            vals = prob_map[comp_mask]
            score = float(vals.mean()) if vals.size else 0.0

        comps.append(
            {
                "bbox": (int(x), int(y), int(x + w), int(y + h)),
                "area": int(area),
                "score": float(score) if score is not None else None,
            }
        )

    comps.sort(key=lambda d: d["area"], reverse=True)
    return comps


def _draw_yolo_style_label(img: np.ndarray, x1: int, y1: int, text: str, color: Tuple[int, int, int]) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)

    pad_x, pad_y = 6, 4
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y + baseline

    # Try above the bbox; if it doesn't fit, draw inside
    y_top = y1 - box_h
    if y_top < 0:
        y_top = y1

    x2 = min(img.shape[1] - 1, x1 + box_w)
    y2 = min(img.shape[0] - 1, y_top + box_h)

    cv2.rectangle(img, (x1, y_top), (x2, y2), color, thickness=-1)
    cv2.putText(
        img,
        text,
        (x1 + pad_x, y_top + pad_y + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def run_segmentation(image: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Binary segmentation pipeline:
      - overlay + bbox + label("trash <score>")
      - stats compatible with your current API headers
    """
    if not is_model_loaded():
        raise RuntimeError("Model is not loaded on the server.")

    image_bgr = _pil_to_bgr(image)
    H, W = image_bgr.shape[:2]
    total_pixels = int(H * W)

    pred_small, obj_prob_small = _infer_binary(image_bgr)  # (256,256), (256,256)

    # Upscale to original size
    bin_full = cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    prob_full = cv2.resize(obj_prob_small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    trash_pixels = int(bin_full.sum())
    contains_object = trash_pixels > 0
    total_object_ratio = (trash_pixels / total_pixels) if total_pixels else 0.0

    mean_confidence = 0.0
    if contains_object:
        vals = prob_full[bin_full == 1]
        mean_confidence = float(vals.mean()) if vals.size else 0.0

    # Overlay
    annotated = image_bgr.copy()
    overlay = annotated.copy()
    mask = (bin_full == 1)
    if mask.any():
        overlay[mask] = TRASH_COLOR_BGR
    annotated = cv2.addWeighted(overlay, OVERLAY_ALPHA, annotated, 1.0 - OVERLAY_ALPHA, 0.0)

    # BBoxes
    min_area = max(20, int(MIN_AREA_RATIO * total_pixels))
    comps = _extract_components(bin_full, prob_full, min_area=min_area)

    for comp in comps:
        score = comp["score"]
        if score is not None and score < MIN_CONFIDENCE:
            continue

        x1, y1, x2, y2 = comp["bbox"]
        label = "trash" if score is None else f"trash {score:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), TRASH_COLOR_BGR, thickness=BOX_THICKNESS)
        _draw_yolo_style_label(annotated, x1, y1, label, TRASH_COLOR_BGR)

    stats: Dict[str, Any] = {
        "contains_object": bool(contains_object),
        "total_object_ratio": float(total_object_ratio),
        "mean_confidence": float(mean_confidence),
        "dominant_object_class_id": None,
        "dominant_object_class_name": None,
        "class_pixel_ratios": {"trash": float(total_object_ratio)},
    }

    return _bgr_to_pil(annotated), stats
