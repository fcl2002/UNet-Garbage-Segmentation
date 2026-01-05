# app/services/segmentation_classes.py

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any, List

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import MODEL_PATH, DEVICE

# ---------- Labels ----------
CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "plastic",
    2: "paper",
    3: "metal",
    4: "others",
}

# ---------- Colors (BGR) ----------
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    1: (0, 224, 224),   # plastic (yellow)
    2: (0, 204, 0),     # paper (green)
    3: (0, 0, 255),     # metal (red)
    4: (153, 0, 153),   # others (purple)
}

# ---------- Model / Input ----------
# Model returns (1,5,256,256)
INPUT_SIZE_WH = (256, 256)  # (W,H)

# IMPORTANT: Must match your training transform (ImageNet normalization)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]  # (3,1,1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]   # (3,1,1)

# ---------- Filters ----------
MIN_AREA_RATIO = 0.0008
MIN_CONFIDENCE = 0.35

# ---------- Drawing ----------
BOX_THICKNESS = 2
FONT_SCALE = 1.5
FONT_THICKNESS = 2
OVERLAY_ALPHA = 0.35

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
      2) convert BGR->RGB
      3) scale to [0,1]
      4) normalize with ImageNet mean/std
      5) CHW + batch dimension
    """
    w, h = INPUT_SIZE_WH
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # (3,H,W)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.expand_dims(x, 0)        # (1,3,H,W)

    return torch.from_numpy(x).to(_torch_device())


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Model output is not a torch.Tensor.")


@torch.inference_mode()
def _infer(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred_small: (256,256) uint8 (0..4)
      probs_small: (5,256,256) float32
    """
    if _model is None:
        _load_model()
    if _model is None:
        raise RuntimeError("Model not loaded.")

    x = _preprocess(image_bgr)

    logits = _as_tensor(_model(x))
    if logits.ndim != 4:
        raise RuntimeError(f"Unexpected model output shape: {tuple(logits.shape)}")

    probs = torch.softmax(logits, dim=1)[0]  # (5,H,W)
    pred = torch.argmax(probs, dim=0)        # (H,W)

    return pred.to(torch.uint8).cpu().numpy(), probs.to(torch.float32).cpu().numpy()


def _extract_components(
    bin_mask_01: np.ndarray,
    prob_map: Optional[np.ndarray],
    min_area: int
) -> List[Dict[str, Any]]:
    """
    bin_mask_01: (H,W) uint8 0/1
    prob_map:    (H,W) float prob of this class (optional)
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

    # Try above; if it doesn't fit, draw inside
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
    Multi-class segmentation, but renders ONLY ONE result:
      - pick the single component (across all classes) with the highest mean probability
      - draw overlay + bbox + label ONLY for that best component/class
    """
    if not is_model_loaded():
        raise RuntimeError("Model is not loaded on the server.")

    image_bgr = _pil_to_bgr(image)
    H, W = image_bgr.shape[:2]
    total_pixels = int(H * W)

    pred_small, probs_small = _infer(image_bgr)  # (256,256), (5,256,256)

    # Upscale prediction to original size
    pred_full = cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # Resize probability maps to original size for classes 1..4
    prob_full: Dict[int, np.ndarray] = {}
    for cls_id in (1, 2, 3, 4):
        p = probs_small[cls_id]  # (256,256)
        prob_full[cls_id] = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)

    # --------- Choose the BEST single component across classes ----------
    min_area = max(20, int(MIN_AREA_RATIO * total_pixels))

    best: Optional[Dict[str, Any]] = None
    # best = {"cls_id": int, "bbox": (x1,y1,x2,y2), "score": float, "area": int}

    for cls_id in (1, 2, 3, 4):
        cls_mask_01 = (pred_full == cls_id).astype(np.uint8)
        if cls_mask_01.sum() == 0:
            continue

        comps = _extract_components(cls_mask_01, prob_full.get(cls_id), min_area=min_area)

        for comp in comps:
            score = comp["score"]
            if score is None:
                continue
            if score < MIN_CONFIDENCE:
                continue

            candidate = {
                "cls_id": cls_id,
                "bbox": comp["bbox"],
                "score": float(score),
                "area": int(comp["area"]),
            }

            # Choose highest score; tie-breaker: larger area
            if best is None:
                best = candidate
            else:
                if candidate["score"] > best["score"]:
                    best = candidate
                elif candidate["score"] == best["score"] and candidate["area"] > best["area"]:
                    best = candidate

    # --------- Render ONLY the selected class/component ----------
    annotated = image_bgr.copy()

    if best is None:
        # No confident component found -> return original image + empty stats
        stats: Dict[str, Any] = {
            "contains_object": False,
            "total_object_ratio": 0.0,
            "mean_confidence": 0.0,
            "dominant_object_class_id": None,
            "dominant_object_class_name": None,
            "class_pixel_ratios": {},
        }
        return _bgr_to_pil(annotated), stats

    cls_id = int(best["cls_id"])
    x1, y1, x2, y2 = best["bbox"]
    score = float(best["score"])

    # Overlay only for the selected class pixels
    overlay = annotated.copy()
    sel_mask = (pred_full == cls_id)
    color = CLASS_COLORS_BGR.get(cls_id, (255, 0, 0))
    overlay[sel_mask] = color
    annotated = cv2.addWeighted(overlay, OVERLAY_ALPHA, annotated, 1.0 - OVERLAY_ALPHA, 0.0)

    # Draw bbox + label
    label_name = CLASS_NAMES.get(cls_id, str(cls_id))
    label = f"{label_name} {score:.2f}"
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=BOX_THICKNESS)
    _draw_yolo_style_label(annotated, x1, y1, label, color)

    # --------- Stats (now based only on the selected class) ----------
    selected_pixels = int(sel_mask.sum())
    total_object_ratio = (selected_pixels / total_pixels) if total_pixels else 0.0

    mean_confidence = 0.0
    vals = prob_full[cls_id][sel_mask]
    if vals.size:
        mean_confidence = float(vals.mean())

    stats = {
        "contains_object": True,
        "total_object_ratio": float(total_object_ratio),
        "mean_confidence": float(mean_confidence),
        "dominant_object_class_id": cls_id,
        "dominant_object_class_name": label_name,
        "class_pixel_ratios": {label_name: float(total_object_ratio)},
    }

    return _bgr_to_pil(annotated), stats
