from io import BytesIO
import base64

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.v1.image_schemas import ImageBase64Request
from app.utils.image_io import load_image_from_bytes, load_image_from_base64
# from app.services.segmentation_binaire import run_segmentation, is_model_loaded
from app.services.segmentation_classes import run_segmentation, is_model_loaded

# All v1 endpoints will be under /api/v1/...
router = APIRouter(prefix="/api/v1", tags=["v1"])


@router.post(
    "/predict/file",
    responses={200: {"content": {"image/png": {}}, "description": "Annotated PNG with class-colored overlay"}},
    summary="Segment an image sent as file",
)
async def predict_from_file(file: UploadFile = File(...)):
    """
    v1 – Receive an image as multipart/form-data and return
    a new PNG image with the segmentation contour in green.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    if not is_model_loaded():
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

    data = await file.read()

    try:
        image = load_image_from_bytes(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    annotated_img, stats = run_segmentation(image)

    # Convert annotated image to PNG bytes
    buf = BytesIO()
    annotated_img.save(buf, format="PNG")
    buf.seek(0)

    # Expose simple stats in HTTP headers
    headers = {
    "X-Contains-Object": str(stats["contains_object"]),
    "X-Total-Object-Ratio": str(stats["total_object_ratio"]),
    "X-Mean-Confidence": str(stats["mean_confidence"]),
    "X-Dominant-Class-Id": "" if stats["dominant_object_class_id"] is None else str(stats["dominant_object_class_id"]),
    "X-Dominant-Class-Name": "" if stats["dominant_object_class_name"] is None else str(stats["dominant_object_class_name"]),
    # string curta para não estourar header
    "X-Class-Ratios": ";".join([f"{k}={v:.4f}" for k, v in stats["class_pixel_ratios"].items()]),
}

    return StreamingResponse(buf, media_type="image/png", headers=headers)


@router.post(
    "/predict/base64",
    summary="Segment an image sent as base64 string",
)
async def predict_from_base64(payload: ImageBase64Request):
    """
    v1 – Receive an image as base64 in JSON and return:
      - annotated image encoded as base64 PNG
      - segmentation statistics
    """
    if not is_model_loaded():
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

    try:
        image = load_image_from_base64(payload.image_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    annotated_img, stats = run_segmentation(image)

    buf = BytesIO()
    annotated_img.save(buf, format="PNG")
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "image_base64": b64_img,
        "stats": stats,
    }
