from io import BytesIO
import base64
from PIL import Image


def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Read a PIL.Image from raw bytes and convert it to RGB.
    Raise ValueError if the image cannot be decoded.
    """
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Could not decode image from bytes: {e}")


def load_image_from_base64(b64_string: str) -> Image.Image:
    """
    Read a PIL.Image from a base64-encoded string.
    Accepts strings with 'data:image/...;base64,' prefix or plain base64.
    """
    try:
        # Remove `data:image/...;base64,` prefix if present
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        data = base64.b64decode(b64_string)
        return load_image_from_bytes(data)
    except Exception as e:
        raise ValueError(f"Invalid or corrupted base64 image string: {e}")
