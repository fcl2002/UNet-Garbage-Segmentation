from pydantic import BaseModel


class ImageBase64Request(BaseModel):
    """Request body for v1 base64 endpoint."""
    image_base64: str
