from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import STATIC_DIR
from app.api.v1.endpoints import router as v1_router

app = FastAPI(
    title="Segmentation API",
    description="API that receives images (file or base64) and runs a UNet segmentation model.",
    version="1.0.0",
)

# CORS configuration (open for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the minimal demo frontend."""
    return FileResponse(STATIC_DIR / "index.html")


# Include v1 API routes
app.include_router(v1_router)
