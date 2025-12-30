# Trash Segmentation API

## 1. Overview

This project implements a **trash / waste segmentation API** using a pre-trained **UNet** model exported as **TorchScript (`.pt`)**.  
The server is built with **FastAPI** and exposes HTTP endpoints that:

- Receive an image (file upload or base64 string),
- Run semantic segmentation on the image,
- Return an **annotated image** with the mask contours drawn in green,
- Provide basic segmentation statistics (mask ratio, mean probability, etc.).

A minimal web frontend is included under `static/` to make it easy to test the API from the browser (uploading or pasting images).

---

## 2. Features

- **UNet-based segmentation** loaded from a TorchScript model (`model_unet.pt`).
- **FastAPI backend** with CORS enabled for local development.
- **Image upload** via `multipart/form-data` (`POST /api/v1/predict/file`).
- **Base64 JSON endpoint** (`POST /api/v1/predict/base64`).
- **Annotated output image**:
  - Original image with segmentation contours drawn in green.
- **Segmentation statistics** returned as:
  - HTTP headers (for file endpoint), or
  - JSON payload (for base64 endpoint).
- Modular project structure, ready for future versions (e.g. webcam capture in v2).

---

## 3. Project Structure

```text
.
├─ app/
│  ├─ main.py                    # FastAPI app, CORS, static files, routers
│  ├─ core/
│  │  └─ config.py               # Paths, device selection (CPU / CUDA)
│  ├─ api/
│  │  └─ v1/
│  │     └─ endpoints.py         # v1 HTTP endpoints (file & base64)
│  ├─ services/
│  │  └─ segmentation.py # UNet loading and inference
│  ├─ schemas/
│  │  └─ v1/
│  │     └─ image_schemas.py     # Pydantic models for v1
│  └─ utils/
│     └─ image_io.py             # Image loading helpers (bytes / base64)
│
├─ static/
│  └─ index.html                 # Simple frontend to test the API
├─ model_unet.pt                 # TorchScript UNet model
├─ requirements.txt
└─ README.md
```

## 4. Getting Started

### 4.1. Requirements

- Python 3.10+ (recommended)
- Virtual environment (optional but recommended)
- `model_unet.pt` (TorchScript UNet model) placed at the project root

Python dependencies are listed in `requirements.txt`.

---

### 4.2. Installation

From the project root:

```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### 4.3. Running the API server

```bash
uvicorn app.main:app --reload
```

## 5. Frontend Demo

A minimal frontend is included to make testing easier.

- Open: `http://127.0.0.1:8000/`

From the UI you can:

- Upload an image file (`.jpg`, `.png`, etc.),
- Paste an image from the clipboard (`Ctrl+V`),
- See:
  - the **input image** preview,
  - the **segmentation stats** returned by the API.

The static files are served from `/static`:

```text
GET /           -> static/index.html
GET /           -> static/index_realtime.html
```

---

## 6. API Documentation

All v1 endpoints are grouped under the router in `app/api/v1/image_endpoints.py`.

Base URL for v1:

```text
/api/v1
```

### 6.1. `POST /api/v1/predict/file`

**Description**  
Segment an image sent as a **file** (multipart/form-data) and return an **annotated PNG image**.

**Request**

- Method: `POST`
- URL: `/api/v1/predict/file`
- Content-Type: `multipart/form-data`
- Field:
  - `file`: image file (`image/*`)

**Response**

- Status: `200 OK`
- Content-Type: `image/png`
- Body: annotated image (overlay + bbox + label + probability)

### 6.2. `POST /api/v1/predict/base64`

**Description**  
Segment an image sent as a **base64-encoded string** in JSON and return:

- Annotated image encoded in base64 (PNG),
- Segmentation statistics.

**Request**

- Method: `POST`
- URL: `/api/v1/predict/base64`
- Content-Type: `application/json`
- Body:

```json
{
  "image_base64": "<base64 string>"
}
```

**Response**

- JSON:
- Image_base64: annotqated PNG encoded in base64
- Stats: same structure returned by the backend service

## 7. Versioning

### v1 – Current Version

- Modular project structure:
  - `core`, `api/v1`, `services`, `schemas/v1`, `utils`, `static`
- Segmentation service (`app/services/segmentation.py`) with:
  - TorchScript model loading (`model_unet.pt`)
  - Multi-class inference (5 classes) + post-processing
  - Subtle per-class overlay + bounding boxes from connected components
  - Labels with class + probability (2 decimals)
- API endpoints:
  - `POST /api/v1/predict/file` (returns annotated PNG + stats in headers)
  - `POST /api/v1/predict/base64` (returns base64 PNG + stats in JSON)
- Frontend:
  - `GET /` or `GET /static/index.html` (upload/paste image demo)
  - `GET /static/index_realtime.html` (webcam pseudo real-time demo)

### v2 – Planned (optional)

Possible future improvements:

- True real-time inference mode (continuous streaming with minimal latency), focused only on real-time operation.
- Expanded class coverage (increase the number of classes the model can segment/classify).

---

## 8. Development Notes

- The model is loaded once at startup (TorchScript via `torch.jit.load`) and reused for all incoming requests.
- The device (`CPU` or `CUDA`) is selected automatically in `app/core/config.py` based on `torch.cuda.is_available()`.
- The inference output is multi-class logits `(1, 5, 256, 256)` with classes:
  - `0 background`, `1 plastic`, `2 paper`, `3 metal`, `4 others`
- The API output image includes:
  - subtle per-class overlay,
  - bounding boxes computed from connected components,
  - labels containing class + probability (2 decimals).

---

## 9. License

Define your license here (e.g. MIT, Apache 2.0) once you decide how you want to distribute the project.  
If you plan to publish the code publicly, it is recommended to explicitly include a `LICENSE` file at the root of the repository.

---

## 10. Team and Academic Context

Project team – Trash Segmentation API

| Role        | Name                                         |
|-------------|----------------------------------------------|
| Student     | [Fernando COSTA LASMAR](https://www.linkedin.com/in/fernando-costa-lasmar/)                    |
| Student     | [Flávio ROSIM DE SOUSA](https://www.linkedin.com/in/flávio-rosim-de-sousa/)                               |
| Student     | [Matheus SISTON GALDINO](https://www.linkedin.com/in/matheussistongaldino/)                               |
| Student     | [Yan DING](mailto:dingyan02040608@gmail.com)                                                              |

---

This project explores the use of deep learning for trash segmentation, combining a UNet-based computer vision model with a modular FastAPI backend. It serves both as a practical study of machine learning deployment and as a foundation for future experiments in environmental and waste-management applications.




