import io
import os
import hashlib
import logging
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Request, Query
from fastapi.responses import Response
from PIL import Image, ImageOps, ImageEnhance, ImageCms

# =========================
# App & Logging
# =========================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocessor")

app = FastAPI(title="Image Preprocessor", version="preproc-1.0")
BUILD_ID = "2025-10-16-no-multipart-v2"

# Optional request/response logger
@app.middleware("http")
async def log_request(request: Request, call_next):
    try:
        body = await request.body()
        log.info(
            f"REQUEST {request.method} {request.url.path} "
            f"CT={request.headers.get('content-type')} Len={len(body)}"
        )
    except Exception:
        pass
    resp = await call_next(request)
    log.info(f"RESPONSE {resp.status_code} CT={resp.headers.get('content-type')}")
    return resp


# =========================
# Helpers
# =========================
def attach_srgb(img: Image.Image) -> Image.Image:
    """Convert/attach sRGB profile when available (avoid color shifts)."""
    try:
        icc = img.info.get("icc_profile")
        if icc:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(img, src, dst, outputMode=img.mode)
        return img
    except Exception as e:
        log.warning(f"sRGB attach failed: {e}")
        return img  # fall back silently

def decode_pil_from_bytes(raw: bytes) -> Image.Image:
    if not raw:
        raise ValueError("Empty body")
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img) or img
    img = attach_srgb(img)
    return img.convert("RGBA")

def decode_pil_from_upload(file: UploadFile) -> Image.Image:
    raw = file.file.read()
    return decode_pil_from_bytes(raw)

def autocrop_alpha(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    return img if not bbox else img.crop(bbox)

def pad_to_square(img: Image.Image, bg=(0, 0, 0, 0)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), bg)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas

def scale_to_max(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    s = max(w, h)
    if s <= 0:
        return img, 1.0
    if s == max_side:
        return img, 1.0
    r = max_side / s
    new_size = (max(1, int(round(w * r))), max(1, int(round(h * r))))
    return img.resize(new_size, Image.LANCZOS), r

def apply_tone(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

def parse_bg(bg_hex: Optional[str]) -> Tuple[int, int, int, int]:
    if not bg_hex:
        return (0, 0, 0, 0)
    s = bg_hex.strip().lstrip("#")
    if len(s) == 6:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255)
    if len(s) == 8:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16))
    raise HTTPException(status_code=400, detail=f"Invalid bg hex: {bg_hex}")

def sha1_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# =========================
# Utility Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.get("/version")
def version():
    return {"build": BUILD_ID}

@app.get("/raw-test", response_class=Response)
def raw_test():
    """1x1 transparent PNG to verify raw image/png responses end-to-end."""
    import base64
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFWgKQWc5TFwAAAABJRU5ErkJggg=="
    )
    return Response(
        png_1x1,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="probe.png"', "X-Build": BUILD_ID},
    )


# =========================
# Main Endpoint
# =========================
@app.post("/normalize", response_class=Response)
async def normalize(
    # Accept EITHER multipart (field 'file') OR raw body:
    file: UploadFile | None = File(None, description="Upload file as multipart field named 'file'"),
    body: bytes | None = Body(None, description="Raw image bytes if not sending multipart"),
    # Options as QUERY PARAMS (works with both multipart and raw)
    max_side: int = Query(2048, description="Largest output side in px"),
    pad_square: bool = Query(True, description="Pad result to square canvas"),
    bg: Optional[str] = Query(None, description="Hex bg color, e.g. #FFFFFF; None keeps alpha"),
    autocrop: bool = Query(True, description="Trim transparent/near-transparent borders"),
    brightness: float = Query(1.0, description="1.0 = no change"),
    contrast: float = Query(1.0, description="1.0 = no change"),
    filename: Optional[str] = Query("normalized.png"),
):
    """
    Normalize the image and return ONLY PNG bytes.
    - Body: multipart with 'file' OR raw image bytes
    - Options: query params (so they work with either body type)
    - Metadata exposed via headers (no multipart wrapping)
    """
    try:
        # 1) Decode source
        if file is not None:
            img = decode_pil_from_upload(file)
        elif body:
            img = decode_pil_from_bytes(body)
        else:
            raise HTTPException(status_code=422, detail="No image provided. Send multipart field 'file' or raw bytes.")

        # 2) Pipeline
        if autocrop:
            img = autocrop_alpha(img)

        bg_rgba = parse_bg(bg)
        if pad_square:
            img = pad_to_square(img, bg_rgba)

        img, scale = scale_to_max(img, max_side)
        img = apply_tone(img, brightness=brightness, contrast=contrast)

        # 3) Encode PNG (flatten if opaque bg requested)
        buf = io.BytesIO()
        if bg_rgba[3] == 255:  # Opaque background requested
            flat = Image.new("RGB", img.size, bg_rgba[:3])
            flat.paste(img, mask=img.split()[-1])
            flat.save(buf, format="PNG", optimize=True)
        else:
            img.save(buf, format="PNG", optimize=True)

        png_bytes = buf.getvalue()
        headers = {
            "Content-Disposition": f'inline; filename="{filename or "normalized.png"}"',
            "X-Scale-Factor": f"{scale:.6f}",
            "X-Filesize-Bytes": str(len(png_bytes)),
            "X-Content-Hash": f"sha1:{sha1_digest(png_bytes)}",
            "X-Build": BUILD_ID,
        }

        # Return ONLY the PNG bytes
        return Response(content=png_bytes, media_type="image/png", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Normalize failed")
        raise HTTPException(status_code=400, detail=f"Normalize failed: {e}")


# =========================
# Local dev
# =========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
