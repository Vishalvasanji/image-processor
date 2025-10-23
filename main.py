import io
import numpy as np
import cv2
import os
import hashlib
import logging
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, ImageCms, ImageEnhance

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocessor")

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="Image Preprocessor", version="preproc-1.0")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def attach_srgb(img: Image.Image) -> Image.Image:
    """
    Convert/attach sRGB profile to avoid color shifts.
    Falls back to RGBA conversion if profiles are missing.
    """
    try:
        # If there's an embedded profile, convert to sRGB.
        if "icc_profile" in img.info:
            icc_bytes = img.info.get("icc_profile")
            if icc_bytes:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                dst = ImageCms.createProfile("sRGB")
                intent = ImageCms.INTENT_PERCEPTUAL
                img = ImageCms.profileToProfile(img, src, dst, renderingIntent=intent, outputMode=img.mode)
                return img
        # If no embedded profile, at least set mode consistently.
        return img.convert("RGBA")
    except Exception as e:
        log.warning(f"sRGB attach failed: {e}")
        return img.convert("RGBA")


def decode_to_rgba(file: UploadFile) -> Image.Image:
    """Read an uploaded file and normalize orientation + color space to RGBA."""
    try:
        raw = file.file.read()
        if not raw:
            raise ValueError("Empty upload.")

        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img) or img  # fix EXIF orientation
        img = attach_srgb(img)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")


def autocrop_alpha(img: Image.Image, bg=(0, 0, 0, 0)) -> Image.Image:
    """
    Trim uniform transparent/near-transparent borders.
    Works on RGBA—uses alpha as primary signal.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if not bbox:
        # Completely transparent; just return as-is to avoid empty crop.
        return img
    return img.crop(bbox)


def pad_to_square(img: Image.Image, bg=(0, 0, 0, 0)) -> Image.Image:
    """Pad the image to a square canvas (centered) using bg color."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), bg)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def scale_to_max(img: Image.Image, max_side: int) -> Tuple[Image.Image, float]:
    """Scale so that the largest side == max_side, preserving aspect ratio."""
    w, h = img.size
    side = max(w, h)
    if side == 0:
        return img, 1.0
    scale = max_side / side
    if scale == 1.0:
        return img, 1.0
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS), scale


def apply_tone(img: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    """Optionally adjust brightness/contrast (no saturation shift)."""
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def parse_bg(bg_hex: Optional[str]) -> Tuple[int, int, int, int]:
    """Parse hex color (like '#FFFFFF' or 'FFFFFF') to RGBA with full alpha."""
    if not bg_hex:
        return (0, 0, 0, 0)
    s = bg_hex.strip().lstrip("#")
    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b, 255)
    elif len(s) == 8:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        a = int(s[6:8], 16)
        return (r, g, b, a)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid bg hex: {bg_hex}")


def sha1_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": "preproc-1.0"}


@app.post("/normalize", response_class=Response)
async def normalize(
    file: UploadFile = File(..., description="Source image file."),
    # Canvas & transform options
    max_side: int = Form(2048, description="Largest output side in px (default 2048)."),
    pad_square: bool = Form(True, description="Pad result to square canvas."),
    bg: Optional[str] = Form(None, description="Hex bg color (e.g., '#FFFFFF'). None=transparent."),
    autocrop: bool = Form(True, description="Trim transparent/near-transparent borders."),
    # Simple tone controls (optional)
    brightness: float = Form(1.0, description="1.0 = no change"),
    contrast: float = Form(1.0, description="1.0 = no change"),
    # Output filename hint (header only; content is raw PNG)
    filename: Optional[str] = Form("normalized.png"),
):
    """
    Normalize an uploaded image and return ONLY the PNG bytes.

    Pipeline (defaults):
      1) Decode + EXIF-fix + sRGB -> RGBA
      2) (Optional) Autocrop transparent borders
      3) (Optional) Pad to square (centered)
      4) Scale largest side to `max_side`
      5) (Optional) Apply simple tone tweaks
      6) Encode to PNG and RETURN BYTES (media_type='image/png')
    """
    try:
        # 1) Decode & color-normalize
        rgba = decode_to_rgba(file)

        # 2) Autocrop
        if autocrop:
            rgba = autocrop_alpha(rgba)

        # 3) Pad to square
        bg_rgba = parse_bg(bg)
        if pad_square:
            rgba = pad_to_square(rgba, bg_rgba)

        # 4) Scale to target
        rgba, scale = scale_to_max(rgba, max_side=max_side)

        # 5) Tone
        rgba = apply_tone(rgba, brightness=brightness, contrast=contrast)

        # 6) Encode to PNG bytes
        buf = io.BytesIO()
        # If caller asked for opaque background but we still have alpha, flatten
        if bg_rgba[3] == 255:
            # Flatten to RGB
            flat = Image.new("RGB", rgba.size, bg_rgba[:3])
            flat.paste(rgba, mask=rgba.split()[-1])
            flat.save(buf, format="PNG", optimize=True)
        else:
            rgba.save(buf, format="PNG", optimize=True)

        png_bytes = buf.getvalue()
        content_hash = sha1_digest(png_bytes)

        headers = {
            "Content-Disposition": f'inline; filename="{filename or "normalized.png"}"',
            "X-Scale-Factor": str(round(scale, 4)),
            "X-Filesize-Bytes": str(len(png_bytes)),
            "X-Content-Hash": f"sha1:{content_hash}",
        }

        # RETURN ONLY PNG BYTES
        return Response(content=png_bytes, media_type="image/png", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Normalize failed")
        raise HTTPException(status_code=400, detail=f"Normalize failed: {e}")

# ---------- Helpers for bbox detection/cropping ----------
def pil_to_cv_bgra(img: Image.Image) -> np.ndarray:
    """Pillow RGBA -> OpenCV BGRA numpy array."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)

def clamp_box(x, y, w, h, W, H, pad_px=0):
    x = max(0, x - pad_px)
    y = max(0, y - pad_px)
    w = min(W - x, w + 2 * pad_px)
    h = min(H - y, h + 2 * pad_px)
    return x, y, w, h

# ---------- 1) Detect bounding box ----------
@app.post("/detect-bbox")
async def detect_bbox(
    file: UploadFile = File(...),
    pad_ratio: float = Form(0.02),       # 2% padding around the garment
    min_area_ratio: float = Form(0.05)   # ignore contours < 5% of image area
):
    # Uses your existing helper to normalize to RGBA
    pil_rgba = decode_to_rgba(file)
    W, H = pil_rgba.size

    cv_bgra = pil_to_cv_bgra(pil_rgba)
    bgr = cv2.cvtColor(cv_bgra, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Otsu + morphology (robust on studio mockups)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return JSONResponse({"ok": False, "reason": "no_contours"})

    img_area = W * H
    min_area = max(1, int(min_area_ratio * img_area))
    big = [c for c in cnts if cv2.contourArea(c) >= min_area]

    if big:
        c = max(big, key=cv2.contourArea)
        confidence = 0.9
    else:
        c = max(cnts, key=cv2.contourArea)
        confidence = 0.4  # small shapes only (logo/noise)

    x, y, w, h = cv2.boundingRect(c)

    # Padding
    pad_px = int(round(pad_ratio * max(W, H)))
    x, y, w, h = clamp_box(x, y, w, h, W, H, pad_px)

    return {
        "ok": True,
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "image_size": {"w": W, "h": H},
        "confidence": float(confidence),
    }

# ---------- 2) Crop to a bounding box ----------
@app.post("/crop-to-bbox")
async def crop_to_bbox(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...),
    filename: str = Form("crop.png"),
):
    pil_rgba = decode_to_rgba(file)
    W, H = pil_rgba.size

    # Clamp to image bounds
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    cropped = pil_rgba.crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=True)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )

# ------------------------------------------------------------
# Local dev
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
