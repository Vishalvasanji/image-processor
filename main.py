import io
import os
import hashlib
import logging
from typing import Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageOps, ImageEnhance, ImageCms

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocessor")

# ------------------------------------------------------------
# App
# ------------------------------------------------------------
app = FastAPI(title="Image Preprocessor", version="preproc-1.0")

# (Optional) request logger for debugging input shape
@app.middleware("http")
async def log_request(request: Request, call_next):
    try:
        body = await request.body()
        log.info(
            f"REQUEST {request.method} {request.url.path} "
            f"CT={request.headers.get('content-type')} "
            f"Len={len(body)}"
        )
    except Exception:
        pass
    resp = await call_next(request)
    log.info(f"RESPONSE {resp.status_code} CT={resp.headers.get('content-type')}")
    return resp

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def attach_srgb(img: Image.Image) -> Image.Image:
    try:
        if "icc_profile" in img.info and img.info["icc_profile"]:
            src = ImageCms.ImageCmsProfile(io.BytesIO(img.info["icc_profile"]))
            dst = ImageCms.createProfile("sRGB")
            img = ImageCms.profileToProfile(img, src, dst, outputMode=img.mode)
        return img
    except Exception as e:
        log.warning(f"sRGB attach failed: {e}")
        return img

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
    if s <= 0 or s == max_side:
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
        return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16), 255)
    if len(s) == 8:
        return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16), int(s[6:8],16))
    raise HTTPException(status_code=400, detail=f"Invalid bg hex: {bg_hex}")

def sha1_digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "version": "preproc-1.0"}

@app.post("/normalize", response_class=Response)
async def normalize(
    # Accept EITHER multipart or raw bytes
    file: UploadFile | None = File(None, description="Upload file as multipart field named 'file'"),
    body: bytes | None = Body(None, description="Raw image bytes if not sending multipart"),
    # Options (can be form fields or query params)
    max_side: int = Form(2048),
    pad_square: bool = Form(True),
    bg: Optional[str] = Form(None),
    autocrop: bool = Form(True),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    filename: Optional[str] = Form("normalized.png"),
):
    """
    Returns ONLY PNG bytes (no multipart). Any metadata is placed into headers.
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

        # 3) Encode to PNG (flatten if opaque bg requested)
        buf = io.BytesIO()
        if bg_rgba[3] == 255:  # opaque
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
        }

        return Response(content=png_bytes, media_type="image/png", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Normalize failed")
        raise HTTPException(status_code=400, detail=f"Normalize failed: {e}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
