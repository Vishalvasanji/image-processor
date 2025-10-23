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

U2NET_ONNX_PATH = os.getenv("U2NET_ONNX_PATH", "models/u2netp.onnx")
u2net = None
if os.path.exists(U2NET_ONNX_PATH):
    try:
        u2net = cv2.dnn.readNetFromONNX(U2NET_ONNX_PATH)
        print(f"[init] Loaded U2Net from {U2NET_ONNX_PATH}")
    except Exception as e:
        print("[init] Failed to load U2Net:", e)

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

def _is_plausible_garment_box(x, y, w, h, W, H,
                              min_w_ratio=0.40,   # at least 40% of width
                              min_h_ratio=0.35,   # at least 35% of height
                              min_area_ratio=0.20 # at least 20% of area
                             ):
    wr = w / float(W)
    hr = h / float(H)
    ar = (w * h) / float(W * H)
    return (wr >= min_w_ratio) and (hr >= min_h_ratio) and (ar >= min_area_ratio)

def _largest_contour_bbox(binary_mask):
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return cv2.boundingRect(c)  # (x,y,w,h)

def _detect_bbox_threshold(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  kernel, iterations=1)
    return _largest_contour_bbox(th)

def _detect_bbox_edge(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 150)
    # thicken edges slightly, then fill contour
    dil = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    return _largest_contour_bbox(dil)

def _detect_bbox_grabcut(bgr, rect_scale=0.75, iters=4):
    H, W = bgr.shape[:2]
    w0, h0 = int(W*rect_scale), int(H*rect_scale)
    x0, y0 = (W - w0)//2, (H - h0)//2
    mask = np.zeros((H, W), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x0,y0,w0,h0), bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    kernel = np.ones((5,5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _largest_contour_bbox(fg)

def _u2net_saliency_mask(bgr):
    """
    Returns a float mask [0..1], same HxW as input. Requires global u2net.
    """
    if u2net is None:
        return None
    H, W = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1/255.0, size=(320,320),
                                 mean=(0,0,0), swapRB=True, crop=False)
    u2net.setInput(blob)
    out = u2net.forward()            # shape: 1x1x320x320
    sal = out[0,0]
    sal = 1 / (1 + np.exp(-sal))     # sigmoid
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    # normalize to [0,1]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal

def _detect_bbox_ml(bgr, thr=0.35, dilate_ratio=0.01):
    """
    Saliency -> threshold -> (optional) dilation -> largest contour -> bbox
    dilate_ratio controls sleeve/hem inclusion.
    """
    sal = _u2net_saliency_mask(bgr)
    if sal is None:
        return None
    H, W = bgr.shape[:2]
    binm = (sal >= thr).astype(np.uint8) * 255
    # Expand a hair to catch sleeves/neckline
    k = max(3, int(round(dilate_ratio * max(W, H))))
    kernel = np.ones((k, k), np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel, iterations=1)
    binm = cv2.dilate(binm, kernel, iterations=1)

    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return cv2.boundingRect(c)  # (x,y,w,h)

# ---------- 1) Detect bounding box ----------
@app.post("/detect-bbox")
async def detect_bbox(
    file: UploadFile = File(...),
    pad_ratio: float = Form(0.005),       # your tuned default
    min_area_ratio: float = Form(0.05),   # filter tiny blobs in the first pass
    mode: str = Form("auto")              # "auto" | "threshold" | "edge" | "grabcut"
):
    pil_rgba = decode_to_rgba(file)
    W, H = pil_rgba.size
    bgr = cv2.cvtColor(np.array(pil_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)

    def pad_and_clamp(x, y, w, h):
        pad_px = int(round(pad_ratio * max(W, H)))
        x = max(0, x - pad_px); y = max(0, y - pad_px)
        w = min(W - x, w + 2*pad_px); h = min(H - y, h + 2*pad_px)
        return x, y, w, h

    tried = []
    bbox = None
    confidence = 0.9

    def attempt(method_name):
        if method_name == "threshold":
            return _detect_bbox_threshold(bgr)
        if method_name == "edge":
            return _detect_bbox_edge(bgr)
        if method_name == "grabcut":
            # speed optimization: downscale for grabcut and rescale back
            scale = 1000.0 / max(W, H) if max(W, H) > 1000 else 1.0
            if scale < 1.0:
                small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
                bb = _detect_bbox_grabcut(small, rect_scale=0.75, iters=4)
                if bb:
                    sx, sy, sw, sh = [int(round(v/scale)) for v in bb]
                    return (sx, sy, sw, sh)
                return None
            else:
                return _detect_bbox_grabcut(bgr, rect_scale=0.75, iters=4)
        return None

    order = ([mode] if mode in ("threshold", "edge", "grabcut", "ml") else ["threshold", "edge", "ml", "grabcut"])
    def attempt(meth):
        if meth == "ml":
            return _detect_bbox_ml(bgr, thr=0.35, dilate_ratio=0.01)
    
    for meth in order:
        tried.append(meth)
        bb = attempt(meth)
        if not bb:
            continue
        x,y,w,h = bb
        # Reject obviously-too-small boxes (e.g., front print only)
        if not _is_plausible_garment_box(x, y, w, h, W, H):
            # low confidence; try next method
            confidence = 0.5
            continue
        bbox = (x,y,w,h)
        confidence = 0.9 if meth != "grabcut" else 0.95
        break

    if not bbox:
        # as a last resort, return the largest box from the last successful method (even if small)
        for meth in order:
            bb = attempt(meth)
            if bb:
                x,y,w,h = bb
                bbox = (x,y,w,h)
                confidence = 0.4
                break

    if not bbox:
        return JSONResponse({"ok": False, "reason": "no_bbox", "tried": tried})

    x,y,w,h = pad_and_clamp(*bbox)
    return {
        "ok": True,
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "image_size": {"w": W, "h": H},
        "confidence": float(confidence),
        "mode_tried": tried
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

@app.post("/debug-bbox")
async def debug_bbox(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    w: int = Form(...),
    h: int = Form(...)
):
    import numpy as np, cv2, io
    from fastapi import Response
    from PIL import Image

    data = file.file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 6)
    cv2.putText(bgr, f"{x},{y},{w},{h}", (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    png = cv2.imencode(".png", bgr)[1].tobytes()
    return Response(content=png, media_type="image/png")


# ------------------------------------------------------------
# Local dev
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
