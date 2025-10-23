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
# Model load (U2Net)
# ------------------------------------------------------------
U2NET_ONNX_PATH = os.getenv("U2NET_ONNX_PATH", "models/u2netp.onnx")
u2net = None
try:
    if os.path.exists(U2NET_ONNX_PATH):
        u2net = cv2.dnn.readNetFromONNX(U2NET_ONNX_PATH)
        print(f"[init] Loaded U2Net from {U2NET_ONNX_PATH}")
    else:
        print(f"[init] U2Net not found at {U2NET_ONNX_PATH}")
except Exception as e:
    print("[init] Failed to load U2Net:", e)
    u2net = None

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocessor")

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
app = FastAPI(title="Image Preprocessor", version="preproc-1.1-ml")

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def attach_srgb(img: Image.Image) -> Image.Image:
    try:
        if "icc_profile" in img.info:
            icc_bytes = img.info.get("icc_profile")
            if icc_bytes:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                dst = ImageCms.createProfile("sRGB")
                img = ImageCms.profileToProfile(img, src, dst, renderingIntent=ImageCms.INTENT_PERCEPTUAL)
                return img
        return img.convert("RGBA")
    except Exception as e:
        log.warning(f"sRGB attach failed: {e}")
        return img.convert("RGBA")

def decode_to_rgba(file: UploadFile) -> Image.Image:
    try:
        raw = file.file.read()
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img = attach_srgb(img)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

# ------------------------------------------------------------
# ML + Classic detection helpers
# ------------------------------------------------------------
def _largest_contour_bbox(binary_mask):
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    return cv2.boundingRect(c)

def _u2net_saliency_mask(bgr):
    if u2net is None:
        return None
    H, W = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, scalefactor=1/255.0, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    u2net.setInput(blob)
    out = u2net.forward()[0, 0]
    sal = 1 / (1 + np.exp(-out))
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_CUBIC)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal

def _detect_bbox_ml(bgr, thr=0.33, dilate_ratio=0.015):
    sal = _u2net_saliency_mask(bgr)
    if sal is None:
        return None
    H, W = bgr.shape[:2]
    binm = (sal >= thr).astype(np.uint8) * 255
    k = max(3, int(round(dilate_ratio * max(W, H))))
    kernel = np.ones((k, k), np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel, iterations=1)
    binm = cv2.dilate(binm, kernel, iterations=1)
    return _largest_contour_bbox(binm)

def _detect_bbox_threshold(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return _largest_contour_bbox(th)

def _detect_bbox_edge(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 150)
    dil = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return _largest_contour_bbox(dil)

def _detect_bbox_grabcut(bgr, rect_scale=0.70, iters=4):
    H, W = bgr.shape[:2]
    w0, h0 = int(W * rect_scale), int(H * rect_scale)
    x0, y0 = (W - w0)//2, (H - h0)//2
    mask = np.zeros((H, W), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x0, y0, w0, h0), bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) + (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    kern = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kern, iterations=1)
    return _largest_contour_bbox(fg)

def _box_plausible(x, y, w, h, W, H, min_w_ratio=0.55, min_h_ratio=0.45, min_area_ratio=0.30):
    wr = w / float(W)
    hr = h / float(H)
    ar = (w * h) / float(W * H)
    return (wr >= min_w_ratio) and (hr >= min_h_ratio) and (ar >= min_area_ratio)

# ------------------------------------------------------------
# Bounding box detection endpoint
# ------------------------------------------------------------
@app.post("/detect-bbox")
async def detect_bbox(
    file: UploadFile = File(...),
    pad_ratio: float = Form(0.005),
    min_area_ratio: float = Form(0.05),
    mode: str = Form("auto"),
    min_box_w_ratio: float = Form(0.55),
    min_box_h_ratio: float = Form(0.45),
    min_box_area_ratio: float = Form(0.30),
    ml_thr: float = Form(0.33),
    ml_dilate_ratio: float = Form(0.015)
):
    pil_rgba = decode_to_rgba(file)
    W, H = pil_rgba.size
    bgr = cv2.cvtColor(np.array(pil_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)

    def pad_and_clamp(x, y, w, h):
        pad_px = int(round(pad_ratio * max(W, H)))
        x = max(0, x - pad_px); y = max(0, y - pad_px)
        w = min(W - x, w + 2*pad_px); h = min(H - y, h + 2*pad_px)
        return x, y, w, h

    def attempt(meth):
        if meth == "threshold":
            return _detect_bbox_threshold(bgr)
        if meth == "edge":
            return _detect_bbox_edge(bgr)
        if meth == "ml":
            return _detect_bbox_ml(bgr, thr=ml_thr, dilate_ratio=ml_dilate_ratio)
        if meth == "grabcut":
            side = max(W, H)
            if side > 1200:
                scale = 1200.0 / side
                small = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
                bb = _detect_bbox_grabcut(small)
                if bb:
                    sx, sy, sw, sh = [int(round(v/scale)) for v in bb]
                    return (sx, sy, sw, sh)
                return None
            else:
                return _detect_bbox_grabcut(bgr)
        return None

    order = [mode] if mode in ("threshold", "edge", "ml", "grabcut") else ["threshold", "edge", "ml", "grabcut"]
    tried = []
    bbox = None
    confidence = 0.4
    chosen = None

    for meth in order:
        tried.append(meth)
        bb = attempt(meth)
        if not bb:
            continue
        x, y, w, h = bb
        if _box_plausible(x, y, w, h, W, H, min_box_w_ratio, min_box_h_ratio, min_box_area_ratio):
            bbox = (x, y, w, h)
            chosen = meth
            confidence = 0.95 if meth == "ml" else 0.9
            break

    if not bbox:
        for meth in order:
            bb = attempt(meth)
            if bb:
                bbox = bb
                chosen = f"{meth}-fallback"
                confidence = 0.5
                break

    if not bbox:
        return JSONResponse({"ok": False, "reason": "no_bbox", "mode_tried": tried})

    x, y, w, h = pad_and_clamp(*bbox)
    return {
        "ok": True,
        "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "image_size": {"w": W, "h": H},
        "confidence": float(confidence),
        "mode_tried": tried,
        "chosen": chosen
    }

# ------------------------------------------------------------
# Crop and debug endpoints
# ------------------------------------------------------------
@app.post("/crop-to-bbox")
async def crop_to_bbox(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...), filename: str = Form("crop.png")):
    pil_rgba = decode_to_rgba(file)
    W, H = pil_rgba.size
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    cropped = pil_rgba.crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png", headers={"Content-Disposition": f'inline; filename="{filename}"'})

@app.post("/debug-bbox")
async def debug_bbox(file: UploadFile = File(...), x: int = Form(...), y: int = Form(...), w: int = Form(...), h: int = Form(...)):
    data = file.file.read()
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 6)
    cv2.putText(bgr, f"{x},{y},{w},{h}", (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    png = cv2.imencode(".png", bgr)[1].tobytes()
    return Response(content=png, media_type="image/png")

# ------------------------------------------------------------
# Local dev run
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
