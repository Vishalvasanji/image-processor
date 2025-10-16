import io
import os
import hashlib
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageEnhance, ImageCms, ImageOps
from starlette.responses import StreamingResponse
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("preprocessor")

# Create FastAPI app
app = FastAPI(title="Image Preprocessor", version="1.0")

# Load settings from environment variables
CANVAS_SIZE = int(os.getenv("CANVAS_SIZE", "2048"))
FILL_RATIO = float(os.getenv("FILL_RATIO", "0.68"))
BUFFER_PCT = float(os.getenv("BUFFER_PCT", "0.025"))
BRIGHTNESS = float(os.getenv("BRIGHTNESS", "1.01"))
CONTRAST = float(os.getenv("CONTRAST", "1.03"))
MAX_FILE_MB = float(os.getenv("MAX_FILESIZE_MB", "1.2"))
TIMEOUT_S = int(os.getenv("TIMEOUT_S", "20"))

def attach_srgb(img: Image.Image) -> Image.Image:
    """Attach sRGB color profile if missing"""
    try:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        # Try to get existing profile
        if 'icc_profile' not in img.info:
            # Create sRGB profile
            srgb_profile = ImageCms.createProfile("sRGB")
            img_with_profile = ImageCms.profileToProfile(
                img.convert("RGB"), 
                srgb_profile, 
                srgb_profile,
                outputMode="RGB"
            ).convert("RGBA")
            return img_with_profile
        return img
    except Exception as e:
        log.warning(f"Could not attach sRGB profile: {e}")
        return img.convert("RGBA")

def decode_to_rgba(file: UploadFile) -> Image.Image:
    """Read uploaded file and convert to RGBA format"""
    try:
        data = file.file.read()
        img = Image.open(io.BytesIO(data))
        
        # Handle EXIF orientation
        img = ImageOps.exif_transpose(img) or img
        
        # Convert to RGBA
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        img = attach_srgb(img)
        log.info(f"Decoded image: {img.size}, mode: {img.mode}")
        return img
    except Exception as e:
        log.error(f"Failed to decode image: {e}")
        raise HTTPException(status_code=400, detail=f"decode_failed: {str(e)}")

def get_bbox(img: Image.Image):
    """Find bounding box of the main object in the image"""
    w, h = img.size
    
    # Try alpha channel first (for transparent backgrounds)
    if img.mode == "RGBA":
        alpha = img.split()[-1]
        # Threshold: alpha > 25 (about 10% opacity)
        mask = alpha.point(lambda a: 255 if a > 25 else 0)
        bbox = mask.getbbox()
        if bbox:
            log.info(f"Found bbox using alpha channel: {bbox}")
            return bbox, "alpha"
    
    # Fallback: use luminance (brightness)
    luma = img.convert("L")
    # Threshold: brightness > 13 (about 5%)
    mask = luma.point(lambda y: 255 if y > 13 else 0)
    bbox = mask.getbbox()
    
    if not bbox:
        log.error("Could not find object in image")
        raise HTTPException(status_code=400, detail="bbox_not_found")
    
    log.info(f"Found bbox using luminance: {bbox}")
    return bbox, "luma"

def expand_bbox(bbox, w, h, buffer_pct):
    """Add padding around the bounding box"""
    x0, y0, x1, y1 = bbox
    # Calculate buffer size as percentage of largest dimension
    buf = int(max(w, h) * buffer_pct)
    
    # Expand bbox and clip to image boundaries
    expanded = (
        max(0, x0 - buf),
        max(0, y0 - buf),
        min(w, x1 + buf),
        min(h, y1 + buf)
    )
    log.info(f"Expanded bbox by {buf}px: {expanded}")
    return expanded, buf

def tone_normalize(rgba: Image.Image) -> Image.Image:
    """Apply subtle brightness and contrast adjustments"""
    # Split alpha channel
    rgb = rgba.convert("RGB")
    alpha = rgba.split()[-1]
    
    # Apply adjustments
    rgb = ImageEnhance.Brightness(rgb).enhance(BRIGHTNESS)
    rgb = ImageEnhance.Contrast(rgb).enhance(CONTRAST)
    
    # Recombine with alpha
    result = Image.new("RGBA", rgba.size)
    result.paste(rgb, (0, 0))
    result.putalpha(alpha)
    
    return result

def generate_filename(team_handle: str, design_slug: str, product_id: str, view_type: str) -> str:
    """Generate standardized filename with hash"""
    # Ensure lowercase
    base = f"{team_handle.lower()}_{design_slug.lower()}_{product_id}_{view_type.lower()}"
    
    # Generate short hash for versioning
    hash_input = base.encode("utf-8")
    short_hash = hashlib.sha1(hash_input).hexdigest()[:6]
    
    filename = f"{base}_v{short_hash}.png"
    
    # Enforce max length
    if len(filename) > 80:
        log.warning(f"Filename too long ({len(filename)} chars), truncating")
        filename = filename[:76] + ".png"
    
    return filename

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "version": "preproc-1.0"}

@app.post("/normalize")
async def normalize(
    file: UploadFile = File(...),
    team_handle: str = Form(...),
    design_slug: str = Form(...),
    product_id: str = Form(...),
    view_type: str = Form(...),
    fill_ratio: Optional[float] = Form(None),
    buffer_pct: Optional[float] = Form(None),
    canvas_size: Optional[int] = Form(None),
):
    """
    Main endpoint: normalize and process an image
    
    Accepts: multipart form with image file and metadata
    Returns: multipart response with processed PNG + JSON metadata
    """
    # Validate view_type
    if view_type.lower() not in ("front", "back", "detail"):
        raise HTTPException(status_code=400, detail="invalid_view_type: must be front, back, or detail")
    
    # Use custom values or defaults
    fr = fill_ratio if fill_ratio is not None else FILL_RATIO
    bp = buffer_pct if buffer_pct is not None else BUFFER_PCT
    cs = canvas_size if canvas_size is not None else CANVAS_SIZE
    
    log.info(f"Processing: {team_handle}/{design_slug}/{product_id}/{view_type}")
    
    # STEP 1: Decode and normalize input
    img = decode_to_rgba(file)
    w, h = img.size
    
    # STEP 2: Find bounding box
    bbox, bbox_source = get_bbox(img)
    
    # STEP 3: Add buffer
    bbox, buffer_px = expand_bbox(bbox, w, h, bp)
    
    # STEP 4: Crop to buffered bbox
    img = img.crop(bbox)
    log.info(f"Cropped to: {img.size}")
    
    # STEP 5: Scale proportionally
    target_w = int(cs * fr)
    scale = target_w / img.width
    new_h = max(1, int(img.height * scale))
    new_size = (target_w, new_h)
    
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    log.info(f"Resized to: {img.size}, scale: {scale:.3f}")
    
    # STEP 6: Center on canvas
    canvas = Image.new("RGBA", (cs, cs), (0, 0, 0, 0))
    offset_x = (cs - img.width) // 2
    offset_y = (cs - img.height) // 2
    canvas.paste(img, (offset_x, offset_y), img)
    log.info(f"Centered on {cs}x{cs} canvas at offset ({offset_x}, {offset_y})")
    
    # STEP 7: Tone normalization
    canvas = tone_normalize(canvas)
    
    # STEP 8: Generate filename and save
    filename = generate_filename(team_handle, design_slug, product_id, view_type)
    
    # Save to bytes with optimization
    buf = io.BytesIO()
    canvas.save(
        buf, 
        format="PNG", 
        optimize=True, 
        compress_level=7,
        dpi=(300, 300)
    )
    png_bytes = buf.getvalue()
    
    filesize_mb = len(png_bytes) / (1024 * 1024)
    log.info(f"Output size: {filesize_mb:.2f} MB")
    
    # Warn if file is too large
    if filesize_mb > MAX_FILE_MB:
        log.warning(f"File size {filesize_mb:.2f} MB exceeds target {MAX_FILE_MB} MB")
    
    # STEP 9: Build metadata
    content_hash = hashlib.sha1(png_bytes).hexdigest()
    
    meta = {
        "width": cs,
        "height": cs,
        "bbox_source": bbox_source,
        "bbox": list(bbox),
        "buffer_px": buffer_px,
        "scale_factor": round(scale, 4),
        "fill_ratio": fr,
        "canvas_size": cs,
        "brightness": BRIGHTNESS,
        "contrast": CONTRAST,
        "filesize_bytes": len(png_bytes),
        "filesize_mb": round(filesize_mb, 3),
        "filename": filename,
        "content_hash": f"sha1:{content_hash}",
        "version": "preproc-1.0"
    }
    
    log.info(f"Processing complete: {filename}")
    
    # STEP 10: Return PNG image with metadata in headers
    headers = {
        "Content-Type": "image/png",
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Image-Width": str(cs),
        "X-Image-Height": str(cs),
        "X-Bbox-Source": bbox_source,
        "X-Scale-Factor": str(round(scale, 4)),
        "X-Filesize-Bytes": str(len(png_bytes)),
        "X-Content-Hash": f"sha1:{content_hash}",
        "X-Filename": filename
    }
    
    return Response(content=png_bytes, media_type="image/png", headers=headers)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
