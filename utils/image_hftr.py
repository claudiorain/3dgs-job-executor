#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

from PIL import Image, ImageOps


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    gy = gx.T
    pad = 1
    gpad = np.pad(gray, pad_width=pad, mode='reflect')
    H, W = gray.shape
    Gx = (
        -1*gpad[0:H,   0:W] + 1*gpad[0:H,   2:W+2] +
        -2*gpad[1:H+1,0:W] + 2*gpad[1:H+1,2:W+2] +
        -1*gpad[2:H+2,0:W] + 1*gpad[2:H+2,2:W+2]
    )
    Gy = (
        -1*gpad[0:H,   0:W] -2*gpad[0:H,   1:W+1] -1*gpad[0:H,   2:W+2] +
         1*gpad[2:H+2,0:W] +2*gpad[2:H+2,1:W+1] +1*gpad[2:H+2,2:W+2]
    )
    return np.hypot(Gx, Gy)

def compute_hftr(images_dir: str, sample_k: int = 20, short_side: int = 720, agg: str = "blend"):
    """
    High-Frequency Texture Ratio (HFTR) over a sample of images.
    Returns float in ~[0.02, 0.5] or None if not computable.
    agg: 'median' | 'p90' | 'blend' | 'trimmed'
    """
    if Image is None or not images_dir:
        print("prova")
        print(Image)
        return None
    p = Path(images_dir)
    if not p.exists() or not p.is_dir():
        return None
    exts = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
    files = [q for q in sorted(p.iterdir()) if q.suffix.lower() in exts and q.is_file()]
    if not files:
        return None
    M = len(files)
    K = min(max(16, (M + 9)//10), 48)  # ≈10% del dataset, clamp a [16,48]
    step = max(1, M // K)
    idxs = [min(M-1, int(round(j*step + step/2))) for j in range(K)]
    vals = []
    for i in idxs:
        fp = files[i]
        try:
            im = Image.open(fp)
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            smin = min(w, h)
            if smin > short_side:
                scale = short_side / smin
                im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.BILINEAR)
            y = np.asarray(im.convert('L'), dtype=np.float32)
            mag = _sobel_mag(y)
            med = float(np.median(mag))
            mad = float(np.median(np.abs(mag - med))) + 1e-6
            tau = med + 1.5*mad
            hf = float((mag >= tau).mean())
            vals.append(hf)
        except Exception:
            continue
    if not vals:
        return None
    vals = np.array(vals, dtype=np.float32)
    if agg == "median":
        return float(np.median(vals))
    elif agg == "p90":
        return float(np.quantile(vals, 0.9))
    elif agg == "trimmed":
        lo, hi = np.quantile(vals, [0.1, 0.9])
        sel = vals[(vals>=lo) & (vals<=hi)]
        return float(sel.mean()) if sel.size>0 else float(vals.mean())
    else:  # blend
        return float(0.7*np.median(vals) + 0.3*np.quantile(vals, 0.9))

def norm_hftr(hftr: float | None, lo: float = 0.05, hi: float = 0.35) -> float:
    if hftr is None:
        return 0.0
    x = max(lo, min(hi, hftr))
    return (x - lo) / (hi - lo)
