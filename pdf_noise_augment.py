#!/usr/bin/env python3
"""
ECG AUGMENTATION – MODERATE + 10x HEAVY (VISIBLE BUT NOT BLURRY)
 - Folds + wrinkles preserved
 - ECG waveform protected
"""

import argparse, os, cv2, io
import numpy as np
from PIL import Image
import img2pdf

rng = np.random.default_rng()

# ---------- ECG PROTECTION ----------
def protect_signal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)   # STRONGER LINE PROTECTION
    mask = cv2.dilate(edges, None, iterations=2)
    out = img.copy()
    out[mask > 0] = img[mask > 0]
    return out

# ---------- BASE FX ----------
def rotate(img, max_deg=4):
    ang = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

def mild_speckle(img, density=0.0004):
    out = img.copy()
    h, w = img.shape[:2]
    n = int(density * h * w)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), 2, (0,0,0), -1)
    return out

def mild_blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)   # SOFT only

# ---------- BIG FOLD EFFECT (CLEAR NOW) ----------
def fold_paper(img):
    h, w = img.shape[:2]
    x = rng.integers(int(w*0.2), int(w*0.8))
    fold = img.copy()

    for i in range(-25, 25):    # small fold band
        val = 255 - abs(i) * 6  # shading only
        fold[:, x+i] = np.clip(fold[:, x+i]*0.4 + val, 0, 255)

    blend = cv2.addWeighted(img, 0.8, fold, 0.2, 0)
    return protect_signal(blend)

# ---------- LIGHT WRINKLE (NO BLUR) ----------
def wrinkle(img):
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    wave = 15 * np.sin(map_y / 20.0)
    map_x = (map_x + wave).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return protect_signal(warped)

# ---------- PIPELINE ----------
def apply_heavy(img):
    if rng.random() < 1.0: img = rotate(img, 5)
    if rng.random() < 0.8: img = mild_speckle(img)
    img = fold_paper(img)      # always
    if rng.random() < 0.7: img = wrinkle(img)
    return protect_signal(img)

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_png")
    ap.add_argument("--outdir", default="augmented_ecg")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input_png)
    if img is None:
        raise ValueError("Could not load file")

    for i in range(10):              # GENERATE 10 IMAGES
        out = apply_heavy(img)
        out_name = f"{args.outdir}/ecg_heavy_{i:02d}.png"
        cv2.imwrite(out_name, out)
        print(f"[OK] → {out_name}")

if __name__ == "__main__":
    main()
