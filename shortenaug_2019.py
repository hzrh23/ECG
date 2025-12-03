#!/usr/bin/env python3
"""
Augmentation techniques implemented directly from:

C. Shorten and T. M. Khoshgoftaar,
"A Survey on Image Data Augmentation for Deep Learning,"
Journal of Big Data, vol. 6, no. 1, pp. 1–48, 2019.

The survey organizes augmentations into:
  - Geometric transformations
  - Pixel-level transformations
  - Kernel filter transforms
  - Noise injection
  - Random erasing (cutout)
  - Color-space augmentation

This file implements representative examples of each.
"""

import cv2
import numpy as np
import argparse, os

rng = np.random.default_rng()

# ======================================================
# (1) GEOMETRIC TRANSFORMS
# ======================================================

def rotate(img, max_deg=20):
    """Geometric transform: random rotation (Shorten 2019 Section 3.1)."""
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

def random_crop(img, crop_percent=0.1):
    """Random cropping (Shorten 2019 Section 3.1)."""
    h, w = img.shape[:2]
    ch = int(h * crop_percent)
    cw = int(w * crop_percent)

    top = rng.integers(0, ch)
    left = rng.integers(0, cw)

    cropped = img[top:h - (ch - top), left:w - (cw - left)]
    return cv2.resize(cropped, (w, h))

def flip_horizontal(img):
    """Horizontal flipping (Shorten 2019 Section 3.1)."""
    return cv2.flip(img, 1)

def affine_transform(img):
    """Affine scaling + shear (Shorten 2019 Section 3.2)."""
    h, w = img.shape[:2]

    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([
        [0, 0],
        [w * rng.uniform(0.9, 1.1), h * rng.uniform(-0.05, 0.05)],
        [w * rng.uniform(-0.05, 0.05), h * rng.uniform(0.9, 1.1)]
    ])

    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

# ======================================================
# (2) PIXEL-LEVEL TRANSFORMS
# ======================================================

def adjust_brightness_contrast(img, alpha=1.0, beta=10):
    """Brightness & contrast jitter (Shorten 2019 Section 3.3)."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def convert_grayscale(img):
    """Colorspace transform → grayscale (Shorten 2019 Section 3.3)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ======================================================
# (3) KERNEL FILTER TRANSFORMS
# ======================================================

def gaussian_blur(img, k=5):
    """Gaussian blur augmentation (Shorten 2019 Section 3.3)."""
    return cv2.GaussianBlur(img, (k, k), 0)

# ======================================================
# (4) NOISE INJECTION
# ======================================================

def gaussian_noise(img, sigma=12):
    """Additive Gaussian noise (Shorten 2019 Section 3.4)."""
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def salt_pepper(img, p=0.002):
    """Salt-and-pepper noise (Shorten 2019 Section 3.4)."""
    noisy = img.copy()
    h, w = img.shape[:2]
    num = int(h * w * p)

    # salt
    xs = rng.integers(0, w, num)
    ys = rng.integers(0, h, num)
    noisy[ys, xs] = 255

    # pepper
    xs = rng.integers(0, w, num)
    ys = rng.integers(0, h, num)
    noisy[ys, xs] = 0

    return noisy

# ======================================================
# (5) RANDOM ERASING (CUTOUT)
# ======================================================

def random_erasing(img, erase_pct=0.15):
    """
    Implements random erasing described in Shorten 2019 Section 3.5
    (originally from Zhong et al., 2017).
    """
    out = img.copy()
    h, w = img.shape[:2]

    mask_h = int(h * erase_pct)
    mask_w = int(w * erase_pct)

    y = rng.integers(0, h - mask_h)
    x = rng.integers(0, w - mask_w)

    out[y:y+mask_h, x:x+mask_w] = 255
    return out

# ======================================================
# PIPELINE
# ======================================================

def apply_shorten2019_pipeline(img):
    """Combination of classical methods recommended in Shorten & Khoshgoftaar (2019)."""
    out = img.copy()

    # Geometric
    if rng.random() < 0.7: out = rotate(out)
    if rng.random() < 0.5: out = random_crop(out)
    if rng.random() < 0.4: out = affine_transform(out)

    # Pixel-level
    if rng.random() < 0.6: out = gaussian_noise(out)
    if rng.random() < 0.6: out = gaussian_blur(out)
    if rng.random() < 0.5: out = adjust_brightness_contrast(out, alpha=rng.uniform(0.8,1.2))

    # Salt & pepper
    if rng.random() < 0.3: out = salt_pepper(out)

    # Cutout
    if rng.random() < 0.3: out = random_erasing(out)

    return out

# ======================================================
# MAIN
# ======================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_png")
    ap.add_argument("--outdir", default="shorten2019_results")
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input_png)

    for i in range(args.count):
        aug = apply_shorten2019_pipeline(img)
        outname = f"{args.outdir}/aug_shorten2019_{i:02d}.png"
        cv2.imwrite(outname, aug)
        print("[OK] →", outname)


if __name__ == "__main__":
    main()
