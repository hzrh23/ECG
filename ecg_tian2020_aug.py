#!/usr/bin/env python3
"""
Augmentation pipeline inspired by:
Y. Tian and S. Lu, "Data augmentation for document image classification
using generative adversarial networks," CVPR Workshops, 2020.

We replicate GAN-generated document artifacts using procedural transforms:
 - Illumination bias
 - Paper folds
 - Local deformation
 - Speckle noise
 - Blur
 - Grayscale/contrast decay
"""

import os, argparse
import numpy as np
import cv2

rng = np.random.default_rng()

# ==========================
# 1. Illumination (Shadow)
# ==========================

def add_shadow(img, strength=0.6):
    h, w = img.shape[:2]
    shadow = np.zeros((h, w), dtype=np.float32)

    direction = rng.choice(["left","right","top","bottom"])

    if direction == "left":
        for i in range(w):
            shadow[:, i] = 1 - strength * (i / w)
    elif direction == "right":
        for i in range(w):
            shadow[:, i] = 1 - strength * ((w - i) / w)
    elif direction == "top":
        for i in range(h):
            shadow[i, :] = 1 - strength * (i / h)
    else:
        for i in range(h):
            shadow[i, :] = 1 - strength * ((h - i) / h)

    shadow = np.stack([shadow]*3, axis=2)
    shaded = (img.astype(np.float32) * shadow).astype(np.uint8)
    return shaded

# ==========================
# 2. Paper Fold Simulation
# ==========================

def add_fold(img):
    h, w = img.shape[:2]
    out = img.copy()

    vertical = rng.random() < 0.5

    if vertical:
        x = rng.integers(int(w*0.2), int(w*0.8))
        width = rng.integers(20, 50)
        for i in range(-width, width):
            alpha = (width - abs(i)) / width
            shade = int(255 * (0.75 + 0.20 * alpha))
            cv2.line(out, (x + i, 0), (x + i, h), (shade, shade, shade), 1)
    else:
        y = rng.integers(int(h*0.2), int(h*0.8))
        width = rng.integers(20, 50)
        for i in range(-width, width):
            alpha = (width - abs(i)) / width
            shade = int(255 * (0.75 + 0.20 * alpha))
            cv2.line(out, (0, y + i), (w, y + i), (shade, shade, shade), 1)

    return out

# ==========================
# 3. Local Wrinkle
# ==========================

def wrinkle(img):
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    wave = 10 * np.sin(map_y / 20.0)
    map_x = (map_x + wave).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return warped

# ==========================
# 4. Classic Noise Models
# ==========================

def speckle(img, density=0.0004):
    out = img.copy()
    h, w = img.shape[:2]
    n = int(h*w*density)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), 1, (0,0,0), -1)
    return out

def blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)

# ==========================
# 5. Grayscale Decay
# ==========================

def gray_decay(img, alpha=0.85, beta=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faded = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return cv2.cvtColor(faded, cv2.COLOR_GRAY2BGR)

# ==========================
# Pipeline
# ==========================

def apply_tian2020_pipeline(img):
    out = img.copy()

    if rng.random() < 0.8: out = add_shadow(out)
    if rng.random() < 0.8: out = add_fold(out)
    if rng.random() < 0.5: out = wrinkle(out)
    if rng.random() < 0.8: out = speckle(out)
    if rng.random() < 0.7: out = blur(out)
    if rng.random() < 0.7: out = gray_decay(out)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_png")
    ap.add_argument("--outdir", default="tian2020_results")
    ap.add_argument("--count", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input_png)

    for i in range(args.count):
        aug = apply_tian2020_pipeline(img)
        outname = f"{args.outdir}/aug_tian2020_{i:02d}.png"
        cv2.imwrite(outname, aug)
        print("[OK] →", outname)


if __name__ == "__main__":
    main()
