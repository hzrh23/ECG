#!/usr/bin/env python3
"""
ECG PDF → realistic scan style augmentation (moderate noise)
Adds:
 - slightly stronger blur
 - more random small rotations
 - variable contrast fade
 - random subtle cropping
"""

import argparse, os, io
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
import img2pdf

rng = np.random.default_rng()

# =================== Effects ===================

def rotate(img, max_deg=4):
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

def tiny_speckle(img, density=0.00025):
    out = img.copy()
    h, w = out.shape[:2]
    n = int(density * h * w)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    for x, y in zip(xs, ys):
        cv2.circle(out, (int(x), int(y)), 1, (0,0,0), -1)
    return out

def mild_blur(img, strength):
    k = 3 + int(strength * 4)  # 3–7 kernel
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k,k), sigmaX=1.0 + strength*2)

def soft_grayscale(img, strength):
    alpha = 0.92 - 0.25 * strength   # contrast fade
    beta = 4 + int(12 * strength)    # slight brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)

def subtle_crop(img, strength):
    pct = 0.01 + rng.uniform(0, 0.04 + strength*0.04)  # variable crop
    h, w = img.shape[:2]
    t = int(pct * h * rng.uniform(0.3,1))
    b = int(pct * h * rng.uniform(0.3,1))
    l = int(pct * w * rng.uniform(0.3,1))
    r = int(pct * w * rng.uniform(0.3,1))
    cropped = img[t:h-b, l:w-r]
    return cv2.copyMakeBorder(
        cropped, t, b, l, r,
        cv2.BORDER_CONSTANT, value=(255,255,255)
    )

def soft_jpeg(img, strength):
    q = int(92 - 15*strength)  # slightly softer edges
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, 1)

# =================== Pipeline ===================

def apply_pipeline(img, strength=0.45):
    out = img.copy()

    # randomness per transform
    if rng.random() < 0.98:
        out = rotate(out, max_deg=2 + 3*strength)

    if rng.random() < 0.90:
        out = tiny_speckle(out, density=0.0002 + strength*0.0006)

    if rng.random() < 0.95:
        out = mild_blur(out, strength)

    if rng.random() < 0.90:
        out = soft_grayscale(out, strength)

    if rng.random() < 0.85:
        out = subtle_crop(out, strength)

    if rng.random() < 0.95:
        out = soft_jpeg(out, strength)

    return out

# =================== PDF utilities ===================

def pdf_to_imgs(pdf, dpi=300):
    return convert_from_path(pdf, dpi=dpi)

def imgs_to_pdf(imgs, outfile):
    img_bytes = []
    for im in imgs:
        if im.mode != "RGB": im = im.convert("RGB")
        b = io.BytesIO(); im.save(b, format="JPEG", quality=95)
        img_bytes.append(b.getvalue())
    with open(outfile, "wb") as f:
        f.write(img2pdf.convert(img_bytes))

# =================== Batch generation ===================

def generate_batch(pdf_path, n=10):
    os.makedirs("moderate_outputs", exist_ok=True)
    pages = pdf_to_imgs(pdf_path)

    for i in range(n):
        strength = np.clip(0.25 + 0.07*i, 0.25, 0.75)  # varying strength
        imgs = []
        os.makedirs(f"moderate_outputs/previews_{i:02d}", exist_ok=True)

        for p_i, pil in enumerate(pages):
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            aug = apply_pipeline(bgr, strength)
            rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img.save(f"moderate_outputs/previews_{i:02d}/page_{p_i:02d}.png")
            imgs.append(img)

        outfile = f"moderate_outputs/ecg_mod_{i:02d}.pdf"
        imgs_to_pdf(imgs, outfile)
        print(f"[OK] {outfile}")

# =================== Main ===================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_pdf")
    ap.add_argument("out_pdf", nargs="?")
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--preview", type=str, default=None)
    args = ap.parse_args()

    if args.in_pdf == "BATCH":
        generate_batch(args.out_pdf)
        return

    pages = pdf_to_imgs(args.in_pdf)
    imgs = []
    os.makedirs(args.preview, exist_ok=True) if args.preview else None

    for i, pil in enumerate(pages):
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        aug = apply_pipeline(bgr, args.strength)
        rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if args.preview:
            img.save(f"{args.preview}/page_{i:02d}.png")
        imgs.append(img)

    imgs_to_pdf(imgs, args.out_pdf)
    print(f"[DONE] {args.out_pdf}")

if __name__ == "__main__":
    main()
