#!/usr/bin/env python3
"""
ECG PDF → heavy scan simulation (strong but still readable)
Adds:
 - big dark speckle dots (hardware dust)
 - strong blur (camera shake / bad scanner optics)
 - rotation up to ~7 degrees
 - stronger contrast loss / gray cast
 - larger random crop offsets
"""

import argparse, os, io
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
import img2pdf

rng = np.random.default_rng()

# =================== Heavy degradations ===================

def rotate(img, max_deg=7):
    angle = rng.uniform(-max_deg, max_deg)
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))

def heavy_speckle(img, density=0.0012, rmin=1, rmax=3):
    out = img.copy()
    h,w = out.shape[:2]
    n = int(density * h * w)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    for x,y in zip(xs, ys):
        r = rng.integers(rmin, rmax+1)
        cv2.circle(out, (int(x),int(y)), r, (0,0,0), -1)
    return out

def strong_blur(img, strength):
    # kernel from 5 to 11, gaussian sigma heavier
    k = 5 + int(strength*6)
    if k % 2 == 0: k+=1
    return cv2.GaussianBlur(img, (k,k), sigmaX=2 + strength*3)

def gray_low_contrast(img, strength):
    alpha = 0.85 - 0.35*strength   # significant contrast drop
    beta  = 5 + int(20*strength)   # slight bright/haze
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)

def heavy_crop(img, strength):
    pct = rng.uniform(0.02, 0.08 + 0.08*strength)  # bigger misalignments
    h,w = img.shape[:2]
    t = int(pct*h * rng.uniform(0.3,1))
    b = int(pct*h * rng.uniform(0.3,1))
    l = int(pct*w * rng.uniform(0.3,1))
    r = int(pct*w * rng.uniform(0.3,1))
    cropped = img[t:h-b, l:w-r]
    return cv2.copyMakeBorder(cropped, t,b,l,r, cv2.BORDER_CONSTANT, value=(255,255,255))

def jpeg_artifacts_heavy(img, strength):
    q = int(80 - 40*strength)  # strong softening but not blocky garbage
    q = max(25, q)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,q])
    return cv2.imdecode(enc,1)

# =================== Pipeline ===================

def apply_pipeline(img, strength=0.8):
    out = img.copy()

    if rng.random() < 0.98:
        out = rotate(out, max_deg=3 + strength*6)

    if rng.random() < 0.95:
        out = heavy_speckle(out, density=0.0008 + strength*0.002, rmin=1, rmax=3)

    if rng.random() < 0.98:
        out = strong_blur(out, strength)

    if rng.random() < 0.95:
        out = gray_low_contrast(out, strength)

    if rng.random() < 0.9:
        out = heavy_crop(out, strength)

    if rng.random() < 0.98:
        out = jpeg_artifacts_heavy(out, strength)

    return out

# =================== PDF utils ===================

def pdf_to_imgs(pdf, dpi=300):
    return convert_from_path(pdf, dpi=dpi)

def imgs_to_pdf(imgs, outfile):
    img_bytes=[]
    for im in imgs:
        if im.mode!="RGB": im=im.convert("RGB")
        b=io.BytesIO(); im.save(b, format="JPEG", quality=95)
        img_bytes.append(b.getvalue())
    with open(outfile,"wb") as f:
        f.write(img2pdf.convert(img_bytes))

# =================== Batch ===================

def generate_batch(pdf_path, n=10):
    os.makedirs("heavy_outputs", exist_ok=True)
    pages = pdf_to_imgs(pdf_path)

    for i in range(n):
        strength = np.clip(0.45 + 0.055*i, 0.45, 1.0)
        imgs=[]
        os.makedirs(f"heavy_outputs/previews_{i:02d}", exist_ok=True)

        for p_i, pil in enumerate(pages):
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            aug = apply_pipeline(bgr, strength)
            rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.save(f"heavy_outputs/previews_{i:02d}/page_{p_i:02d}.png")
            imgs.append(im)

        outfile = f"heavy_outputs/ecg_heavy_{i:02d}.pdf"
        imgs_to_pdf(imgs, outfile)
        print(f"[OK] {outfile}")

# =================== Main ===================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_pdf")
    ap.add_argument("out_pdf", nargs="?")
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--preview", type=str, default=None)
    args=ap.parse_args()

    if args.in_pdf=="BATCH":
        generate_batch(args.out_pdf)
        return

    pages = pdf_to_imgs(args.in_pdf)
    imgs=[]
    if args.preview: os.makedirs(args.preview, exist_ok=True)

    for i,pil in enumerate(pages):
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        aug = apply_pipeline(bgr, args.strength)
        rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        if args.preview: im.save(f"{args.preview}/page_{i:02d}.png")
        imgs.append(im)

    imgs_to_pdf(imgs, args.out_pdf)
    print(f"[DONE] {args.out_pdf}")

if __name__=="__main__":
    main()
