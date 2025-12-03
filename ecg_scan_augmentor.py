#!/usr/bin/env python3
"""
Full ECG PDF Augmentation Engine

Modes:
  subtle   – light scanner noise
  moderate – print-scan level
  heavyA   – extreme rotation + zoom + crop (damaged scan)
  heavyB   – phone-shot style (perspective + shadow)
  heavyC   – half-crop + mega-blur + strong speckle

Usage:
  python ecg_augment_all.py --mode heavyA in.pdf out.pdf
  python ecg_augment_all.py --mode heavyB --preview prev in.pdf out.pdf
  python ecg_augment_all.py BATCH in.pdf   (generates 10 variations)
"""

import argparse
import os, io
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
import img2pdf

rng = np.random.default_rng()

# ============================================================
#  Helper functions
# ============================================================

def speckle(img, density=0.0003, rmin=1, rmax=2):
    out = img.copy()
    h, w = out.shape[:2]
    n = int(h * w * density)
    ys = rng.integers(0, h, n)
    xs = rng.integers(0, w, n)
    for x, y in zip(xs, ys):
        r = rng.integers(rmin, rmax + 1)
        cv2.circle(out, (int(x), int(y)), r, (0, 0, 0), -1)
    return out


def pdf_to_imgs(pdf, dpi=300):
    return convert_from_path(pdf, dpi=dpi)


def imgs_to_pdf(imgs, outfile):
    img_bytes = []
    for im in imgs:
        if im.mode != "RGB":
            im = im.convert("RGB")
        b = io.BytesIO()
        im.save(b, format="JPEG", quality=95)
        img_bytes.append(b.getvalue())
    with open(outfile, "wb") as f:
        f.write(img2pdf.convert(img_bytes))


# ============================================================
#  MODE 1 — SUBTLE
# ============================================================

def pipeline_subtle(img, strength=0.35):
    out = img.copy()

    # slight rotation
    angle = rng.uniform(-3, 3)
    h, w = out.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(255,255,255))

    # tiny speckle
    out = speckle(out, density=0.0002, rmin=1, rmax=1)

    # light blur
    out = cv2.GaussianBlur(out, (3,3), sigmaX=1.1)

    # grayscale fade
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=0.92, beta=5)
    out = cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)

    # tiny crop
    pct = rng.uniform(0.01, 0.03)
    t = int(pct * h)
    b = int(pct * h)
    l = int(pct * w)
    r = int(pct * w)
    cropped = out[t:h-b, l:w-r]
    out = cv2.copyMakeBorder(cropped, t,b,l,r, cv2.BORDER_CONSTANT, value=(255,255,255))

    return out


# ============================================================
#  MODE 2 — MODERATE
# ============================================================

def pipeline_moderate(img, strength=0.55):
    out = img.copy()
    h, w = out.shape[:2]

    # moderate rotation
    angle = rng.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(255,255,255))

    # moderate speckle
    out = speckle(out, density=0.0008, rmin=1, rmax=2)

    # blur
    k = 5
    out = cv2.GaussianBlur(out, (k,k), sigmaX=2)

    # grayscale fade
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=0.80, beta=10)
    out = cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)

    # moderate crop
    pct = rng.uniform(0.02, 0.06)
    t = int(pct*h); b=int(pct*h)
    l = int(pct*w); r=int(pct*w)
    cropped = out[t:h-b, l:w-r]
    out = cv2.copyMakeBorder(cropped, t,b,l,r, cv2.BORDER_CONSTANT, value=(255,255,255))

    return out


# ============================================================
#  MODE 3 — HEAVY A
# Extreme rotation + zoom + crop
# ============================================================

def pipeline_heavyA(img, strength=0.8):
    out = img.copy()
    h, w = out.shape[:2]

    # rotation (±12–20°)
    max_deg = 12 + strength * 8
    angle = rng.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(255,255,255))

    # zoom (very light to avoid blur)
    zoom = rng.uniform(1.0, 1.15)
    zh, zw = int(h/zoom), int(w/zoom)
    y0 = rng.integers(0, h-zh)
    x0 = rng.integers(0, w-zw)
    cropped = out[y0:y0+zh, x0:x0+zw]
    out = cv2.resize(cropped, (h, w), interpolation=cv2.INTER_CUBIC)

    # ✅ small crop (1–12% max)
    pct = rng.uniform(0.01, 0.12)
    t = int(pct * h * rng.uniform(0.3, 1))
    b = int(pct * h * rng.uniform(0.3, 1))
    l = int(pct * w * rng.uniform(0.3, 1))
    r = int(pct * w * rng.uniform(0.3, 1))

    cropped = out[t:h-b, l:w-r]
    out = cv2.copyMakeBorder(cropped, t, b, l, r,
                             cv2.BORDER_CONSTANT, value=(255,255,255))

    # blur (controlled)
    k = rng.choice([5, 7])
    out = cv2.GaussianBlur(out, (k, k), sigmaX=1.5 + strength)

    # dust
    out = speckle(out, density=0.001, rmin=1, rmax=2)

    # small contrast fade
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=0.80, beta=10)
    return cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)




# ============================================================
#  MODE 4 — HEAVY B
# Perspective warp + shadow (phone-shot)
# ============================================================

def pipeline_heavyB(img, strength=0.8):
    out = img.copy()
    h, w = out.shape[:2]

    # perspective warp (mild)
    shift = int(min(h,w) * (0.03 + strength * 0.08))
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([
        [rng.integers(0,shift), rng.integers(0,shift)],
        [w-rng.integers(0,shift), rng.integers(0,shift)],
        [rng.integers(0,shift), h-rng.integers(0,shift)],
        [w-rng.integers(0,shift), h-rng.integers(0,shift)],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    out = cv2.warpPerspective(out, M, (w, h), borderValue=(255,255,255))

    # soft shadow
    shadow = np.zeros_like(out)
    sx = rng.integers(w//3, 2*w//3)
    cv2.rectangle(shadow, (sx,0), (w, h), (150,150,150), -1)
    out = cv2.addWeighted(out, 1.0, shadow, 0.18, 0)

    # rotation (±8°)
    angle = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(255,255,255))

    # blur (light)
    out = cv2.GaussianBlur(out, (5,5), sigmaX=1.2)

    # dust
    out = speckle(out, density=0.0012, rmin=1, rmax=2)

    # grayscale
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ============================================================
#  MODE 5 — HEAVY C
# half-crop + mega-blur + dark speckles
# ============================================================

def pipeline_heavyC(img, strength=0.8):
    out = img.copy()
    h, w = out.shape[:2]

    # controlled partial crop (keep 80–95% of content)
    cut_pct = rng.uniform(0.05, 0.20)
    side = rng.choice(["left","right","top","bottom"])

    if side == "left":
        out = out[:, int(w * cut_pct):]
    elif side == "right":
        out = out[:, :int(w * (1 - cut_pct))]
    elif side == "top":
        out = out[int(h * cut_pct):, :]
    else:
        out = out[:int(h * (1 - cut_pct)), :]

    # resize back
    out = cv2.resize(out, (w,h), interpolation=cv2.INTER_CUBIC)

    # blur (medium)
    k = rng.choice([5,7])
    out = cv2.GaussianBlur(out, (k,k), sigmaX=1.5 + strength)

    # speckle
    out = speckle(out, density=0.003, rmin=1, rmax=2)

    # faint dirty patch
    if rng.random() < 0.4:
        pw = rng.integers(w//10, w//6)
        ph = rng.integers(h//10, h//6)
        x0 = rng.integers(0, w-pw)
        y0 = rng.integers(0, h-ph)
        out[y0:y0+ph, x0:x0+pw] = out[y0:y0+ph, x0:x0+pw] * rng.uniform(0.85, 0.95)

    # light contrast drop
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=0.70, beta=12)
    return cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)


# ============================================================
#  Main pipeline runner
# ============================================================

def apply_pipeline(img, mode):
    if mode == "subtle":
        return pipeline_subtle(img)
    if mode == "moderate":
        return pipeline_moderate(img)
    if mode == "heavyA":
        return pipeline_heavyA(img)
    if mode == "heavyB":
        return pipeline_heavyB(img)
    if mode == "heavyC":
        return pipeline_heavyC(img)
    raise ValueError("Invalid mode")


def generate_batch(in_pdf, mode):
    os.makedirs(f"{mode}_batch", exist_ok=True)
    pages = pdf_to_imgs(in_pdf)

    for i in range(10):
        imgs=[]
        for p_i, pil in enumerate(pages):
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            aug = apply_pipeline(bgr, mode)
            rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(rgb))
        outname = f"{mode}_batch/{mode}_{i:02d}.pdf"
        imgs_to_pdf(imgs, outname)
        print("[OK] ", outname)


def load_input(in_path):
    """
    Loads input file:
      - PDF → list of PIL images (one per page)
      - PNG/JPG → single PIL image (wrapped as list)
    """
    ext = in_path.lower()
    if ext.endswith(".pdf"):
        return pdf_to_imgs(in_path)
    else:
        img = Image.open(in_path).convert("RGB")
        return [img]


def generate_batch_any(in_file, out_prefix, mode):
    """
    Batch mode that works for PDF and PNG/JPG.
    Produces 10 augmented files.
    """
    os.makedirs(f"{mode}_batch", exist_ok=True)
    pages = load_input(in_file)

    for i in range(10):
        imgs = []

        for p_i, pil in enumerate(pages):
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            aug = apply_pipeline(bgr, mode)
            rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            imgs.append(Image.fromarray(rgb))

        # File extension based on out_prefix (pdf or png/jpg)
        if out_prefix.lower().endswith(".pdf"):
            outfile = f"{mode}_batch/{out_prefix[:-4]}_{i:02d}.pdf"
            imgs_to_pdf(imgs, outfile)
        else:
            outfile = f"{mode}_batch/{out_prefix}_{i:02d}.png"
            imgs[0].save(outfile)

        print("[BATCH OK]", outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_file", help="Input PDF or image")
    ap.add_argument("out_file", help="Output PDF or image")
    ap.add_argument("--mode", type=str, default="subtle",
                    choices=["subtle","moderate","heavyA","heavyB","heavyC"])
    ap.add_argument("--preview", type=str)
    ap.add_argument("--batch", action="store_true")
    args = ap.parse_args()

    # ---------------------------------------------------------
    # BATCH MODE (NOW WORKS FOR PNG/JPG TOO)
    # ---------------------------------------------------------
    if args.batch:
        generate_batch_any(args.in_file, args.out_file, args.mode)
        return

    # ---------------------------------------------------------
    # NORMAL SINGLE-PASS MODE
    # ---------------------------------------------------------
    pages = load_input(args.in_file)
    imgs = []

    # preview folder
    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    # process each page / image
    for i, pil in enumerate(pages):
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        aug = apply_pipeline(bgr, args.mode)
        rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)

        if args.preview:
            im.save(f"{args.preview}/preview_{i:02d}.png")

        imgs.append(im)

    # SAVE OUTPUT
    if args.out_file.lower().endswith(".pdf"):
        imgs_to_pdf(imgs, args.out_file)
    else:
        imgs[0].save(args.out_file)

    print("[DONE]", args.out_file)



if __name__ == "__main__":
    main()
