

import argparse, os
import numpy as np
import cv2
from PIL import Image

rng = np.random.default_rng()

# =========================================================
#  ECG LINE PROTECTION (VERY IMPORTANT)
# =========================================================
def protect_signal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    mask = cv2.dilate(edges, None, iterations=2)
    out = img.copy()
    out[mask > 0] = img[mask > 0]
    return out

# =========================================================
#  BASIC EFFECTS
# =========================================================
def rotate(img, max_deg=4):
    angle = rng.uniform(-max_deg, max_deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
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
    return cv2.GaussianBlur(img, (3,3), 0)

def gray_contrast(img, alpha=0.85, beta=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adj = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return cv2.cvtColor(adj, cv2.COLOR_GRAY2BGR)

# =========================================================
#  FOLD (IMPROVED - CLEAR & VISIBLE)
# =========================================================
def add_realistic_fold(img):
    h, w = img.shape[:2]
    out = img.copy()

    # Choose fold direction
    vertical = rng.random() < 0.5

    if vertical:
        x = rng.integers(int(w*0.25), int(w*0.75))
        fold_width = rng.integers(20, 50)
        for i in range(-fold_width, fold_width):
            alpha = (fold_width - abs(i)) / fold_width
            shade = int(255 * (0.78 + 0.20 * alpha))
            cv2.line(out, (x + i, 0), (x + i, h), (shade, shade, shade), 1)
    else:
        y = rng.integers(int(h*0.25), int(h*0.75))
        fold_width = rng.integers(20, 50)
        for i in range(-fold_width, fold_width):
            alpha = (fold_width - abs(i)) / fold_width
            shade = int(255 * (0.78 + 0.20 * alpha))
            cv2.line(out, (0, y + i), (w, y + i), (shade, shade, shade), 1)

    return protect_signal(out)

# =========================================================
#  LIGHT WRINKLE (OPTIONAL)
# =========================================================
def wrinkle(img):
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    wave = 10 * np.sin(map_y / 22.0)  # gentle
    map_x = (map_x + wave).astype(np.float32)
    warped = cv2.remap(img, map_x, map_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return protect_signal(warped)

# =========================================================
#  PHONE SHADOW / LIGHTING BIAS
# =========================================================
def add_shadow(img, direction="left", strength=0.6):
    h, w = img.shape[:2]
    shadow = np.zeros((h, w), dtype=np.float32)

    if direction == "left":
        for i in range(w):
            shadow[:,i] = 1 - strength * (i / w)
    elif direction == "right":
        for i in range(w):
            shadow[:,i] = 1 - strength * ((w-i) / w)
    elif direction == "top":
        for i in range(h):
            shadow[i,:] = 1 - strength * (i / h)
    else:  # bottom
        for i in range(h):
            shadow[i,:] = 1 - strength * ((h-i) / h)

    shadow_3d = np.stack([shadow]*3, axis=2)
    shaded = (img.astype(np.float32) * shadow_3d).astype(np.uint8)
    return protect_signal(shaded)

# =========================================================
#  PIPELINE
# =========================================================
def apply_pipeline(img):
    if rng.random() < 0.8: img = rotate(img, 5)
    if rng.random() < 0.7: img = mild_blur(img)
    if rng.random() < 0.8: img = mild_speckle(img)
    if rng.random() < 0.7: img = gray_contrast(img, alpha=0.83, beta=18)
    if rng.random() < 0.8: img = add_realistic_fold(img)
    if rng.random() < 0.4: img = wrinkle(img)
    if rng.random() < 0.6:
        img = add_shadow(img, direction=rng.choice(["left","right","top","bottom"]))
    return protect_signal(img)

# =========================================================
#  MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_png")
    ap.add_argument("--outdir", default="aug_results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    img = cv2.imread(args.input_png)
    if img is None:
        raise ValueError("Failed to load input image.")

    for i in range(10):
        augmented = apply_pipeline(img)
        outname = f"{args.outdir}/ecg_aug_{i:02d}.png"
        cv2.imwrite(outname, augmented)
        print(f"[OK] → {outname}")

if __name__ == "__main__":
    main()
