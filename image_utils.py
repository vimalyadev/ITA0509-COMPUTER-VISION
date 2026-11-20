"""
image_utils.py – Pillow‑based OpenCV shim (same as Assignment 1)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

def imread(path, flags=1):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.uint8)
    return arr[..., ::-1].copy()

def imwrite(path, arr):
    rgb = arr[..., ::-1]
    Image.fromarray(rgb).save(path)
    return True

# Histogram equalization (global) using scikit-image for 8‑bit images
def equalize_hist(arr):
    from skimage import exposure
    # Ensure uint8 grayscale
    if arr.ndim == 3:
        # Convert to gray first
        gray = cvtColor(arr, 6)
    else:
        gray = arr
    eq = exposure.equalize_hist(gray)
    # skimage returns float in [0,1]; convert back to uint8
    return (eq * 255).astype(np.uint8)

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe(arr, clip_limit=2.0, tile_grid_size=(8,8)):
    from skimage import exposure
    if arr.ndim == 3:
        gray = cvtColor(arr, 6)
    else:
        gray = arr
    cl = exposure.equalize_adapthist(gray, clip_limit=clip_limit, kernel_size=tile_grid_size)
    return (cl * 255).astype(np.uint8)

def resize(arr, size, interpolation=1):
    method_map = {0: Image.NEAREST, 1: Image.BILINEAR, 2: Image.BOX, 3: Image.BICUBIC}
    img = Image.fromarray(arr[..., ::-1]).resize(size, method_map.get(interpolation, Image.BILINEAR))
    return np.asarray(img)[..., ::-1]

def cvtColor(arr, code):
    if code == 6:  # BGR2GRAY
        rgb = arr[..., ::-1] / 255.0
        gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])
        return (gray * 255).astype(np.uint8)
    if code == 40:  # BGR2HSV
        rgb = arr[..., ::-1] / 255.0
        maxc = rgb.max(-1)
        minc = rgb.min(-1)
        v = maxc
        delta = maxc - minc
        s = np.where(maxc == 0, 0, delta / maxc)
        rc = (maxc - rgb[..., 0]) / (delta + 1e-10)
        gc = (maxc - rgb[..., 1]) / (delta + 1e-10)
        bc = (maxc - rgb[..., 2]) / (delta + 1e-10)
        h = np.select([rgb[..., 0] == maxc, rgb[..., 1] == maxc], [60 * (bc - gc), 60 * (2.0 + rc)], default=60 * (4.0 + gc))
        h = (h % 360).astype(np.uint8)
        s = (s * 255).astype(np.uint8)
        v = (v * 255).astype(np.uint8)
        return np.stack([h, s, v], axis=-1)
    if code == 44:  # BGR2LAB (approx via Pillow)
        img = Image.fromarray(arr[..., ::-1]).convert('LAB')
        return np.asarray(img)
    raise NotImplementedError

def threshold(src, thresh, maxval, type_flag=0):
    if type_flag == 1:  # BINARY_INV
        dst = np.where(src > thresh, 0, maxval).astype(np.uint8)
    else:
        dst = np.where(src > thresh, maxval, 0).astype(np.uint8)
    return dst, dst

def inRange(src, lower, upper):
    mask = np.logical_and.reduce([src[..., i] >= lower[i] for i in range(len(lower))] + [src[..., i] <= upper[i] for i in range(len(upper))])
    return (mask * 255).astype(np.uint8)

def getStructuringElement(shape, ksize):
    return np.ones(ksize, dtype=np.uint8)

def morphologyEx(src, op, kernel):
    if op == 2:  # OPEN
        pil = Image.fromarray(src).filter(ImageFilter.MinFilter(kernel.shape[0]))
        pil = pil.filter(ImageFilter.MaxFilter(kernel.shape[0]))
        return np.asarray(pil)
    raise NotImplementedError

def Canny(img, t1, t2):
    gray = cvtColor(img, 6)
    pil = Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(pil)
    edges = np.where((arr > t1) & (arr < t2), 255, 0).astype(np.uint8)
    return edges

def distanceTransform(src, dist_type, mask_size):
    from scipy import ndimage
    bool_src = src.astype(bool)
    dt = ndimage.distance_transform_edt(~bool_src)
    return dt.astype(np.float32)

def connectedComponentsWithStats(binary, connectivity=8):
    from scipy import ndimage
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labeled, num = ndimage.label(binary, structure=structure)
    stats = np.zeros((num + 1, 5), dtype=np.int32)
    centroids = np.zeros((num + 1, 2), dtype=np.float64)
    for i in range(1, num + 1):
        mask = labeled == i
        coords = np.column_stack(np.where(mask))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        area = mask.sum()
        stats[i] = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1, area]
        centroids[i] = [x_min + (x_max - x_min) / 2.0, y_min + (y_max - y_min) / 2.0]
    return num + 1, labeled, stats, centroids
