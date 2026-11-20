# image_utils.py – Pillow‑based OpenCV shim (same as Assignment 2)
"""A very small subset of OpenCV functions implemented with Pillow.
Only the functions needed by the placeholder scripts are provided.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def imread(path, as_gray=False):
    img = Image.open(path)
    if as_gray:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    arr = np.asarray(img, dtype=np.uint8)
    # Return BGR like OpenCV
    if not as_gray:
        arr = arr[..., ::-1]
    return arr


def imwrite(path, arr):
    # Assume arr is BGR; convert to RGB
    if arr.ndim == 3:
        arr = arr[..., ::-1]
    Image.fromarray(arr).save(path)
    return True


def resize(arr, size, interpolation=1):
    # size = (width, height)
    method_map = {0: Image.NEAREST, 1: Image.BILINEAR, 2: Image.BICUBIC, 3: Image.LANCZOS}
    img = Image.fromarray(arr[..., ::-1] if arr.ndim == 3 else arr)
    img = img.resize(size, method_map.get(interpolation, Image.BILINEAR))
    out = np.asarray(img)
    if arr.ndim == 3:
        out = out[..., ::-1]
    return out


def cvtColor(arr, code):
    # Only implement BGR2GRAY (code 6)
    if code == 6:
        rgb = arr[..., ::-1]
        gray = np.dot(rgb, [0.2989, 0.5870, 0.1140])
        return gray.astype(np.uint8)
    raise NotImplementedError


def GaussianBlur(arr, ksize, sigmaX=0):
    img = Image.fromarray(arr[..., ::-1] if arr.ndim == 3 else arr)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=sigmaX if sigmaX>0 else 1))
    out = np.asarray(blurred)
    if arr.ndim == 3:
        out = out[..., ::-1]
    return out


def Canny(arr, thresh1, thresh2):
    # Very naive Canny: use edge detection from Pillow
    gray = cvtColor(arr, 6)
    img = Image.fromarray(gray)
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(edges)
    return (edges > thresh1).astype(np.uint8) * 255
