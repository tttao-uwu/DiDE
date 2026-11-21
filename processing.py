from __future__ import annotations

import os

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def palette_voronoi(
    image_path: str | os.PathLike,
    n_colors: int = 5,
    k: int = 5,
    width: int = 518,
    height: int = 518,
) -> Image.Image:
    """读取颜色图像，提取主色并返回随机 Voronoi 色块图（PIL RGB 图像）。

    Args:
        image_path: 输入图像路径。
        n_colors: 使用 KMeans 提取的主色数量，默认 5。
        k: 控制随机中心数量，总数为 n_colors * k，默认 5。
        width: 输出图像宽度，默认 518。
        height: 输出图像高度，默认 518。

    Returns:
        PIL.Image.Image，可直接使用或调用 .save(path) 保存。
        每次调用都会重新随机生成色块布局。
    """
    for name, value in (
        ("n_colors", n_colors), ("k", k), ("width", width), ("height", height)
    ):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    with Image.open(image_path) as image:
        pixels = np.array(image.convert("RGB")).reshape(-1, 3).astype(np.float32)

    if n_colors > len(pixels):
        raise ValueError("n_colors must not exceed the number of input pixels")

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)

    n_centers = n_colors * k
    rng = np.random.default_rng()
    cx = rng.integers(0, width, size=n_centers)
    cy = rng.integers(0, height, size=n_centers)
    color_indices = rng.integers(0, n_colors, size=n_centers)

    ys, xs = np.mgrid[0:height, 0:width]
    dx = xs[:, :, None] - cx[None, None, :]
    dy = ys[:, :, None] - cy[None, None, :]
    nearest = np.argmin(dx**2 + dy**2, axis=2)

    return Image.fromarray(colors[color_indices[nearest]])


def turn_gray(
    image_path: str | os.PathLike,
    brightness_factor: float = 1.15,
) -> Image.Image:
    """读取图像并返回保留原始透明度的灰度 RGBA 图像。

    Args:
        image_path: 输入图像路径。
        brightness_factor: 灰度亮度倍数，默认 1.15，与 turn_gray.py 一致。

    Returns:
        PIL.Image.Image，尺寸与输入相同，可调用 .save(path) 保存为 PNG。
        RGB 三通道均为灰度值，alpha 通道保持不变；原图 RGB 三通道
        均大于 240 的像素恢复为纯白，不进行颜色反相。
    """
    with Image.open(image_path) as image:
        rgba = image.convert("RGBA")

    r, g, b, a = rgba.split()
    gray = Image.merge("RGB", (r, g, b)).convert("L")
    gray_np = np.array(gray, dtype=np.float32)
    gray_np = np.clip(gray_np * brightness_factor, 0, 255).astype(np.uint8)

    r_np = np.array(r, dtype=np.uint8)
    g_np = np.array(g, dtype=np.uint8)
    b_np = np.array(b, dtype=np.uint8)
    white_mask = (r_np > 240) & (g_np > 240) & (b_np > 240)
    gray_np[white_mask] = 255

    gray_fixed = Image.fromarray(gray_np)
    return Image.merge("LA", (gray_fixed, a)).convert("RGBA")
