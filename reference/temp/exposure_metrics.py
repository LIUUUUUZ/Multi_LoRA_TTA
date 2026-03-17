#!/usr/bin/env python3
"""
过曝 / 欠曝 / 动态范围异常（Exposure / Clipping）相关指标实现。

根据表中的指标，针对单张图像计算：
- hi_clip_ratio       : 高光饱和比例 r_hi = 1/N Σ 1(Y >= 1-ε_hi)
- lo_clip_ratio       : 暗部饱和比例 r_lo = 1/N Σ 1(Y <= ε_lo)
- mean_luma           : 亮度均值 μ_Y
- median_luma         : 亮度中位数 median(Y)
- hist_skewness       : 亮度直方图偏度
- hist_kurtosis       : 亮度直方图峰度
- entropy_norm        : 归一化熵 H(Y)/log(K)，0~1
- dynamic_range       : 有效动态范围 DR = P_99(Y) - P_1(Y)
- rgb_hi_clip_ratio   : RGB 通道高光饱和比例（3 通道平均）

约定：
- 输入 img 为 (H,W,3) 或 (3,H,W)，uint8 或 float；内部统一转为 [0,1]。
"""

import numpy as np

from blur_metrics import rgb_to_luminance


def _to_luma_and_rgb01(img):
    """返回 (Y, rgb01)，都在 [0,1]。rgb01 shape 为 (H,W,3)。"""
    if img.ndim == 3 and img.shape[-1] == 3:
        rgb = img.astype(np.float32)
    elif img.ndim == 3 and img.shape[0] == 3:
        rgb = np.transpose(img, (1, 2, 0)).astype(np.float32)
    else:
        # 单通道，当成 Y
        Y = img.astype(np.float32)
        if Y.max() > 1.0:
            Y = Y / 255.0
        return Y, None

    if rgb.max() > 1.0:
        rgb01 = rgb / 255.0
    else:
        rgb01 = rgb
    Y = rgb_to_luminance(rgb01)  # 已在 [0,1] 上
    return Y, rgb01


def hi_lo_clip_ratio(Y, eps_hi=0.02, eps_lo=0.02):
    """高光/暗部饱和比例（相对于亮度 Y∈[0,1]）。"""
    N = Y.size
    hi_thr = 1.0 - eps_hi
    lo_thr = eps_lo
    hi = float(np.sum(Y >= hi_thr)) / N
    lo = float(np.sum(Y <= lo_thr)) / N
    return hi, lo


def brightness_stats(Y):
    """亮度均值与中位数。"""
    return float(Y.mean()), float(np.median(Y))


def histogram_stats(Y, num_bins=256):
    """
    直方图统计：偏度、峰度、熵（归一化）。
    """
    Y_flat = Y.ravel().astype(np.float64)
    hist, bin_edges = np.histogram(Y_flat, bins=num_bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    p = hist.astype(np.float64) / total
    # 熵（归一化到 0~1）
    eps = 1e-12
    H = -np.sum(p * np.log(p + eps))
    H_norm = H / np.log(num_bins)

    # 偏度、峰度：基于像素值本身
    mu = Y_flat.mean()
    var = Y_flat.var()
    if var <= 0:
        return 0.0, 0.0, float(H_norm)
    std = np.sqrt(var)
    z = (Y_flat - mu) / std
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    return skew, kurt, float(H_norm)


def dynamic_range(Y, p_low=1.0, p_high=99.0):
    """有效动态范围：DR = P_high - P_low（百分位），基于 Y∈[0,1]。"""
    lo = float(np.percentile(Y, p_low))
    hi = float(np.percentile(Y, p_high))
    return max(0.0, hi - lo)


def rgb_hi_clip_ratio(rgb01, eps_hi=0.02):
    """
    RGB 通道高光饱和比例：3 通道分别 > 1-eps_hi 的比例再平均。
    反映“某一颜色通道剪切”的严重程度。
    """
    if rgb01 is None:
        return 0.0
    H, W, C = rgb01.shape
    thr = 1.0 - eps_hi
    ratios = []
    for c in range(C):
        ch = rgb01[:, :, c]
        ratios.append(float(np.sum(ch >= thr)) / (H * W))
    return float(np.mean(ratios))


def compute_exposure_metrics(img):
    """
    对单张图像计算曝光/剪切相关指标，返回 dict。
    """
    Y, rgb01 = _to_luma_and_rgb01(img)
    hi, lo = hi_lo_clip_ratio(Y)
    mean_y, med_y = brightness_stats(Y)
    skew, kurt, H_norm = histogram_stats(Y)
    dr = dynamic_range(Y)
    rgb_hi = rgb_hi_clip_ratio(rgb01)
    return {
        "hi_clip_ratio": hi,
        "lo_clip_ratio": lo,
        "mean_luma": mean_y,
        "median_luma": med_y,
        "hist_skewness": skew,
        "hist_kurtosis": kurt,
        "entropy_norm": H_norm,
        "dynamic_range": dr,
        "rgb_hi_clip_ratio": rgb_hi,
    }


def compute_exposure_batch_metrics(images):
    """
    images: (N, C, H, W) 或 (N, H, W, C)。
    返回 (N, 9) 数组，对应：
      [hi_clip_ratio, lo_clip_ratio, mean_luma, median_luma,
       hist_skewness, hist_kurtosis, entropy_norm, dynamic_range,
       rgb_hi_clip_ratio]
    """
    N = images.shape[0]
    out = np.zeros((N, 9), dtype=np.float64)
    for i in range(N):
        m = compute_exposure_metrics(images[i])
        out[i, 0] = m["hi_clip_ratio"]
        out[i, 1] = m["lo_clip_ratio"]
        out[i, 2] = m["mean_luma"]
        out[i, 3] = m["median_luma"]
        out[i, 4] = m["hist_skewness"]
        out[i, 5] = m["hist_kurtosis"]
        out[i, 6] = m["entropy_norm"]
        out[i, 7] = m["dynamic_range"]
        out[i, 8] = m["rgb_hi_clip_ratio"]
    return out


if __name__ == "__main__":
    img = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
    print(compute_exposure_metrics(img))

