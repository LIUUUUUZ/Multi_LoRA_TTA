#!/usr/bin/env python3
"""
噪声（Additive / Sensor Noise & Low-light Noise）相关传统指标实现。

对应表中 4 类方法：
1) 高通残差 MAD 估计噪声 σ
2) 平坦块方差法
3) 频谱高频“白噪声抬升”
4) 暗部噪声指标（只在低亮度区域估计 σ）

输入 img: (H,W,3) 或 (3,H,W)，uint8 或 float。
内部统一转换到亮度 Y∈[0,1]。
"""

import numpy as np

from blur_metrics import rgb_to_luminance, _convolve2d


def _to_luma01(img):
    """将输入转为亮度 Y∈[0,1]。"""
    if img.ndim == 3 and img.shape[-1] == 3:
        rgb = img.astype(np.float32)
    elif img.ndim == 3 and img.shape[0] == 3:
        rgb = np.transpose(img, (1, 2, 0)).astype(np.float32)
    else:
        Y = img.astype(np.float32)
        if Y.max() > 1.0:
            Y = Y / 255.0
        return Y
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return rgb_to_luminance(rgb)


def _mean_blur(Y):
    """简单 3×3 均值滤波，作为 blur(Y)。"""
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    return _convolve2d(Y.astype(np.float32), k)


def mad_sigma(Y):
    """
    高通残差 MAD 估计噪声 σ：
      R = Y - blur(Y)
      σ_hat = median(|R|) / 0.6745
    """
    Y = Y.astype(np.float32)
    blur = _mean_blur(Y)
    R = Y - blur
    sigma_hat = np.median(np.abs(R)) / 0.6745
    return float(sigma_hat)


def flat_patch_sigma(Y, patch_size=7):
    """
    平坦块方差法：寻找局部方差最小的一块 patch，用其方差估计噪声。
    返回估计的 σ（标准差，而不是方差）。
    """
    Y = Y.astype(np.float32)
    H, W = Y.shape
    k = np.ones((patch_size, patch_size), dtype=np.float32) / (patch_size * patch_size)
    # E[X], E[X^2]
    mean = _convolve2d(Y, k)
    mean_sq = _convolve2d(Y ** 2, k)
    var = np.clip(mean_sq - mean ** 2, 0.0, None)
    # 仅在能放下完整 patch 的内部区域取最小方差
    pad = patch_size // 2
    inner = var[pad : H - pad, pad : W - pad] if H > 2 * pad and W > 2 * pad else var
    sigma_hat = float(np.sqrt(np.min(inner)))
    return sigma_hat


def spectrum_highfreq_lift(Y, r_low=0.1, r_high=0.5):
    """
    频谱高频“白噪声抬升”：
      计算频谱中高频环带 (r_low~r_high) 的平均能量，
      再除以全频能量，得到高频能量占比。
      噪声越大，高频能量整体抬升且更接近各向同性，该比例越高。
    """
    Y = Y.astype(np.float64)
    H, W = Y.shape
    F = np.fft.fft2(Y)
    F = np.fft.fftshift(F)
    P = np.abs(F) ** 2
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.indices((H, W))
    dy = y_idx - cy
    dx = x_idx - cx
    r = np.sqrt(dx ** 2 + dy ** 2)
    r_norm = r / (max(H, W) / 2.0 + 1e-6)
    mask_hf = (r_norm >= r_low) & (r_norm <= r_high)
    hf_energy = P[mask_hf].sum()
    total_energy = P.sum() + 1e-12
    return float(hf_energy / total_energy)


def dark_region_sigma(Y, dark_thresh=0.2):
    """
    暗部噪声指标：仅在 Y < dark_thresh 的区域上，用 MAD 残差估计 σ。
    在低照度区域对噪声更敏感。
    """
    Y = Y.astype(np.float32)
    mask = Y < dark_thresh
    if not np.any(mask):
        return 0.0
    blur = _mean_blur(Y)
    R = (Y - blur)[mask]
    sigma_hat = np.median(np.abs(R)) / 0.6745
    return float(sigma_hat)


def compute_noise_metrics(img):
    """
    对单张图像计算 4 个噪声指标，返回 dict：
      {
        'mad_sigma',           # 全局 MAD 噪声
        'flat_patch_sigma',    # 平坦块噪声估计
        'hf_lift_ratio',       # 频谱高频能量占比
        'dark_region_sigma',   # 暗部区域噪声
      }
    """
    Y = _to_luma01(img)
    return {
        "mad_sigma": mad_sigma(Y),
        "flat_patch_sigma": flat_patch_sigma(Y),
        "hf_lift_ratio": spectrum_highfreq_lift(Y),
        "dark_region_sigma": dark_region_sigma(Y),
    }


def compute_noise_batch_metrics(images):
    """
    images: (N, C, H, W) 或 (N, H, W, C)。
    返回 (N, 4) 数组，列顺序：
      [mad_sigma, flat_patch_sigma, hf_lift_ratio, dark_region_sigma]
    """
    N = images.shape[0]
    out = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        m = compute_noise_metrics(images[i])
        out[i, 0] = m["mad_sigma"]
        out[i, 1] = m["flat_patch_sigma"]
        out[i, 2] = m["hf_lift_ratio"]
        out[i, 3] = m["dark_region_sigma"]
    return out


if __name__ == "__main__":
    img = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
    print(compute_noise_metrics(img))

