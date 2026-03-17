#!/usr/bin/env python3
"""
图中四种模糊指标（传统数学方法）的纯 NumPy 实现。
用于对 CIFAR-10-C 等图像做模糊/清晰度评估与对比。

趋势说明（四种指标均为「越模糊越小」）：
- 仅对「同一场景、模糊加重」成立；跨污染类型比较时不一定。
- 噪声类（gaussian_noise 等）会引入高频，可能抬高 Laplacian/Tenengrad 等，故噪声图指标偏大≠更清晰。
- 表格中的主要陷阱：平坦场景天然小、强噪声抬高、阈值需校准、缩放敏感、纹理依赖等，都会导致「越模糊越小」在部分数据上不明显。
"""

import numpy as np


def rgb_to_luminance(img):
    """RGB -> 亮度 Y (单通道)，支持 (H,W,3) 或 (C,H,W)。"""
    if img.ndim == 3:
        if img.shape[-1] == 3:
            # (H, W, 3)
            return np.dot(img.astype(np.float32), [0.299, 0.587, 0.114])
        else:
            # (C, H, W)
            return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).astype(np.float32)
    return img.astype(np.float32)


# --------------- 1. Laplacian 方差 B_lap = Var(∇²Y) ---------------
def laplacian_variance(Y):
    """
    拉普拉斯方差：越模糊越小。
    定义: B_lap = Var(∇²Y)，离散 Laplacian 为 4 邻域形式。
    Y: (H,W) 亮度图，float。
    """
    # 离散 Laplacian: ∇²Y ≈ Y(i-1,j)+Y(i+1,j)+Y(i,j-1)+Y(i,j+1) - 4*Y(i,j)
    lap = np.zeros_like(Y, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        -4 * Y[1:-1, 1:-1].astype(np.float64)
        + Y[0:-2, 1:-1].astype(np.float64)
        + Y[2:, 1:-1].astype(np.float64)
        + Y[1:-1, 0:-2].astype(np.float64)
        + Y[1:-1, 2:].astype(np.float64)
    )
    # 只对有效区域（内部）求方差，避免边界 0 稀释结果
    return np.var(lap[1:-1, 1:-1])


# --------------- 2. Tenengrad (Sobel 梯度能量) ---------------
# 定义: T = Σ(G_x² + G_y²)，求和仅对满足 √(G_x² + G_y²) > η 的像素；η 可用百分位数给定
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _convolve2d(Y, k):
    """2D 卷积，边界 0 填充，用 stride 一次算整图。"""
    H, W = Y.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    Yp = np.pad(Y, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    # 所有 (kh,kw) 窗口
    s0, s1 = Yp.strides
    win = np.lib.stride_tricks.as_strided(
        Yp, shape=(H, W, kh, kw),
        strides=(s0, s1, s0, s1), writeable=False
    )
    return np.tensordot(win, k, axes=((2, 3), (0, 1)))


def tenengrad(Y, eta_percentile=50.0):
    """
    Tenengrad：越模糊越小。
    eta: 梯度幅值阈值，可用百分位数（0~100）指定，只统计幅值 > eta 的像素。
    """
    Gx = _convolve2d(Y, _SOBEL_X)
    Gy = _convolve2d(Y, _SOBEL_Y)
    mag_sq = Gx.astype(np.float64) ** 2 + Gy.astype(np.float64) ** 2
    mag = np.sqrt(mag_sq)
    eta = np.percentile(mag, eta_percentile)
    mask = mag > eta
    if np.sum(mask) == 0:
        return 0.0
    return np.sum(mag_sq[mask])


# --------------- 3. Brenner 梯度 ---------------
# 定义: Σ (Y(x+2,y) - Y(x,y))²（水平）；实现中同时加垂直方向以增强稳定性
def brenner_gradient(Y):
    """
    Brenner 梯度：越模糊越小。
    水平: (Y(x+2,y)-Y(x,y))² 与 垂直: (Y(x,y+2)-Y(x,y))² 之和。
    """
    Y = Y.astype(np.float64)
    dx = (Y[2:, :] - Y[:-2, :]) ** 2   # 水平步长 2
    dy = (Y[:, 2:] - Y[:, :-2]) ** 2  # 垂直步长 2
    return np.sum(dx) + np.sum(dy)


# --------------- 4. 小波高频子带能量 (Haar 一层) ---------------
# 定义: HH、LH、HL 子带能量和（频域高频），越模糊越小
def wavelet_high_freq_energy(Y):
    """
    小波一层 Haar 的 HH、LH、HL 高频子带能量和：越模糊越小。
    若 H/W 为奇则截断到偶数尺寸，保证 0::2 与 1::2 等长。
    """
    Y = Y.astype(np.float64)
    H, W = Y.shape
    if H % 2 != 0:
        Y = Y[:-1, :]
        H -= 1
    if W % 2 != 0:
        Y = Y[:, :-1]
        W -= 1
    # 行方向：低 = (偶+奇)/2，高 = (偶-奇)/2
    L = (Y[:, 0::2] + Y[:, 1::2]) * 0.5
    H_row = (Y[:, 0::2] - Y[:, 1::2]) * 0.5
    # 列方向得 LL, LH, HL, HH
    LH = (L[0::2, :] - L[1::2, :]) * 0.5
    HL = (H_row[0::2, :] + H_row[1::2, :]) * 0.5
    HH = (H_row[0::2, :] - H_row[1::2, :]) * 0.5
    return np.sum(LH ** 2) + np.sum(HL ** 2) + np.sum(HH ** 2)


# --------------- 对单张图像统一接口 ---------------
def compute_all_metrics(img, tenengrad_eta_percentile=50.0):
    """
    对单张图像计算四种指标。
    img: (H,W,3) 或 (3,H,W)，uint8 或 float。
    返回: dict { 'laplacian_var', 'tenengrad', 'brenner', 'wavelet_hf' }
    """
    Y = rgb_to_luminance(img)
    return {
        'laplacian_var': laplacian_variance(Y),
        'tenengrad': tenengrad(Y, eta_percentile=tenengrad_eta_percentile),
        'brenner': brenner_gradient(Y),
        'wavelet_hf': wavelet_high_freq_energy(Y),
    }


def compute_batch_metrics(images, tenengrad_eta_percentile=50.0):
    """
    images: (N, C, H, W) 或 (N, H, W, C)。
    返回: (N, 4) 的数组，列顺序 [laplacian_var, tenengrad, brenner, wavelet_hf]。
    """
    N = images.shape[0]
    out = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        m = compute_all_metrics(images[i], tenengrad_eta_percentile=tenengrad_eta_percentile)
        out[i, 0] = m['laplacian_var']
        out[i, 1] = m['tenengrad']
        out[i, 2] = m['brenner']
        out[i, 3] = m['wavelet_hf']
    return out
