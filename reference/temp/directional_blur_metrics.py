#!/usr/bin/env python3
"""
运动模糊 / 方向性模糊的传统数学指标（基于前面四个通用 blur 指标之后的第二层）。

实现 3 类方向性指标，对应图中：
1) 梯度方向熵 / 方向性
2) 频谱方向能量集中度
3) 边缘方向性（Radon/Hough 的简化版：基于边缘像素的方向直方图各向异性）

约定：
- 对单张图像输入 (H,W,3) 或 (3,H,W)，uint8/float 均可。
- 输出 3 个标量：
    grad_dir_anisotropy   : 梯度方向各向异性，越大越“单一方向”；
    fft_dir_concentration : 频谱方向能量集中度，越大越“各向异性”；
    edge_dir_anisotropy   : 仅基于边缘像素的方向各向异性，越大越“单一方向”。

这些指标本身不是“好/坏”的绝对度量，而是用来区分
    - defocus blur（各向同性模糊）
    - motion blur（强方向性模糊）
等。
"""

import numpy as np

from blur_metrics import rgb_to_luminance, _convolve2d, _SOBEL_X, _SOBEL_Y


def _gaussian_kernel_2d(sigma, truncate=3.0):
    """sigma 对应的高斯核，用于平滑。"""
    r = int(np.ceil(truncate * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g1d = np.exp(-0.5 * (x / sigma) ** 2)
    g1d /= g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]
    return g2d.astype(np.float32)


def _orientation_hist_from_grad(Y, num_bins=36, use_edges=False, mag_percentile=50.0):
    """
    基于 Sobel 梯度构建方向直方图。

    - num_bins: 方向 bin 数，覆盖 [0, π)。
    - use_edges: 若为 True，仅使用梯度幅值大于 mag_percentile 分位数的“边缘像素”。
    - mag_percentile: 边缘选取阈值。

    返回:
        hist: shape (num_bins,) ，已归一化为概率分布（和为 1），若无有效像素则返回均匀分布。
    """
    Gx = _convolve2d(Y, _SOBEL_X)
    Gy = _convolve2d(Y, _SOBEL_Y)
    mag = np.sqrt(Gx.astype(np.float64) ** 2 + Gy.astype(np.float64) ** 2)

    if use_edges:
        thr = np.percentile(mag, mag_percentile)
        mask = mag > thr
    else:
        mask = mag > 0

    if not np.any(mask):
        return np.ones(num_bins, dtype=np.float64) / num_bins

    ang = np.arctan2(Gy[mask], Gx[mask])  # [-pi, pi]
    # 方向模 π：梯度方向和反方向视为同一方向
    ang = np.mod(ang, np.pi)
    bins = np.linspace(0.0, np.pi, num_bins + 1, endpoint=True)
    hist, _ = np.histogram(ang, bins=bins, weights=mag[mask])
    if hist.sum() <= 0:
        return np.ones(num_bins, dtype=np.float64) / num_bins
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    return hist


def grad_direction_anisotropy(Y, num_bins=36, mag_pct=50.0, smooth_sigma=1.0):
    """
    梯度方向熵 → 各向异性得分（与 gradient_direction_entropy_score 一致逻辑，返回 1 - 熵）。

    步骤：高斯预平滑 → Sobel → 仅保留幅值 > mag_pct 分位的强边缘 → 对方向 [0°, 180°) 建直方图（按像素个数，不加权）
    → 归一化熵 H/log(n_bins) → 返回 1 - H_norm（值越高 = 方向越集中，运动模糊候选）。
    强梯度像素 < 10 时返回 0（无方向性）。
    """
    Y = Y.astype(np.float64)
    if smooth_sigma > 0:
        kernel = _gaussian_kernel_2d(smooth_sigma)
        Y = _convolve2d(Y.astype(np.float32), kernel).astype(np.float64)
    Gx = _convolve2d(Y.astype(np.float32), _SOBEL_X)
    Gy = _convolve2d(Y.astype(np.float32), _SOBEL_Y)
    mag = np.hypot(Gx.astype(np.float64), Gy.astype(np.float64))

    thr = np.percentile(mag, mag_pct)
    mask = mag > thr
    if np.sum(mask) < 10:
        return 0.0  # 几乎无梯度，退化为无方向性（各向异性=0）

    angle = np.rad2deg(np.arctan2(Gy[mask], Gx[mask])) % 180.0  # [0°, 180°)
    hist, _ = np.histogram(angle, bins=num_bins, range=(0.0, 180.0))  # 按个数，不加权
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    if p.size == 0:
        return 0.0
    entropy = -float(np.sum(p * np.log(p)))
    H_norm = entropy / np.log(num_bins)  # [0, 1]
    return float(1.0 - H_norm)  # 各向异性：越高越单方向


def gradient_direction_entropy_score(
    img_np, n_bins=36, mag_pct=50.0, smooth_sigma=1.0
):
    """
    梯度方向熵：通过梯度方向分布计算表示方向均匀性的分数。

    实现步骤：
    1) 高斯预平滑（抑制噪声引入的随机方向）
    2) Sobel 计算 Gx / Gy
    3) 仅保留幅值高于 mag_pct 百分位的“强边缘”像素（排除低幅值噪声点）
    4) 对保留像素的梯度方向 θ = arctan2(Gy, Gx) % 180° 建直方图，映射到 [0°, 180°)，无符号方向
    5) 计算归一化熵 H / log(n_bins)，值域 [0, 1]

    注意：场景自身有强方向结构（建筑/栅栏/斑马线）时熵也会偏低，建议与 local_contrast 联合判断：
    - 低熵 + 低对比度 → 运动模糊
    - 低熵 + 高对比度 → 场景自身方向性强（非模糊）

    参数:
        img_np: 输入图像，(H,W,C) 或 (C,H,W)，uint8 或 float，像素值范围无要求（内部转灰度后仅用相对关系）
        n_bins: 直方图区间数，默认 36（每区间 5°）
        mag_pct: 强边缘筛选的梯度幅值百分位，默认 50（保留上半区）
        smooth_sigma: 高斯预平滑 sigma；0 表示不平滑

    返回:
        score: 梯度方向熵（归一化），值越高=方向越均匀；值越低=方向越集中（运动模糊或强方向结构）。
        强梯度像素 < 10 时返回 1.0。
    """
    if img_np.ndim == 3 and img_np.shape[-1] == 3:
        Y = rgb_to_luminance(img_np)
    elif img_np.ndim == 3 and img_np.shape[0] == 3:
        Y = rgb_to_luminance(img_np)
    else:
        Y = np.asarray(img_np, dtype=np.float64)
    Y = Y.astype(np.float64)
    if smooth_sigma > 0:
        kernel = _gaussian_kernel_2d(smooth_sigma)
        Y = _convolve2d(Y.astype(np.float32), kernel).astype(np.float64)
    Gx = _convolve2d(Y.astype(np.float32), _SOBEL_X)
    Gy = _convolve2d(Y.astype(np.float32), _SOBEL_Y)
    mag = np.hypot(Gx.astype(np.float64), Gy.astype(np.float64))
    thr = np.percentile(mag, mag_pct)
    mask = mag > thr
    if np.sum(mask) < 10:
        return 1.0
    angle = np.rad2deg(np.arctan2(Gy[mask], Gx[mask])) % 180.0
    hist, _ = np.histogram(angle, bins=n_bins, range=(0.0, 180.0))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    if p.size == 0:
        return 1.0
    entropy = -float(np.sum(p * np.log(p)))
    return float(entropy / np.log(n_bins))


def compute_gradient_direction_entropy_batch(images, n_bins=36, mag_pct=50.0, smooth_sigma=1.0):
    """
    对一批图像计算梯度方向熵，返回 (N,) 的 score 数组。
    images: (N, C, H, W) 或 (N, H, W, C)
    """
    N = images.shape[0]
    out = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        out[i] = gradient_direction_entropy_score(
            images[i], n_bins=n_bins, mag_pct=mag_pct, smooth_sigma=smooth_sigma
        )
    return out


def edge_direction_anisotropy(Y, num_bins=18, mag_percentile=70.0):
    """
    边缘方向各向异性（Radon/Hough on edges 的简化版）。
    仅统计梯度幅值高于给定分位数的像素，其他与 grad_direction_anisotropy 相同。
    """
    hist = _orientation_hist_from_grad(Y, num_bins=num_bins, use_edges=True, mag_percentile=mag_percentile)
    eps = 1e-12
    H = -np.sum(hist * np.log(hist + eps))
    H_norm = H / np.log(len(hist))
    return float(1.0 - H_norm)


def fft_direction_concentration(Y, num_bins=18):
    """
    频谱方向能量集中度（越大越“各向异性”）。

    步骤：
      1) 对 Y 做 2D FFT，取幅值谱。
      2) 忽略 DC 和极低频（中心小圆盘）。
      3) 按频率方向 angle ∈ [0, π) 做加权直方图（权重=谱幅值）。
      4) 集中度定义为 max(p_k)，0~1：
         - 各向同性（能量在所有方向上近似均匀） → 接近 1/K；
         - 强方向性模糊 → 某一方向能量远大于其它方向 → 接近 1。
    """
    Y = Y.astype(np.float64)
    H, W = Y.shape
    F = np.fft.fft2(Y)
    F = np.fft.fftshift(F)
    mag = np.abs(F)

    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.indices((H, W))
    dy = y_idx - cy
    dx = x_idx - cx
    r = np.sqrt(dx ** 2 + dy ** 2)
    # 忽略中心低频小圆盘（例如 r <= 2 像素）
    mask = r > 2
    if not np.any(mask):
        return 0.0

    ang = np.arctan2(dy[mask], dx[mask])
    ang = np.mod(ang, np.pi)  # [0, π)
    weights = mag[mask]

    bins = np.linspace(0.0, np.pi, num_bins + 1, endpoint=True)
    hist, _ = np.histogram(ang, bins=bins, weights=weights)
    if hist.sum() <= 0:
        return 0.0
    hist = hist.astype(np.float64)
    hist /= hist.sum()
    return float(hist.max())


def compute_directional_metrics(img):
    """
    对单张图像计算 3 个方向性指标。
    返回 dict:
      {
        'grad_dir_anisotropy': ...,
        'fft_dir_concentration': ...,
        'edge_dir_anisotropy': ...,
      }
    """
    if img.ndim == 3 and img.shape[-1] == 3:
        Y = rgb_to_luminance(img)
    elif img.ndim == 3 and img.shape[0] == 3:
        # (C,H,W)
        Y = rgb_to_luminance(img)
    else:
        Y = img.astype(np.float32)
    return {
        'grad_dir_anisotropy': grad_direction_anisotropy(Y),
        'fft_dir_concentration': fft_direction_concentration(Y),
        'edge_dir_anisotropy': edge_direction_anisotropy(Y),
    }


def compute_directional_batch_metrics(images):
    """
    images: (N, C, H, W) 或 (N, H, W, C)。
    返回: (N, 3) 的数组，列顺序：
      [grad_dir_anisotropy, fft_dir_concentration, edge_dir_anisotropy]
    """
    N = images.shape[0]
    out = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        m = compute_directional_metrics(images[i])
        out[i, 0] = m['grad_dir_anisotropy']
        out[i, 1] = m['fft_dir_concentration']
        out[i, 2] = m['edge_dir_anisotropy']
    return out


def compute_grad_anisotropy_only(images):
    """仅计算 grad_direction_anisotropy，返回 (N,) 数组。用于单独跑该指标。"""
    N = images.shape[0]
    out = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        if images[i].ndim == 3 and images[i].shape[0] == 3:
            Y = rgb_to_luminance(images[i])
        else:
            Y = rgb_to_luminance(images[i])
        out[i] = grad_direction_anisotropy(Y, num_bins=36, mag_pct=50.0, smooth_sigma=1.0)
    return out


if __name__ == '__main__':
    # 简单自测
    img = np.random.randint(0, 256, (3, 32, 32), dtype=np.uint8)
    print(compute_directional_metrics(img))

