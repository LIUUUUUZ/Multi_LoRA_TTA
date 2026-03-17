import os
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.font_manager as fm
import colorsys
import scipy.ndimage as ndimage
from scipy.ndimage import minimum_filter, uniform_filter, convolve, gaussian_filter
import torch
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

# ── 中文字体配置 ──────────────────────────────────────────────────────────────
# 按优先级依次尝试常见中文字体，找到第一个可用的即用
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei",      # 微软雅黑（Windows）
    "SimHei",               # 黑体（Windows）
    "PingFang SC",          # 苹方（macOS）
    "WenQuanYi Micro Hei",  # 文泉驿微米黑（Linux）
    "Noto Sans CJK SC",     # Noto（Linux/跨平台）
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
_cn_font = next((f for f in _CN_FONT_CANDIDATES if f in _available_fonts), None)
if _cn_font:
    plt.rcParams["font.family"]        = _cn_font
    plt.rcParams["axes.unicode_minus"] = False   # 防止负号显示为方块

# ── 常量 ──────────────────────────────────────────────────────────────────────
# 15 种标准污染类型，顺序与 conf.py 中的 CORRUPTION_LIST 一致
CORRUPTION_LIST = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

DATA_ROOT = "./dataset/CIFAR-10-C"   # CIFAR-10-C 数据集根目录


# ══════════════════════════════════════════════════════════════════════════════
# 数据集类
# ══════════════════════════════════════════════════════════════════════════════

class CIFAR10C_ByCorrType(Dataset):
    """
    加载单个 (污染类型, severity) 组合的数据。

    每个样本返回四元组:
        (img_tensor, corr_idx, corr_name, severity)
        - img_tensor : (3, 32, 32) float32，已归一化到 [0, 1]
        - corr_idx   : int，污染种类在 CORRUPTION_LIST 中的索引（0-14），用作 label
        - corr_name  : str，污染种类名称
        - severity   : int，污染强度等级（1-5）
    """

    def __init__(self, corr_name: str, severity: int):
        assert corr_name in CORRUPTION_LIST, f"未知污染类型: {corr_name}"
        assert 1 <= severity <= 5, "severity 必须在 1~5 之间"

        self.corr_name = corr_name
        self.corr_idx  = CORRUPTION_LIST.index(corr_name)  # 污染种类索引，作为分类 label
        self.severity  = severity

        # 从磁盘加载 .npy 文件，格式为 (N, H, W, C) uint8
        path = os.path.join(DATA_ROOT, "corrupted", f"severity-{severity}")
        self.images = np.load(os.path.join(path, f"{corr_name}.npy"))
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_nhwc = self.images[idx]              # (H, W, C) uint8
        # NHWC → NCHW，uint8 → float32，归一化到 [0, 1]
        img_tensor = torch.from_numpy(
            img_nhwc.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        return img_tensor, self.corr_idx, self.corr_name, self.severity


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载与展示
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset(severities=(1, 2, 3, 4, 5), corruptions=CORRUPTION_LIST):
    """
    加载 CIFAR-10-C 中指定污染类型 × 指定 severity 等级的全部数据。

    参数
    ----
    severities  : 要加载的 severity 等级列表，默认 (1,2,3,4,5)
    corruptions : 要加载的污染类型列表，默认全部 15 种

    返回
    ----
    dataset : ConcatDataset
        将所有 (污染, severity) 子集合并，样本格式见 CIFAR10C_ByCorrType。
    """
    subsets = []
    for sev in severities:
        for corr in corruptions:
            subsets.append(CIFAR10C_ByCorrType(corr_name=corr, severity=sev))
            print(f"  已加载: {corr}  severity-{sev}  ({len(subsets[-1])} 张)")

    dataset = ConcatDataset(subsets)
    print(f"\n共加载 {len(dataset)} 张图片，"
          f"{len(corruptions)} 种污染 × {len(severities)} 个 severity 等级\n")
    return dataset


def display(dataset, n=5, seed=None):
    """
    从 dataset 中随机抽取 n 张图片并并排展示。
    图片标题显示污染种类索引、名称及 severity 等级。

    参数
    ----
    dataset : 由 get_dataset() 返回的 ConcatDataset
    n       : 展示张数，默认 5
    seed    : 随机种子，传入可复现抽样结果
    """
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(len(dataset)), k=n)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        img_tensor, corr_idx, corr_name, severity = dataset[idx]

        # (C, H, W) → (H, W, C)，供 matplotlib 显示
        img_np = np.clip(img_tensor.numpy().transpose(1, 2, 0), 0.0, 1.0)
        ax.imshow(img_np)
        ax.set_title(
            f"[{corr_idx}] {corr_name}\nseverity={severity}",
            fontsize=8, wrap=True,
        )
        ax.axis("off")

    plt.suptitle("CIFAR-10-C  (label = corruption type)", fontsize=10)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 方法一：暗通道先验得分
# 公式：D(x) = min_{c∈{r,g,b}} min_{y∈Ω(x)} I_c(y)
# 物理含义：有雾时暗通道值升高；夜景/低照度也可能抬升
# ══════════════════════════════════════════════════════════════════════════════

def dark_channel_score(img_np: np.ndarray, omega: int = 5,
                       stat: str = 'mean') -> float:
    """
    计算单张图片的暗通道得分（标量）。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    omega  : 局部窗口半径，窗口大小 = (2*omega+1)²
    stat   : 汇总统计量，可选 'mean' | 'median' | 'q75' | 'q90'
             有雾/雪/亮度污染时均值及高分位数显著升高

    返回
    ----
    score : float
        值越大 → 暗通道越抬升（越可能存在雾/高亮污染）
    """
    def _dark_channel(img: np.ndarray, w: int) -> np.ndarray:
        # 第一步：逐像素取三通道最小值 → (H, W)
        min_c = img.min(axis=2)
        # 第二步：在 Ω(x) 邻域内再取最小值（等价于 min-pooling）
        return minimum_filter(min_c, size=2 * w + 1, mode='reflect')

    dark = _dark_channel(img_np, omega)
    if stat == 'mean':
        return float(dark.mean())
    elif stat == 'median':
        return float(np.median(dark))
    elif stat == 'q75':
        return float(np.percentile(dark, 75))
    elif stat == 'q90':
        return float(np.percentile(dark, 90))
    else:
        raise ValueError(f"未知 stat='{stat}'，可选: mean / median / q75 / q90")


# ══════════════════════════════════════════════════════════════════════════════
# 方法二：HSV 饱和度得分
# 公式：S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
# 物理含义：有雾/雪时颜色褪色，饱和度下降；纯灰色场景本身饱和度低会误判
# ══════════════════════════════════════════════════════════════════════════════

def hsv_saturation_score(img_np: np.ndarray, stat: str = 'mean') -> float:
    """
    计算单张图片的 HSV 饱和度得分（标量）。
    使用纯 numpy 实现，无需 cv2 / skimage 依赖。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    stat   : 汇总统计量，可选 'mean' | 'median' | 'q25' | 'q10'
             'q25'/'q10' 聚焦低饱和区域，对雾化更敏感

    返回
    ----
    score : float
        值越低 → 饱和度越低（越可能存在雾/雪褪色污染）
    """
    def _saturation(img: np.ndarray) -> np.ndarray:
        # S = (max - min) / max；max 为 0 时 S 定义为 0
        cmax = img.max(axis=2)
        cmin = img.min(axis=2)
        return np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0.0).astype(np.float32)

    sat = _saturation(img_np)
    if stat == 'mean':
        return float(sat.mean())
    elif stat == 'median':
        return float(np.median(sat))
    elif stat == 'q25':
        return float(np.percentile(sat, 25))
    elif stat == 'q10':
        return float(np.percentile(sat, 10))
    else:
        raise ValueError(f"未知 stat='{stat}'，可选: mean / median / q25 / q10")


# ══════════════════════════════════════════════════════════════════════════════
# 方法三：局部对比度得分（RMS 局部对比 / Laplacian 能量）
# 物理含义：有雾/失焦时高频被压制，对比度下降；噪声类污染反而升高
# ══════════════════════════════════════════════════════════════════════════════

# ITU-R BT.601 灰度转换权重
_RGB2GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# 4-邻接 Laplacian 卷积核
_LAP_KERNEL = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=np.float32)


def local_contrast_score(img_np: np.ndarray,
                         mode: str  = 'laplacian',
                         omega: int = 2,
                         stat: str  = 'mean') -> float:
    """
    计算单张图片的局部对比度得分（标量）。

    两种模式
    --------
    'laplacian'
        Laplacian 能量：E = mean( (∇²I)² )
        对失焦模糊和雾化均敏感；比 RMS 更受高频细节驱动。
        · 雾/模糊  → 高频被压制 → E 显著下降
        · 噪声污染 → 引入假高频 → E 可能升高（可与其他指标联判）

    'rms'
        局部 RMS 对比：σ_local(x) = sqrt( E[I²]_{Ω(x)} − E[I]²_{Ω(x)} )
        对雾和模糊均下降；窗口大小由 omega 控制。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    mode   : 'laplacian' | 'rms'
    omega  : 仅 'rms' 模式有效，局部窗口半径，窗口大小 = (2*omega+1)²
    stat   : 汇总统计量 'mean' | 'median' | 'q75' | 'q90'
             高分位数对保留局部纹理的污染（如噪声）更敏感

    返回
    ----
    score : float
        值越低 → 对比度越弱（雾/模糊特征越强）
        值越高 → 高频越丰富（噪声类污染特征）
    """
    gray = img_np @ _RGB2GRAY   # RGB → 灰度 (H, W)

    if mode == 'laplacian':
        lap = convolve(gray, _LAP_KERNEL, mode='reflect')
        energy_map = lap ** 2                                    # 逐像素 Laplacian²

    elif mode == 'rms':
        win = 2 * omega + 1
        local_mean    = uniform_filter(gray,      size=win, mode='reflect')
        local_sq_mean = uniform_filter(gray ** 2, size=win, mode='reflect')
        # 浮点误差可能产生微小负方差，clamp 到 0
        local_var  = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
        energy_map = np.sqrt(local_var)                          # 局部标准差图

    else:
        raise ValueError(f"未知 mode='{mode}'，可选: 'laplacian' | 'rms'")

    if stat == 'mean':
        return float(energy_map.mean())
    elif stat == 'median':
        return float(np.median(energy_map))
    elif stat == 'q75':
        return float(np.percentile(energy_map, 75))
    elif stat == 'q90':
        return float(np.percentile(energy_map, 90))
    else:
        raise ValueError(f"未知 stat='{stat}'，可选: mean / median / q75 / q90")


# ══════════════════════════════════════════════════════════════════════════════
# 方法四：边缘可见性 / 边缘密度得分
# 物理含义：有雾/模糊时边缘被抑制，得分下降；噪声类污染在梯度模式下得分升高
# 注意：纹理稀少的场景（如天空、墙面）本身边缘少，可能误判；阈值对 Canny 模式影响大
# ══════════════════════════════════════════════════════════════════════════════

# 归一化 Sobel 核（除以 8，使 [0,1] 图像的响应幅值落在 [0,1]）
_SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32) / 8.0
_SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32) / 8.0


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """计算灰度图 (H, W) 的 Sobel 梯度幅值图。"""
    gx = convolve(gray, _SOBEL_X, mode='reflect')
    gy = convolve(gray, _SOBEL_Y, mode='reflect')
    return np.hypot(gx, gy)


def gradient_direction_entropy_score(img_np: np.ndarray,
                                     n_bins: int    = 36,
                                     mag_pct: float = 50.0,
                                     smooth_sigma: float = 1.0) -> float:
    """
    梯度方向熵得分（标量）。

    原理
    ----
    对图像梯度方向建立直方图，计算归一化香农熵：
    · 均匀分布（各向同性）→ 熵 = 1.0  → 无方向性（无运动模糊）
    · 集中在某一方向    → 熵 → 0.0  → 强方向性（运动模糊候选）

    实现步骤
    --------
    1. 高斯预平滑（抑制噪声引入的随机方向）
    2. Sobel 计算 Gx / Gy
    3. 仅保留幅值高于 mag_pct 百分位的"强边缘"像素
       （排除低幅值噪声点，它们的方向信息可靠性低）
    4. 对保留像素的梯度方向 θ = arctan2(Gy, Gx) % 180° 建直方图
       （映射到 [0°, 180°)，方向对称，无符号方向）
    5. 计算归一化熵 H / log(n_bins)，值域 [0, 1]

    注意
    ----
    场景本身含强单一方向的结构（建筑物、栅栏、斑马线）
    同样会产生低熵，可能被误判为运动模糊。
    建议与 local_contrast / edge_visibility 联合使用：
    · 低熵 + 低对比度 → 运动模糊
    · 低熵 + 高对比度 → 场景自身方向性强（非模糊）

    参数
    ----
    img_np       : (H, W, C) float32，值域 [0, 1]
    n_bins       : 直方图区间数（默认 36，即每 5° 一个 bin）
    mag_pct      : 仅保留幅值高于此百分位的像素（默认 50，即上半区间）
    smooth_sigma : 高斯预平滑 sigma（设为 0 则跳过平滑）

    返回
    ----
    score : float
        值越高 → 方向越均匀（接近各向同性，无运动模糊特征）
        值越低 → 方向越集中（强方向性，运动模糊或场景方向结构明显）
    """
    gray = img_np @ _RGB2GRAY   # (H, W)

    if smooth_sigma > 0:
        gray = gaussian_filter(gray, sigma=smooth_sigma)

    gx  = convolve(gray, _SOBEL_X, mode='reflect')
    gy  = convolve(gray, _SOBEL_Y, mode='reflect')
    mag = np.hypot(gx, gy)

    # 仅取强梯度像素（过滤低幅值噪声点）
    thr  = np.percentile(mag, mag_pct)
    mask = mag > thr
    if mask.sum() < 10:
        return 1.0   # 几乎无梯度（纯色图），退化为最高熵

    angle = np.rad2deg(np.arctan2(gy[mask], gx[mask])) % 180.0  # [0°, 180°)

    hist, _ = np.histogram(angle, bins=n_bins, range=(0.0, 180.0))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()   # 归一化，去除零项（避免 log(0)）

    entropy = -float(np.sum(p * np.log(p)))
    return entropy / np.log(n_bins)   # 归一化到 [0, 1]


def _canny_nms_hysteresis(gray: np.ndarray,
                          sigma: float,
                          low_thr: float,
                          high_thr: float) -> np.ndarray:
    """
    向量化 Canny 边缘检测（不依赖 cv2）。

    流程
    ----
    1. 高斯平滑（抑制噪声）
    2. Sobel 梯度幅值 + 梯度方向
    3. 非极大值抑制（4 方向量化，全向量化无像素级循环）
    4. 双阈值 + 一次性 8-邻接滞后连接

    返回
    ----
    edges : bool 数组 (H, W)，True 表示该像素为边缘点
    """
    # 步骤 1：高斯平滑
    blurred = gaussian_filter(gray, sigma=sigma)

    # 步骤 2：Sobel 梯度
    gx    = convolve(blurred, _SOBEL_X, mode='reflect')
    gy    = convolve(blurred, _SOBEL_Y, mode='reflect')
    mag   = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180   # 映射到 [0°, 180°)

    # 步骤 3：向量化非极大值抑制
    # 将梯度方向量化为 0°/45°/90°/135° 四个方向
    mp     = np.pad(mag, 1, mode='edge')             # 边界填充，便于切片取邻域
    dir0   = (angle <  22.5) | (angle >= 157.5)      # 水平方向（左/右邻域）
    dir45  = (angle >= 22.5) & (angle <  67.5)       # 右上/左下方向
    dir90  = (angle >= 67.5) & (angle < 112.5)       # 垂直方向（上/下邻域）
    dir135 = (angle >=112.5) & (angle < 157.5)       # 左上/右下方向

    # 各方向对应的两个邻域像素（切片操作，O(1) 内存，无循环）
    neighbors = {
        0:   (mp[1:-1, 2:],  mp[1:-1, :-2]),   # 右、左
        45:  (mp[:-2,  2:],  mp[2:,   :-2]),   # 右上、左下
        90:  (mp[:-2, 1:-1], mp[2:,  1:-1]),   # 上、下
        135: (mp[:-2, :-2],  mp[2:,   2:]),    # 左上、右下
    }
    nms = mag.copy()
    for d, mask in zip([0, 45, 90, 135], [dir0, dir45, dir90, dir135]):
        a, b = neighbors[d]
        # 非局部极大值 → 置零（抑制）
        nms[mask & ((mag < a) | (mag < b))] = 0.0

    # 步骤 4：双阈值分类
    strong = nms >= high_thr                          # 强边缘
    weak   = (nms >= low_thr) & ~strong               # 弱边缘

    # 滞后连接：弱边缘像素若 8-邻接内存在强边缘则提升为边缘
    sp = np.pad(strong.astype(np.uint8), 1, mode='constant')
    neighbor_strong = (
        sp[:-2, :-2] + sp[:-2, 1:-1] + sp[:-2, 2:] +
        sp[1:-1, :-2]               + sp[1:-1, 2:] +
        sp[2:,  :-2] + sp[2:,  1:-1] + sp[2:,  2:]
    )
    return strong | (weak & (neighbor_strong > 0))


def edge_visibility_score(img_np: np.ndarray,
                          mode: str     = 'gradient',
                          sigma: float  = 1.0,
                          low_thr: float  = 0.04,
                          high_thr: float = 0.10,
                          stat: str     = 'mean') -> float:
    """
    计算单张图片的边缘可见性得分（标量）。

    两种模式
    --------
    'gradient'
        Sobel 梯度幅值的统计量（均值/分位数）。
        · 雾/运动模糊/失焦模糊 → 边缘被平滑 → 得分下降
        · 噪声污染 → 引入伪边缘 → 得分升高（可与局部对比度联判）

    'canny'
        完整 Canny 流程：高斯平滑 → Sobel → 非极大值抑制 → 滞后连接。
        返回边缘像素密度（占全图比例，范围 [0,1]）。
        · 对噪声不如梯度模式敏感；但双阈值参数对结果影响较大

    参数
    ----
    img_np   : (H, W, C) float32，值域 [0, 1]
    mode     : 'gradient' | 'canny'
    sigma    : 高斯预平滑标准差（仅 canny 模式有效；越大越粗糙）
    low_thr  : Canny 弱边缘阈值（仅 canny 模式；需满足 low_thr < high_thr）
    high_thr : Canny 强边缘阈值（仅 canny 模式）
    stat     : 统计量，仅 'gradient' 模式有效：'mean' | 'median' | 'q75' | 'q90'
               （canny 模式忽略此参数，始终返回边缘密度）

    返回
    ----
    score : float
        值越高 → 边缘越清晰（图像越干净）
        值越低 → 边缘被抑制（雾/模糊污染特征）
    """
    gray = img_np @ _RGB2GRAY   # RGB → 灰度 (H, W)

    if mode == 'gradient':
        mag = _sobel_magnitude(gray)
        if stat == 'mean':
            return float(mag.mean())
        elif stat == 'median':
            return float(np.median(mag))
        elif stat == 'q75':
            return float(np.percentile(mag, 75))
        elif stat == 'q90':
            return float(np.percentile(mag, 90))
        else:
            raise ValueError(f"未知 stat='{stat}'，可选: mean / median / q75 / q90")

    elif mode == 'canny':
        edges = _canny_nms_hysteresis(gray, sigma=sigma,
                                      low_thr=low_thr, high_thr=high_thr)
        return float(edges.mean())   # 边缘像素占比，值域 [0, 1]

    else:
        raise ValueError(f"未知 mode='{mode}'，可选: 'gradient' | 'canny'")


# ══════════════════════════════════════════════════════════════════════════════
# 方法五：块效应得分（Blockiness）
# 原理：比较 8×8 块边界处像素差 vs 非边界处像素差
#       JPEG 压缩时 DCT 以 8×8 为单元独立编码，压缩越重块边界越明显
# 注意：图像本身含规则网格纹理（棋盘格、栅栏）时可能误判
# ══════════════════════════════════════════════════════════════════════════════

def blockiness_score(img_np: np.ndarray,
                     block_size: int = 8,
                     mode: str       = 'ratio',
                     stat: str       = 'mean') -> float:
    """
    计算单张图片的块效应得分（标量）。

    算法
    ----
    1. 将 RGB 图像转为灰度（ITU-R BT.601 权重）
    2. 分别计算水平/垂直方向的相邻像素绝对差分图
    3. 提取落在 8×8 块边界（列/行索引满足 (idx+1) % block_size == 0）
       的差分值，以及非边界处的差分值
    4. 对边界差分和非边界差分各自施加 stat 统计量，再按 mode 合并为标量

    两种 mode
    ---------
    'ratio'
        score = agg(boundary_diff) / agg(inner_diff)
        · 无压缩时 ≈ 1.0；压缩越重 > 1.0
        · 对图像整体亮度不敏感，跨图可比

    'diff'
        score = agg(boundary_diff) - agg(inner_diff)
        · 绝对差值，与图像亮度和纹理量相关；适合同场景对比

    参数
    ----
    img_np     : (H, W, C) float32，值域 [0, 1]
    block_size : DCT 块大小，JPEG 标准为 8
    mode       : 'ratio' | 'diff'
    stat       : 聚合统计量 'mean' | 'median' | 'q75' | 'q90'
                 高分位数对突出边界更敏感；'mean' 更稳健

    返回
    ----
    score : float
        值越高 → 块效应越明显（JPEG 压缩伪影越严重）
        ratio 模式中无压缩基线约为 1.0
    """
    gray  = img_np @ _RGB2GRAY          # (H, W)
    H, W  = gray.shape

    # ── 水平/垂直相邻像素绝对差分 ─────────────────────────────────────────
    h_diff = np.abs(np.diff(gray, axis=1))   # (H, W-1)
    v_diff = np.abs(np.diff(gray, axis=0))   # (H-1, W)

    # ── 边界掩码：(idx+1) % block_size == 0，即 idx = 7, 15, 23 … ──────────
    h_cols = np.arange(W - 1)
    v_rows = np.arange(H - 1)
    h_mask = (h_cols + 1) % block_size == 0   # 水平块边界列
    v_mask = (v_rows + 1) % block_size == 0   # 垂直块边界行

    boundary_vals = np.concatenate([
        h_diff[:, h_mask].ravel(),             # 水平边界处差分
        v_diff[v_mask, :].ravel(),             # 垂直边界处差分
    ])
    inner_vals = np.concatenate([
        h_diff[:, ~h_mask].ravel(),            # 水平非边界处差分
        v_diff[~v_mask, :].ravel(),            # 垂直非边界处差分
    ])

    def _agg(arr: np.ndarray) -> float:
        if stat == 'mean':
            return float(arr.mean())
        elif stat == 'median':
            return float(np.median(arr))
        elif stat == 'q75':
            return float(np.percentile(arr, 75))
        elif stat == 'q90':
            return float(np.percentile(arr, 90))
        else:
            raise ValueError(f"未知 stat='{stat}'，可选: mean / median / q75 / q90")

    b_val = _agg(boundary_vals)
    i_val = _agg(inner_vals)

    if mode == 'ratio':
        return b_val / (i_val + 1e-8)
    elif mode == 'diff':
        return b_val - i_val
    else:
        raise ValueError(f"未知 mode='{mode}'，可选: 'ratio' | 'diff'")


# ══════════════════════════════════════════════════════════════════════════════
# 方法六：振铃得分（Ringing）
# 原理：强边缘附近的平滑区域出现过冲/回摆（Gibbs 现象），
#       体现为 Laplacian 绝对值在近边缘区域异常偏高
# 注意：需要边缘定位，计算成本略高于前几种方法；
#       纹理丰富的图像自身 Laplacian 较高，ratio 模式可缓解误判
# ══════════════════════════════════════════════════════════════════════════════

def ringing_score(img_np: np.ndarray,
                  edge_thresh: float = 0.15,
                  radius: int        = 3,
                  mode: str          = 'ratio',
                  stat: str          = 'mean') -> float:
    """
    计算单张图片的振铃得分（标量）。

    算法
    ----
    1. 将 RGB 转为灰度（ITU-R BT.601）
    2. 用 Sobel 算子计算梯度幅值，归一化到 [0, 1] 后阈值化得到强边缘掩码
    3. 以半径 radius 膨胀边缘掩码，得到"近边缘区域"，
       再排除边缘本身，仅保留紧邻边缘的"应平滑区域"
    4. 在该区域内计算 Laplacian 绝对值的统计量
       ——振铃时此区域会出现高频振荡，Laplacian 异常偏高

    两种 mode
    ---------
    'ratio'
        score = agg(|Lap| in near_zone) / agg(|Lap| in far_zone)
        · 对图像整体纹理做归一化，≈1 表示无振铃；越大振铃越严重
        · 跨不同内容图像可比性更强

    'abs'
        score = agg(|Lap| in near_zone)
        · 近边缘区 Laplacian 的绝对量；受图像纹理影响较大，
          适合同类场景内的相对比较

    参数
    ----
    img_np      : (H, W, C) float32，值域 [0, 1]
    edge_thresh : Sobel 梯度幅值（归一化后）的阈值，用于确定"强边缘"
                  · 典型范围 0.10–0.25；过低噪声被当边缘，过高边缘稀疏
    radius      : 边缘膨胀像素数，定义近边缘区域宽度
                  · CIFAR 32×32 建议 2–4；高分辨率图像可用 5–8
    mode        : 'ratio' | 'abs'
    stat        : 'mean' | 'median' | 'q75' | 'q90'
                  · 高分位数对局部突出振铃更敏感

    返回
    ----
    score : float
        值越高 → 振铃越明显（JPEG/压缩伪影在边缘附近越严重）
        ratio 模式中无振铃基线约为 1.0
    """
    gray = img_np @ _RGB2GRAY          # (H, W)

    # ── Sobel 梯度幅值，归一化到 [0, 1] ─────────────────────────────────────
    gx       = ndimage.sobel(gray, axis=1)
    gy       = ndimage.sobel(gray, axis=0)
    grad_mag = np.hypot(gx, gy)
    g_max    = grad_mag.max()
    if g_max > 1e-8:
        grad_mag /= g_max

    # ── 强边缘掩码 & 近边缘区域 ──────────────────────────────────────────────
    edge_mask  = grad_mag > edge_thresh                          # 强边缘
    struct     = np.ones((2 * radius + 1, 2 * radius + 1), bool)
    dilated    = ndimage.binary_dilation(edge_mask, structure=struct)
    near_zone  = dilated & ~edge_mask                            # 近边缘但非边缘本身
    far_zone   = ~dilated                                        # 远离边缘区域

    # ── 近边缘区域为空时返回 0（极简图像或阈值过高）────────────────────────
    if near_zone.sum() == 0:
        return 0.0

    # ── Laplacian 绝对值 ─────────────────────────────────────────────────────
    lap_abs = np.abs(ndimage.laplace(gray))

    def _agg(arr: np.ndarray) -> float:
        if stat == 'mean':
            return float(arr.mean())
        elif stat == 'median':
            return float(np.median(arr))
        elif stat == 'q75':
            return float(np.percentile(arr, 75))
        elif stat == 'q90':
            return float(np.percentile(arr, 90))
        else:
            raise ValueError(f"未知 stat='{stat}'，可选: mean / median / q75 / q90")

    near_val = _agg(lap_abs[near_zone])

    if mode == 'ratio':
        if far_zone.sum() == 0:
            return near_val          # 退化：整张图均为近边缘，直接返回绝对量
        far_val = _agg(lap_abs[far_zone])
        return near_val / (far_val + 1e-8)
    elif mode == 'abs':
        return near_val
    else:
        raise ValueError(f"未知 mode='{mode}'，可选: 'ratio' | 'abs'")


# ══════════════════════════════════════════════════════════════════════════════
# 方法七：Gray-World 偏离（色偏得分）
# 原理：理想自然图像中 E[R] ≈ E[G] ≈ E[B]（Gray-World 假设）；
#       三通道均值偏离越大，色偏越严重（如雾偏白、雪偏蓝、夜景偏黄）
# 注意：场景本身主色调强（草地偏绿、海洋偏蓝）会造成误判，
#       此时偏离并非由污染引起而是场景固有属性
# ══════════════════════════════════════════════════════════════════════════════

def gray_world_score(img_np: np.ndarray,
                     mode: str = 'l2') -> float:
    """
    计算单张图片的 Gray-World 偏离得分（标量）。

    算法
    ----
    1. 分别计算 R、G、B 三通道的空间均值 μ_R, μ_G, μ_B
    2. 计算三通道全局均值 μ = (μ_R + μ_G + μ_B) / 3
    3. 按指定 mode 度量三通道均值相对于 μ 的偏差

    三种 mode
    ---------
    'l2'
        score = sqrt(mean((μ_c - μ)²))，c ∈ {R, G, B}
        · RMS 偏差，对大色偏更敏感（二次惩罚）；跨图可比性强（推荐）

    'l1'
        score = mean(|μ_c - μ|)
        · 平均绝对偏差，对异常通道惩罚较温和

    'max_range'
        score = max(μ_R, μ_G, μ_B) - min(μ_R, μ_G, μ_B)
        · 最大与最小通道均值之差；直觉最强，但仅用两个通道信息

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    mode   : 'l2' | 'l1' | 'max_range'

    返回
    ----
    score : float
        值越高 → 色偏越严重（三通道均值越不均衡）
        Gray-World 满足时 score ≈ 0
    """
    # 各通道空间均值，shape (C,) = (3,)
    mu        = img_np.mean(axis=(0, 1))
    mu_global = float(mu.mean())           # (μ_R + μ_G + μ_B) / 3
    dev       = mu - mu_global             # 各通道相对全局均值的偏差

    if mode == 'l2':
        return float(np.sqrt((dev ** 2).mean()))
    elif mode == 'l1':
        return float(np.abs(dev).mean())
    elif mode == 'max_range':
        return float(mu.max() - mu.min())
    else:
        raise ValueError(f"未知 mode='{mode}'，可选: 'l2' | 'l1' | 'max_range'")


# ══════════════════════════════════════════════════════════════════════════════
# 方法八：White-Patch / Max-RGB 偏离（高光色偏得分）
# 原理：White-Patch 假设图像中最亮区域应为白色（R_max ≈ G_max ≈ B_max）；
#       各通道最大值（或高分位数）偏离越大，色偏越严重
# 注意：高光过曝（像素值截断至 1.0）会导致各通道齐平而误判偏离为 0；
#       使用 q99/q95 分位数代替纯 max 可缓解此问题
# ══════════════════════════════════════════════════════════════════════════════

def white_patch_score(img_np: np.ndarray,
                      stat: str = 'q99',
                      mode: str = 'l2') -> float:
    """
    计算单张图片的 White-Patch / Max-RGB 偏离得分（标量）。

    算法
    ----
    1. 对每个颜色通道提取"亮端代表值" M_c（由 stat 参数控制）
    2. 计算三通道亮端代表值的全局均值 M̄ = mean(M_R, M_G, M_B)
    3. 按 mode 度量各通道偏离 M̄ 的程度

    stat 选项（亮端代表值提取方式）
    --------------------------------
    'max'
        M_c = max of channel c，纯最大值
        · 最符合 White-Patch 原始定义，但对单像素高光/过曝极敏感

    'q99' / 'q95'
        M_c = 第 99 / 95 百分位数
        · 对少量过曝像素鲁棒（推荐）；分位数越低越保守

    mode 选项（偏差度量方式）
    -------------------------
    'l2'
        score = sqrt(mean((M_c - M̄)²))，RMS 偏差（推荐）
    'l1'
        score = mean(|M_c - M̄|)，平均绝对偏差
    'max_range'
        score = max(M_c) - min(M_c)，通道亮端极差

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    stat   : 'max' | 'q99' | 'q95'
    mode   : 'l2' | 'l1' | 'max_range'

    返回
    ----
    score : float
        值越高 → 高光色偏越严重（最亮区域三通道越不均衡）
        White-Patch 满足时 score ≈ 0
    """
    # ── 各通道亮端代表值，shape (C,) = (3,) ────────────────────────────────
    if stat == 'max':
        ch_val = img_np.max(axis=(0, 1))
    elif stat == 'q99':
        ch_val = np.percentile(img_np, 99, axis=(0, 1))
    elif stat == 'q95':
        ch_val = np.percentile(img_np, 95, axis=(0, 1))
    else:
        raise ValueError(f"未知 stat='{stat}'，可选: 'max' | 'q99' | 'q95'")

    ref = float(ch_val.mean())          # M̄
    dev = ch_val - ref                  # 各通道偏离

    if mode == 'l2':
        return float(np.sqrt((dev ** 2).mean()))
    elif mode == 'l1':
        return float(np.abs(dev).mean())
    elif mode == 'max_range':
        return float(ch_val.max() - ch_val.min())
    else:
        raise ValueError(f"未知 mode='{mode}'，可选: 'l2' | 'l1' | 'max_range'")


# ══════════════════════════════════════════════════════════════════════════════
# 方法九：色度向量集中度 / 偏移（Chroma Shift）
# 原理：在 rg 色度空间或感知均匀的 CIE Lab ab 平面内，
#       计算图像均值色度向量相对于"中性灰"原点的偏移幅度；
#       色偏越重，均值色度点偏离中性越远
# 注意：场景本身主色调强（草地/海洋）会造成自然偏移而非污染引起；
#       可与 gray_world / white_patch 联判以区分场景偏移与污染偏移
# ══════════════════════════════════════════════════════════════════════════════

# sRGB → CIE XYZ (D65) 线性变换矩阵
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

# CIE D65 标准光源白点 (Xn, Yn, Zn)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


def _rgb_to_lab(img_np: np.ndarray) -> np.ndarray:
    """
    将 (H, W, 3) float32/float64 sRGB 图像（值域 [0,1]）转换为 CIE Lab，
    返回同形状 (H, W, 3) 数组：[L*, a*, b*]。

    转换链：sRGB → 线性 RGB（去伽马）→ CIE XYZ D65 → CIE Lab
    """
    img = img_np.astype(np.float64)
    # sRGB 去伽马（IEC 61966-2-1 标准）
    linear = np.where(img <= 0.04045,
                      img / 12.92,
                      ((img + 0.055) / 1.055) ** 2.4)
    # 线性 RGB → XYZ D65
    xyz = linear @ _SRGB_TO_XYZ.T           # (H, W, 3)
    # XYZ → Lab：f(t) 分段函数
    t      = xyz / _D65_WHITE
    delta  = 6.0 / 29.0
    ft     = np.where(t > delta ** 3,
                      np.cbrt(t),
                      t / (3.0 * delta ** 2) + 4.0 / 29.0)
    L  = 116.0 * ft[..., 1] - 16.0
    a  = 500.0 * (ft[..., 0] - ft[..., 1])
    b  = 200.0 * (ft[..., 1] - ft[..., 2])
    return np.stack([L, a, b], axis=-1)


def chroma_shift_score(img_np: np.ndarray,
                       mode:           str   = 'lab',
                       min_brightness: float = 0.02) -> float:
    """
    计算单张图片的色度向量偏移得分（标量）。

    算法
    ----
    在色度空间内，中性灰图像的均值色度向量应落在"原点"附近；
    色偏越重，均值色度向量偏离原点越远。

    四种 mode
    ---------
    'rg'
        rg 色度空间（r = R/(R+G+B), g = G/(R+G+B)）
        score = ||mean(r,g) - (1/3, 1/3)||₂
        · 计算最快，无需颜色空间变换；中性灰的 rg 坐标恰为 (1/3, 1/3)
        · sRGB 非线性不影响方向判断，仅轻微影响量值精度

    'rg_spread'
        rg 色度分布离散度：mean(||rg_i - mean(rg)||₂)
        · 度量色度"集中程度"：雾/白色污染使所有像素收敛到相似色度 → 值降低
        · 此模式"值越低 = 色度越集中 = 往往污染越严重"
          → _SORT_ASCENDING=True，绘图时展示 1-score

    'lab'
        CIE Lab ab 平面（感知均匀色彩空间）
        score = ||(ā, b̄)||₂，其中 ā, b̄ 为全图 a*, b* 均值
        · 感知最均匀；ab=(0,0) 表示无色偏（推荐默认）

    'lab_chroma'
        每像素彩度 C* = sqrt(a*² + b*²) 的全图均值
        · 反映图像整体"有多彩"；雾天 C* 降低 → 值越低越严重
          → _SORT_ASCENDING=True，绘图时展示 1-score

    参数
    ----
    img_np         : (H, W, C) float32，值域 [0, 1]
    mode           : 'rg' | 'rg_spread' | 'lab' | 'lab_chroma'
    min_brightness : 仅 rg / rg_spread 有效：过滤亮度总和低于此值的近黑像素，
                     防止极暗像素引发除零和色度抖动（默认 0.02）

    返回
    ----
    score : float
        'rg' / 'lab'          → 值越高 = 色偏偏离中性越严重
        'rg_spread' / 'lab_chroma' → 值越高 = 色度越分散/越饱和
                                     （需结合 _SORT_ASCENDING 解读）
    """
    if mode in ('rg', 'rg_spread'):
        # ── 亮度过滤，避免近黑像素拉偏色度 ────────────────────────────────
        brightness = img_np.sum(axis=-1)               # (H, W)
        valid      = img_np[brightness > min_brightness]  # (N, 3)
        if valid.shape[0] == 0:
            return 0.0
        rg = valid[:, :2] / (valid.sum(axis=-1, keepdims=True) + 1e-8)  # (N, 2)

        if mode == 'rg':
            mean_rg = rg.mean(axis=0)                           # (2,)
            return float(np.linalg.norm(mean_rg - np.array([1/3, 1/3])))
        else:  # 'rg_spread'
            dists = np.linalg.norm(rg - rg.mean(axis=0), axis=1)
            return float(dists.mean())

    elif mode in ('lab', 'lab_chroma'):
        lab = _rgb_to_lab(img_np)                       # (H, W, 3)
        a   = lab[..., 1]                               # a* 通道
        b   = lab[..., 2]                               # b* 通道

        if mode == 'lab':
            return float(np.sqrt(a.mean() ** 2 + b.mean() ** 2))
        else:  # 'lab_chroma'
            chroma = np.sqrt(a ** 2 + b ** 2)          # C* per pixel
            return float(chroma.mean())

    else:
        raise ValueError(
            f"未知 mode='{mode}'，可选: 'rg' | 'rg_spread' | 'lab' | 'lab_chroma'"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 方法十一至十四：模糊度量（Blur Metrics，纯 NumPy 实现）
# 四种指标均为「越模糊越小」，在 _SORT_ASCENDING 中标记为 True，
# 绘图时统一展示 1-score（值越高 = 图像越模糊）。
# 注意：噪声类污染会引入高频，可能抬高各指标，使其指标偏大≠更清晰。
# ══════════════════════════════════════════════════════════════════════════════

def _bm_luminance(img_np: np.ndarray) -> np.ndarray:
    """(H,W,C) float [0,1] → (H,W) 亮度 Y，BT.601 加权。"""
    return (0.299 * img_np[..., 0]
            + 0.587 * img_np[..., 1]
            + 0.114 * img_np[..., 2]).astype(np.float32)


_BM_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_BM_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _bm_convolve2d(Y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """2D 卷积，边界 0 填充，用 stride_tricks 一次算整图。"""
    H, W = Y.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    Yp = np.pad(Y, ((ph, ph), (pw, pw)), mode='constant', constant_values=0)
    s0, s1 = Yp.strides
    win = np.lib.stride_tricks.as_strided(
        Yp, shape=(H, W, kh, kw),
        strides=(s0, s1, s0, s1), writeable=False,
    )
    return np.tensordot(win, k, axes=((2, 3), (0, 1)))


def laplacian_var_score(img_np: np.ndarray, **kwargs) -> float:
    """
    拉普拉斯方差 Var(∇²Y)：越模糊越小。
    离散 Laplacian 为 4 邻域形式，仅对内部像素求方差（避免边界 0 稀释）。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float，值越大 = 图像越清晰
    """
    Y = _bm_luminance(img_np)
    lap = np.zeros_like(Y, dtype=np.float64)
    lap[1:-1, 1:-1] = (
        -4 * Y[1:-1, 1:-1].astype(np.float64)
        + Y[0:-2, 1:-1].astype(np.float64)
        + Y[2:,   1:-1].astype(np.float64)
        + Y[1:-1, 0:-2].astype(np.float64)
        + Y[1:-1, 2:  ].astype(np.float64)
    )
    return float(np.var(lap[1:-1, 1:-1]))


def tenengrad_score(img_np: np.ndarray,
                    eta_percentile: float = 50.0, **kwargs) -> float:
    """
    Tenengrad（Sobel 梯度能量）：越模糊越小。
    返回幅值高于 eta_percentile 百分位的像素上的 **平均** 梯度能量（逐像素均值），
    而非累积求和，保证值域与图像尺寸无关，且通常落在 [0, 32] 内（[0,1] 图像时
    Sobel 最大响应 4，故 mag_sq 上界 = 4²+4² = 32）。

    参数
    ----
    img_np          : (H, W, C) float32，值域 [0, 1]
    eta_percentile  : 梯度幅值阈值百分位数（0~100），默认 50.0

    返回
    ----
    score : float，值越大 = 图像越清晰
    """
    Y = _bm_luminance(img_np)
    Gx = _bm_convolve2d(Y, _BM_SOBEL_X)
    Gy = _bm_convolve2d(Y, _BM_SOBEL_Y)
    mag_sq = Gx.astype(np.float64) ** 2 + Gy.astype(np.float64) ** 2
    mag    = np.sqrt(mag_sq)
    eta    = np.percentile(mag, eta_percentile)
    mask   = mag > eta
    n = int(np.sum(mask))
    if n == 0:
        return 0.0
    return float(np.sum(mag_sq[mask]) / n)


def brenner_score(img_np: np.ndarray, **kwargs) -> float:
    """
    Brenner 梯度：越模糊越小。
    水平: Σ(Y(x+2,y)-Y(x,y))² + 垂直: Σ(Y(x,y+2)-Y(x,y))²，
    再除以参与计算的像素总数，得到**逐像素均值**。
    由于 Y ∈ [0,1]，每项差值平方 ∈ [0,1]，故均值天然落在 [0, 1]。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ∈ [0, 1]，值越大 = 图像越清晰
    """
    Y  = _bm_luminance(img_np).astype(np.float64)
    H, W = Y.shape
    dx = (Y[2:, :] - Y[:-2, :]) ** 2   # (H-2, W)
    dy = (Y[:, 2:] - Y[:, :-2]) ** 2   # (H, W-2)
    n  = (H - 2) * W + H * (W - 2)
    return float((np.sum(dx) + np.sum(dy)) / n)


def wavelet_hf_score(img_np: np.ndarray, **kwargs) -> float:
    """
    小波高频子带能量（Haar 一层）：越模糊越小。
    返回 HH、LH、HL 三个高频子带的**逐系数平均**能量。
    若 H/W 为奇则截断到偶数尺寸。

    Haar 一层变换后，每个子带系数 ∈ [-0.25, 0.25]（因两次 ×0.5），
    故系数平方 ∈ [0, 0.0625]，平均值天然有界。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ∈ [0, 0.0625]，值越大 = 图像越清晰
    """
    Y = _bm_luminance(img_np).astype(np.float64)
    H, W = Y.shape
    if H % 2 != 0:
        Y = Y[:-1, :]
        H -= 1
    if W % 2 != 0:
        Y = Y[:, :-1]
        W -= 1
    L     = (Y[:, 0::2] + Y[:, 1::2]) * 0.5
    H_row = (Y[:, 0::2] - Y[:, 1::2]) * 0.5
    LH = (L[0::2, :]     - L[1::2, :])     * 0.5
    HL = (H_row[0::2, :] + H_row[1::2, :]) * 0.5
    HH = (H_row[0::2, :] - H_row[1::2, :]) * 0.5
    n  = (H // 2) * (W // 2)   # 每个子带的系数个数
    return float((np.sum(LH ** 2) + np.sum(HL ** 2) + np.sum(HH ** 2)) / (3 * n))


# ══════════════════════════════════════════════════════════════════════════════
# 方法十五至十六：方向性模糊度量（Directional Blur Metrics）
# 两种指标均为「越方向性越大」（运动模糊候选越强），
# 在 _SORT_ASCENDING 中标记为 False，直接展示原始得分（高分 = 危险）。
# 与 gradient_dir_entropy 的区别：
#   - edge_dir_anisotropy    : 仅统计强边缘像素，直方图按幅值加权，侧重"最强边缘的方向一致性"
#   - fft_dir_concentration  : 在频域直接测量能量的方向集中度，对周期性运动模糊更灵敏
# ══════════════════════════════════════════════════════════════════════════════

def edge_dir_anisotropy_score(img_np: np.ndarray,
                              n_bins: int   = 18,
                              mag_pct: float = 70.0, **kwargs) -> float:
    """
    边缘方向各向异性（Radon/Hough on edges 的简化版）：越高 = 边缘方向越集中。

    算法
    ----
    1. 用 scipy.ndimage.convolve (mode='reflect') 计算 Sobel Gx/Gy，
       避免 zero-padding 在边界产生虚假强梯度（已修复原版 bug）。
    2. 仅保留幅值 > mag_pct 百分位的强边缘像素。
    3. 对保留像素按 **幅值加权** 构建方向直方图，覆盖 [0°, 180°)（无符号方向）。
    4. 计算归一化香农熵 H_norm = H / log(n_bins)，返回 1 - H_norm（各向异性）。

    与 gradient_dir_entropy_score 的区别：
    - 本函数直方图按梯度幅值加权（强边缘贡献更大）；
    - gradient_dir_entropy_score 按像素个数等权计数。

    无效退化条件（强边缘 < 10 像素）：返回 0.0（无方向性）。

    参数
    ----
    img_np  : (H, W, C) float32，值域 [0, 1]
    n_bins  : 方向直方图区间数（默认 18，即每 10° 一个 bin）
    mag_pct : 强边缘筛选百分位（默认 70，保留最强 30% 边缘）

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 边缘方向越集中（运动模糊候选）
    """
    gray = img_np @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    if smooth_sigma := kwargs.get('smooth_sigma', 0.0):
        gray = gaussian_filter(gray, sigma=smooth_sigma)

    gx  = convolve(gray, _SOBEL_X, mode='reflect')
    gy  = convolve(gray, _SOBEL_Y, mode='reflect')
    mag = np.hypot(gx.astype(np.float64), gy.astype(np.float64))

    thr  = np.percentile(mag, mag_pct)
    mask = mag > thr
    if int(mask.sum()) < 10:
        return 0.0

    ang = np.rad2deg(np.arctan2(gy[mask], gx[mask])) % 180.0  # [0°, 180°)
    hist, _ = np.histogram(ang, bins=n_bins, range=(0.0, 180.0),
                           weights=mag[mask])                   # 幅值加权
    if hist.sum() <= 0:
        return 0.0
    p = hist.astype(np.float64) / hist.sum()
    p = p[p > 0]   # 滤掉零项，避免 log(0)
    entropy = -float(np.sum(p * np.log(p)))
    H_norm  = entropy / np.log(n_bins)
    return float(1.0 - H_norm)


def fft_dir_concentration_score(img_np: np.ndarray,
                                n_bins: int    = 18,
                                dc_radius: float = 2.0, **kwargs) -> float:
    """
    频谱方向能量集中度：越高 = 频谱越各向异性（运动模糊候选）。

    算法
    ----
    1. 对亮度图做 2D FFT，取幅值谱并 fftshift 使 DC 居中。
    2. 排除半径 ≤ dc_radius 的低频圆盘（含 DC）。
    3. 按频率向量方向 ∈ [0, π) 做幅值加权方向直方图。
    4. 归一化后以 (max(p_k) - 1/n_bins) / (1 - 1/n_bins) 衡量集中度：
         - 各向同性（均匀分布）→ 0；
         - 完全集中到一个方向  → 1。
       （已修复原版 bug：原版返回 hist.max() ∈ [1/n_bins, 1]，基线非 0）

    物理含义
    --------
    水平运动模糊的 PSF 在频域使垂直频率分量集中（水平高频被压制），
    因此集中度升高可指示运动模糊方向性。

    参数
    ----
    img_np    : (H, W, C) float32，值域 [0, 1]
    n_bins    : 方向直方图区间数（默认 18）
    dc_radius : 排除低频圆盘的像素半径（默认 2.0）

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 频谱方向越集中（运动模糊候选）
    """
    Y = (img_np @ np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.float64)
    H_im, W_im = Y.shape

    F   = np.fft.fftshift(np.fft.fft2(Y))
    mag = np.abs(F)

    cy, cx    = H_im // 2, W_im // 2
    y_idx, x_idx = np.indices((H_im, W_im))
    dy = y_idx - cy
    dx = x_idx - cx
    r  = np.sqrt(dy ** 2 + dx ** 2)
    mask = r > dc_radius
    if not np.any(mask):
        return 0.0

    ang = np.mod(np.arctan2(dy[mask], dx[mask]), np.pi)   # [0, π)
    bins = np.linspace(0.0, np.pi, n_bins + 1, endpoint=True)
    hist, _ = np.histogram(ang, bins=bins, weights=mag[mask])
    if hist.sum() <= 0:
        return 0.0

    p = hist.astype(np.float64) / hist.sum()
    uniform_baseline = 1.0 / n_bins
    peak = float(p.max())
    # 归一化到 [0, 1]：均匀分布时为 0，完全集中时为 1
    return float((peak - uniform_baseline) / (1.0 - uniform_baseline + 1e-12))


# ══════════════════════════════════════════════════════════════════════════════
# 方法十七至二十五：曝光/亮度分布度量（Exposure Metrics）
#
# 原始指标来自 exposure_metrics.py，改写为统一 (img_np) → float 接口。
# 修复说明：
#   1. 原 hi_lo_clip_ratio / brightness_stats / histogram_stats 返回元组 →
#      拆成独立函数，每个返回单个 float。
#   2. 偏度（skewness）有正负，直接用原始值会使柱状图出现负柱，语义不清 →
#      改为 abs(skewness)，"绝对偏度越大 = 分布越异常"，恒 >= 0。
#   3. histogram_stats 中 p * log(p+eps) 的写法改为先过滤零项再计算熵，
#      与框架其他熵计算保持一致。
# ══════════════════════════════════════════════════════════════════════════════

def hi_clip_score(img_np: np.ndarray, eps_hi: float = 0.02, **kwargs) -> float:
    """
    高光饱和比例 r_hi = 1/N × Σ 1(Y ≥ 1−ε_hi)：越高 = 过曝越严重。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    eps_hi : 高光阈值，默认 0.02（即 Y ≥ 0.98 计为饱和）

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 高光饱和像素比例越大（过曝越严重）
    """
    Y = _bm_luminance(img_np)
    return float(np.sum(Y >= (1.0 - eps_hi))) / Y.size


def lo_clip_score(img_np: np.ndarray, eps_lo: float = 0.02, **kwargs) -> float:
    """
    暗部饱和比例 r_lo = 1/N × Σ 1(Y ≤ ε_lo)：越高 = 欠曝越严重。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    eps_lo : 暗部阈值，默认 0.02（即 Y ≤ 0.02 计为饱和）

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 暗部饱和像素比例越大（欠曝越严重）
    """
    Y = _bm_luminance(img_np)
    return float(np.sum(Y <= eps_lo)) / Y.size


def mean_luma_score(img_np: np.ndarray, **kwargs) -> float:
    """
    亮度均值 μ_Y：越高 = 图像越亮（brightness / fog / snow 等污染常使均值升高）。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 平均亮度越高
    """
    return float(_bm_luminance(img_np).mean())


def median_luma_score(img_np: np.ndarray, **kwargs) -> float:
    """
    亮度中位数 median(Y)：对极端像素鲁棒，更稳定的全局亮度估计。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 中位亮度越高
    """
    return float(np.median(_bm_luminance(img_np)))


def hist_skewness_score(img_np: np.ndarray, **kwargs) -> float:
    """
    亮度直方图绝对偏度 |skew(Y)|：越高 = 亮度分布偏斜越严重。

    使用绝对值以统一方向（原始偏度有正负：过曝→负偏，欠曝→正偏，均为异常）。
    对 var=0 的纯色图返回 0.0。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ≥ 0，值越高 = 亮度分布偏斜越严重（过曝/欠曝均可触发）
    """
    Y = _bm_luminance(img_np).ravel().astype(np.float64)
    var = float(Y.var())
    if var <= 0:
        return 0.0
    z = (Y - Y.mean()) / np.sqrt(var)
    return float(abs((z ** 3).mean()))


def hist_kurtosis_score(img_np: np.ndarray, **kwargs) -> float:
    """
    亮度直方图峰度（4 阶标准矩）：越高 = 分布越尖峭（饱和/剪切区域集中）。

    正态分布的峰度 ≈ 3；过曝图像像素集中于高亮端，峰度升高。
    对 var=0 的纯色图返回 0.0。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ≥ 0（正态≈3），值越高 = 分布越尖峭
    """
    Y = _bm_luminance(img_np).ravel().astype(np.float64)
    var = float(Y.var())
    if var <= 0:
        return 0.0
    z = (Y - Y.mean()) / np.sqrt(var)
    return float((z ** 4).mean())


def entropy_norm_score(img_np: np.ndarray,
                       num_bins: int = 256, **kwargs) -> float:
    """
    亮度直方图归一化熵 H(Y) / log(K)：越低 = 分布越集中（过/欠曝候选）。

    修复：改为先过滤零概率 bin 再计算熵，避免 0×log(0+eps) 数值干扰。

    参数
    ----
    img_np   : (H, W, C) float32，值域 [0, 1]
    num_bins : 直方图区间数，默认 256

    返回
    ----
    score : float ∈ [0, 1]，值越低 = 亮度分布越集中（曝光异常候选）
    """
    Y = _bm_luminance(img_np).ravel().astype(np.float64)
    hist, _ = np.histogram(Y, bins=num_bins, range=(0.0, 1.0))
    total = int(hist.sum())
    if total == 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]   # 过滤零 bin，避免 log(0)
    H = -float(np.sum(p * np.log(p)))
    return float(H / np.log(num_bins))


def dynamic_range_score(img_np: np.ndarray,
                        p_low: float  = 1.0,
                        p_high: float = 99.0, **kwargs) -> float:
    """
    有效动态范围 DR = P_p_high(Y) − P_p_low(Y)：越低 = 对比度越差（雾/低对比度污染）。

    参数
    ----
    img_np  : (H, W, C) float32，值域 [0, 1]
    p_low   : 下百分位数（默认 1.0）
    p_high  : 上百分位数（默认 99.0）

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 动态范围越宽（图像对比度越好）
    """
    Y = _bm_luminance(img_np)
    lo = float(np.percentile(Y, p_low))
    hi = float(np.percentile(Y, p_high))
    return float(max(0.0, hi - lo))


def rgb_hi_clip_score(img_np: np.ndarray, eps_hi: float = 0.02, **kwargs) -> float:
    """
    RGB 通道高光饱和比例（三通道各自统计后取均值）：越高 = 某通道剪切越严重。

    与 hi_clip_score 的区别：本函数按颜色通道分别统计，
    可检测"单通道过曝"（例如强色偏导致的通道饱和），而亮度版本会被其他通道均值稀释。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]
    eps_hi : 高光阈值，默认 0.02

    返回
    ----
    score : float ∈ [0, 1]，值越高 = RGB 通道高光饱和越严重
    """
    thr = 1.0 - eps_hi
    H, W = img_np.shape[:2]
    n = H * W
    return float(np.mean([
        np.sum(img_np[..., c] >= thr) / n for c in range(3)
    ]))


# ══════════════════════════════════════════════════════════════════════════════
# 方法二十六至二十九：噪声度量（Noise Metrics）
#
# 原始逻辑来自 noise_metrics.py，改写为统一 (img_np) → float 接口。
# 修复说明：
#   原版 _mean_blur 使用 _convolve2d（零填充），导致边界像素的 blur 值
#   偏低，使 R = Y - blur(Y) 在边界处虚假偏大，过高估计边界噪声。
#   mad_sigma / dark_region_sigma / flat_patch_sigma 均受影响。
#   修复：统一改用 scipy.ndimage.uniform_filter(mode='reflect')，
#   已与 pre_knowledge.py 中其余方法的边界处理保持一致。
# ══════════════════════════════════════════════════════════════════════════════

def mad_sigma_score(img_np: np.ndarray, **kwargs) -> float:
    """
    高通残差 MAD 噪声估计 σ_hat = median(|R|) / 0.6745：越高 = 噪声越强。

    算法：R = Y − blur(Y)，blur 为 3×3 均值滤波（mode='reflect'，已修复原版零填充 bug）。
    MAD 估计对少数异常像素鲁棒，是常用的鲁棒噪声估计方法。

    参数
    ----
    img_np : (H, W, C) float32，值域 [0, 1]

    返回
    ----
    score : float ≥ 0，值越高 = 全图噪声 σ 越大
    """
    Y    = _bm_luminance(img_np).astype(np.float32)
    blur = uniform_filter(Y, size=3, mode='reflect')
    R    = Y - blur
    return float(np.median(np.abs(R)) / 0.6745)


def flat_patch_sigma_score(img_np: np.ndarray,
                           patch_size: int = 7, **kwargs) -> float:
    """
    平坦块方差法噪声估计：寻找局部方差最小的内部 patch，用其标准差估计背景噪声。

    算法：
      1. 用 uniform_filter 计算每个位置的 E[Y] 和 E[Y²]（mode='reflect'，已修复零填充）
      2. local_var = E[Y²] − E[Y]²，clip ≥ 0 处理浮点精度
      3. 取内部区域（排除 pad 圈边缘）的最小 var，开方得 σ

    参数
    ----
    img_np     : (H, W, C) float32，值域 [0, 1]
    patch_size : 局部窗口大小，默认 7

    返回
    ----
    score : float ≥ 0，值越高 = 最平坦区域的噪声 σ 越大
    """
    Y    = _bm_luminance(img_np).astype(np.float32)
    H, W = Y.shape
    pad  = patch_size // 2

    e_x  = uniform_filter(Y,      size=patch_size, mode='reflect')
    e_x2 = uniform_filter(Y ** 2, size=patch_size, mode='reflect')
    var  = np.clip(e_x2 - e_x ** 2, 0.0, None)

    # 排除边缘 pad 圈，只在能放下完整 patch 的内部区域取最小值
    if H > 2 * pad and W > 2 * pad:
        inner = var[pad: H - pad, pad: W - pad]
    else:
        inner = var
    return float(np.sqrt(np.min(inner)))


def hf_lift_ratio_score(img_np: np.ndarray,
                        r_low: float  = 0.1,
                        r_high: float = 0.5, **kwargs) -> float:
    """
    频谱高频"白噪声抬升"：高频环带能量占全频总能量的比例。

    算法：
      1. 2D FFT 取功率谱 P = |F|²，fftshift 使 DC 居中。
      2. 归一化半径 r_norm = r / (max(H,W)/2)。
      3. 高频环带 mask: r_low ≤ r_norm ≤ r_high。
      4. score = Σ P[mask] / (Σ P + ε)。

    注意：total_energy 含 DC 分量，图像越亮则 DC 越大，
    相同噪声水平的明亮图像得分会略低（已知的亮度耦合限制）。

    参数
    ----
    img_np  : (H, W, C) float32，值域 [0, 1]
    r_low   : 环带内径（归一化），默认 0.1
    r_high  : 环带外径（归一化），默认 0.5

    返回
    ----
    score : float ∈ [0, 1]，值越高 = 高频能量占比越大（白噪声抬升越明显）
    """
    Y = _bm_luminance(img_np).astype(np.float64)
    H, W = Y.shape

    P   = np.abs(np.fft.fftshift(np.fft.fft2(Y))) ** 2
    cy, cx    = H // 2, W // 2
    y_idx, x_idx = np.indices((H, W))
    r_norm = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2) / (max(H, W) / 2.0 + 1e-6)

    mask_hf = (r_norm >= r_low) & (r_norm <= r_high)
    return float(P[mask_hf].sum() / (P.sum() + 1e-12))


def dark_region_sigma_score(img_np: np.ndarray,
                            dark_thresh: float = 0.2, **kwargs) -> float:
    """
    暗部区域噪声估计 σ（MAD 法）：仅在亮度 < dark_thresh 的像素上估计噪声。

    暗部噪声对低照度污染（impulse_noise / shot_noise 在暗部更明显）更敏感。
    无暗部像素（全图亮度 ≥ dark_thresh）时返回 0.0。
    边界处理已修复：改用 uniform_filter(mode='reflect')。

    参数
    ----
    img_np      : (H, W, C) float32，值域 [0, 1]
    dark_thresh : 暗部亮度阈值，默认 0.2

    返回
    ----
    score : float ≥ 0，值越高 = 暗部噪声 σ 越大
    """
    Y    = _bm_luminance(img_np).astype(np.float32)
    mask = Y < dark_thresh
    if not np.any(mask):
        return 0.0
    blur = uniform_filter(Y, size=3, mode='reflect')
    R    = (Y - blur)[mask]
    return float(np.median(np.abs(R)) / 0.6745)


# ══════════════════════════════════════════════════════════════════════════════
# 统一打分接口
# ══════════════════════════════════════════════════════════════════════════════

# 方法注册表：方法名 → 打分函数
_SCORE_REGISTRY = {
    'dark_channel':              dark_channel_score,
    'hsv':                       hsv_saturation_score,
    'local_contrast':            local_contrast_score,
    'edge_visibility':           edge_visibility_score,
    'blockiness':                blockiness_score,
    'ringing':                   ringing_score,
    'gray_world':                gray_world_score,
    'white_patch':               white_patch_score,
    'chroma_shift':              chroma_shift_score,
    'gradient_dir_entropy':      gradient_direction_entropy_score,
    'laplacian_var':             laplacian_var_score,
    'tenengrad':                 tenengrad_score,
    'brenner':                   brenner_score,
    'wavelet_hf':                wavelet_hf_score,
    'edge_dir_anisotropy':       edge_dir_anisotropy_score,
    'fft_dir_concentration':     fft_dir_concentration_score,
    'mad_sigma':                 mad_sigma_score,
    'flat_patch_sigma':          flat_patch_sigma_score,
    'hf_lift_ratio':             hf_lift_ratio_score,
    'dark_region_sigma':         dark_region_sigma_score,
    'hi_clip':                   hi_clip_score,
    'lo_clip':                   lo_clip_score,
    'mean_luma':                 mean_luma_score,
    'median_luma':               median_luma_score,
    'hist_skewness':             hist_skewness_score,
    'hist_kurtosis':             hist_kurtosis_score,
    'entropy_norm':              entropy_norm_score,
    'dynamic_range':             dynamic_range_score,
    'rgb_hi_clip':               rgb_hi_clip_score,
}

# 每种方法的说明（用于图表标题）
_METHOD_DESC = {
    'dark_channel': (
        "Dark Channel Score  D(x)=min_{{c}} min_{{y in Omega}} I_c(y)\n"
        "值越高 = 暗通道越抬升（雾/亮度污染特征越明显）"
    ),
    'hsv': (
        "1 - HSV Saturation Score  (原始: S=(max-min)/max)\n"
        "展示 1-score：值越高 = 饱和度越低（雾/雪褪色特征越明显）"
    ),
    'local_contrast': (
        "1 - Local Contrast Score  (原始: laplacian 或 rms)\n"
        "展示 1-score：值越高 = 对比度越弱（雾/失焦模糊特征越明显）"
    ),
    'edge_visibility': (
        "1 - Edge Visibility Score  (原始: gradient 或 canny)\n"
        "展示 1-score：值越高 = 边缘越少（雾/模糊特征越明显）"
    ),
    'blockiness': (
        "Blockiness Score  boundary_diff / inner_diff  (或 diff 模式)\n"
        "值越高 = 块效应越明显（JPEG 压缩伪影越严重）"
    ),
    'ringing': (
        "Ringing Score  |Lap|(near_edge) / |Lap|(far_edge)  (或 abs 模式)\n"
        "值越高 = 振铃越明显（强边缘附近过冲/回摆越严重）"
    ),
    'gray_world': (
        "Gray-World Deviation Score  sqrt(mean((mu_c - mu)^2))\n"
        "值越高 = 色偏越严重（三通道均值越不均衡）"
    ),
    'white_patch': (
        "White-Patch / Max-RGB Deviation Score  sqrt(mean((M_c - M_bar)^2))\n"
        "值越高 = 高光色偏越严重（最亮区域三通道越不均衡）"
    ),
    'chroma_shift': (
        "Chroma Shift Score  ||mean(a*, b*)||  or  ||mean(rg) - (1/3,1/3)||\n"
        "值越高 = 均值色度向量偏离中性越远（色偏越严重）"
    ),
    'gradient_dir_entropy': (
        "1 - Gradient Direction Entropy  H(θ) / log(n_bins)\n"
        "展示 1-score：值越高 = 梯度方向越集中（强方向性，运动模糊候选特征越明显）\n"
        "注意：场景自身有强方向结构（建筑/栅栏）时可能误判，建议与 local_contrast 联合判断"
    ),
    'laplacian_var': (
        "1 - Laplacian Variance  Var(∇²Y)\n"
        "展示 1-score：值越高 = 拉普拉斯方差越低（图像越模糊）\n"
        "注意：噪声类污染会引入高频，可能使该指标偏大，不代表更清晰"
    ),
    'tenengrad': (
        "1 - Tenengrad  mean(Gx²+Gy²) [幅值 > η 百分位像素的逐像素均值]\n"
        "展示 1-score：值越高 = 平均梯度能量越低（图像越模糊）\n"
        "注意：噪声类同上；η 阈值（百分位）影响结果，默认 50"
    ),
    'brenner': (
        "1 - Brenner Gradient  mean((Y(x+2)-Y(x))²)  值域 [0,1]\n"
        "展示 1-score：值越高 = 逐像素平均 Brenner 梯度越小（图像越模糊）\n"
        "注意：步长 2 对中低频模糊更敏感，对噪声鲁棒性略优于单步差分"
    ),
    'wavelet_hf': (
        "1 - Wavelet High-Freq Energy  mean(HH²+LH²+HL²)  值域 [0, 0.0625]\n"
        "展示 1-score：值越高 = 逐系数平均高频能量越低（图像越模糊）\n"
        "注意：噪声类可能抬高高频子带能量，使该指标偏大"
    ),
    'edge_dir_anisotropy': (
        "Edge Direction Anisotropy  1 - H_norm(mag-weighted hist)  值域 [0, 1]\n"
        "值越高 = 强边缘方向越集中（运动模糊候选）\n"
        "直方图按梯度幅值加权；场景本身有强单一方向结构时可能误判"
    ),
    'fft_dir_concentration': (
        "FFT Direction Concentration  (max(p) - 1/K) / (1 - 1/K)  值域 [0, 1]\n"
        "值越高 = 频谱能量方向越集中（运动模糊候选）\n"
        "对周期性运动模糊灵敏；DC 低频圆盘（dc_radius 以内）被排除"
    ),
    'mad_sigma': (
        "MAD Sigma  median(|Y - blur(Y)|) / 0.6745\n"
        "值越高 = 高通残差越大（全图加性噪声 σ 越大）\n"
        "已修复：blur 改用 uniform_filter(reflect) 避免边界虚假高估"
    ),
    'flat_patch_sigma': (
        "Flat-Patch Sigma  sqrt(min_patch var(Y))  值域 ≥ 0\n"
        "值越高 = 最平坦区域噪声 σ 越大（纹理/边缘等高方差区域不影响结果）\n"
        "已修复：局部均值/方差用 uniform_filter(reflect) 计算"
    ),
    'hf_lift_ratio': (
        "HF Lift Ratio  ΣP[r_low≤r_norm≤r_high] / ΣP  值域 [0, 1]\n"
        "值越高 = 高频环带能量占比越大（白噪声抬升越明显）\n"
        "注意：total_energy 含 DC，亮图相同噪声下得分略低"
    ),
    'dark_region_sigma': (
        "Dark-Region Sigma  MAD σ 仅在 Y < dark_thresh 像素上估计\n"
        "值越高 = 暗部噪声越强（低照度/impulse/shot noise 敏感）\n"
        "已修复：blur 改用 uniform_filter(reflect)"
    ),
    'hi_clip': (
        "Hi-Clip Ratio  1/N Σ 1(Y ≥ 1−ε_hi)  值域 [0, 1]\n"
        "值越高 = 高光饱和像素比例越大（过曝/brightness 污染特征越明显）"
    ),
    'lo_clip': (
        "Lo-Clip Ratio  1/N Σ 1(Y ≤ ε_lo)  值域 [0, 1]\n"
        "值越高 = 暗部饱和像素比例越大（欠曝/低照度污染特征越明显）"
    ),
    'mean_luma': (
        "Mean Luminance  μ_Y  值域 [0, 1]\n"
        "值越高 = 平均亮度越高（brightness / fog / snow 等亮度抬升污染）"
    ),
    'median_luma': (
        "Median Luminance  median(Y)  值域 [0, 1]\n"
        "对极端像素鲁棒；值越高 = 整体亮度越高"
    ),
    'hist_skewness': (
        "Abs Histogram Skewness  |skew(Y)|  值域 ≥ 0\n"
        "值越高 = 亮度分布偏斜越严重（过曝→负偏，欠曝→正偏，均以绝对值展示）"
    ),
    'hist_kurtosis': (
        "Histogram Kurtosis  E[(Y-μ)⁴/σ⁴]  正态≈3\n"
        "值越高 = 亮度分布越尖峭/集中（饱和/剪切区域集中时升高）"
    ),
    'entropy_norm': (
        "1 - Norm Entropy  H(Y)/log(K)  值域 [0, 1]\n"
        "展示 1-score：值越高 = 亮度分布越集中（过曝/欠曝候选）"
    ),
    'dynamic_range': (
        "1 - Dynamic Range  P_99(Y) - P_1(Y)  值域 [0, 1]\n"
        "展示 1-score：值越高 = 有效动态范围越窄（雾化/低对比度污染特征越明显）"
    ),
    'rgb_hi_clip': (
        "RGB Hi-Clip Ratio  mean_c(1/N Σ 1(I_c ≥ 1−ε))  值域 [0, 1]\n"
        "值越高 = RGB 通道高光饱和越严重（可检测单通道色偏导致的通道饱和）"
    ),
}

# 标记哪些方法原始得分方向为"低分=污染严重"。
# 在绘图和控制台输出中，这些方法统一展示 (1 - score)，
# 使所有方法对齐到"值越高 = 污染特征越明显"的统一语义。
_SORT_ASCENDING = {
    'dark_channel':    False,   # 原始高=危，直接展示
    'hsv':             True,    # 原始低=危，展示 1 - score
    'local_contrast':  True,    # 原始低=危，展示 1 - score
    'edge_visibility': True,    # 原始低=危，展示 1 - score
    'blockiness':      False,   # 原始高=危（ratio/diff 越大越严重），直接展示
    'ringing':         False,   # 原始高=危（近边缘振荡越强越严重），直接展示
    'gray_world':      False,   # 原始高=危（色偏越大越严重），直接展示
    'white_patch':     False,   # 原始高=危（高光色偏越大越严重），直接展示
    'chroma_shift':         False,   # 原始高=危（均值色度偏离中性越远越严重），直接展示
    'gradient_dir_entropy': True,    # 原始低=危（方向越集中越可疑），展示 1 - score
    'laplacian_var':        True,    # 原始低=危（越模糊越小），展示 1 - score
    'tenengrad':            True,    # 原始低=危（越模糊越小），展示 1 - score
    'brenner':              True,    # 原始低=危（越模糊越小），展示 1 - score
    'wavelet_hf':           True,    # 原始低=危（越模糊越小），展示 1 - score
    'edge_dir_anisotropy':  False,   # 原始高=危（方向越集中越可疑），直接展示
    'fft_dir_concentration': False,  # 原始高=危（频谱越集中越可疑），直接展示
    'mad_sigma':         False,  # 原始高=危（噪声 σ 越大越严重），直接展示
    'flat_patch_sigma':  False,  # 原始高=危（最平坦区域噪声越大越严重），直接展示
    'hf_lift_ratio':     False,  # 原始高=危（高频抬升越高=噪声越强），直接展示
    'dark_region_sigma': False,  # 原始高=危（暗部噪声越大越严重），直接展示
    'hi_clip':       False,   # 原始高=危（高光饱和比例越高越严重），直接展示
    'lo_clip':       False,   # 原始高=危（暗部饱和比例越高越严重），直接展示
    'mean_luma':     False,   # 原始高=危（亮度均值越高=过曝特征越明显），直接展示
    'median_luma':   False,   # 原始高=危（中位亮度越高=过曝特征越明显），直接展示
    'hist_skewness': False,   # 原始高=危（绝对偏度越大=分布偏斜越严重），直接展示
    'hist_kurtosis': False,   # 原始高=危（峰度越高=分布越尖峭），直接展示
    'entropy_norm':  True,    # 原始低=危（熵越低=分布越集中=曝光异常），展示 1-score
    'dynamic_range': True,    # 原始低=危（动态范围越窄=对比度越差），展示 1-score
    'rgb_hi_clip':   False,   # 原始高=危（通道饱和越严重），直接展示
}


def _compute_weighted_score(score_accum: dict, ws_cfg: dict) -> list:
    """
    对各基础方法的逐图得分进行归一化、加权合并，生成每个 group 的综合得分。

    归一化策略
    ----------
    1. 对每个基础方法，在 **全数据集** 范围（所有污染类型 × 所有图片）
       做 min-max 归一化到 [0, 1]
    2. 若该方法为 ascending（低分=污染严重），取补集 1 - normalized，
       统一使"值越高 = 污染特征越强"
    3. 在图片级别按权重加权求和，最后对每种污染类型取均值

    参数
    ----
    score_accum : {方法名: {污染名称: [逐图得分列表]}}，由 process_method 在遍历时积累
    ws_cfg      : weighted_score 的配置字典，包含：
        'mode'    : 'mean'（等权） | 'weighted'（用 'weights' 字典）
        'weights' : {方法名: float}，仅 mode='weighted' 时有效
        'groups'  : list[list[str]]，每个子列表为一个 group，产生一张图

    返回
    ----
    group_results : list of (group, weight_dict, per_corr_weighted)
        - group             : list[str]，该 group 包含的方法名列表
        - weight_dict       : {方法名: 归一化后的权重}
        - per_corr_weighted : {污染名称: 加权均分}，值域约为 [0, 1]
    """
    groups      = ws_cfg.get('groups', [list(score_accum.keys())])
    mode        = ws_cfg.get('mode', 'mean')
    weights_cfg = ws_cfg.get('weights', {})

    # 检查所有 group 中的方法均已在 score_accum 中
    all_methods_needed = {m for group in groups for m in group}
    missing = all_methods_needed - set(score_accum.keys())
    if missing:
        raise ValueError(
            f"weighted_score 依赖的方法 {missing} 尚未计算，"
            f"请将其加入 process_method 的 method 列表"
        )

    # ── 第一步：对每个基础方法做全局 min-max 归一化 ──────────────────────────
    # 维持原始列表顺序，以便后续按图片级别加权
    normalized = {}   # {方法名: {污染名称: np.ndarray(逐图归一化得分)}}
    for m in all_methods_needed:
        # 将该方法所有污染/所有图片的得分拼接为一个大数组
        corr_names = list(score_accum[m].keys())
        all_scores = np.concatenate([score_accum[m][c] for c in corr_names]).astype(np.float32)

        vmin, vmax = all_scores.min(), all_scores.max()
        norm_all   = (all_scores - vmin) / (vmax - vmin + 1e-8)

        # 若该方法升序（低分=污染严重），翻转使高分=危险
        if _SORT_ASCENDING.get(m, False):
            norm_all = 1.0 - norm_all

        # 按污染类型切分回各自的数组
        offset = 0
        normalized[m] = {}
        for c in corr_names:
            n = len(score_accum[m][c])
            normalized[m][c] = norm_all[offset: offset + n]
            offset += n

    # ── 第二步：对每个 group，逐图加权求和，再按污染取均分 ───────────────────
    group_results = []
    for group in groups:
        # 计算归一化权重
        if mode == 'weighted':
            raw_w = {m: float(weights_cfg.get(m, 1.0)) for m in group}
        else:   # 'mean'：等权
            raw_w = {m: 1.0 for m in group}
        total_w  = sum(raw_w.values())
        weight_d = {m: raw_w[m] / total_w for m in group}

        # 所有方法共享相同的污染类型列表（以 group[0] 为基准）
        corr_names = list(normalized[group[0]].keys())
        per_corr_weighted = {}
        for c in corr_names:
            # 逐图加权：shape = (N_images,)
            img_weighted = sum(
                weight_d[m] * normalized[m][c] for m in group
            )
            per_corr_weighted[c] = float(img_weighted.mean())

        group_results.append((group, weight_d, per_corr_weighted))

    return group_results


def _plot_heatmap(per_corr_sev: dict, method: str, cfg: dict):
    """
    热力图：X 轴 = severity，Y 轴 = corruption type，颜色 = 展示得分。

    参数
    ----
    per_corr_sev : {corr_name: {severity: mean_score}}（存储原始得分）
    method       : 方法名称字符串
    cfg          : 该方法实际使用的参数字典
    """
    ascending   = _SORT_ASCENDING.get(method, False)
    corr_names  = list(per_corr_sev.keys())
    severities  = sorted({sev for d in per_corr_sev.values() for sev in d})

    # 构建得分矩阵 (rows=corruption, cols=severity)
    matrix = np.full((len(corr_names), len(severities)), np.nan)
    for i, c in enumerate(corr_names):
        for j, s in enumerate(severities):
            v = per_corr_sev[c].get(s, np.nan)
            matrix[i, j] = (1.0 - v) if ascending else v

    # 按行均值降序排列（污染特征最强的排最上）
    row_means      = np.nanmean(matrix, axis=1)
    order          = np.argsort(row_means)[::-1]
    matrix         = matrix[order]
    corr_names_s   = [corr_names[i] for i in order]

    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    fig, ax = plt.subplots(figsize=(max(5, len(severities) * 1.4),
                                    max(5, len(corr_names_s) * 0.55 + 2)))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # 格内数值标注
    for i in range(len(corr_names_s)):
        for j in range(len(severities)):
            v = matrix[i, j]
            if not np.isnan(v):
                brightness = (v - vmin) / (vmax - vmin + 1e-8)
                txt_color  = 'white' if brightness > 0.65 or brightness < 0.2 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=8, color=txt_color)

    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f'sev {s}' for s in severities], fontsize=10)
    ax.set_yticks(range(len(corr_names_s)))
    ax.set_yticklabels(corr_names_s, fontsize=9)
    ax.set_xlabel("Severity", fontsize=11)
    ax.set_ylabel("Corruption Type", fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    score_str = '1 - Score' if ascending else 'Score'
    desc      = _METHOD_DESC.get(method, method)
    extra     = "  |  " + ",  ".join(f"{k}={v}" for k, v in cfg.items())
    ax.set_title(f"[{method}]  {score_str}\n{desc}{extra}", fontsize=8)
    plt.tight_layout()
    plt.show()


def _plot_weighted_heatmap(group_results_sev: list):
    """
    热力图版 weighted_score：X 轴 = severity，Y 轴 = corruption type。

    参数
    ----
    group_results_sev : list of (group, weight_dict, per_corr_sev_weighted)
        per_corr_sev_weighted : {(corr_name, severity): weighted_score}
    """
    for g_idx, (group, weight_d, per_cs) in enumerate(group_results_sev):
        # 提取所有 corruption / severity
        corr_set = sorted({k[0] for k in per_cs})
        sev_set  = sorted({k[1] for k in per_cs})

        matrix       = np.full((len(corr_set), len(sev_set)), np.nan)
        corr_idx_map = {c: i for i, c in enumerate(corr_set)}
        sev_idx_map  = {s: j for j, s in enumerate(sev_set)}
        for (c, s), v in per_cs.items():
            matrix[corr_idx_map[c], sev_idx_map[s]] = v

        # 按行均值降序
        row_means    = np.nanmean(matrix, axis=1)
        order        = np.argsort(row_means)[::-1]
        matrix       = matrix[order]
        corr_names_s = [corr_set[i] for i in order]

        vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

        fig, ax = plt.subplots(figsize=(max(5, len(sev_set) * 1.4),
                                        max(5, len(corr_names_s) * 0.55 + 2.5)))
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        for i in range(len(corr_names_s)):
            for j in range(len(sev_set)):
                v = matrix[i, j]
                if not np.isnan(v):
                    brightness = (v - vmin) / (vmax - vmin + 1e-8)
                    txt_color  = 'white' if brightness > 0.65 or brightness < 0.2 else 'black'
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=8, color=txt_color)

        ax.set_xticks(range(len(sev_set)))
        ax.set_xticklabels([f'sev {s}' for s in sev_set], fontsize=10)
        ax.set_yticks(range(len(corr_names_s)))
        ax.set_yticklabels(corr_names_s, fontsize=9)
        ax.set_xlabel("Severity", fontsize=11)
        ax.set_ylabel("Corruption Type", fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

        methods = " + ".join(group)
        w_str   = ",  ".join(f"{m}: {w:.2f}" for m, w in weight_d.items())
        ax.set_title(
            f"[weighted_score  Group {g_idx + 1}]  {methods}\n"
            f"权重: {w_str}\n"
            f"归一化加权综合得分（高危 = 红色）",
            fontsize=9
        )
        plt.tight_layout()
        plt.show()

        # 控制台摘要（按 severity 展示）
        print(f"\n[weighted_score  Group {g_idx + 1}]  {methods}")
        print(f"  权重: {w_str}")
        header = f"  {'Corruption':<22}" + "".join(f"  sev{s:>2}" for s in sev_set)
        print(header)
        print("  " + "-" * (22 + 8 * len(sev_set)))
        for c in corr_names_s:
            row = f"  {c:<22}"
            for s in sev_set:
                v = per_cs.get((c, s), float('nan'))
                row += f"  {v:>6.4f}"
            print(row)


def _plot_weighted_bar(group_results: list):
    """
    为 weighted_score 的每个 group 绘制一张柱状图。

    参数
    ----
    group_results : _compute_weighted_score 的返回值
        list of (group, weight_dict, per_corr_weighted)
    """
    for g_idx, (group, weight_d, per_corr) in enumerate(group_results):
        names  = list(per_corr.keys())
        values = [per_corr[n] for n in names]

        # 综合得分越高 = 污染特征越强 → 降序，高危排左
        order    = np.argsort(values)[::-1]
        names_s  = [names[i]  for i in order]
        values_s = [values[i] for i in order]

        # 颜色映射：高危红色
        nv     = np.array(values_s)
        nv_norm = (nv - nv.min()) / (nv.max() - nv.min() + 1e-8)
        colors  = [mplcm.RdYlGn_r(v) for v in nv_norm]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(names_s, values_s, color=colors,
                      edgecolor='black', linewidth=0.6)

        # 柱顶标注数值
        gap = (max(values_s) - min(values_s)) * 0.01
        for bar, val in zip(bars, values_s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + gap,
                f"{val:.4f}",
                ha='center', va='bottom', fontsize=7.5
            )

        ax.set_xlabel("Corruption Type", fontsize=11)
        ax.set_ylabel("Weighted Score  (归一化加权综合得分)", fontsize=11)

        # 标题：显示 group 编号、包含方法及各方法权重
        w_str   = ",  ".join(f"{m}: {w:.2f}" for m, w in weight_d.items())
        methods = " + ".join(group)
        ax.set_title(
            f"[weighted_score  Group {g_idx + 1}]  {methods}\n"
            f"权重: {w_str}\n"
            f"归一化后加权，值越高 = 污染特征越强（高危柱排左，红色）",
            fontsize=9
        )
        plt.xticks(rotation=35, ha='right', fontsize=9)
        plt.tight_layout()
        plt.show()

        # 控制台摘要
        print(f"\n[weighted_score  Group {g_idx + 1}]  {methods}")
        print(f"  权重: {w_str}")
        print(f"  {'Corruption':<22} {'WeightedScore':>15}")
        print("  " + "-" * 40)
        for name, val in zip(names_s, values_s):
            print(f"  {name:<22} {val:>15.6f}")


def _plot_bar(per_corr_scores: dict, method: str, cfg: dict):
    """
    绘制单个方法的柱状图。

    统一语义：所有方法均以"值越高 = 污染特征越明显"展示。
    对 _SORT_ASCENDING=True 的方法（hsv / local_contrast / edge_visibility），
    绘图时取 (1 - 原始得分)，并在 Y 轴标签和柱顶数值处标注 "1 - score"。

    参数
    ----
    per_corr_scores : {污染名称: 均分} 字典（存储原始得分）
    method          : 方法名称字符串
    cfg             : 该方法实际使用的参数字典（显示在标题中）
    """
    ascending  = _SORT_ASCENDING.get(method, False)
    names      = list(per_corr_scores.keys())
    raw_values = np.array([per_corr_scores[n] for n in names], dtype=np.float64)

    # 对 ascending 方法取补集，统一为"高=危"
    display_values = (1.0 - raw_values) if ascending else raw_values

    # 统一降序排列：污染特征最强的排最左
    order    = np.argsort(display_values)[::-1]
    names_s  = [names[i]       for i in order]
    disp_s   = [display_values[i] for i in order]

    # 颜色映射：RdYlGn_r，值越高越红（危险程度越高）
    nv      = np.array(disp_s)
    nv_norm = (nv - nv.min()) / (nv.max() - nv.min() + 1e-8)
    colors  = [mplcm.RdYlGn_r(v) for v in nv_norm]

    stat_label = cfg.get('stat', '-')
    # Y 轴标签：ascending 方法注明展示的是 1-score
    y_label = (f"1 - Score  (stat={stat_label})"
               if ascending else f"Score  (stat={stat_label})")

    fig, ax = plt.subplots(figsize=(14, 5))
    bars    = ax.bar(names_s, disp_s, color=colors,
                     edgecolor='black', linewidth=0.6)

    # 柱顶标注展示值
    gap = (disp_s[0] - disp_s[-1]) * 0.01 if len(disp_s) > 1 else 0.005
    for bar, val in zip(bars, disp_s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + gap,
            f"{val:.4f}",
            ha='center', va='bottom', fontsize=7.5
        )

    ax.set_xlabel("Corruption Type", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    desc  = _METHOD_DESC.get(method, method)
    extra = "  |  " + ",  ".join(f"{k}={v}" for k, v in cfg.items())
    ax.set_title(f"[{method}]  {desc}{extra}", fontsize=9)
    plt.xticks(rotation=35, ha='right', fontsize=9)
    plt.tight_layout()
    plt.show()


def process_method(dataset, method=None, method_config=None,
                   plot: str = 'bar'):
    """
    对 dataset 中每张图片用指定方法打分，统计各污染类型的均分，并绘图。

    参数
    ----
    dataset       : 由 get_dataset() 返回的 ConcatDataset
                    样本格式: (img_tensor, corr_idx, corr_name, severity)

    method        : str 或 list[str]，要运行的方法，可选：
                      'dark_channel'         — 暗通道先验（值高 → 雾/亮度污染）
                      'hsv'                  — HSV 饱和度（值低 → 雾/雪褪色）
                      'local_contrast'       — 局部对比度（值低 → 雾/失焦模糊）
                      'edge_visibility'      — 边缘可见性（值低 → 雾/模糊）
                      'gradient_dir_entropy' — 梯度方向熵（值低 → 强方向性/运动模糊）
                      'laplacian_var'          — 拉普拉斯方差（值低 → 图像越模糊）
                      'tenengrad'              — Tenengrad Sobel 梯度能量（值低 → 越模糊）
                      'brenner'                — Brenner 步长2梯度（值低 → 越模糊）
                      'wavelet_hf'             — Haar 小波高频子带能量（值低 → 越模糊）
                      'edge_dir_anisotropy'    — 边缘方向各向异性（值高 → 方向越集中/运动模糊）
                      'fft_dir_concentration'  — 频谱方向集中度（值高 → 频谱越各向异性/运动模糊）
                      'mad_sigma'              — MAD 高通噪声估计（值高 → 噪声 σ 越大）
                      'flat_patch_sigma'       — 平坦块噪声估计（值高 → 背景噪声越大）
                      'hf_lift_ratio'          — 频谱高频抬升（值高 → 白噪声越强）
                      'dark_region_sigma'      — 暗部区域噪声（值高 → 暗部噪声越大）
                      'hi_clip'                — 高光饱和比例（值高 → 过曝越严重）
                      'lo_clip'                — 暗部饱和比例（值高 → 欠曝越严重）
                      'mean_luma'              — 亮度均值（值高 → 整体越亮）
                      'median_luma'            — 亮度中位数（值高 → 整体越亮，对极端值鲁棒）
                      'hist_skewness'          — 绝对偏度（值高 → 亮度分布偏斜越严重）
                      'hist_kurtosis'          — 峰度（值高 → 亮度分布越尖峭/集中）
                      'entropy_norm'           — 归一化熵（值低 → 分布越集中/曝光异常）
                      'dynamic_range'          — 有效动态范围（值低 → 对比度越差/雾化）
                      'rgb_hi_clip'            — RGB 通道饱和（值高 → 通道过曝越严重）
                    默认 ['dark_channel', 'hsv']

    method_config : dict[str, dict]，各方法的超参数，未指定方法使用内置默认值

    plot          : 绘图模式，可选：
                      'bar'     — 仅输出柱状图（X=corruption type, Y=score）
                      'heatmap' — 仅输出热力棋盘图（X=severity, Y=corruption type）
                      'both'    — 同时输出柱状图与热力图
                    默认 'bar'

    返回
    ----
    results : dict[str, dict[str, float]]
        {方法名: {污染名称: 该污染类型所有图片的均分}}
    """
    if method is None:
        method = ['dark_channel', 'hsv']
    if isinstance(method, str):
        method = [method]

    # 各方法的内置默认参数
    default_configs = {
        'dark_channel':    {'omega': 5,  'stat': 'mean'},
        'hsv':             {'stat': 'mean'},
        'local_contrast':  {'mode': 'laplacian', 'omega': 2, 'stat': 'mean'},
        'edge_visibility': {'mode': 'gradient', 'sigma': 1.0,
                            'low_thr': 0.04, 'high_thr': 0.10, 'stat': 'mean'},
        'blockiness':      {'block_size': 8, 'mode': 'ratio', 'stat': 'mean'},
        'ringing':         {'edge_thresh': 0.15, 'radius': 3,
                            'mode': 'ratio', 'stat': 'mean'},
        'gray_world':      {'mode': 'l2'},
        'white_patch':     {'stat': 'q99', 'mode': 'l2'},
        'chroma_shift':         {'mode': 'lab', 'min_brightness': 0.02},
        'gradient_dir_entropy': {'n_bins': 36, 'mag_pct': 50.0, 'smooth_sigma': 1.0},
        'laplacian_var':        {},
        'tenengrad':            {'eta_percentile': 50.0},
        'brenner':              {},
        'wavelet_hf':           {},
        'edge_dir_anisotropy':  {'n_bins': 18, 'mag_pct': 70.0},
        'fft_dir_concentration': {'n_bins': 18, 'dc_radius': 2.0},
        'mad_sigma':         {},
        'flat_patch_sigma':  {'patch_size': 7},
        'hf_lift_ratio':     {'r_low': 0.1, 'r_high': 0.5},
        'dark_region_sigma': {'dark_thresh': 0.2},
        'hi_clip':       {'eps_hi': 0.02},
        'lo_clip':       {'eps_lo': 0.02},
        'mean_luma':     {},
        'median_luma':   {},
        'hist_skewness': {},
        'hist_kurtosis': {},
        'entropy_norm':  {'num_bins': 256},
        'dynamic_range': {'p_low': 1.0, 'p_high': 99.0},
        'rgb_hi_clip':   {'eps_hi': 0.02},
    }
    if method_config is None:
        method_config = {}
    # 用户配置覆盖默认值（浅合并）
    cfgs = {
        m: {**default_configs.get(m, {}), **method_config.get(m, {})}
        for m in method
    }

    # weighted_score 不是基础评分方法，单独提取其配置，不进入打分循环
    ws_cfg       = cfgs.pop('weighted_score', None)
    base_methods = [m for m in method if m != 'weighted_score']

    # 校验基础方法名合法性
    for m in base_methods:
        if m not in _SCORE_REGISTRY:
            raise ValueError(
                f"未知方法 '{m}'，已注册方法: {list(_SCORE_REGISTRY.keys())}"
            )

    need_heatmap = plot in ('heatmap', 'both')
    need_bar     = plot in ('bar',     'both')

    # ── 单次遍历数据集，为所有基础方法累积逐图得分（避免重复 I/O）──────────
    # score_accum[方法名][污染名]              = [score, ...]
    # score_accum_sev[方法名][污染名][severity] = [score, ...]
    score_accum     = {m: defaultdict(list) for m in base_methods}
    score_accum_sev = {m: defaultdict(lambda: defaultdict(list))
                       for m in base_methods}

    print(f"正在计算得分，方法: {base_methods} ...")
    for img_tensor, corr_idx, corr_name, severity in tqdm(dataset, ncols=80):
        img_np = img_tensor.numpy().transpose(1, 2, 0)   # (C,H,W) → (H,W,C)
        sev    = int(severity)
        for m in base_methods:
            score = _SCORE_REGISTRY[m](img_np, **cfgs[m])
            score_accum[m][corr_name].append(score)
            if need_heatmap:
                score_accum_sev[m][corr_name][sev].append(score)

    # ── 汇总各基础方法污染均分，绘图，打印控制台摘要 ─────────────────────────
    results = {}
    for m in base_methods:
        per_corr = {
            name: float(np.mean(scores))
            for name, scores in score_accum[m].items()
        }
        results[m] = per_corr

        if need_bar:
            _plot_bar(per_corr, m, cfgs[m])

        if need_heatmap:
            per_corr_sev = {
                c: {sev: float(np.mean(score_accum_sev[m][c][sev]))
                    for sev in score_accum_sev[m][c]}
                for c in score_accum_sev[m]
            }
            _plot_heatmap(per_corr_sev, m, cfgs[m])

        ascending = _SORT_ASCENDING.get(m, False)
        display_items = sorted(
            [(name, (1.0 - val) if ascending else val)
             for name, val in per_corr.items()],
            key=lambda x: x[1], reverse=True
        )
        col_title = (f"1 - Score ({cfgs[m].get('stat', '-')})"
                     if ascending else f"Score ({cfgs[m].get('stat', '-')})")
        print(f"\n[{m}]  {col_title}  (高 = 污染特征越明显)")
        print(f"  {'Corruption':<22} {col_title:>22}")
        print("  " + "-" * 46)
        for name, val in display_items:
            print(f"  {name:<22} {val:>22.6f}")

    # ── weighted_score：全局归一化 + 逐图加权 + 按 group 绘图 ────────────────
    if ws_cfg is not None:
        print("\n正在计算 weighted_score（全局归一化 + 逐图加权）...")
        score_accum_np = {
            m: {c: np.array(v, dtype=np.float32)
                for c, v in score_accum[m].items()}
            for m in base_methods
        }
        group_results = _compute_weighted_score(score_accum_np, ws_cfg)
        if need_bar:
            _plot_weighted_bar(group_results)

        if need_heatmap:
            # 将 (corr_name, severity) 展开为复合键，复用同一归一化逻辑
            score_accum_cs_np = {
                m: {(c, sev): np.array(score_accum_sev[m][c][sev], dtype=np.float32)
                    for c in score_accum_sev[m]
                    for sev in score_accum_sev[m][c]}
                for m in base_methods
            }
            group_results_sev = _compute_weighted_score(score_accum_cs_np, ws_cfg)
            _plot_weighted_heatmap(group_results_sev)

        # 将每个 group 的结果写入 results
        results['weighted_score'] = {
            f"group{i + 1}": gr[2]
            for i, gr in enumerate(group_results)
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 入口示例
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 加载全部 15 种污染类型的 severity 1~5 数据
    ds = get_dataset(severities=(1, 2, 3, 4, 5), corruptions=CORRUPTION_LIST)

    # 各方法超参数配置
    METHOD_CONFIG = {
        'dark_channel': {
            'omega': 5,             # 局部窗口半径，对应 11×11 窗口
            'stat':  'mean',        # 可选: 'mean' | 'median' | 'q75' | 'q90'
        },
        'hsv': {
            'stat': 'mean',         # 可选: 'mean' | 'median' | 'q25' | 'q10'
        },
        'local_contrast': {
            'mode':  'laplacian',   # 可选: 'laplacian' | 'rms'
            'omega': 2,             # 仅 rms 模式有效，对应 5×5 窗口
            'stat':  'mean',        # 可选: 'mean' | 'median' | 'q75' | 'q90'
        },
        'edge_visibility': {
            'mode':     'gradient', # 可选: 'gradient' | 'canny'
            'sigma':    1.0,        # 仅 canny：高斯预平滑标准差
            'low_thr':  0.04,       # 仅 canny：弱边缘阈值
            'high_thr': 0.10,       # 仅 canny：强边缘阈值（须 > low_thr）
            'stat':     'mean',     # 仅 gradient：'mean'|'median'|'q75'|'q90'
        },
        'blockiness': {
            'block_size': 8,       # DCT 块大小，JPEG 标准为 8
            'mode':       'ratio', # 可选: 'ratio'（边界/非边界差分比） | 'diff'（差值）
            'stat':       'mean',  # 可选: 'mean' | 'median' | 'q75' | 'q90'
        },
        'ringing': {
            'edge_thresh': 0.15,   # Sobel 梯度幅值（归一化）阈值，决定"强边缘"
                                   # 可选范围 0.10–0.25；过低噪声误判，过高边缘稀疏
            'radius':      3,      # 边缘膨胀半径（像素），决定近边缘区域宽度
                                   # CIFAR 32×32 建议 2–4
            'mode':       'ratio', # 可选: 'ratio'（近/远边缘 Lap 比值）| 'abs'（绝对量）
            'stat':       'mean',  # 可选: 'mean' | 'median' | 'q75' | 'q90'
        },
        'gray_world': {
            'mode': 'l2',          # 可选: 'l2'（RMS偏差）| 'l1'（MAD）| 'max_range'（通道极差）
        },
        'white_patch': {
            'stat': 'q99',         # 亮端代表值：'max'（纯最大值）| 'q99'（推荐）| 'q95'（更保守）
            'mode': 'l2',          # 可选: 'l2'（RMS偏差）| 'l1'（MAD）| 'max_range'（通道极差）
        },
        'chroma_shift': {
            # 'rg'        : rg色度均值距中性点(1/3,1/3)的L2距离（计算快）
            # 'rg_spread' : rg色度分布离散度（雾天降低，_SORT_ASCENDING=True）
            # 'lab'       : CIE Lab ab平面均值色度向量范数（感知均匀，推荐）
            # 'lab_chroma': 每像素彩度C*均值（雾天降低，_SORT_ASCENDING=True）
            'mode':           'lab',   # 推荐 'lab' 或 'rg'
            'min_brightness': 0.02,    # rg/rg_spread 模式：过滤近黑像素的亮度阈值
        },
        'gradient_dir_entropy': {
            # 梯度方向直方图熵（运动模糊检测）
            # n_bins       : 方向直方图区间数（36 = 每5°一个bin）
            # mag_pct      : 仅使用幅值高于此百分位的像素（滤噪）
            # smooth_sigma : 高斯预平滑 sigma（0 = 不平滑）
            'n_bins':       36,
            'mag_pct':      50.0,
            'smooth_sigma': 1.0,
        },
        'weighted_score': {
            # mode='mean'    : 所有方法等权合并
            # mode='weighted': 按 weights 字典中的值分配权重
            'mode': 'weighted',

            # 各基础方法的原始权重（自动归一化为权重之和=1）
            'weights': {
                'dark_channel':         1.0,   # 暗通道：雾化/亮度指标
                'hsv':                  1.0,   # 饱和度：辅助指标
                'local_contrast':       1.0,   # 局部对比度：模糊/雾化指标
                'edge_visibility':      1.0,   # 边缘可见性：辅助指标
                'blockiness':           1.0,   # 块效应：JPEG 压缩伪影指标
                'ringing':              1.0,   # 振铃：强边缘附近过冲/回摆指标
                'gray_world':           1.0,   # Gray-World 偏离：均值色偏指标
                'white_patch':          1.0,   # White-Patch 偏离：高光色偏指标
                'chroma_shift':         1.0,   # 色度向量偏移：Lab/rg 空间色偏指标
                'gradient_dir_entropy': 1.0,   # 梯度方向熵：运动模糊方向性指标
                'laplacian_var':        1.0,   # 拉普拉斯方差：模糊度指标
                'tenengrad':            1.0,   # Tenengrad：Sobel 梯度能量模糊度指标
                'brenner':              1.0,   # Brenner 梯度：模糊度指标
                'wavelet_hf':           1.0,   # Haar 小波高频能量：模糊度指标
                'edge_dir_anisotropy':  1.0,   # 边缘方向各向异性：运动模糊方向性指标
                'fft_dir_concentration': 1.0,  # 频谱方向集中度：运动模糊方向性指标
                'mad_sigma':         1.0,   # MAD 全局噪声估计
                'flat_patch_sigma':  1.0,   # 平坦块噪声估计
                'hf_lift_ratio':     1.0,   # 频谱高频抬升：白噪声指标
                'dark_region_sigma': 1.0,   # 暗部噪声估计
                'hi_clip':       1.0,   # 高光饱和比例：过曝指标
                'lo_clip':       1.0,   # 暗部饱和比例：欠曝指标
                'mean_luma':     1.0,   # 亮度均值：亮度偏移指标
                'median_luma':   1.0,   # 亮度中位数：鲁棒亮度指标
                'hist_skewness': 1.0,   # 绝对偏度：分布偏斜指标
                'hist_kurtosis': 1.0,   # 峰度：分布尖峭指标
                'entropy_norm':  1.0,   # 归一化熵：分布集中度指标
                'dynamic_range': 1.0,   # 有效动态范围：对比度指标
                'rgb_hi_clip':   1.0,   # RGB 通道饱和：通道级过曝指标
            },

            # 每个子列表为一个 group，各自产生一张柱状图
            # group1：全量指标综合得分
            # group2：雾化/色偏相关指标（暗通道 + 饱和度 + 色偏）
            # group3：模糊相关指标（对比度 + 边缘）
            # group4：压缩相关指标（块效应 + 振铃）
            # group5：四种模糊度量（Laplacian/Tenengrad/Brenner/小波高频）
            # group6：两种方向性模糊度量（边缘方向各向异性 + 频谱方向集中度）
            # group7：曝光剪切指标（高光/暗部/RGB 通道饱和）
            # group8：亮度分布指标（均值/中位数/偏度/峰度/熵/动态范围）
            # group9：噪声度量（MAD/平坦块/高频抬升/暗部噪声）
            'groups': [
                ['dark_channel', 'hsv', 'local_contrast', 'edge_visibility'],
                ['dark_channel', 'hsv'],
                ['local_contrast', 'edge_visibility'],
                ['blockiness', 'ringing'],
                ['gray_world', 'white_patch', 'chroma_shift'],
                ['laplacian_var', 'tenengrad', 'brenner', 'wavelet_hf'],
                ['edge_dir_anisotropy', 'fft_dir_concentration'],
                ['hi_clip', 'lo_clip', 'rgb_hi_clip'],
                ['mean_luma', 'median_luma', 'hist_skewness',
                 'hist_kurtosis', 'entropy_norm', 'dynamic_range'],
                ['mad_sigma', 'flat_patch_sigma',
                 'hf_lift_ratio', 'dark_region_sigma'],
            ],
        },
    }

    # 运行全部十种基础方法 + weighted_score，每种方法独立输出图表及控制台摘要
    # plot 可选：'bar'（柱状图）| 'heatmap'（热力棋盘图）| 'both'（两者都输出）
    results = process_method(
        ds,
        method=['dark_channel', 'hsv', 'local_contrast', 'edge_visibility',
                'blockiness', 'ringing', 'gray_world', 'white_patch',
                'chroma_shift', 'gradient_dir_entropy',
                'laplacian_var', 'tenengrad', 'brenner', 'wavelet_hf',
                'edge_dir_anisotropy', 'fft_dir_concentration',
                'mad_sigma', 'flat_patch_sigma', 'hf_lift_ratio', 'dark_region_sigma',
                'hi_clip', 'lo_clip', 'mean_luma', 'median_luma',
                'hist_skewness', 'hist_kurtosis', 'entropy_norm',
                'dynamic_range', 'rgb_hi_clip',
                'weighted_score'],
        method_config=METHOD_CONFIG,
        plot='heatmap',
    )
