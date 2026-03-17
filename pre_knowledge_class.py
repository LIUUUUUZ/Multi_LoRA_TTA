"""
Pre_Knowledge 类：对单张图片计算多维度污染特征得分。

所有 score 函数只输出原始得分，不在函数内做归一化。
归一化、分组加权等逻辑统一由类方法处理。
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter, uniform_filter, convolve, gaussian_filter
import scipy.ndimage as ndimage

# 从 CIFAR-10-C 全量数据预计算得到的各方法原始分 min/max
_NORM_PARAMS_PATH = Path(__file__).parent / 'normalization_params.json'
_NORM_PARAMS: dict = {}
if _NORM_PARAMS_PATH.exists():
    with _NORM_PARAMS_PATH.open('r', encoding='utf-8') as _f:
        _NORM_PARAMS = json.load(_f)


# ══════════════════════════════════════════════════════════════════════════════
# 模块级常量
# ══════════════════════════════════════════════════════════════════════════════

# ITU-R BT.601 灰度转换权重
_RGB2GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# 4-邻接 Laplacian 核
_LAP_KERNEL = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]], dtype=np.float32)

# 归一化 Sobel（÷8，使 [0,1] 图像响应落在 [0,1]）
_SOBEL_X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32) / 8.0
_SOBEL_Y = np.array([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=np.float32) / 8.0

# 非归一化 Sobel（用于模糊/方向性指标）
_BM_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_BM_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

# sRGB → CIE XYZ (D65)
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 模块级辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _bm_luminance(img_np: np.ndarray) -> np.ndarray:
    """(H,W,C) float [0,1] → (H,W) 亮度 Y，BT.601 加权。"""
    return (0.299 * img_np[..., 0]
            + 0.587 * img_np[..., 1]
            + 0.114 * img_np[..., 2]).astype(np.float32)


def _bm_convolve2d(Y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """2D 卷积，边界 0 填充（stride_tricks 实现）。"""
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


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel 梯度幅值（reflect 边界）。"""
    gx = convolve(gray, _SOBEL_X, mode='reflect')
    gy = convolve(gray, _SOBEL_Y, mode='reflect')
    return np.hypot(gx, gy)


def _rgb_to_lab(img_np: np.ndarray) -> np.ndarray:
    """(H,W,3) sRGB [0,1] → CIE Lab (H,W,3)。"""
    img = img_np.astype(np.float64)
    linear = np.where(img <= 0.04045,
                      img / 12.92,
                      ((img + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T
    t     = xyz / _D65_WHITE
    delta = 6.0 / 29.0
    ft    = np.where(t > delta ** 3,
                     np.cbrt(t),
                     t / (3.0 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * ft[..., 1] - 16.0
    a = 500.0 * (ft[..., 0] - ft[..., 1])
    b = 200.0 * (ft[..., 1] - ft[..., 2])
    return np.stack([L, a, b], axis=-1)


def _canny_nms_hysteresis(gray: np.ndarray,
                          sigma: float,
                          low_thr: float,
                          high_thr: float) -> np.ndarray:
    """向量化 Canny 边缘检测，返回 bool 边缘图。"""
    blurred = gaussian_filter(gray, sigma=sigma)
    gx    = convolve(blurred, _SOBEL_X, mode='reflect')
    gy    = convolve(blurred, _SOBEL_Y, mode='reflect')
    mag   = np.hypot(gx, gy)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180

    mp    = np.pad(mag, 1, mode='edge')
    dir0   = (angle <  22.5) | (angle >= 157.5)
    dir45  = (angle >= 22.5) & (angle <  67.5)
    dir90  = (angle >= 67.5) & (angle < 112.5)
    dir135 = (angle >= 112.5) & (angle < 157.5)

    neighbors = {
        0:   (mp[1:-1, 2:],  mp[1:-1, :-2]),
        45:  (mp[:-2,  2:],  mp[2:,   :-2]),
        90:  (mp[:-2, 1:-1], mp[2:,  1:-1]),
        135: (mp[:-2, :-2],  mp[2:,   2:]),
    }
    nms = mag.copy()
    for d, mask in zip([0, 45, 90, 135], [dir0, dir45, dir90, dir135]):
        a, b = neighbors[d]
        nms[mask & ((mag < a) | (mag < b))] = 0.0

    strong = nms >= high_thr
    weak   = (nms >= low_thr) & ~strong
    sp = np.pad(strong.astype(np.uint8), 1, mode='constant')
    neighbor_strong = (
        sp[:-2, :-2] + sp[:-2, 1:-1] + sp[:-2, 2:] +
        sp[1:-1, :-2]               + sp[1:-1, 2:] +
        sp[2:,  :-2] + sp[2:,  1:-1] + sp[2:,  2:]
    )
    return strong | (weak & (neighbor_strong > 0))


# ══════════════════════════════════════════════════════════════════════════════
# 配置字典
# ══════════════════════════════════════════════════════════════════════════════

_METHODS_HYPERPARAMS = {
    'dark_channel':          {'omega': 5, 'stat': 'mean'},
    'hsv':                   {'stat': 'mean'},
    'local_contrast':        {'mode': 'laplacian', 'omega': 2, 'stat': 'mean'},
    'edge_visibility':       {'mode': 'gradient', 'sigma': 1.0,
                              'low_thr': 0.04, 'high_thr': 0.10, 'stat': 'mean'},
    'blockiness':            {'block_size': 8, 'mode': 'ratio', 'stat': 'mean'},
    'ringing':               {'edge_thresh': 0.15, 'radius': 3,
                              'mode': 'ratio', 'stat': 'mean'},
    'gray_world':            {'mode': 'l2'},
    'white_patch':           {'stat': 'q99', 'mode': 'l2'},
    'chroma_shift':          {'mode': 'lab', 'min_brightness': 0.02},
    'gradient_dir_entropy':  {'n_bins': 36, 'mag_pct': 50.0, 'smooth_sigma': 1.0},
    'laplacian_var':         {},
    'tenengrad':             {'eta_percentile': 50.0},
    'brenner':               {},
    'wavelet_hf':            {},
    'edge_dir_anisotropy':   {'n_bins': 18, 'mag_pct': 70.0},
    'fft_dir_concentration': {'n_bins': 18, 'dc_radius': 2.0},
    'mad_sigma':             {},
    'flat_patch_sigma':      {'patch_size': 7},
    'hf_lift_ratio':         {'r_low': 0.1, 'r_high': 0.5},
    'dark_region_sigma':     {'dark_thresh': 0.2},
    'hi_clip':               {'eps_hi': 0.02},
    'lo_clip':               {'eps_lo': 0.02},
    'mean_luma':             {},
    'median_luma':           {},
    'hist_skewness':         {},
    'hist_kurtosis':         {},
    'entropy_norm':          {'num_bins': 256},
    'dynamic_range':         {'p_low': 1.0, 'p_high': 99.0},
    'rgb_hi_clip':           {'eps_hi': 0.02},
}

_SCORE_WEIGHTS = {
    'dark_channel':          1.0,
    'hsv':                   1.0,
    'local_contrast':        1.0,
    'edge_visibility':       1.0,
    'blockiness':            1.0,
    'ringing':               1.0,
    'gray_world':            1.0,
    'white_patch':           1.0,
    'chroma_shift':          1.0,
    'gradient_dir_entropy':  1.0,
    'laplacian_var':         1.0,
    'tenengrad':             1.0,
    'brenner':               1.0,
    'wavelet_hf':            1.0,
    'edge_dir_anisotropy':   1.0,
    'fft_dir_concentration': 1.0,
    'mad_sigma':             1.0,
    'flat_patch_sigma':      1.0,
    'hf_lift_ratio':         1.0,
    'dark_region_sigma':     1.0,
    'hi_clip':               1.0,
    'lo_clip':               1.0,
    'mean_luma':             1.0,
    'median_luma':           1.0,
    'hist_skewness':         1.0,
    'hist_kurtosis':         1.0,
    'entropy_norm':          1.0,
    'dynamic_range':         1.0,
    'rgb_hi_clip':           1.0,
}

# True = 原始得分"越低=越危险"；False = 越高=越危险
_SORT_ASCENDING = {
    'dark_channel':          False,
    'hsv':                   True,
    'local_contrast':        True,
    'edge_visibility':       True,
    'blockiness':            False,
    'ringing':               False,
    'gray_world':            False,
    'white_patch':           False,
    'chroma_shift':          False,
    'gradient_dir_entropy':  True,
    'laplacian_var':         True,
    'tenengrad':             True,
    'brenner':               True,
    'wavelet_hf':            True,
    'edge_dir_anisotropy':   False,
    'fft_dir_concentration': False,
    'mad_sigma':             False,
    'flat_patch_sigma':      False,
    'hf_lift_ratio':         False,
    'dark_region_sigma':     False,
    'hi_clip':               False,
    'lo_clip':               False,
    'mean_luma':             False,
    'median_luma':           False,
    'hist_skewness':         False,
    'hist_kurtosis':         False,
    'entropy_norm':          True,
    'dynamic_range':         True,
    'rgb_hi_clip':           False,
}

_DEFAULT_GROUPS = [
    ['dark_channel', 'hsv', 'local_contrast', 'edge_visibility'],
    ['blockiness', 'ringing'],
    ['gray_world', 'white_patch', 'chroma_shift'],
    ['gradient_dir_entropy', 'laplacian_var', 'tenengrad', 'brenner','wavelet_hf', 'edge_dir_anisotropy', 'fft_dir_concentration'],
    ['hi_clip', 'lo_clip', 'mean_luma', 'median_luma','hist_skewness', 'hist_kurtosis', 'entropy_norm','dynamic_range', 'rgb_hi_clip'],
    ['mad_sigma', 'flat_patch_sigma', 'hf_lift_ratio', 'dark_region_sigma'],
]


class Pre_Knowledge:
    """
    对单张图片计算多维度污染特征得分的工具类。

    典型用法
    --------
    pk = Pre_Knowledge()
    scores = pk.calculate_original_feature_score(img_np)  # (H,W,C) float32 [0,1]
    vec    = pk.handle_return_vectore(mode='group')        # 分组加权得分向量
    """

    def __init__(self) -> None:
        # 注册表：方法名 → 打分函数
        self._method_funcs: dict = {}
        # 当前图片的原始得分缓存
        self._scores: dict = {}

        # 参与计算的方法（按 _METHODS_HYPERPARAMS 键顺序）
        self.methods: list = list(_METHODS_HYPERPARAMS.keys())

        # 分组
        self.groups: list = [list(g) for g in _DEFAULT_GROUPS]

        # 各方法权重（可在外部修改）
        self.weights: dict = dict(_SCORE_WEIGHTS)

        # 归一化方法1： CIFAR10C计算的min max进行归一
        self.normalization_params_1 = {}

        # 注册所有方法
        self._reg_low_contrast_methods()
        self._reg_jpeg_methods()
        self._reg_color_cast_methods()
        self._reg_blur_methods()
        self._reg_brightness_methods()
        self._reg_noise_methods()

    def get_score(self, img_np: np.ndarray,
                  selected_methods: list = None,
                  normalization_mode: str = 'original',
                  reture_mode: str = 'origin') -> dict:
        """
        一站式接口：计算得分并以指定模式返回结果字典。

        参数
        ----
        img_np            : (H, W, C) float32，值域 [0, 1]
        selected_methods  : 要计算的方法名列表；为空则计算全部已注册方法
        normalization_mode: 传递给 _score_normalization，当前仅支持 'original'
        reture_mode       : 'origin' | 'group' | 'all'，传递给 handle_return_vectore

        返回
        ----
        dict，结构取决于 reture_mode
        """
        raw_score      = self.calculate_original_feature_score(img_np, selected_methods)
        normed_score   = self._score_normalization(mode=normalization_mode, scores=raw_score)
        reverted_score = self._convert_score(normed_score)
        return self.handle_return_vectore(mode=reture_mode, reverted_scores=reverted_score)




    def calculate_original_feature_score(self,
                                         img_np: np.ndarray,
                                         selected_methods: list = None) -> dict:
        """
        计算选定方法的原始得分，结果缓存在 self._scores 中并返回。

        参数
        ----
        img_np          : (H, W, C) float32，值域 [0, 1]
        selected_methods: 要计算的方法名列表；为空则计算全部已注册方法

        返回
        ----
        scores : {方法名: 原始得分 (float)} 字典
        """
        if not selected_methods:
            selected_methods = self.methods

        scores = {}
        for m in selected_methods:
            fn = self._method_funcs.get(m)
            if fn is None:
                continue
            cfg = _METHODS_HYPERPARAMS.get(m, {})
            scores[m] = fn(img_np, **cfg)

        self._scores = scores
        return scores

    def _score_normalization(self, mode: str = 'original', scores: dict = None) -> dict:
        """
        对 self._scores 进行归一化并返回新字典。

        mode
        ----
        'original' : 直接返回原始得分，不做任何变换。
                     （其余归一化模式可在后续版本中扩展）
        'method1' : 通过CIFAR 10 C上图像计算得出的最大小值进行每个method的归一

        返回
        ----
        normalized : {方法名: 得分 (float)} 字典
        """
        if scores is None:
            scores = self._scores
        if mode == 'original':
            return dict(scores)

        if mode == 'method1':
            if not _NORM_PARAMS:
                raise RuntimeError(
                    f"归一化参数文件未找到或为空: {_NORM_PARAMS_PATH}\n"
                    "请先运行 calculate_normalization_params.py 生成该文件。"
                )
            result = {}
            for m, v in scores.items():
                params = _NORM_PARAMS.get(m)
                if params is None:
                    result[m] = float(v)
                    continue
                lo, hi = params['min'], params['max']
                span = hi - lo
                if span <= 0:
                    result[m] = 0.0
                else:
                    result[m] = float(np.clip((v - lo) / span, 0.0, 1.0))
            return result

        raise ValueError(
            f"不支持的归一化模式: '{mode}'。可选: 'original' | 'method1'。"
        )

    def _convert_score(self, normed_score: dict) -> dict:
        """
        将归一化得分统一转换为"越大 = 污染越严重"的方向。 请在归一化后使用。

        参数
        ----
        normed_score : {方法名: 归一化得分 (float)}

        返回
        ----
        converted : {方法名: 转换后得分 (float)}，统一满足"越大 = 越严重"
        """
        return {
            m: (1-v if _SORT_ASCENDING.get(m, False) else v)
            for m, v in normed_score.items()
        }



    def _compute_group_score(self, norm_scores: dict = None) -> dict:
        """
        对每个 group 计算各方法的加权平均得分。

        使用 _score_normalization(mode='original') 获取当前得分，
        再按 self.weights 做加权平均，返回各组的标量得分。

        注意：使用原始得分时不同方法的量纲不同，
        建议在 _score_normalization 支持归一化后再调用本方法做比较。

        返回
        ----
        group_scores : {'group_1': float, 'group_2': float, ...} 字典
        """
        if norm_scores is None:
            norm_scores = self._score_normalization()
            norm_scores = self._convert_score(norm_scores)
        group_scores = {}

        for i, group in enumerate(self.groups):
            available = [m for m in group if m in norm_scores]
            if not available:
                continue

            total_w = sum(self.weights.get(m, 1.0) for m in available)
            if total_w <= 0:
                continue

            weighted_sum = sum(
                self.weights.get(m, 1.0) * norm_scores[m]
                for m in available
            )
            group_scores[f'group_{i + 1}'] = weighted_sum / total_w

        return group_scores

    def handle_return_vectore(self, mode: str = 'group', reverted_scores: dict = None) -> dict:
        """
        将当前图片的得分整理为字典返回。

        需先调用 calculate_original_feature_score 以填充 self._scores。

        mode
        ----
        'origin' : {方法名: 原始得分}，顺序与 self.methods 一致
        'group'  : {group_i: 加权均分}，顺序与 self.groups 一致
        'all'    : origin 与 group 合并后的字典

        norm_scores
        -----------
        外部传入的归一化得分字典；为 None 时使用 self._scores。

        返回
        ----
        dict
        """
        not_normed = False
        if reverted_scores is None:
            reverted_scores = self._scores
            not_normed = True

        if mode == 'origin':
            return dict(reverted_scores)

        if mode == 'group':
            if not_normed:
                return self._compute_group_score()
            else:
                return self._compute_group_score(reverted_scores)

        if mode == 'all':
            origin = dict(reverted_scores)
            group  = self._compute_group_score() if not_normed else self._compute_group_score(reverted_scores)
            return {**origin, **group}

        raise ValueError(
            f"未知 mode='{mode}'，可选: 'origin' | 'group' | 'all'"
        )

    # ── 方法注册 ──────────────────────────────────────────────────────────────

    def _reg_low_contrast_methods(self) -> None:
        """注册低对比度 / 雾霾 / 散射型退化相关方法。"""

        def dark_channel_score(img_np: np.ndarray,
                               omega: int = 5,
                               stat: str = 'mean', **kwargs) -> float:
            """暗通道先验得分：值越高 = 暗通道越抬升（雾/亮度污染）。"""
            min_c = img_np.min(axis=2)
            dark  = minimum_filter(min_c, size=2 * omega + 1, mode='reflect')
            if stat == 'mean':
                return float(dark.mean())
            elif stat == 'median':
                return float(np.median(dark))
            elif stat == 'q75':
                return float(np.percentile(dark, 75))
            elif stat == 'q90':
                return float(np.percentile(dark, 90))
            raise ValueError(f"未知 stat='{stat}'")

        def hsv_saturation_score(img_np: np.ndarray,
                                 stat: str = 'mean', **kwargs) -> float:
            """HSV 饱和度得分：值越低 = 饱和度越低（雾/雪褪色）。"""
            cmax = img_np.max(axis=2)
            cmin = img_np.min(axis=2)
            sat  = np.where(cmax > 0,
                            (cmax - cmin) / (cmax + 1e-8),
                            0.0).astype(np.float32)
            if stat == 'mean':
                return float(sat.mean())
            elif stat == 'median':
                return float(np.median(sat))
            elif stat == 'q25':
                return float(np.percentile(sat, 25))
            elif stat == 'q10':
                return float(np.percentile(sat, 10))
            raise ValueError(f"未知 stat='{stat}'")

        def local_contrast_score(img_np: np.ndarray,
                                 mode: str = 'laplacian',
                                 omega: int = 2,
                                 stat: str = 'mean', **kwargs) -> float:
            """局部对比度得分：值越低 = 对比度越弱（雾/模糊）。"""
            gray = img_np @ _RGB2GRAY
            if mode == 'laplacian':
                energy_map = convolve(gray, _LAP_KERNEL, mode='reflect') ** 2
            elif mode == 'rms':
                win  = 2 * omega + 1
                lm   = uniform_filter(gray,      size=win, mode='reflect')
                lm2  = uniform_filter(gray ** 2, size=win, mode='reflect')
                energy_map = np.sqrt(np.maximum(lm2 - lm ** 2, 0.0))
            else:
                raise ValueError(f"未知 mode='{mode}'")
            if stat == 'mean':
                return float(energy_map.mean())
            elif stat == 'median':
                return float(np.median(energy_map))
            elif stat == 'q75':
                return float(np.percentile(energy_map, 75))
            elif stat == 'q90':
                return float(np.percentile(energy_map, 90))
            raise ValueError(f"未知 stat='{stat}'")

        def edge_visibility_score(img_np: np.ndarray,
                                  mode: str = 'gradient',
                                  sigma: float = 1.0,
                                  low_thr: float = 0.04,
                                  high_thr: float = 0.10,
                                  stat: str = 'mean', **kwargs) -> float:
            """边缘可见性得分：值越低 = 边缘越少（雾/模糊）。"""
            gray = img_np @ _RGB2GRAY
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
                raise ValueError(f"未知 stat='{stat}'")
            elif mode == 'canny':
                edges = _canny_nms_hysteresis(gray, sigma=sigma,
                                              low_thr=low_thr, high_thr=high_thr)
                return float(edges.mean())
            raise ValueError(f"未知 mode='{mode}'")

        self._method_funcs.update({
            'dark_channel':    dark_channel_score,
            'hsv':             hsv_saturation_score,
            'local_contrast':  local_contrast_score,
            'edge_visibility': edge_visibility_score,
        })

    def _reg_jpeg_methods(self) -> None:
        """注册 JPEG 压缩 / 编码伪影相关方法。"""

        def blockiness_score(img_np: np.ndarray,
                             block_size: int = 8,
                             mode: str = 'ratio',
                             stat: str = 'mean', **kwargs) -> float:
            """块效应得分：值越高 = 块效应越明显（JPEG 伪影）。"""
            gray = img_np @ _RGB2GRAY
            H, W = gray.shape
            h_diff = np.abs(np.diff(gray, axis=1))
            v_diff = np.abs(np.diff(gray, axis=0))
            h_cols = np.arange(W - 1)
            v_rows = np.arange(H - 1)
            h_mask = (h_cols + 1) % block_size == 0
            v_mask = (v_rows + 1) % block_size == 0
            boundary_vals = np.concatenate([
                h_diff[:, h_mask].ravel(),
                v_diff[v_mask, :].ravel(),
            ])
            inner_vals = np.concatenate([
                h_diff[:, ~h_mask].ravel(),
                v_diff[~v_mask, :].ravel(),
            ])

            def _agg(arr):
                if stat == 'mean':   return float(arr.mean())
                if stat == 'median': return float(np.median(arr))
                if stat == 'q75':   return float(np.percentile(arr, 75))
                if stat == 'q90':   return float(np.percentile(arr, 90))
                raise ValueError(f"未知 stat='{stat}'")

            b, i = _agg(boundary_vals), _agg(inner_vals)
            if mode == 'ratio':
                return b / (i + 1e-8)
            elif mode == 'diff':
                return b - i
            raise ValueError(f"未知 mode='{mode}'")

        def ringing_score(img_np: np.ndarray,
                          edge_thresh: float = 0.15,
                          radius: int = 3,
                          mode: str = 'ratio',
                          stat: str = 'mean', **kwargs) -> float:
            """振铃得分：值越高 = 边缘附近过冲越严重。"""
            gray     = img_np @ _RGB2GRAY
            gx       = ndimage.sobel(gray, axis=1)
            gy       = ndimage.sobel(gray, axis=0)
            grad_mag = np.hypot(gx, gy)
            g_max    = grad_mag.max()
            if g_max > 1e-8:
                grad_mag /= g_max
            edge_mask = grad_mag > edge_thresh
            struct    = np.ones((2 * radius + 1, 2 * radius + 1), bool)
            dilated   = ndimage.binary_dilation(edge_mask, structure=struct)
            near_zone = dilated & ~edge_mask
            far_zone  = ~dilated
            if near_zone.sum() == 0:
                return 0.0
            lap_abs = np.abs(ndimage.laplace(gray))

            def _agg(arr):
                if stat == 'mean':   return float(arr.mean())
                if stat == 'median': return float(np.median(arr))
                if stat == 'q75':   return float(np.percentile(arr, 75))
                if stat == 'q90':   return float(np.percentile(arr, 90))
                raise ValueError(f"未知 stat='{stat}'")

            near_val = _agg(lap_abs[near_zone])
            if mode == 'ratio':
                if far_zone.sum() == 0:
                    return near_val
                return near_val / (_agg(lap_abs[far_zone]) + 1e-8)
            elif mode == 'abs':
                return near_val
            raise ValueError(f"未知 mode='{mode}'")

        self._method_funcs.update({
            'blockiness': blockiness_score,
            'ringing':    ringing_score,
        })

    def _reg_color_cast_methods(self) -> None:
        """注册色偏 / 白平衡异常相关方法。"""

        def gray_world_score(img_np: np.ndarray,
                             mode: str = 'l2', **kwargs) -> float:
            """Gray-World 偏离：值越高 = 色偏越严重。"""
            mu        = img_np.mean(axis=(0, 1))
            mu_global = float(mu.mean())
            dev       = mu - mu_global
            if mode == 'l2':
                return float(np.sqrt((dev ** 2).mean()))
            elif mode == 'l1':
                return float(np.abs(dev).mean())
            elif mode == 'max_range':
                return float(mu.max() - mu.min())
            raise ValueError(f"未知 mode='{mode}'")

        def white_patch_score(img_np: np.ndarray,
                              stat: str = 'q99',
                              mode: str = 'l2', **kwargs) -> float:
            """White-Patch 偏离：值越高 = 高光色偏越严重。"""
            if stat == 'max':
                ch_val = img_np.max(axis=(0, 1))
            elif stat == 'q99':
                ch_val = np.percentile(img_np, 99, axis=(0, 1))
            elif stat == 'q95':
                ch_val = np.percentile(img_np, 95, axis=(0, 1))
            else:
                raise ValueError(f"未知 stat='{stat}'")
            ref = float(ch_val.mean())
            dev = ch_val - ref
            if mode == 'l2':
                return float(np.sqrt((dev ** 2).mean()))
            elif mode == 'l1':
                return float(np.abs(dev).mean())
            elif mode == 'max_range':
                return float(ch_val.max() - ch_val.min())
            raise ValueError(f"未知 mode='{mode}'")

        def chroma_shift_score(img_np: np.ndarray,
                               mode: str = 'lab',
                               min_brightness: float = 0.02, **kwargs) -> float:
            """色度向量偏移：值越高 = 色偏越偏离中性。"""
            if mode in ('rg', 'rg_spread'):
                brightness = img_np.sum(axis=-1)
                valid = img_np[brightness > min_brightness]
                if valid.shape[0] == 0:
                    return 0.0
                rg = valid[:, :2] / (valid.sum(axis=-1, keepdims=True) + 1e-8)
                if mode == 'rg':
                    return float(np.linalg.norm(rg.mean(axis=0) - np.array([1/3, 1/3])))
                dists = np.linalg.norm(rg - rg.mean(axis=0), axis=1)
                return float(dists.mean())
            elif mode in ('lab', 'lab_chroma'):
                lab = _rgb_to_lab(img_np)
                a, b = lab[..., 1], lab[..., 2]
                if mode == 'lab':
                    return float(np.sqrt(a.mean() ** 2 + b.mean() ** 2))
                return float(np.sqrt(a ** 2 + b ** 2).mean())
            raise ValueError(f"未知 mode='{mode}'")

        self._method_funcs.update({
            'gray_world':   gray_world_score,
            'white_patch':  white_patch_score,
            'chroma_shift': chroma_shift_score,
        })

    def _reg_blur_methods(self) -> None:
        """注册模糊相关方法（通用模糊度 + 方向性模糊）。"""

        def gradient_dir_entropy(img_np: np.ndarray,
                                 n_bins: int = 36,
                                 mag_pct: float = 50.0,
                                 smooth_sigma: float = 1.0, **kwargs) -> float:
            """梯度方向熵：值越低 = 方向越集中（运动模糊候选）。"""
            gray = img_np @ _RGB2GRAY
            if smooth_sigma > 0:
                gray = gaussian_filter(gray, sigma=smooth_sigma)
            gx  = convolve(gray, _SOBEL_X, mode='reflect')
            gy  = convolve(gray, _SOBEL_Y, mode='reflect')
            mag = np.hypot(gx, gy)
            thr  = np.percentile(mag, mag_pct)
            mask = mag > thr
            if mask.sum() < 10:
                return 1.0
            angle = np.rad2deg(np.arctan2(gy[mask], gx[mask])) % 180.0
            hist, _ = np.histogram(angle, bins=n_bins, range=(0.0, 180.0))
            p = hist.astype(np.float64)
            p = p[p > 0] / p.sum()
            return float(-np.sum(p * np.log(p)))

        def laplacian_var_score(img_np: np.ndarray, **kwargs) -> float:
            """拉普拉斯方差：值越大 = 图像越清晰。"""
            Y   = _bm_luminance(img_np)
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
            """Tenengrad 平均梯度能量：值越大 = 图像越清晰。"""
            Y      = _bm_luminance(img_np)
            Gx     = _bm_convolve2d(Y, _BM_SOBEL_X)
            Gy     = _bm_convolve2d(Y, _BM_SOBEL_Y)
            mag_sq = Gx.astype(np.float64) ** 2 + Gy.astype(np.float64) ** 2
            mag    = np.sqrt(mag_sq)
            eta    = np.percentile(mag, eta_percentile)
            mask   = mag > eta
            n = int(np.sum(mask))
            if n == 0:
                return 0.0
            return float(np.sum(mag_sq[mask]) / n)

        def brenner_score(img_np: np.ndarray, **kwargs) -> float:
            """Brenner 梯度均值：值越大 = 图像越清晰。"""
            Y    = _bm_luminance(img_np).astype(np.float64)
            H, W = Y.shape
            dx   = (Y[2:, :] - Y[:-2, :]) ** 2
            dy   = (Y[:, 2:] - Y[:, :-2]) ** 2
            n    = (H - 2) * W + H * (W - 2)
            return float((np.sum(dx) + np.sum(dy)) / n)

        def wavelet_hf_score(img_np: np.ndarray, **kwargs) -> float:
            """Haar 小波高频子带平均能量：值越大 = 图像越清晰。"""
            Y = _bm_luminance(img_np).astype(np.float64)
            H, W = Y.shape
            if H % 2 != 0:
                Y = Y[:-1, :]; H -= 1
            if W % 2 != 0:
                Y = Y[:, :-1]; W -= 1
            L     = (Y[:, 0::2] + Y[:, 1::2]) * 0.5
            H_row = (Y[:, 0::2] - Y[:, 1::2]) * 0.5
            LH = (L[0::2, :]     - L[1::2, :])     * 0.5
            HL = (H_row[0::2, :] + H_row[1::2, :]) * 0.5
            HH = (H_row[0::2, :] - H_row[1::2, :]) * 0.5
            n  = (H // 2) * (W // 2)
            return float((np.sum(LH**2) + np.sum(HL**2) + np.sum(HH**2)) / (3 * n))

        def edge_dir_anisotropy(img_np: np.ndarray,
                                n_bins: int = 18,
                                mag_pct: float = 70.0, **kwargs) -> float:
            """边缘方向各向异性：值越高 = 边缘方向越集中（运动模糊候选）。"""
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
            ang = np.rad2deg(np.arctan2(gy[mask], gx[mask])) % 180.0
            hist, _ = np.histogram(ang, bins=n_bins, range=(0.0, 180.0),
                                   weights=mag[mask])
            if hist.sum() <= 0:
                return 0.0
            p = hist.astype(np.float64) / hist.sum()
            p = p[p > 0]
            H_raw = -float(np.sum(p * np.log(p)))
            return float(1.0 - H_raw)

        def fft_dir_concentration(img_np: np.ndarray,
                                  n_bins: int = 18,
                                  dc_radius: float = 2.0, **kwargs) -> float:
            """频谱方向集中度：值越高 = 频谱越各向异性（运动模糊候选）。"""
            Y = (img_np @ np.array([0.299, 0.587, 0.114],
                                   dtype=np.float32)).astype(np.float64)
            H_im, W_im = Y.shape
            F   = np.fft.fftshift(np.fft.fft2(Y))
            mag = np.abs(F)
            cy, cx       = H_im // 2, W_im // 2
            y_idx, x_idx = np.indices((H_im, W_im))
            dy = y_idx - cy
            dx = x_idx - cx
            r  = np.sqrt(dy ** 2 + dx ** 2)
            mask = r > dc_radius
            if not np.any(mask):
                return 0.0
            ang  = np.mod(np.arctan2(dy[mask], dx[mask]), np.pi)
            bins = np.linspace(0.0, np.pi, n_bins + 1, endpoint=True)
            hist, _ = np.histogram(ang, bins=bins, weights=mag[mask])
            if hist.sum() <= 0:
                return 0.0
            p    = hist.astype(np.float64) / hist.sum()
            return float(p.max())

        self._method_funcs.update({
            'gradient_dir_entropy':  gradient_dir_entropy,
            'laplacian_var':         laplacian_var_score,
            'tenengrad':             tenengrad_score,
            'brenner':               brenner_score,
            'wavelet_hf':            wavelet_hf_score,
            'edge_dir_anisotropy':   edge_dir_anisotropy,
            'fft_dir_concentration': fft_dir_concentration,
        })

    def _reg_brightness_methods(self) -> None:
        """注册过曝 / 欠曝 / 亮度分布异常相关方法。"""

        def hi_clip_score(img_np: np.ndarray,
                          eps_hi: float = 0.02, **kwargs) -> float:
            """高光饱和像素数：值越高 = 过曝越严重。"""
            Y = _bm_luminance(img_np)
            return float(np.sum(Y >= (1.0 - eps_hi)))

        def lo_clip_score(img_np: np.ndarray,
                          eps_lo: float = 0.02, **kwargs) -> float:
            """暗部饱和像素数：值越高 = 欠曝越严重。"""
            Y = _bm_luminance(img_np)
            return float(np.sum(Y <= eps_lo))

        def mean_luma_score(img_np: np.ndarray, **kwargs) -> float:
            """亮度均值：值越高 = 整体越亮。"""
            return float(_bm_luminance(img_np).mean())

        def median_luma_score(img_np: np.ndarray, **kwargs) -> float:
            """亮度中位数：对极端像素鲁棒。"""
            return float(np.median(_bm_luminance(img_np)))

        def hist_skewness_score(img_np: np.ndarray, **kwargs) -> float:
            """亮度绝对偏度：值越高 = 分布偏斜越严重（过曝/欠曝均可触发）。"""
            Y   = _bm_luminance(img_np).ravel().astype(np.float64)
            var = float(Y.var())
            if var <= 0:
                return 0.0
            z = (Y - Y.mean()) / np.sqrt(var)
            return float(abs((z ** 3).mean()))

        def hist_kurtosis_score(img_np: np.ndarray, **kwargs) -> float:
            """亮度峰度（4 阶标准矩）：值越高 = 分布越尖峭（饱和区域集中）。"""
            Y   = _bm_luminance(img_np).ravel().astype(np.float64)
            var = float(Y.var())
            if var <= 0:
                return 0.0
            z = (Y - Y.mean()) / np.sqrt(var)
            return float((z ** 4).mean())

        def entropy_norm_score(img_np: np.ndarray,
                               num_bins: int = 256, **kwargs) -> float:
            """亮度分布香农熵（nats）：值越低 = 分布越集中（曝光异常候选）。"""
            Y = _bm_luminance(img_np).ravel().astype(np.float64)
            hist, _ = np.histogram(Y, bins=num_bins, range=(0.0, 1.0))
            total = int(hist.sum())
            if total == 0:
                return 0.0
            p = hist.astype(np.float64) / total
            p = p[p > 0]
            return float(-np.sum(p * np.log(p)))

        def dynamic_range_score(img_np: np.ndarray,
                                p_low: float = 1.0,
                                p_high: float = 99.0, **kwargs) -> float:
            """有效动态范围 P_high - P_low：值越低 = 对比度越差。"""
            Y  = _bm_luminance(img_np)
            lo = float(np.percentile(Y, p_low))
            hi = float(np.percentile(Y, p_high))
            return float(max(0.0, hi - lo))

        def rgb_hi_clip(img_np: np.ndarray,
                        eps_hi: float = 0.02, **kwargs) -> float:
            """RGB 通道高光饱和（三通道像素数均值）：值越高 = 通道剪切越严重。"""
            thr = 1.0 - eps_hi
            return float(np.mean([
                float(np.sum(img_np[..., c] >= thr)) for c in range(3)
            ]))

        self._method_funcs.update({
            'hi_clip':       hi_clip_score,
            'lo_clip':       lo_clip_score,
            'mean_luma':     mean_luma_score,
            'median_luma':   median_luma_score,
            'hist_skewness': hist_skewness_score,
            'hist_kurtosis': hist_kurtosis_score,
            'entropy_norm':  entropy_norm_score,
            'dynamic_range': dynamic_range_score,
            'rgb_hi_clip':   rgb_hi_clip,
        })

    def _reg_noise_methods(self) -> None:
        """注册噪声与低照度噪声相关方法。"""

        def mad_sigma_score(img_np: np.ndarray, **kwargs) -> float:
            """MAD 高通噪声估计 σ：值越高 = 全图噪声越强。"""
            Y    = _bm_luminance(img_np).astype(np.float32)
            blur = uniform_filter(Y, size=3, mode='reflect')
            R    = Y - blur
            return float(np.median(np.abs(R)) / 0.6745)

        def flat_patch_sigma_score(img_np: np.ndarray,
                                   patch_size: int = 7, **kwargs) -> float:
            """平坦块噪声估计 σ：值越高 = 背景噪声越大。"""
            Y    = _bm_luminance(img_np).astype(np.float32)
            H, W = Y.shape
            pad  = patch_size // 2
            e_x  = uniform_filter(Y,      size=patch_size, mode='reflect')
            e_x2 = uniform_filter(Y ** 2, size=patch_size, mode='reflect')
            var  = np.clip(e_x2 - e_x ** 2, 0.0, None)
            if H > 2 * pad and W > 2 * pad:
                inner = var[pad: H - pad, pad: W - pad]
            else:
                inner = var
            return float(np.sqrt(np.min(inner)))

        def hf_lift_ratio_score(img_np: np.ndarray,
                                r_low: float = 0.1,
                                r_high: float = 0.5, **kwargs) -> float:
            """频谱高频环带能量：值越高 = 白噪声抬升越明显。"""
            Y = _bm_luminance(img_np).astype(np.float64)
            H, W = Y.shape
            P    = np.abs(np.fft.fftshift(np.fft.fft2(Y))) ** 2
            cy, cx       = H // 2, W // 2
            y_idx, x_idx = np.indices((H, W))
            r_norm = (np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
                      / (max(H, W) / 2.0 + 1e-6))
            mask_hf = (r_norm >= r_low) & (r_norm <= r_high)
            return float(P[mask_hf].sum())

        def dark_region_sigma_score(img_np: np.ndarray,
                                    dark_thresh: float = 0.2, **kwargs) -> float:
            """暗部区域 MAD 噪声估计：值越高 = 暗部噪声越大。"""
            Y    = _bm_luminance(img_np).astype(np.float32)
            mask = Y < dark_thresh
            if not np.any(mask):
                return 0.0
            blur = uniform_filter(Y, size=3, mode='reflect')
            R    = (Y - blur)[mask]
            return float(np.median(np.abs(R)) / 0.6745)

        self._method_funcs.update({
            'mad_sigma':         mad_sigma_score,
            'flat_patch_sigma':  flat_patch_sigma_score,
            'hf_lift_ratio':     hf_lift_ratio_score,
            'dark_region_sigma': dark_region_sigma_score,
        })

