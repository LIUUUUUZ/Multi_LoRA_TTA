from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pre_knowledge import CORRUPTION_LIST, get_dataset
from pre_knowledge_class import Pre_Knowledge

# ── 配置 ──────────────────────────────────────────────────────────────────────
SEVERITIES   = (1, 2, 3, 4, 5)
CORRUPTIONS  = CORRUPTION_LIST          # 全部 15 种，如需快速测试可缩短
RETURN_MODE  = 'all'                    # 'origin' | 'group' | 'all'
MAX_IMGS_PER_COMB = None                # None = 不限制；设整数可加速调试

# ── 数据加载 ──────────────────────────────────────────────────────────────────
ds = get_dataset(severities=SEVERITIES, corruptions=CORRUPTIONS)

# ── 打分 ──────────────────────────────────────────────────────────────────────
pk = Pre_Knowledge()

# score_accum[metric][corr_name][severity] = [float, ...]
score_accum: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

print(f"\n正在用 Pre_Knowledge 计算得分（mode='{RETURN_MODE}'）...")
counter: dict = defaultdict(int)        # 用于限制每个 (corr, sev) 的样本数

for img_tensor, corr_idx, corr_name, severity in tqdm(ds, ncols=80):
    corr  = str(corr_name)
    sev   = int(severity)
    key   = (corr, sev)

    if MAX_IMGS_PER_COMB is not None:
        if counter[key] >= MAX_IMGS_PER_COMB:
            continue
        counter[key] += 1

    img_np = img_tensor.numpy().transpose(1, 2, 0)   # (C,H,W) → (H,W,C)
    scores = pk.get_score(img_np, reture_mode=RETURN_MODE, normalization_mode='method1')

    for metric, value in scores.items():
        score_accum[metric][corr][sev].append(float(value))

# ── 汇总均值 ──────────────────────────────────────────────────────────────────
# mean_scores[metric][corr][sev] = float
mean_scores: dict = {
    metric: {
        corr: {
            sev: float(np.mean(vals))
            for sev, vals in sev_dict.items()
        }
        for corr, sev_dict in corr_dict.items()
    }
    for metric, corr_dict in score_accum.items()
}

# 控制台摘要：各 metric 按全 severity 均值降序排列
print("\n─── 控制台摘要（各 metric 的 corruption 均分） ───")
for metric, corr_dict in mean_scores.items():
    per_corr_mean = {
        corr: float(np.mean(list(sev_dict.values())))
        for corr, sev_dict in corr_dict.items()
    }
    ranked = sorted(per_corr_mean.items(), key=lambda x: x[1], reverse=True)
    print(f"\n[{metric}]")
    print(f"  {'corruption':<26} {'mean score':>12}")
    print("  " + "-" * 40)
    for corr, val in ranked:
        print(f"  {corr:<26} {val:>12.5f}")

# ── 热力图绘制 ────────────────────────────────────────────────────────────────
def plot_heatmap(per_corr_sev: dict, metric: str) -> None:
    """
    热力图：X = severity，Y = corruption type，颜色 = 均分。

    参数
    ----
    per_corr_sev : {corr_name: {severity: mean_score}}
    metric       : 指标名称（用于标题）
    """
    corr_names = list(per_corr_sev.keys())
    severities = sorted({s for d in per_corr_sev.values() for s in d})

    matrix = np.full((len(corr_names), len(severities)), np.nan)
    for i, c in enumerate(corr_names):
        for j, s in enumerate(severities):
            matrix[i, j] = per_corr_sev[c].get(s, np.nan)

    # 按行均值降序排列
    row_means = np.nanmean(matrix, axis=1)
    order     = np.argsort(row_means)[::-1]
    matrix    = matrix[order]
    sorted_corr = [corr_names[i] for i in order]

    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    fig, ax = plt.subplots(
        figsize=(max(5, len(severities) * 1.4),
                 max(5, len(sorted_corr) * 0.55 + 2))
    )
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    for i in range(len(sorted_corr)):
        for j in range(len(severities)):
            v = matrix[i, j]
            if not np.isnan(v):
                brightness = (v - vmin) / (vmax - vmin + 1e-8)
                txt_color  = 'white' if brightness > 0.65 or brightness < 0.2 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=8, color=txt_color)

    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f'sev {s}' for s in severities], fontsize=10)
    ax.set_yticks(range(len(sorted_corr)))
    ax.set_yticklabels(sorted_corr, fontsize=9)
    ax.set_xlabel('Severity', fontsize=11)
    ax.set_ylabel('Corruption Type', fontsize=11)
    ax.set_title(f'[Pre_Knowledge]  {metric}', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    plt.tight_layout()
    plt.show()


print("\n─── 热力图输出 ───")
for metric, corr_dict in mean_scores.items():
    plot_heatmap(corr_dict, metric)
