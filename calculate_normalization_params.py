import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from pre_knowledge import CORRUPTION_LIST, get_dataset
from pre_knowledge_class import Pre_Knowledge

# ── 配置 ──────────────────────────────────────────────────────────────────────
SEVERITIES        = (1, 2, 3, 4, 5)
CORRUPTIONS       = CORRUPTION_LIST   # 全部 15 种，如需快速调试可缩短
MAX_IMGS_PER_COMB = None              # None = 不限制；设整数可加速调试
OUTPUT_JSON       = 'normalization_params.json'

# ── 数据加载 ──────────────────────────────────────────────────────────────────
ds = get_dataset(severities=SEVERITIES, corruptions=CORRUPTIONS)

# ── 打分 ──────────────────────────────────────────────────────────────────────
pk = Pre_Knowledge()

# score_accum[metric] = [float, ...]  （跨所有图片展平）
score_accum: dict[str, list] = defaultdict(list)

print("\n正在计算原始得分以统计归一化参数...")
counter: dict = defaultdict(int)

for img_tensor, corr_idx, corr_name, severity in tqdm(ds, ncols=80):
    corr = str(corr_name)
    sev  = int(severity)
    key  = (corr, sev)

    if MAX_IMGS_PER_COMB is not None:
        if counter[key] >= MAX_IMGS_PER_COMB:
            continue
        counter[key] += 1

    img_np = img_tensor.numpy().transpose(1, 2, 0)   # (C,H,W) → (H,W,C)
    scores = pk.calculate_original_feature_score(img_np)

    for metric, value in scores.items():
        score_accum[metric].append(float(value))

# ── 计算 min / max ────────────────────────────────────────────────────────────
norm_params: dict[str, dict] = {}
print("\n─── 各方法归一化参数 ───")
print(f"  {'metric':<30} {'min':>14} {'max':>14}")
print("  " + "-" * 60)

for metric, values in score_accum.items():
    arr = np.array(values, dtype=np.float64)
    v_min = float(np.percentile(arr, 0.5))
    v_max = float(np.percentile(arr, 99.5))
    norm_params[metric] = {'min': v_min, 'max': v_max}
    print(f"  {metric:<30} {v_min:>14.6f} {v_max:>14.6f}")

# ── 保存 JSON ─────────────────────────────────────────────────────────────────
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(norm_params, f, indent=2, ensure_ascii=False)

print(f"\n归一化参数已保存至 {OUTPUT_JSON}")
