"""
MlTTA – Multi-LoRA Test-Time Adaptation
========================================
核心逻辑
--------
1. 将基础模型所有 nn.Linear 层替换为 LoRALinear（含 NUM_LORAS 套独立适配器）。
   初始状态所有 LoRA 增量为 0（B 矩阵全零），基础模型权重完全冻结。

2. 对每张到来的测试图片：
   a. Pre_Knowledge 计算归一化后的 6 组污染得分（group_1~group_6）。
   b. 选取得分最高的组，激活对应的 LoRA 适配器。
   c. 用 HUS 策略将样本存入该 LoRA 的专属记忆池。

3. 每累积 update_every_x 个样本后：
   a. 对所有非空的 LoRA 记忆池分别构建 DataLoader。
   b. 只开启对应 LoRA 的梯度，用熵最小化损失（HLoss）微调。

4. 评估时逐样本路由，使每条样本经过与训练路由一致的 LoRA 推断。

污染分组（与 _DEFAULT_GROUPS 对应）
-------------------------------------
  0 – low_contrast  (dark_channel / hsv / local_contrast / edge_visibility)
  1 – jpeg          (blockiness / ringing)
  2 – color_cast    (gray_world / white_patch / chroma_shift)
  3 – blur          (gradient_dir_entropy / laplacian_var / … / fft_dir_concentration)
  4 – brightness    (hi_clip / lo_clip / mean_luma / … / rgb_hi_clip)
  5 – noise         (mad_sigma / flat_patch_sigma / hf_lift_ratio / dark_region_sigma)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import conf
from pre_knowledge_class import Pre_Knowledge
from utils import memory
from utils.loss_functions import HLoss
from .dnn import DNN

device = torch.device(
    "cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu"
)

NUM_LORAS = 6  # 对应 6 个污染大类

LORA_GROUP_NAMES = [
    'low_contrast',
    'jpeg',
    'color_cast',
    'blur',
    'brightness',
    'noise',
]


# ══════════════════════════════════════════════════════════════════════════════
# LoRA 模块
# ══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    将一个 nn.Linear 包装为支持 n_loras 套独立 LoRA 适配器的层。

    参数
    ----
    linear  : 被包装的原始 Linear 层（权重将被冻结）
    r       : LoRA 秩
    alpha   : 缩放因子，delta = (alpha/r) * B @ A @ x
    n_loras : 适配器套数
    """

    def __init__(self, linear: nn.Linear, r: int = 4,
                 alpha: float = 1.0, n_loras: int = NUM_LORAS):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.r     = r
        self.scale = alpha / r

        # 冻结原始权重
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = (
            nn.Parameter(linear.bias.data.clone(), requires_grad=False)
            if linear.bias is not None else None
        )

        # n_loras 套 LoRA 参数：A ~ N(0, 0.02)，B = 0（初始增量为 0）
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.empty(r, self.in_features).normal_(0, 0.02))
            for _ in range(n_loras)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, r))
            for _ in range(n_loras)
        ])

        self._active: int = 0

    # ── 激活控制 ──────────────────────────────────────────────────────────────

    def set_active(self, idx: int) -> None:
        assert 0 <= idx < len(self.lora_A), f"LoRA 索引 {idx} 超出范围"
        self._active = idx

    # ── 前向传播 ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base   = F.linear(x, self.weight, self.bias)
        lora_x = F.linear(x, self.lora_A[self._active])          # (*, r)
        delta  = F.linear(lora_x, self.lora_B[self._active])     # (*, out)
        return base + delta * self.scale


# ══════════════════════════════════════════════════════════════════════════════
# Multi-LoRA TTA
# ══════════════════════════════════════════════════════════════════════════════

class MlTTA(DNN):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ── 冻结骨干网络所有参数 ───────────────────────────────────────────────
        for param in self.net.parameters():
            param.requires_grad = False

        # ── 注入 LoRA（替换所有 Linear 层） ───────────────────────────────────
        lora_r     = getattr(conf.args, 'lora_r',     4)
        lora_alpha = getattr(conf.args, 'lora_alpha', 1.0)
        self._inject_lora(r=lora_r, alpha=lora_alpha)

        # ── 每个 LoRA 的独立 HUS 记忆池 ───────────────────────────────────────
        self.lora_memories: list[memory.HUS] = [
            memory.HUS(
                capacity=conf.args.memory_size,
                threshold=conf.args.high_threshold,
            )
            for _ in range(NUM_LORAS)
        ]

        # ── FIFO（用于 batch-based 评估，与 SoTTA 一致） ──────────────────────
        self.fifo = memory.FIFO(capacity=conf.args.update_every_x)

        # ── Pre_Knowledge（污染特征提取 + 分组路由） ───────────────────────────
        self.pk = Pre_Knowledge()

        # ── 每个 LoRA 的独立 Adam 优化器 ──────────────────────────────────────
        lr = conf.args.opt['learning_rate']
        self.lora_optimizers: list[torch.optim.Optimizer] = [
            torch.optim.Adam(self._lora_params(i), lr=lr)
            for i in range(NUM_LORAS)
        ]

        self.previous_train_loss = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # LoRA 工具方法
    # ══════════════════════════════════════════════════════════════════════════

    def _inject_lora(self, r: int, alpha: float) -> None:
        """递归将 net 中所有 nn.Linear 替换为 LoRALinear。"""
        def _replace(module: nn.Module) -> None:
            for name, child in list(module.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(module, name,
                            LoRALinear(child, r=r, alpha=alpha).to(device))
                else:
                    _replace(child)
        _replace(self.net)

    def _lora_params(self, idx: int) -> list:
        """返回第 idx 套 LoRA 的全部可训练参数（A + B）。"""
        params = []
        for m in self.net.modules():
            if isinstance(m, LoRALinear):
                params.append(m.lora_A[idx])
                params.append(m.lora_B[idx])
        return params

    def _set_active_lora(self, idx: int) -> None:
        """全网络切换激活 LoRA 适配器。"""
        for m in self.net.modules():
            if isinstance(m, LoRALinear):
                m.set_active(idx)

    def _enable_grad_for_lora(self, idx: int) -> None:
        """只开启第 idx 套 LoRA 的梯度，关闭其余。"""
        for lora_idx in range(NUM_LORAS):
            requires = (lora_idx == idx)
            for p in self._lora_params(lora_idx):
                p.requires_grad_(requires)

    # ══════════════════════════════════════════════════════════════════════════
    # 路由：Pre_Knowledge → LoRA 索引
    # ══════════════════════════════════════════════════════════════════════════

    def _select_lora(self, feat_tensor: torch.Tensor) -> int:
        """
        对单张图片 (C,H,W) float32 Tensor 计算 6 组污染得分，
        返回得分最高的组索引（0~5）。
        """
        img_np = feat_tensor.cpu().float().numpy().transpose(1, 2, 0)  # (H,W,C)
        group_scores = self.pk.get_score(
            img_np,
            normalization_mode='method1',
            reture_mode='group',
        )
        scores = [group_scores.get(f'group_{i + 1}', 0.0) for i in range(NUM_LORAS)]
        return int(np.argmax(scores))

    # ══════════════════════════════════════════════════════════════════════════
    # 在线训练主循环
    # ══════════════════════════════════════════════════════════════════════════

    def train_online(self, current_num_sample,
                     add_memory: bool = True,
                     evaluation: bool = True):
        TRAINED  = 0
        SKIPPED  = 1
        FINISHED = 2

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        feats, cls, dls = self.target_train_set
        current_sample = (
            feats[current_num_sample - 1],
            cls[current_num_sample - 1],
            dls[current_num_sample - 1],
        )

        # ── 样本入库 ──────────────────────────────────────────────────────────
        if add_memory:
            self.fifo.add_instance(current_sample)

            with torch.no_grad():
                self.net.eval()
                f = current_sample[0].to(device)
                d = current_sample[2].to(device)

                # 路由：选择得分最高的 LoRA
                lora_idx = self._select_lora(current_sample[0])
                self._set_active_lora(lora_idx)

                # 计算伪标签和置信度，用于 HUS 筛选
                logit       = self.net(f.unsqueeze(0))
                pseudo_cls  = logit.max(1, keepdim=False)[1][0].cpu().numpy()
                pseudo_conf = (F.softmax(logit, dim=1)
                               .max(1, keepdim=False)[0][0].cpu().numpy())

                self.lora_memories[lora_idx].add_instance(
                    [f, pseudo_cls, d, pseudo_conf]
                )

        # ── 未达到更新周期 → 跳过训练 ────────────────────────────────────────
        if current_num_sample % conf.args.update_every_x != 0:
            if not (current_num_sample == len(self.target_train_set[0])
                    and conf.args.update_every_x >= current_num_sample):
                self.log_loss_results(
                    'train_online', epoch=current_num_sample,
                    loss_avg=self.previous_train_loss,
                )
                return SKIPPED

        # ── 评估（用 FIFO 当前 batch，每样本按路由推断） ───────────────────────
        if evaluation:
            self.evaluation_online(current_num_sample, self.fifo.get_memory())

        # ── 对每个非空 LoRA 独立训练 ──────────────────────────────────────────
        entropy_loss = HLoss(conf.args.temperature)

        for lora_idx in range(NUM_LORAS):
            mem_feats, _, _ = self.lora_memories[lora_idx].get_memory()
            if len(mem_feats) == 0:
                continue

            self._set_active_lora(lora_idx)
            self._enable_grad_for_lora(lora_idx)

            stacked_feats = torch.stack(mem_feats)
            loader = DataLoader(
                TensorDataset(stacked_feats),
                batch_size=conf.args.opt['batch_size'],
                shuffle=True, drop_last=False, pin_memory=False,
            )
            opt = self.lora_optimizers[lora_idx]

            for _ in range(conf.args.epoch):
                for (batch_feats,) in loader:
                    batch_feats = batch_feats.to(device)
                    if conf.args.tta_attack_type:
                        batch_feats = batch_feats.clone().detach()

                    self.net.train()
                    preds = self.net(batch_feats)
                    loss  = entropy_loss(preds)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        if add_memory and evaluation:
            self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED

    # ══════════════════════════════════════════════════════════════════════════
    # 在线评估（逐样本路由，保持与训练一致的 LoRA 视角）
    # ══════════════════════════════════════════════════════════════════════════

    def evaluation_online(self, epoch, current_samples):
        """
        对 current_samples 中每个样本分别路由到对应 LoRA 后推断，
        再逐条调用父类的 evaluation_online_body 累积指标。
        """
        self.net.eval()
        features, cl_labels, do_labels = current_samples

        eval_fn = self._eval_single_no_grad

        if conf.args.log_grad:
            # log_grad 模式需要梯度，逐样本计算
            for feat, cl, dl in zip(features, cl_labels, do_labels):
                lora_idx = self._select_lora(feat)
                self._set_active_lora(lora_idx)
                single = ([feat], [cl], [dl])
                feats_t = feat.unsqueeze(0).to(device)
                cls_t   = self._to_tensor(cl).to(device)
                dls_t   = self._to_tensor(dl).to(device)
                self.evaluation_online_body(
                    epoch, current_samples, feats_t, cls_t, dls_t
                )
        else:
            with torch.no_grad():
                for feat, cl, dl in zip(features, cl_labels, do_labels):
                    lora_idx = self._select_lora(feat)
                    self._set_active_lora(lora_idx)
                    feats_t = feat.unsqueeze(0).to(device)
                    cls_t   = self._to_tensor(cl).to(device)
                    dls_t   = self._to_tensor(dl).to(device)
                    self.evaluation_online_body(
                        epoch, current_samples, feats_t, cls_t, dls_t
                    )

    # ── 工具 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0) if x.dim() == 0 else x.unsqueeze(0)
        return torch.tensor([x])

    @staticmethod
    def _eval_single_no_grad(fn, *args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
