# SoTTA 方法深度解析：训练数据集构建与模型更新机制

> 本文档基于代码库实现（`learner/sotta.py`、`utils/memory.py`、`learner/dnn.py` 等）对 SoTTA（Source-free Online Test-Time Adaptation）的核心机制进行详细分析。

---

## 目录

1. [整体架构](#1-整体架构)
2. [数据流与在线处理循环](#2-数据流与在线处理循环)
3. [Support Set（训练数据集）的构建](#3-support-set训练数据集的构建)
4. [数据收集与筛选策略（HUS）](#4-数据收集与筛选策略hus)
5. [模型更新时机](#5-模型更新时机)
6. [模型更新方式](#6-模型更新方式)
7. [BN 统计量更新（EMA 模式）](#7-bn-统计量更新ema-模式)
8. [OOD 噪声样本的处理](#8-ood-噪声样本的处理)
9. [完整端到端流程](#9-完整端到端流程)
10. [关键超参数速查表](#10-关键超参数速查表)
11. [各内存类型对比](#11-各内存类型对比)

---

## 1. 整体架构

```
SoTTA/
├── main.py                    # 主入口，在线训练循环驱动
├── conf.py                    # 超参数配置（CIFAR10Opt / CIFAR100Opt / IMAGENET_C）
│
├── learner/
│   ├── dnn.py                 # 基类：数据预处理、评估、内存初始化
│   └── sotta.py               # SoTTA 核心：train_online / step / 参数选择
│
├── utils/
│   ├── memory.py              # FIFO / HUS / ConfFIFO（Support Set 实现）
│   ├── memory_rotta.py        # CSTU（RoTTA 内存，SoTTA 可选复用）
│   ├── loss_functions.py      # HLoss（温控熵最小化）/ calc_energy / softmax_entropy
│   └── sam_optimizer.py       # SAM 优化器（ESM 所需的两步更新）
│
└── data_loader/
    ├── data_loader.py         # 统一加载入口（batch_size=1, 顺序读取）
    ├── NoisyDataset.py        # OOD 噪声流封装（class_label=10000 标记）
    ├── CIFAR10Dataset.py      # CIFAR-10-C 数据集
    └── CIFAR100Dataset.py     # CIFAR-100-C 数据集
```

**SoTTA 的核心思想**：在无源域数据（Source-free）的在线测试阶段，通过维护一个高质量的 Support Set（仅保留高置信度、类别均衡的样本），对 BN 层参数进行熵最小化优化，同时利用 SAM 优化器抑制 sharp loss landscape，从而在存在 OOD 噪声的数据流中稳定适应。

---

## 2. 数据流与在线处理循环

### 2.1 目标域数据加载

目标域数据以 **`batch_size=1`、不打乱顺序** 的方式加载，模拟真实在线场景中样本逐一到来：

```python
# data_loader/data_loader.py
# 目标域：batch_size=1, shuffle=False
target_loader = DataLoader(dataset, batch_size=1, shuffle=False, ...)
```

所有目标域样本在 `dnn.py` 的 `target_data_processing()` 中被预先整体载入内存（存为 `self.target_train_set`），以提升训练速度。支持多种时序分布模拟：

| `tgt_train_dist` | 分布模式 | 用途 |
|---|---|---|
| `0` | 原始顺序 | 对齐原始评测顺序 |
| `1` | 随机打乱 | i.i.d. 场景 |
| `4` | Dirichlet 采样（`beta=0.1`）| 时序相关（Temporal Correlation）场景 |
| `5` | 不打乱（等同 0）| 顺序评测 |
| `6` | SAR imbalanced | 类别不均衡 label shift 场景 |

### 2.2 支持的测试流类型

通过 `main.py` 的参数控制：

| 参数 | 数据流说明 |
|---|---|
| `--mixed_severity` | 单一 corruption 类型 × 5 个严重度（25000 张）|
| `--mixed_corruption_severity` | 15 种 corruption × 5 个 severity（375000 张）|
| 默认 | 单一 corruption、单一 severity |

### 2.3 在线循环驱动（main.py）

```python
while not finished and current_num_sample < num_sample_end:
    ret_val = learner.train_online(current_num_sample)
    current_num_sample += 1
    # ret_val: TRAINED(0) / SKIPPED(1) / FINISHED(2)
```

每次调用 `train_online(t)` 处理第 `t` 个样本，返回值指示当前是否触发了梯度更新。

---

## 3. Support Set（训练数据集）的构建

SoTTA 维护**两个独立的内存结构**，功能不同：

### 3.1 FIFO 推理窗口（`self.fifo`）

- **容量**：等于 `update_every_x`（默认 64）
- **策略**：无条件 FIFO，每个到来的样本都会入队
- **用途**：只用于**推理评估**（`evaluation_online` 在触发更新时对这批样本计算准确率），不用于训练
- **关键特性**：先评估再训练，保证测试-训练顺序正确，避免 label leakage

### 3.2 HUS 训练内存（`self.mem`）

- **容量**：`memory_size`（默认 64）
- **策略**：基于置信度过滤 + 类别均衡替换（详见第 4 节）
- **用途**：只用于**梯度更新**（提供实际的训练 batch）
- **关键特性**：保证训练数据质量，过滤 OOD 样本

```
每个新样本 x_t 同时进入两个内存：
                ┌─────────────────────────────────────────────┐
                │                                             │
x_t ──►  FIFO（推理窗口，无条件入队，容量=update_every_x）   │  评估用
         x_t ──►  HUS（训练内存，置信度过滤，容量=memory_size）   │  训练用
                │                                             │
                └─────────────────────────────────────────────┘
```

### 3.3 训练 DataLoader 构建

每次触发更新时，从 `self.mem`（HUS）中取出所有保留的样本，构建临时 DataLoader：

```python
# learner/sotta.py
feats, _, _ = self.mem.get_memory()
feats = torch.stack(feats)
data_loader = DataLoader(
    TensorDataset(feats),
    batch_size=conf.args.batch_size,   # 默认 64
    shuffle=True,                       # 打乱顺序
    drop_last=False
)
```

---

## 4. 数据收集与筛选策略（HUS）

HUS（High-Uncertainty Sampling，注意：在 SoTTA 中实际为 **High-Confidence Sampling**，名称有些反直觉）是 SoTTA 的核心数据管理机制。

### 4.1 入队决策流程

```
新样本 x_t 到来
    │
    ▼
计算 pseudo_conf = max(softmax(f_θ(x_t)))
    │
    ├─► pseudo_conf < high_threshold ?
    │         ├─ 是 → 直接丢弃（OOD 过滤）
    │         └─ 否 → 继续判断容量
    │
    ▼
HUS 已满（occupancy >= memory_size）?
    │
    ├─ 否 → 直接加入对应类别槽
    │
    └─ 是 → 执行类均衡替换（remove_instance）
              │
              ├─ 找到占用样本最多的类 C_max
              ├─ 若新样本类 ≠ C_max：替换 C_max 中的随机一个样本
              └─ 若新样本类 = C_max：替换本类中的随机一个样本
```

### 4.2 置信度阈值（`high_threshold`）

各数据集的默认置信度阈值：

| 数据集 | `high_threshold` | 含义 |
|---|---|---|
| CIFAR-10 | **0.99** | 极高置信度，过滤几乎所有 OOD 样本 |
| CIFAR-100 | **0.66** | 中高置信度（100类更难，阈值相应降低）|
| ImageNet | **0.33** | 较宽松（1000类，最高置信度本身偏低）|

**设计动机**：OOD 样本（如随机噪声、来自其他数据集的图像）在预训练模型上通常产生**接近均匀分布**的预测概率，最大 softmax 值很低。设置高阈值可以自然地将 OOD 样本排除在 support set 之外，避免用"毒"数据训练。

### 4.3 类别均衡替换

HUS 按类别分槽存储（每类一个独立列表），满容量时总是替换占用最多的类中的样本，而不是最老的样本。

**效果**：当数据流存在类别不均衡时（如某类 corruption 下某些类别样本占多数），类均衡机制保证 support set 不被少数类支配，训练时各类都有代表性样本。

### 4.4 其他可选内存类型

通过 `--memory_type` 参数切换：

**ConfFIFO**：置信度过滤 + 传统 FIFO 顺序淘汰（与 HUS 区别在于淘汰策略是按时间顺序而非类均衡）：

```python
# utils/memory.py - ConfFIFO
def add_instance(self, instance):
    if instance[3] < self.threshold:   # 置信度过滤
        return                          # 丢弃
    if self.get_occupancy() >= self.capacity:
        self.remove_instance()          # FIFO 删最旧
    # 加入队尾
```

**FIFO（无过滤）**：所有样本无条件入队，满了删最旧，无任何质量控制（基线对比用）。

**CSTU（RoTTA 专用）**：基于时效性（age）+ 不确定性（entropy）的综合打分替换策略：

```
heuristic_score(sample) = λ_t * sigmoid(age/capacity) + λ_u * entropy/log(C)
```

当新样本得分低于被替换候选时才触发替换，兼顾新鲜度和不确定性。

---

## 5. 模型更新时机

### 5.1 批量更新触发条件

```python
# learner/sotta.py - train_online()
if current_num_sample % conf.args.update_every_x != 0:
    return SKIPPED   # 样本数未到达 update_every_x 的整数倍，跳过
# 到达触发点，执行更新
```

**默认值**：`update_every_x = 64`，即**每收到 64 个新测试样本触发一次梯度更新**。

### 5.2 更新-评估的时序关系

在同一次触发点，**必须先评估后训练**：

```python
# learner/sotta.py - train_online()
# 第一步：评估（用 FIFO 中最近 64 个样本的预测结果）
if evaluation:
    self.evaluation_online(current_num_sample, self.fifo.get_memory())

# 第二步：训练（用 HUS 中的高质量样本）
feats, _, _ = self.mem.get_memory()
# ... 构建 DataLoader ...
for e in range(conf.args.epoch):
    for batch in data_loader:
        self.step(loss_fn=entropy_loss, feats=batch)
```

这一顺序保证：评估时使用的是**更新前**的模型参数，反映模型在当前数据流上的真实泛化能力，不受即将到来的训练步骤影响。

### 5.3 内部迭代轮数

每次触发更新后，对 support set 执行 `conf.args.epoch`（默认 **1**）轮完整遍历，避免过拟合当前 support set。

---

## 6. 模型更新方式

### 6.1 可更新参数范围

SoTTA **只更新 BN/IN/LN 层的仿射参数**（γ 和 β），主干网络完全冻结：

```python
# learner/sotta.py - __init__()
# 第一步：关闭所有参数的梯度
for param in self.net.parameters():
    param.requires_grad = False

# 第二步：只开放归一化层的 weight（γ）和 bias（β）
for module in self.net.modules():
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                           nn.InstanceNorm1d, nn.InstanceNorm2d,
                           nn.LayerNorm)):
        module.weight.requires_grad_(True)
        module.bias.requires_grad_(True)
```

**参数量极少**（典型 ResNet18 约 9600 个可训练参数，占总参数量的 0.06%），更新高效且不易过拟合。

### 6.2 损失函数：温控熵最小化（HLoss）

```python
# utils/loss_functions.py
class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        self.temp_factor = temp_factor

    def forward(self, x):
        # x: logits, shape (B, C)
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        return entropy.sum(dim=1).mean()   # 标量损失
```

- **直觉**：对测试样本的预测分布进行熵最小化，使模型对当前领域的预测更"自信"（分布更尖锐）
- **温度系数** `temperature`（默认 1.0）：较低温度使 softmax 分布更尖锐，放大高置信类别的梯度信号
- **为什么有效**：支持集样本均为高置信度样本（pseudo_conf ≥ high_threshold），熵最小化方向与正确标签方向大概率一致

### 6.3 优化器：SAM + Adam（ESM，`--esm` 开启时）

SoTTA 的核心创新之一是 **ESM（Energy-based Sharpness Minimization）**，即在 SAM（Sharpness-Aware Minimization）框架下结合能量函数：

```python
# learner/sotta.py - step()
def step(self, loss_fn, feats):
    self.net.train()
    preds = self.net(feats)
    loss_first = loss_fn(preds)          # 第一次前向：计算当前点 loss

    self.optimizer.zero_grad()
    loss_first.backward()               # 第一次反向：计算梯度方向

    if isinstance(self.optimizer, SAM):
        # SAM 第一步：沿梯度方向"爬坡"到 sharp loss peak
        self.optimizer.first_step(zero_grad=True)

        # SAM 第二步前向：在 peak 处重新计算 loss
        preds = self.net(feats)
        loss_second = loss_fn(preds)
        loss_second.backward()

        # SAM 第二步：在 peak 处计算梯度，但更新回原点（flat minima 方向）
        self.optimizer.second_step(zero_grad=True)
    else:
        self.optimizer.step()           # 标准 Adam 更新（ESM 关闭时）
```

**SAM 的动机**：普通梯度下降容易收敛到 sharp minima（泛化差），SAM 寻找损失曲面平坦的区域（flat minima），在测试时域偏移下更鲁棒。

**SAM 优化器配置**（`utils/sam_optimizer.py`）：

```python
# sam 使用 Adam 作为基础优化器
base_optimizer = torch.optim.Adam
optimizer = SAM(
    params=bn_params,           # 只优化 BN 参数
    base_optimizer=base_optimizer,
    rho=0.05,                   # 爬坡步长（扰动半径）
    lr=conf.args.learning_rate, # 默认 0.001
    weight_decay=conf.args.weight_decay
)
```

### 6.4 梯度更新完整流程

```
support set 样本 {x_1, ..., x_N}（来自 HUS）
        │
        ▼
  DataLoader（shuffle=True, batch_size=64）
        │
        ▼
  for epoch in range(1):              ← 默认只训练 1 epoch
    for batch in dataloader:
      preds = BN_model(batch)         ← 前向（BN 处于 train 模式，使用 EMA 统计量）
      loss = HLoss(preds)             ← 熵最小化
      loss.backward()
      SAM.first_step()                ← 爬坡到 sharp peak
      preds' = BN_model(batch)        ← 在 peak 处再次前向
      loss' = HLoss(preds')
      loss'.backward()
      SAM.second_step()               ← 从 peak 沿 flat 方向更新 γ, β
```

---

## 7. BN 统计量更新（EMA 模式）

SoTTA 与 TENT 等方法的重要区别：**使用 EMA（指数移动平均）统计量**而非每 batch 的即时统计量。

```python
# learner/sotta.py - __init__()
if conf.args.use_learned_stats:   # 默认 True
    module.track_running_stats = True
    module.momentum = conf.args.bn_momentum   # 默认 0.2
```

**BN EMA 更新公式**（PyTorch 内置，在 `model.train()` 模式下前向时自动执行）：

```
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
             = 0.8 * running_mean + 0.2 * batch_mean

running_var  = (1 - momentum) * running_var  + momentum * batch_var
             = 0.8 * running_var  + 0.2 * batch_var
```

**对比其他方式**：

| BN 统计量模式 | 优点 | 缺点 | 使用方法 |
|---|---|---|---|
| 源域统计量（frozen）| 稳定，不会偏移 | 域偏移下不准确 | 标准 BN |
| 即时 batch 统计量 | 快速适应当前 batch | 单样本时方差极大，不稳定 | TENT（batch_size > 1）|
| EMA 统计量（SoTTA）| 平滑更新，兼顾稳定性和适应性 | 需要调 momentum | SoTTA（`momentum=0.2`）|

当 `use_learned_stats=False` 时，SoTTA 退化为使用每 batch 的即时统计量（等同 TENT 的 BN 模式）。

---

## 8. OOD 噪声样本的处理

### 8.1 数据流中的 OOD 样本构造

`NoisyDataset.py` 支持将多种 OOD 噪声样本混入目标域测试流，模拟真实部署场景：

| 噪声类型 | 说明 |
|---|---|
| `original` | 原始（无噪声）CIFAR-10 样本 |
| `divide` | 测试集等分后的子集 |
| `repeat` | 重复的清洁样本 |
| `cifar100` | CIFAR-100 样本（语义外分布）|
| `gaussian` | 纯高斯随机噪声图像 |
| `uniform` | 均匀随机噪声图像 |
| `mnist` | MNIST 数字图像（视觉外分布）|

OOD 样本的类别标签统一设置为 `NOISY_CLASS_IDX = 10000`（特殊标记），与 CIFAR-10 的 0-9 类区分：

```python
# data_loader/NoisyDataset.py
NOISY_CLASS_IDX = 10000

noisy_dataset = TensorDataset(
    torch.from_numpy(noisy_stream),
    torch.from_numpy(NOISY_CLASS_IDX * np.ones(len(noisy_stream))),  # OOD 标签
    ...
)
```

### 8.2 SoTTA 如何天然屏蔽 OOD 样本

SoTTA 不需要显式检测 OOD 样本，而是通过 HUS 的置信度阈值**隐式过滤**：

```
OOD 样本（如高斯噪声）
    │
    ▼
模型预测：softmax 分布接近均匀 → max_prob ≈ 1/C（CIFAR-10: ≈0.1）
    │
    ▼
pseudo_conf < high_threshold（0.99）
    │
    ▼
不入队 HUS → 不参与梯度更新 → 模型不受 OOD 样本污染
```

**与 TENT/CoTTA 对比**：TENT 等方法使用整个测试流（包含 OOD 样本）计算梯度，容易被 OOD 样本"带偏"。SoTTA 的高置信度过滤是其在 noisy stream 下性能优于 baseline 的核心原因之一。

---

## 9. 完整端到端流程

```
初始化阶段
─────────────────────────────────────────────────────────────
1. 加载预训练模型（ResNet18，源域训练好的 checkpoint）
2. 冻结所有参数，只开放 BN/IN/LN 的 γ 和 β
3. BN 切换为 EMA 模式（momentum=0.2）
4. 初始化 FIFO（容量=64）和 HUS（容量=64）
5. 初始化 SAM(Adam) 优化器，lr=0.001


在线测试阶段（每个测试样本 x_t 到来时）
─────────────────────────────────────────────────────────────

t=1,2,...,64:
  ├─ x_t → FIFO.add(x_t)                  [无条件入队推理窗口]
  │
  ├─ net(x_t) → pseudo_conf = max softmax  [单样本推理]
  │
  ├─ HUS.add(x_t, pseudo_conf):
  │     ├─ pseudo_conf ≥ 0.99 → 入队       [高置信度样本保留]
  │     └─ pseudo_conf < 0.99 → 丢弃       [OOD / 低置信样本过滤]
  │
  └─ t % 64 ≠ 0 → SKIPPED（继续收集样本）


t=64,128,...（每 64 个样本触发一次）:
  ├─ 1. 评估（先于更新执行）
  │     └─ evaluation_online(FIFO.get_memory())
  │          → 记录 accuracy, entropy, energy, confidence
  │
  ├─ 2. 从 HUS 取出训练数据
  │     feats, _, _ = mem.get_memory()
  │     dataloader = DataLoader(feats, batch_size=64, shuffle=True)
  │
  └─ 3. 梯度更新（1 epoch）
        for batch in dataloader:
          preds = net(batch)                [BN train 模式，EMA 统计量自动更新]
          loss = HLoss(preds)               [熵最小化]
          loss.backward()
          SAM.first_step()                  [爬到 sharp peak]
          preds' = net(batch)
          HLoss(preds').backward()
          SAM.second_step()                 [flat minima 方向更新 γ, β]
        → TRAINED


结果记录
─────────────────────────────────────────────────────────────
- 每个评估点记录：accuracy / entropy / energy / confidence
- 按 [PerCorruption] 聚合各 corruption 类型的最终准确率
- 输出到 .txt 日志文件
```

---

## 10. 关键超参数速查表

| 超参数 | CLI 参数 | 默认值 | 含义 |
|---|---|---|---|
| 更新触发间隔 | `--update_every_x` | `64` | 每 N 个新样本触发一次梯度更新 |
| Support Set 容量 | `--memory_size` | `64` | HUS 中最多保留的样本数 |
| 内存类型 | `--memory_type` | `HUS` | `FIFO / HUS / ConfFIFO / CSTU` |
| 置信度阈值 | `--high_threshold` | 依数据集 | HUS 入队的最低置信度（CIFAR10: 0.99）|
| 学习率 | `--lr` | `0.001` | Adam / SAM 基础学习率 |
| 权重衰减 | `--weight_decay` | `0.0005` | L2 正则化系数 |
| 损失温度 | `--temperature` | `1.0` | HLoss 的温度系数 |
| 内部轮数 | `--epoch` | `1` | 每次触发更新的 epoch 数 |
| BN EMA 动量 | `--bn_momentum` | `0.2` | BN 统计量的 EMA 动量 |
| 使用 EMA 统计量 | `--use_learned_stats` | `True` | 是否启用 EMA BN（否则用 batch 统计）|
| 启用 ESM | `--esm` | `True` | 是否使用 SAM 优化器（两步更新）|
| SAM 扰动半径 | `rho` | `0.05` | SAM 爬坡步长 |
| 目标域分布 | `--tgt_train_dist` | `1` | `0=原序 / 1=随机 / 4=Dirichlet` |
| Dirichlet 浓度 | `--dirichlet_beta` | `0.1` | Dirichlet 分布参数（越小相关性越强）|
| batch size | `--batch_size` | `64` | 训练 DataLoader 的 batch 大小 |

---

## 11. 各内存类型对比

| 特性 | FIFO | HUS（SoTTA 默认）| ConfFIFO | CSTU（RoTTA）|
|---|---|---|---|---|
| 置信度过滤 | ✗ | ✓（≥ threshold）| ✓（≥ threshold）| ✗ |
| 类别均衡替换 | ✗ | ✓（替换最大类）| ✗ | ✓（类槽独立管理）|
| 时效性考虑 | ✓（删最旧）| ✗（随机替换）| ✓（删最旧）| ✓（age 打分）|
| 不确定性考虑 | ✗ | 入队时过滤 | 入队时过滤 | ✓（entropy 打分）|
| OOD 鲁棒性 | 低 | 高 | 中 | 低 |
| 类别平衡性 | 取决于流 | 主动平衡 | 取决于流 | 主动平衡 |
| 适用场景 | baseline | Noisy stream | 轻噪声流 | RoTTA 复现 |

---

## 附录：关键代码路径索引

| 功能 | 文件 | 核心函数/类 |
|---|---|---|
| 在线训练主循环 | `learner/sotta.py` | `train_online()` |
| 单步梯度更新 | `learner/sotta.py` | `step()` |
| BN 参数选取 | `learner/sotta.py` | `__init__()` 中的参数冻结逻辑 |
| HUS 内存管理 | `utils/memory.py` | `class HUS` |
| FIFO 推理窗口 | `utils/memory.py` | `class FIFO` |
| 熵最小化损失 | `utils/loss_functions.py` | `class HLoss` |
| 能量得分计算 | `utils/loss_functions.py` | `calc_energy()` |
| SAM 两步优化 | `utils/sam_optimizer.py` | `class SAM` |
| 评估逻辑 | `learner/dnn.py` | `evaluation_online()` |
| 目标域数据顺序 | `learner/dnn.py` | `target_data_processing()` |
| OOD 噪声混合 | `data_loader/NoisyDataset.py` | `class NoisyDataset` |
| 超参数配置 | `conf.py` | `CIFAR10Opt / CIFAR100Opt` |
