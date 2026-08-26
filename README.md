# SSP-DMGTimeNet 复现代码包

本目录是论文实验的**独立可复现提交包**：只含源码、配置、测试与一键脚本，不含 HighD / NGSIM 原始数据，也不含已训练权重。按本文步骤即可从原始轨迹重建队列样本、训练、评估，并生成与正文对应的表格和图。

**投稿模板文件（COMMTR / JICV *Replication package explanatory file*）**请用：

- [`Replication_package_explanatory_file.docx`](Replication_package_explanatory_file.docx)（随稿上交）
- [`Replication_package_explanatory_file.md`](Replication_package_explanatory_file.md)（同一内容的 Markdown 备份）

`README.md` 是给复现者看的逐步命令；Word 要求的栏目（access / data / code environment / simulation / experiment design / others）在上述 explanatory file 里。投稿前请把文件中的 `[PLEASE FILL]`（期刊名、作者、稿件编号、代码公开链接）填上。

包内模块说明见 [`ssp_dmgtimenet/README.md`](ssp_dmgtimenet/README.md)。

---

## 1. 本包包含 / 不包含

**包含**

- Python 包 `ssp_dmgtimenet/`（模型、损失、指标、数据管线、训练/评估入口）
- 论文用到的全部 YAML 配置（主模型、10 个对照、8 个消融、N=3/6/7 扩展）
- 稳定性协议 v3 的单元测试
- 数据准备、训练、评估、出表出图脚本

**不包含（许可与体积）**

- HighD / NGSIM 原始 CSV（需自行申请，放置方式见 [`datasets/README.md`](datasets/README.md)）
- `best.pt` 等权重、`.npz` 窗口、TensorBoard 日志

---

## 2. 目录结构

```text
submission/
├── README.md                          # 本文件：实验步骤与复现说明
├── requirements.txt
├── datasets/                          # 自行放入 HighD / NGSIM
├── artifacts/                         # 运行后自动生成
├── scripts/
│   ├── smoke_test.sh                  # 安装自检
│   ├── prepare_highd.sh               # 切 HighD 队列窗口
│   ├── prepare_ngsim.sh               # 切 NGSIM 测试窗口
│   ├── train_all.sh                   # 训练
│   ├── evaluate_highd.sh              # HighD 主实验 + 消融评估
│   ├── evaluate_extensions.sh         # NGSIM / 敏感性 / N 扩展评估
│   ├── make_tables_and_figures.sh     # 论文表与汇总图
│   ├── plot_optional.sh               # 可解释性 / 轨迹案例 / 时延
│   └── reproduce_all.sh               # 上述步骤串起来
└── ssp_dmgtimenet/                    # 可安装 Python 包（在此目录执行 python -m ...）
    ├── configs/
    ├── ssp_dmgtimenet/
    └── tests/
```

配置里的数据路径是相对于 **`ssp_dmgtimenet/` 工作目录** 的，例如 `../artifacts/platoons/highd_N5_h5_p3/train.npz`。请使用本包提供的脚本（脚本会先 `cd` 到该目录），不要在仓库根目录直接调用 `python -m` 除非你同时改了路径。

---

## 3. 环境

| 项目 | 要求 |
| --- | --- |
| Python | 3.10、3.11 或 3.12 |
| GPU | 一块 CUDA GPU，显存建议 ≥ 12 GB（`batch_size=64`） |
| 磁盘 | HighD 原始数据约数十 GB；切出的 `.npz` 与 checkpoint 另计 |
| 系统 | Linux（脚本按 bash 编写） |

单卡完整复现（主实验 10 模型 + 消融 8 模型 + N 扩展 9 模型）通常需要**数天**。可先只跑 `ssp_dmgtimenet_v6` 验证通路，再并行或分批训练其余模型。

---

## 4. 安装

在 **`submission/`** 根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 按本机 CUDA 改写
pip install -e "./ssp_dmgtimenet[dev]"
```

无 GPU 时安装 CPU 版 PyTorch，并把后面命令加上 `DEVICE=cpu`。评估协议本身可在 CPU 上跑，训练会非常慢。

安装后做冒烟测试（不需要数据）：

```bash
bash scripts/smoke_test.sh
```

应打印包版本、`torch` 版本、CUDA 是否可用，并跑通 `tests/`。

---

## 5. 实验协议（必须与论文一致）

下面参数已写进配置和评估脚本，**不要改**，否则数字对不上正文。

### 5.1 窗口与划分

| 项 | 取值 |
| --- | --- |
| 采样率 | 10 Hz |
| 历史 / 预测 | 5 s / 3 s（50 / 30 步） |
| 滑窗步长 | 1.0 s |
| 队列长度（主实验） | N = 5 |
| 非平稳过滤 | 领头车速度/加速度标准差分位数 `q = 0.5` |
| HighD 划分 | 训练 rec. 1–45，验证 46–50，测试 51–60 |
| 随机种子 | `trainer.seed = 42` |
| 主模型训练 | 80 epoch，batch 64，AdamW `3e-4`，AMP |

### 5.2 测试评估口径（稳定性协议 v3）

所有 `evaluate` 调用统一使用：

- `--delta-unstable 0.05`（不稳定阈值为 `1 + δ`）
- `--excitation-floor 0.05`（去趋势领头车 RMS，单位 m/s）

三层协议：

1. **扰动检测**：GT 与预测领头车共用同一去趋势 RMS 阈值。
2. **统一外部响应（主比较）**：窗口只由 GT 领头车选定，所有模型强制同一批窗口，分母为 GT 领头车幅值。
3. **条件内部稳定（诊断）**：仅在 `GT excited AND prediction excited` 上报告沿队列放大率，并同时报告条件样本量。

归一化：优先使用 checkpoint 内保存的均值/方差；仅无缓冲区的物理模型才用当前训练集统计。NGSIM 零样本**不重新归一化**，仍用 HighD `train.npz` 的统计（通过配置里的 `paths.train`）。

---

## 6. 逐步复现

以下命令均在 **`submission/`** 根目录、已激活虚拟环境的前提下执行。

### 步骤 A. 放入原始数据

按 [`datasets/README.md`](datasets/README.md) 放置 HighD 与 NGSIM。最少需要：

- `datasets/highD/01_tracks.csv` … `60_tracks.csv` 及对应 `*Meta.csv`
- `datasets/NGSIM/vehicle-trajectory-data/` 下 US-101 / I-80 六个 15 分钟时段

### 步骤 B. 生成队列样本

```bash
bash scripts/prepare_highd.sh
bash scripts/prepare_ngsim.sh
```

成功后应存在：

```text
artifacts/platoons/highd_N5_h5_p3/{train,val,test}.npz
artifacts/platoons/highd_N{3,6,7}_h5_p3/{train,val,test}.npz
artifacts/platoons/ngsim_N5_h5_p3/{us101,i80}/test.npz
artifacts/platoons/ngsim_sensitivity/{i80_orig_0400,i80_recon_0400}/test.npz
```

中断后续跑可加 `SKIP_EXISTING=1`，已有 `train.npz` / `test.npz` 的集合会被跳过。

### 步骤 C. 训练

```bash
# 只跑论文主模型（建议先验证）
bash scripts/train_all.sh ssp_dmgtimenet_v6

# 主实验 10 个模型（学习对照 + 物理对照）
bash scripts/train_all.sh main

# 8 个消融（均从 ssp_dmgtimenet_v6.yaml 改一项）
bash scripts/train_all.sh ablation

# N=3/6/7 × {SSP-v6, Int-LSTM, Transformer}
bash scripts/train_all.sh n_ext

# 以上全部
bash scripts/train_all.sh all
```

断点续跑：

```bash
SKIP_EXISTING=1 bash scripts/train_all.sh all
```

已存在的 `artifacts/checkpoints/<name>/best.pt` 会被跳过。N 扩展权重在 `artifacts/checkpoints/n_ext_N{3,6,7}/<name>/best.pt`。

指定 GPU：

```bash
DEVICE=cuda TRAIN_WORKERS=4 bash scripts/train_all.sh main
```

训练日志：`artifacts/logs/train/`。

### 步骤 D. HighD 测试评估

```bash
bash scripts/evaluate_highd.sh
```

每个模型写出：

- `artifacts/evaluation_v3/reports/<name>/test_report.md`
- `artifacts/evaluation_v3/reports/<name>/test_report.json`（含数据/配置/权重哈希与归一化来源）

### 步骤 E. 扩展实验评估

```bash
bash scripts/evaluate_extensions.sh
```

覆盖：

- NGSIM US-101 / I-80 零样本（HighD 训练权重 + NGSIM `test.npz`）
- I-80 16:00–16:15 原始轨迹 vs Montanino–Punzo 重构轨迹
- 队列长度 N = 3 / 5 / 6 / 7（N=5 复用主实验权重）

### 步骤 F. 生成论文表与汇总图

```bash
bash scripts/make_tables_and_figures.sh
```

| 产物 | 路径 |
| --- | --- |
| HighD 主表 + 消融表 | `artifacts/evaluation_v3/tables.md` |
| NGSIM / 敏感性 / N 扩展表 | `artifacts/evaluation_v3/extension_tables.md` |
| 协议汇总图 | `artifacts/evaluation_v3/figures/highd_protocol_v3.png` |
| N 扩展图 | `artifacts/evaluation_v3/figures/n_extension_v3.png` |

### 步骤 G.（可选）可解释性、轨迹案例、推理时延

```bash
bash scripts/plot_optional.sh
```

需要至少已经训练完 SSP-v6、Int-LSTM、Transformer。

### 一键全流程

数据放好并且环境已装好之后：

```bash
bash scripts/reproduce_all.sh
```

等价于依次执行：冒烟测试 → HighD 切窗 → NGSIM 切窗 → 全部训练 → HighD 评估 → 扩展评估 → 出表出图。

---

## 7. 模型与配置对照

### 7.1 主实验（HighD N=5）

| 论文名称 | 配置 | checkpoint 目录 |
| --- | --- | --- |
| SSP-DMGTimeNet | `configs/ssp_dmgtimenet_v6.yaml` | `artifacts/checkpoints/ssp_dmgtimenet_v6/` |
| Int-LSTM | `configs/baseline_int_lstm.yaml` | `artifacts/checkpoints/interaction_lstm/` |
| Transformer | `configs/baseline_transformer.yaml` | `artifacts/checkpoints/platoon_transformer/` |
| Full-graph Attention | `configs/baseline_full_graph.yaml` | `artifacts/checkpoints/full_graph_attention/` |
| LSTM | `configs/baseline_lstm.yaml` | `artifacts/checkpoints/platoon_lstm/` |
| CNN-Int-LSTM-IDM | `configs/baseline_cnn_int_lstm_idm.yaml` | `artifacts/checkpoints/cnn_int_lstm_idm/` |
| IDM cascade | `configs/baseline_idm.yaml` | `artifacts/checkpoints/idm_cascade/` |
| DMGTimeNet cascade | `configs/baseline_dmg_cascade.yaml` | `artifacts/checkpoints/dmg_cascade/` |
| OVM cascade | `configs/baseline_ovm.yaml` | `artifacts/checkpoints/ovm_cascade/` |
| FVDM cascade | `configs/baseline_fvdm.yaml` | `artifacts/checkpoints/fvdm_cascade/` |

论文采用 **v6**（`ssp_dmgtimenet_v6.yaml`），不要用 `ssp_dmgtimenet.yaml` / `v5.yaml` 当最终结果。后两者仅保留作消融历史。

### 7.2 消融（均相对 v6 只改一项）

| 消融 | 配置 | 改动 |
| --- | --- | --- |
| w/o delay bias | `ablation_wo_delay_bias.yaml` | 关闭 SP-DACA 时滞偏置，并去掉 delay 损失 |
| w/o adj | `ablation_wo_adj.yaml` | `L_adj = 0` |
| w/o CFE | `ablation_wo_cfe.yaml` | 去掉 CFE 分支与协整损失 |
| full graph | `ablation_full_graph.yaml` | 空间掩码改为全图 |
| w/o sub | `ablation_wo_sub.yaml` | `L_sub = 0` |
| w/o HGF | `ablation_wo_hgf.yaml` | 多尺度门控改为均匀平均 |
| fixed tau | `ablation_fixed_tau.yaml` | `τ` 固定为 1.0 s |
| w/o FFT | `ablation_wo_fft.yaml` | `L_fft = 0` |

### 7.3 N 扩展

`configs/n_ext_N{3,6,7}_{ssp_v6,int_lstm,transformer}.yaml` 只改 `num_vehicles` 和对应 `highd_N*_h5_p3` 路径。N=5 直接复用主实验配置与权重。

---

## 8. 手动单条命令（脚本内部实际调用）

所有训练 / 评估都在 `ssp_dmgtimenet/` 下执行。

```bash
cd ssp_dmgtimenet

python -m ssp_dmgtimenet.scripts.train \
  --config configs/ssp_dmgtimenet_v6.yaml \
  --device cuda --num-workers 4

python -m ssp_dmgtimenet.scripts.evaluate \
  --config configs/ssp_dmgtimenet_v6.yaml \
  --checkpoint ../artifacts/checkpoints/ssp_dmgtimenet_v6/best.pt \
  --split test --device cuda --num-workers 2 \
  --delta-unstable 0.05 --excitation-floor 0.05 \
  --out-markdown ../artifacts/evaluation_v3/reports/ssp_dmgtimenet_v6/test_report.md
```

NGSIM 零样本只需再加 `--test-path`，**不要改**配置里的 `paths.train`：

```bash
python -m ssp_dmgtimenet.scripts.evaluate \
  --config configs/ssp_dmgtimenet_v6.yaml \
  --checkpoint ../artifacts/checkpoints/ssp_dmgtimenet_v6/best.pt \
  --split test \
  --test-path ../artifacts/platoons/ngsim_N5_h5_p3/us101/test.npz \
  --delta-unstable 0.05 --excitation-floor 0.05 \
  --out-markdown ../artifacts/evaluation_v3/extensions/ngsim_us101/ssp_dmgtimenet_v6/test_report.md
```

---

## 9. 验收检查

评估全部完成后，同一 HighD 测试集上应满足：

1. 所有模型的 `detection_n_total` 与 `gt_ref_n_windows` 一致。
2. `conditional_internal_n_windows = detection_tp`。
3. 每份 `test_report.json` 含数据、配置、checkpoint 的 SHA-256，以及实际使用的归一化来源（`checkpoint_buffers` 或 `current_train_data`）。

可用：

```bash
python scripts/build_evaluation_v3_manifest.py
```

生成 `artifacts/evaluation_v3/dataset_manifest.json`，核对当前 `.npz` / 权重指纹。

---

## 10. 常见问题

**Q: 数字和论文不完全相同？**  
A: 训练有 GPU 非确定性。请确认：数据划分与窗口参数未改、评估使用 `δ=0.05` 与 `excitation_floor=0.05`、主模型是 v6、NGSIM 仍用 HighD 训练集统计。数量级与排序应可复现；逐位浮点一致不是目标。

**Q: HighD 找不到录像？**  
A: `*_tracks.csv` 必须直接位于 `datasets/highD/`，不要停在 `highD/data/` 或 `highD-dataset-v1.0/` 里还不拷出来。

**Q: NGSIM 报 “Expected exactly one trajectories-*.csv”？**  
A: 每个时段目录只能有一个非 `RECONSTRUCTED` 的 `trajectories-*.csv`。重构文件名必须以 `RECONSTRUCTED` 开头，敏感性实验会单独指定路径。

**Q: 显存不够？**  
A: 可把对应 YAML 里 `trainer.batch_size` 从 64 改为 32，并按比例增大梯度累积（本仓库 trainer 未内置 accumulation，改 batch 后指标会有小幅漂移）。N 扩展 N=7 更占显存。

**Q: `plot_paper_results.py` 能不能用来出正文图？**  
A: 不要。该脚本内嵌了旧表数字。正文复现请只用 `scripts/plot_evaluation_v3.py`，它读取本次评估写出的 JSON。

**Q: 环境变量有哪些？**

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `DEVICE` | `cuda` | `cuda` 或 `cpu` |
| `TRAIN_WORKERS` | `4` | DataLoader workers |
| `EVAL_WORKERS` | `2` | 评估 workers |
| `SKIP_EXISTING` | `0` | 设为 `1` 时跳过已有 npz / checkpoint |
| `HIGHD_ROOT` | `datasets/highD` | HighD CSV 根目录 |
| `NGSIM_ROOT` | `datasets/NGSIM/vehicle-trajectory-data` | NGSIM 时段根目录 |

---

## 11. 代码入口速查

| 任务 | 模块 |
| --- | --- |
| HighD 加载 | `ssp_dmgtimenet.data.highd` |
| NGSIM 加载 | `ssp_dmgtimenet.data.ngsim` |
| 队列提取 / 切窗 | `ssp_dmgtimenet.data.platoons`, `...windowing` |
| 主模型 | `ssp_dmgtimenet.models.ssp_dmgtimenet` |
| SP-DACA / CFE / HGF | `models/sp_daca.py`, `cross_cfe.py`, `hgf.py` |
| 稳定性损失 | `ssp_dmgtimenet.losses.stability` |
| 稳定性指标（含 v3） | `ssp_dmgtimenet.metrics.stability` |
| 训练 | `python -m ssp_dmgtimenet.scripts.train` |
| 评估 | `python -m ssp_dmgtimenet.scripts.evaluate` |
