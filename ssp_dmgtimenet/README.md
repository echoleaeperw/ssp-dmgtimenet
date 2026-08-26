# SSP-DMGTimeNet

**完整实验步骤与复现说明请先阅读上一级 [`../README.md`](../README.md)。** 本文件只说明包内模块与设计对应关系。

队列尺度扰动传播预测与弦稳定性约束（方案 C）的工程实现。

模型代号：**SSP-DMGTimeNet**（String-Stability-aware Sequential Propagation DMGTimeNet）。

研究目标：在已完成的 DMGTimeNet（DACA + HGF + CFE + MSO，单链非平稳跟驰预测）基础上，将研究对象扩展为同车道连续 N(>=5) 车队列，预测未来 3-5 s 的纵向轨迹，并显式评估预测轨迹是否会让扰动沿队列向后放大。

详细设计依据：[`../scheme_c_detailed_plan.md`](../scheme_c_detailed_plan.md)。

## 目录布局

```
ssp_dmgtimenet/
├── ssp_dmgtimenet/        # Python 包源码
│   ├── data/              # HighD / NGSIM / OpenACC 加载与队列样本生成
│   ├── models/            # SP-DACA, Cross-CFE, HGF, 多任务头, 主模型
│   ├── losses/            # 预测/运动学/安全/稳定性(adj/sub/fft)/协整/时滞正则
│   ├── metrics/           # MAE/RMSE/A_i/A_{j->i}/FFT gain/phase delay/安全指标
│   ├── baselines/         # IDM, OVM/FVDM, LSTM, Int-LSTM, Transformer, Full-graph, DMG cascade, CNN-Int-LSTM-IDM
│   ├── training/          # Trainer + curriculum + evaluator + logging
│   ├── analysis/          # 时滞分布、传播延迟散点、扰动时空热图、Pareto
│   └── utils/             # config / seed / filters / io
├── scripts/               # 命令行入口
│   ├── audit_highd_platoons.py
│   ├── build_platoon_samples.py
│   ├── train.py
│   ├── evaluate.py
│   └── plot_interpretability.py
├── configs/               # YAML 配置
├── tests/                 # 单元测试
├── pyproject.toml
└── README.md
```

数据与产物集中在仓库根的 `datasets/` 与 `artifacts/`：

```
datasets/
├── highD/        # 来自 RWTH Aachen 官方申请，*_tracks.csv / *_tracksMeta.csv / *_recordingMeta.csv
├── NGSIM/        # FHWA / data.gov，建议使用重构/平滑版本
└── OpenACC/      # JRC 数据页

artifacts/
├── platoons/     # 提取的队列样本（.npz / .h5）
├── checkpoints/  # 模型权重
├── figures/      # 可解释性、稳定性图
└── reports/      # 审计报告与实验表格
```

## 安装

依赖管理使用标准的 `pyproject.toml`，建议在独立 venv 或 conda 环境内安装：

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -e .[dev]
```

需要支持 CUDA 的 PyTorch，请按照 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/) 选择匹配 CUDA 版本的轮子覆盖默认 CPU 版本。

## 数据准备

### HighD (主数据集)

1. 在 [RWTH Aachen 官网](https://ika.rwth-aachen.de/en/competences/projects/automated-driving/highd-dataset.html) 申请并下载 HighD（60 段录像）。
2. 将解压后的 `data/` 目录复制到 `datasets/highD/`，确保结构形如 `datasets/highD/01_tracks.csv` 等。
3. 跑数据审计：

```bash
python -m ssp_dmgtimenet.scripts.audit_highd_platoons \
    --highd-root datasets/highD \
    --report-dir artifacts/reports/highd_audit \
    --max-N 7 --min-N 3
```

审计脚本会输出每个 `recording * laneId * N * stationarity-threshold` 条件下的可用窗口数，并写出 CSV / Markdown 报告，作为「主实验 N=5 是否成立」的 Go/No-Go 依据。

4. 构建队列样本：

```bash
python -m ssp_dmgtimenet.scripts.build_platoon_samples \
    --highd-root datasets/highD \
    --out-dir artifacts/platoons/highd_N5_h5_p3 \
    --target-hz 10 \
    --N 5 \
    --history-sec 5 \
    --predict-sec 3 \
    --stride-sec 1.0 \
    --nonstationary-quantile 0.5
```

### NGSIM / OpenACC

NGSIM 使用 I-80 / US-101，必须使用平滑或重构版本（避免速度噪声污染稳定性指标）。OpenACC 用作工程 case study，不做主训练。

## 训练 / 评估

```bash
python -m ssp_dmgtimenet.scripts.train \
    --config configs/ssp_dmgtimenet.yaml

python -m ssp_dmgtimenet.scripts.evaluate \
    --config configs/ssp_dmgtimenet.yaml \
    --checkpoint artifacts/checkpoints/ssp_dmgtimenet_best.pt
```

`configs/ssp_dmgtimenet.yaml` 给出了主实验设置（`N=5`、`history=5 s`、`predict=3 s`、curriculum 三段式损失加权、稳定性正则的 ramp-up 计划）。`configs/baselines.yaml` 给出 IDM / OVM / LSTM / Int-LSTM / Transformer / Full-graph / DMG cascade / CNN-Int-LSTM-IDM 的训练入口。

## 度量与可解释性

- 预测精度：分变量（v / s / a）MAE / RMSE，horizon-wise（1/2/3/5 s），vehicle-wise（C2..CN），队尾误差。
- 稳定性：$A_i$、$A_{j\to i}$、Unstable window ratio、Exceedance area、FFT gain、Phase delay。
- 安全 / 工程：collision risk、jerk-based comfort、gap violation rate、推理时延。
- 可解释性：`scripts/plot_interpretability.py` 输出 $\tau_i$ 沿队列分布、真实 vs 预测传播延迟散点、扰动时空热图、子队列 $A_{j\to i}$ 热图、$G(f)$ 频域增益、Pareto。

## 与方案文档的对应

| 方案 C 段落 | 实现位置 |
| --- | --- |
| §4.4 HighD 队列提取流程 | `ssp_dmgtimenet/data/platoons.py`, `scripts/audit_highd_platoons.py`, `scripts/build_platoon_samples.py` |
| §5.2 SP-DACA | `ssp_dmgtimenet/models/sp_daca.py` |
| §5.3 Cross-Vehicle CFE | `ssp_dmgtimenet/models/cross_cfe.py`, `ssp_dmgtimenet/losses/cointegration.py` |
| §5.4 HGF | `ssp_dmgtimenet/models/hgf.py` |
| §5.5 稳定性损失 | `ssp_dmgtimenet/losses/stability.py` |
| §5.6 总损失 + curriculum | `ssp_dmgtimenet/losses/total.py`, `ssp_dmgtimenet/training/curriculum.py` |
| §6 实验设计 | `scripts/train.py`, `scripts/evaluate.py`, `configs/*.yaml` |
| §7 评价指标 | `ssp_dmgtimenet/metrics/*.py` |
| §8 Baseline | `ssp_dmgtimenet/baselines/*.py` |

## 开发约束

- 严禁模拟数据：未拿到 HighD / NGSIM / OpenACC 之前不构造合成数据掩饰；脚本会在数据缺失时抛错而非走兜底分支。
- 严禁回退机制：训练失败、稳定性损失发散等情形必须暴露，不做 silent fallback。
- 严禁简化：所有方案 C 中提到的损失与指标均按照公式实现，不做近似裁剪。
