# 原始数据放置说明

本目录**不随代码分发**。HighD 与 NGSIM 均受各自许可协议约束，请自行申请后按下列结构放置。

## HighD（主训练 / 主测试）

申请：[highD dataset](https://ika.rwth-aachen.de/en/competences/projects/automated-driving/highd-dataset.html)

将 60 段录像的 CSV 放到 `datasets/highD/`（文件名形如 `01_tracks.csv`）：

```text
datasets/highD/
├── 01_tracks.csv
├── 01_tracksMeta.csv
├── 01_recordingMeta.csv
├── ...
└── 60_tracks.csv
```

官方 zip 解压后若多一层 `data/`，把其中的 CSV 拷到本目录即可。脚本用 `*_tracks.csv` 发现录像，不依赖中间目录名。

## NGSIM（零样本泛化 + 平滑敏感性）

来源：FHWA / data.gov。本实验只用 **US-101** 与 **I-80** 高速公路 15 分钟时段，不要用 Peachtree / Lankershim。

```text
datasets/NGSIM/vehicle-trajectory-data/
├── 0750am-0805am/trajectories-0750am-0805am.csv      # US-101
├── 0805am-0820am/trajectories-0805am-0820am.csv
├── 0820am-0835am/trajectories-0820am-0835am.csv
├── 0400pm-0415pm/trajectories-0400-0415.csv          # I-80
├── 0400pm-0415pm/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv
├── 0500pm-0515pm/trajectories-0500-0515.csv
└── 0515pm-0530pm/trajectories-0515-0530.csv
```

每个时段目录里应恰好有一个不以 `RECONSTRUCTED` 开头的 `trajectories-*.csv`。Montanino–Punzo 重构轨迹仅用于 I-80 16:00–16:15 的敏感性实验。
