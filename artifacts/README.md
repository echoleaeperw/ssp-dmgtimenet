# 运行产物目录

训练与评估脚本会在此写入中间数据与结果。提交代码包时该目录为空，复现过程中会自动创建。

| 子目录 | 内容 |
| --- | --- |
| `platoons/` | 由原始轨迹切出的队列窗口 `.npz` |
| `checkpoints/` | `best.pt` / `last.pt` |
| `reports/` | 训练过程验证报告 |
| `tensorboard/` | TensorBoard 日志 |
| `evaluation_v3/` | 论文主表对应的 test 报告、表格与图 |
| `figures/` | 可解释性图、轨迹案例图 |
