# Replication package explanatory file

本文件对应 COMMTR / JICV 投稿模板 *Replication package explanatory file*。正式上交请使用同目录下的 **`Replication_package_explanatory_file.docx`**。红色/`[PLEASE FILL]` 处需作者填写后删除本说明句。

**Journal title:** [PLEASE FILL: Communications in Transportation Research, or Journal of Intelligent and Connected Vehicles]

**Manuscript title:** String-stability-aware sequential propagation network for vehicle-platoon prediction: SSP-DMGTimeNet

**Manuscript ID:** [PLEASE FILL after Editorial Manager assignment]

**Authors:** [PLEASE FILL: Name1, Name2, ...]

---

## For COMMTR or JICV article

This explanatory file is submitted together with the manuscript and is intended for review by a replication editor. A short summary (no more than 100 words) to be placed in the manuscript immediately before the Acknowledgements is provided at the end of this file. Please insert the public repository URL and replace “[the replication package was approved]” after the editor’s confirmation.

## Replication package access

The replication package consists of two parts.

**(1) Code.** All source code, YAML configs, unit tests, and one-click scripts needed to rebuild platoon windows, train models, evaluate the three-layer stability protocol, and regenerate tables/figures are provided in the folder submitted with this file (the `submission/` package). **[PLEASE FILL the public URL after depositing the same folder on GitHub / Zenodo / ETS-Data.]** If an open platform cannot be used, we will request ETS-Data access through the replication editor.

**(2) Raw trajectory data.** HighD and NGSIM cannot be redistributed with this package because of their licenses and data-use agreements. They must be obtained from the official providers using the steps below. After placement, the bundled scripts reconstruct every training and test split used in the paper, so the key tables can be regenerated without any hidden data.

HighD (primary training and testing data):

- Apply at https://levelxdata.com/highd-dataset/ (RWTH Aachen / ika highD).
- Download the 60 recordings and copy the CSV files to `datasets/highD/` so that `01_tracks.csv`, `01_tracksMeta.csv`, `01_recordingMeta.csv`, …, `60_tracks.csv` are present.

NGSIM US-101 and I-80 (zero-shot evaluation and smoothing sensitivity):

- Obtain the FHWA NGSIM vehicle-trajectory files from data.gov / the NGSIM community archive.
- Place the six 15-minute freeway periods under `datasets/NGSIM/vehicle-trajectory-data/` as listed in `datasets/README.md`.
- For the I-80 16:00–16:15 sensitivity study, also place the Montanino–Punzo reconstructed file `RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv` in the same period folder.

## Data description

Complete raw trajectories are not uploaded. Only derived processing code is provided. This is required by the HighD license and by FHWA NGSIM redistribution terms, not by a desire to withhold results. The uploaded package still generates the key results because every number in the main HighD, NGSIM, ablation, platoon-length, and sensitivity tables is produced by deterministic scripts from the official CSVs.

Derived data produced by the package (not redistributed here, rebuilt locally):

- Format: NumPy `.npz` windows with history/future tensors of shape `[B, T, N, F]` at 10 Hz, plus vehicle identifiers and recording ids.
- HighD splits by recording id: train 01–45, validation 46–50, test 51–60. Main experiment uses N = 5 vehicles, 5 s history, 3 s prediction, 1 s stride, leader-stationarity quantile q = 0.5.
- NGSIM: US-101 and I-80, up to 4,000 windows per site, same window protocol; normalisation statistics remain those of the HighD training split (zero-shot).
- Platoon-length extensions rebuild N = 3, 6, 7 with the same protocol.

After `bash scripts/prepare_highd.sh` and `bash scripts/prepare_ngsim.sh`, the files `artifacts/platoons/highd_N5_h5_p3/{train,val,test}.npz` and `artifacts/platoons/ngsim_N5_h5_p3/{us101,i80}/test.npz` are the exact inputs of the reported experiments.

## Code description

Computer environment required to run the scripts:

- OS: Linux (bash scripts). Python 3.10, 3.11, or 3.12.
- Deep learning: PyTorch ≥ 2.0 with CUDA recommended. One GPU with ≥ 12 GB memory is sufficient for batch size 64; CPU can run evaluation but training is impractical.
- Python dependencies: see `requirements.txt`.
- Install: `python -m venv .venv && source .venv/bin/activate && pip install -U pip && pip install torch && pip install -e "./ssp_dmgtimenet[dev]"`.
- Sanity check without data: `bash scripts/smoke_test.sh`.

What the code contains:

- `ssp_dmgtimenet/`: model (SP-DACA, Cross-CFE, HGF), losses, metrics, HighD/NGSIM loaders, training and evaluation entry points.
- `ssp_dmgtimenet/configs/`: paper configs, including `ssp_dmgtimenet_v6.yaml` (reported model), 9 baselines, 8 ablations, and N-extension YAMLs.
- `scripts/`: data preparation, training, evaluation, and table/figure generation.

Step-by-step commands are in `README.md`. The reported model is **v6**. Evaluation uses δ = 0.05 and excitation floor 0.05 m/s.

## Simulation software description

Not applicable. This manuscript does not use traffic microsimulation software (SUMO, VISSIM, AIMSUN, etc.). All experiments are supervised learning and evaluation on reconstructed trajectory windows in PyTorch.

## Experiment design description

The manuscript does not include surveys, questionnaires, or human-subject experiments. The computational experiment design is as follows.

- Task: given a same-lane platoon of N vehicles, predict the next 3 s of longitudinal state from 5 s of history, and score string stability of the predicted trajectories.
- Main comparison (HighD, N = 5): SSP-DMGTimeNet versus five learned baselines and four physics/hybrid baselines.
- Ablations (8): delay bias, adjacent loss, CFE, sub-platoon loss, HGF, FFT loss, full-graph mask, fixed τ.
- Generalisation: zero-shot NGSIM US-101 / I-80; I-80 original vs reconstructed trajectories; platoon length N ∈ {3, 5, 6, 7}.
- Stability protocol v3 is identical for every model (detection / unified GT-referenced response / conditional internal amplification).

## Others

- After the raw CSVs are in place: `bash scripts/reproduce_all.sh`.
- Full training takes several GPU-days. To verify the pipeline: `bash scripts/train_all.sh ssp_dmgtimenet_v6` then `bash scripts/evaluate_highd.sh`.
- Resume with `SKIP_EXISTING=1`.
- Bit-wise identity of GPU metrics is not expected; ranking and order of magnitude should match.

## Manuscript summary (≤ 100 words; insert before Acknowledgements)

> The code to reconstruct platoon windows, train SSP-DMGTimeNet and all baselines, and regenerate the reported tables is available at [PLEASE FILL URL]. Raw HighD and NGSIM trajectories are not redistributed; they must be obtained from the official providers as described in the replication package. Derived training/test splits are rebuilt by the bundled scripts. [The replication package was approved by the replication editor.]

Word count excluding the two bracketed placeholders: 78.
