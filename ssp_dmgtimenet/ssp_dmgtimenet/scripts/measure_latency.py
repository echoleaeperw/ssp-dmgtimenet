"""Measure single-sample inference latency for every platoon model.

Reports batch=1 forward latency on CPU and GPU using warmup + repeated timing
on a real HighD test sample, with HighD-train normalisation statistics shared
across all models (identical to the evaluation pipeline). GPU timing uses CUDA
events with explicit synchronisation; CPU timing uses ``perf_counter``.

Output: ``artifacts/reports/latency/latency.json`` and a markdown summary table.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch

from ..data.dataset import build_platoon_loaders
from ..training.factory import build_model, initialise_normalisation_for_model
from ..utils.config import load_config

log = logging.getLogger("ssp.latency")

MODELS: list[tuple[str, str, str]] = [
    ("SSP-DMGTimeNet (ours)", "ssp_dmgtimenet_v6", "configs/ssp_dmgtimenet_v6.yaml"),
    ("Int-LSTM", "interaction_lstm", "configs/baseline_int_lstm.yaml"),
    ("Transformer", "platoon_transformer", "configs/baseline_transformer.yaml"),
    ("Full-graph Attention", "full_graph_attention", "configs/baseline_full_graph.yaml"),
    ("LSTM", "platoon_lstm", "configs/baseline_lstm.yaml"),
    ("IDM cascade", "idm_cascade", "configs/baseline_idm.yaml"),
    ("CNN-Int-LSTM-IDM", "cnn_int_lstm_idm", "configs/baseline_cnn_int_lstm_idm.yaml"),
    ("DMGTimeNet cascade", "dmg_cascade", "configs/baseline_dmg_cascade.yaml"),
    ("OVM cascade", "ovm_cascade", "configs/baseline_ovm.yaml"),
    ("FVDM cascade", "fvdm_cascade", "configs/baseline_fvdm.yaml"),
]

OUTPUT_CHANNELS = ("v", "s", "a", "x_rel_leader")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure batch=1 inference latency for all models.")
    parser.add_argument("--base-config", type=Path, default=Path("configs/ssp_dmgtimenet_v6.yaml"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Devices to benchmark; default ['cpu', 'cuda'] when CUDA is available else ['cpu'].",
    )
    parser.add_argument("--cpu-threads", type=int, default=None, help="Override torch CPU thread count.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-json", type=Path, default=Path("../artifacts/reports/latency/latency.json"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    n = len(ordered)
    p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
    return {
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.pstdev(ordered) if n > 1 else 0.0,
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_idx],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "throughput_sps": 1000.0 / statistics.median(ordered),
    }


def _time_forward(
    model: torch.nn.Module,
    history: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    model.eval()
    use_cuda = device.type == "cuda"
    samples_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(history, mask)
        if use_cuda:
            torch.cuda.synchronize()
        for _ in range(repeats):
            if use_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(history, mask)
                end.record()
                torch.cuda.synchronize()
                samples_ms.append(float(start.elapsed_time(end)))
            else:
                t0 = time.perf_counter()
                model(history, mask)
                samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")

    if args.cpu_threads is not None:
        torch.set_num_threads(int(args.cpu_threads))

    if args.devices is not None:
        devices = list(args.devices)
    else:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    base_cfg = load_config(args.base_config)
    paths_section = base_cfg.get("paths", {})
    loaders, normalisation, _ = build_platoon_loaders(
        train_path=Path(paths_section["train"]),
        val_path=Path(paths_section["val"]),
        test_path=Path(paths_section["test"]),
        batch_size=max(int(args.batch_size), 1),
        num_workers=args.num_workers,
        return_raw=True,
    )
    batch = next(iter(loaders["test"]))
    bs = int(args.batch_size)
    history = batch["history_raw"][:bs].contiguous()
    mask = batch["history_mask"][:bs].contiguous()

    output_var_indices = [normalisation.feature_names.index(name) for name in OUTPUT_CHANNELS]
    output_mean = torch.as_tensor(normalisation.mean[output_var_indices], dtype=torch.float32)
    output_std = torch.as_tensor(normalisation.std[output_var_indices], dtype=torch.float32)
    input_mean = torch.as_tensor(normalisation.mean, dtype=torch.float32)
    input_std = torch.as_tensor(normalisation.std, dtype=torch.float32)

    results: list[dict] = []
    for display, dirn, cfg_path in MODELS:
        cfg = load_config(Path(cfg_path))
        model = build_model(cfg["model"])
        ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best.pt"
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict)
        initialise_normalisation_for_model(model, input_mean, input_std, output_mean, output_std)

        params_total = int(sum(p.numel() for p in model.parameters()))
        params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        entry: dict = {
            "display": display,
            "dir": dirn,
            "params_total": params_total,
            "params_trainable": params_trainable,
            "devices": {},
        }
        for dev_name in devices:
            device = torch.device(dev_name)
            model.to(device)
            h = history.to(device)
            m = mask.to(device)
            samples = _time_forward(model, h, m, device, args.warmup, args.repeats)
            entry["devices"][dev_name] = _stats(samples)
            log.info(
                "%-22s %-5s median=%.3f ms  mean=%.3f ms  p95=%.3f ms",
                display,
                dev_name,
                entry["devices"][dev_name]["median_ms"],
                entry["devices"][dev_name]["mean_ms"],
                entry["devices"][dev_name]["p95_ms"],
            )
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(entry)

    out = {
        "batch_size": bs,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "input_shape": list(history.shape),
        "devices": devices,
        "cpu_threads": torch.get_num_threads(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "models": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Latency results written to %s", out_path)

    header = ["model", "params"] + [f"{d}_median_ms" for d in devices]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for entry in results:
        row = [entry["display"], f"{entry['params_total']:,}"]
        for d in devices:
            row.append(f"{entry['devices'][d]['median_ms']:.3f}")
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
