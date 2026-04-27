#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SCGPT_SCRIPT = REPO_ROOT / "real_pbmc68k_finetune.py"
ANNOTATION_SCRIPT = REPO_ROOT / "label_transfer_finetune.py"
BASELINE_SCRIPT = REPO_ROOT / "benchmark_baselines.py"
ANALYSIS_SCRIPT = REPO_ROOT / "benchmark_analysis.py"


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-style benchmark runner for scGPT experiments")
    parser.add_argument("--mode", choices=["auto", "single", "annotation"], default="auto")
    parser.add_argument("--datasets", nargs="*", default=["pbmc68k_reduced", "paul15"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", default=str(REPO_ROOT / "runs" / "benchmark"))
    parser.add_argument("--methods", nargs="*", default=["scgpt", "pca_logreg", "pca_svm", "pca_knn", "leiden", "majority_class"])
    parser.add_argument("--scgpt-epochs", type=int, default=5)
    parser.add_argument("--scgpt-batch-size", type=int, default=128)
    parser.add_argument("--scgpt-lr", type=float, default=5e-4)
    parser.add_argument("--hvg", type=int, default=128)
    parser.add_argument("--n-bins", type=int, default=51)
    return parser.parse_args()


def write_summary(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("method", "")))
        for k, v in row.items():
            if isinstance(v, (int, float)):
                grouped[key][k].append(float(v))

    agg_rows = []
    for (dataset, method), metrics in grouped.items():
        out = {"dataset": dataset, "method": method}
        for metric_name, values in metrics.items():
            if values:
                out[f"{metric_name}_mean"] = statistics.mean(values)
                out[f"{metric_name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        agg_rows.append(out)

    if agg_rows:
        fieldnames = sorted({k for row in agg_rows for k in row.keys()})
        with (out_dir / "summary_agg.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in agg_rows:
                writer.writerow(row)

    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def is_annotation_dataset(dataset: str) -> bool:
    return dataset in {"gutatlas_transfer"}


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(Path(args.output_root))
    rows: list[dict] = []

    for dataset in args.datasets:
        for seed in args.seeds:
            paired = is_annotation_dataset(dataset)
            use_annotation = args.mode == "annotation" or (args.mode == "auto" and paired)
            scgpt_script = ANNOTATION_SCRIPT if use_annotation else SCGPT_SCRIPT

            if "scgpt" in args.methods:
                run_dir = output_root / dataset / "scgpt" / f"seed_{seed}"
                ensure_dir(run_dir)
                cmd = [
                    sys.executable,
                    str(scgpt_script),
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--epochs", str(args.scgpt_epochs),
                    "--batch-size", str(args.scgpt_batch_size),
                    "--lr", str(args.scgpt_lr),
                    "--hvg", str(args.hvg),
                    "--n-bins", str(args.n_bins),
                    "--run-dir", str(run_dir),
                    "--analysis",
                ]
                run_cmd(cmd)
                analysis_input = run_dir / "analysis" / "analysis_input.h5ad"
                if analysis_input.exists():
                    run_cmd([
                        sys.executable,
                        str(ANALYSIS_SCRIPT),
                        "--input-h5ad", str(analysis_input),
                        "--out-dir", str(run_dir / "analysis"),
                        "--embedding-key", "X_scGPT",
                        "--true-label-key", "true_label",
                        "--pred-label-key", "pred_label",
                        "--seed", str(seed),
                    ])
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    rows.append({"dataset": dataset, "method": "scgpt", "seed": seed, **payload.get("final_test", {}), "best_epoch": payload.get("best_epoch")})

            baseline_dir = output_root / dataset / "baselines" / f"seed_{seed}"
            ensure_dir(baseline_dir)
            if any(m in args.methods for m in ["pca_logreg", "pca_svm", "pca_knn", "leiden", "majority_class"]):
                run_cmd([
                    sys.executable,
                    str(BASELINE_SCRIPT),
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--hvg", str(args.hvg),
                    "--out-dir", str(baseline_dir),
                    "--methods", *[m for m in args.methods if m != "scgpt"],
                ])
                for method in [m for m in args.methods if m != "scgpt"]:
                    method_dir = baseline_dir / method
                    analysis_input = method_dir / "analysis_input.h5ad"
                    if analysis_input.exists():
                        run_cmd([
                            sys.executable,
                            str(ANALYSIS_SCRIPT),
                            "--input-h5ad", str(analysis_input),
                            "--out-dir", str(method_dir / "analysis"),
                            "--embedding-key", "X_pca",
                            "--true-label-key", "true_label",
                            "--pred-label-key", "pred_label",
                            "--seed", str(seed),
                        ])
                    metric_path = method_dir / "metrics.json"
                    if metric_path.exists():
                        metric_payload = json.loads(metric_path.read_text(encoding="utf-8"))
                        rows.append({"dataset": dataset, "method": method, "seed": seed, **metric_payload})

    write_summary(rows, output_root)
    print(json.dumps({"output_root": str(output_root), "n_rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
