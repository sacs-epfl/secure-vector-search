#!/usr/bin/env python3
"""Refresh the paper's Fig. 6 (CPU vs GPU latency CDF) + Table 1 (Sustained
throughput at recall@10 = 0.9) data files from a chosen set of artefact runs.

The paper uses paper-local TSV names that the artefact's preprocess.py doesn't
emit (the artefact's `write_latency_cdf` doesn't split by device, and the
paper's `cpu-vs-gpu.tsv` is a 4-row summary derived by linear interpolation at
recall=0.9 between bracketing nprobe values). This helper closes that gap by:

  1. Reading 4×2 = 8 raw.csv files (one per (scheme, device) run-id).
  2. Filtering each to a single nprobe operating point (default 64) at
     batch-size = 1, then emitting `latency-cdf-<scheme>-<device>.tsv` with the
     same 500-point downsampled shape `write_latency_cdf` produces.
  3. Building the full per-scheme recall→qps curves across the entire nprobe
     sweep, then linearly interpolating qps at recall=0.9 along (recall, qps)
     for both CPU and GPU. Emits `cpu-vs-gpu.tsv` (4 rows) with the computed
     speedup column.
  4. Rewriting `cpu-vs-gpu-latency-cdf.manifest.toml` and
     `cpu-vs-gpu.manifest.toml` with the new run-ids + a fresh `ported-at`.

Usage:

  python scripts/build_paper_cpu_vs_gpu.py \\
      --paper ../paper-middleware \\
      --machine d860be76 \\
      --cpu plaintext=1779126972,sap-ivf=1779163192,emvp-ivf=1779208414,bntm-ivf=1779263883 \\
      --gpu plaintext=1779450864,sap-ivf=1779462178,emvp-ivf=1779483201,bntm-ivf=1779492472 \\
      --nprobe 64 \\
      --recall-target 0.9

Re-running with the same args is idempotent — same TSVs, same manifests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

SCHEME_ORDER = ["plaintext", "sap-ivf", "emvp-ivf", "bntm-ivf"]
TABLE_SCHEME_LABEL = {
    "plaintext": "plaintext-ivf",
    "sap-ivf":   "sap-ivf",
    "emvp-ivf":  "emvp-ivf",
    "bntm-ivf":  "bntm-ivf",
}


def parse_scheme_runs(arg: str) -> dict[str, str]:
    """`plaintext=1779126972,sap-ivf=1779163192,...` -> {scheme: run-id}."""
    out: dict[str, str] = {}
    for chunk in arg.split(","):
        if not chunk.strip():
            continue
        k, _, v = chunk.strip().partition("=")
        if not k or not v:
            sys.exit(f"bad --cpu/--gpu element {chunk!r}: expected scheme=run-id")
        out[k.strip()] = v.strip()
    for s in SCHEME_ORDER:
        if s not in out:
            sys.exit(f"missing scheme {s!r} in --cpu/--gpu arg")
    return out


def find_run_dir(results_root: pathlib.Path, machine: str, run_id: str) -> pathlib.Path:
    matches = list((results_root / "runs" / machine).glob(f"*/{run_id}"))
    if not matches:
        sys.exit(f"no run dir found for machine={machine} run-id={run_id}")
    if len(matches) > 1:
        sys.exit(f"multiple run dirs matched run-id={run_id}: {matches}")
    return matches[0]


def load_raw(run_dir: pathlib.Path) -> pd.DataFrame:
    raw = run_dir / "raw.csv"
    if not raw.exists():
        sys.exit(f"no raw.csv under {run_dir}")
    df = pd.read_csv(raw)
    # Normalise kebab → snake to match preprocess.py.
    df.columns = [c.replace("-", "_") for c in df.columns]
    return df


def cdf_downsampled(latencies_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort + downsample to a ~500-point regular grid with an n−1 endpoint.

    Matches the sampling the paper's pre-existing TSVs use:
    `np.arange(0, n, n//500)` (500 evenly stepped indices starting at 0) with
    `n-1` appended so the curve always closes at cdf = 1.0. The result has
    501 points when n is a multiple of 500's step, +/-1 in degenerate cases.
    """
    samples = np.sort(latencies_ms.astype(float))
    n = len(samples)
    if n == 0:
        return np.empty(0), np.empty(0)
    if n > 500:
        step = max(1, n // 500)
        idx = np.concatenate([np.arange(0, n, step), [n - 1]])
        idx = np.unique(idx)
        return samples[idx], (idx + 1) / n
    return samples, np.arange(1, n + 1) / n


def emit_latency_cdf_tsv(
    df: pd.DataFrame, scheme: str, device: str, nprobe: int, dst: pathlib.Path
) -> int:
    """Filter rows for (scheme, device, nprobe, batch_size=1) and write TSV.

    Returns the number of source rows that fed the CDF (informational).
    """
    sub = df[
        (df["scheme"] == scheme)
        & (df["device"] == device)
        & (df["quality_param_name"] == "nprobe")
        & (df["quality_param"] == nprobe)
        & (df["batch_size"] == 1)
    ]
    if sub.empty:
        sys.exit(
            f"{scheme} {device}: no rows for nprobe={nprobe} batch_size=1 — "
            f"check raw.csv contents"
        )
    latency_ms = (sub["latency_us"].astype(float) / 1000.0).to_numpy()
    pts, cdf = cdf_downsampled(latency_ms)
    # %.4f matches the precision the paper's pre-existing TSVs use.
    pd.DataFrame({"latency_ms": pts, "cdf": cdf}).to_csv(
        dst, sep="\t", index=False, float_format="%.4f"
    )
    return len(latency_ms)


def interp_qps_at_recall(
    df: pd.DataFrame, scheme: str, device: str, recall_target: float
) -> float:
    """Linearly interpolate qps at recall=recall_target along (recall, qps).

    For each nprobe in the sweep, compute mean recall and mean latency across
    all rows at (scheme, device, batch_size=1, that nprobe); derive qps =
    1000/latency_ms. Sort by recall ascending; piecewise-linear interpolate at
    recall_target. If recall_target is outside the observed envelope, snap to
    the nearest endpoint (consistent with how the paper's Table 1 values were
    computed against the original CPU-vs-GPU sweep).
    """
    sub = df[
        (df["scheme"] == scheme)
        & (df["device"] == device)
        & (df["quality_param_name"] == "nprobe")
        & (df["batch_size"] == 1)
    ]
    if sub.empty:
        sys.exit(f"{scheme} {device}: no nprobe sweep rows")

    by_qp = sub.groupby("quality_param", as_index=False).agg(
        recall=("recall_at_k", "mean"),
        latency_us=("latency_us", "mean"),
    )
    by_qp["qps"] = 1_000_000.0 / by_qp["latency_us"]
    by_qp = by_qp.sort_values("recall").reset_index(drop=True)
    recalls = by_qp["recall"].to_numpy()
    qpss = by_qp["qps"].to_numpy()
    if recall_target <= recalls[0]:
        return float(qpss[0])
    if recall_target >= recalls[-1]:
        return float(qpss[-1])
    return float(np.interp(recall_target, recalls, qpss))


def emit_table1_tsv(
    rows: list[dict], recall_target: float, machine: str, dst: pathlib.Path
) -> None:
    """Write the 4-row `cpu-vs-gpu.tsv` summary (Table 1)."""
    header = (
        f"# CPU vs GPU throughput at recall@10 = {recall_target} on the 8.8 M MS MARCO corpus.\n"
        f"# Same host (sacs006, machine-id {machine}): dual-socket Xeon Gold 6426Y (64 threads)\n"
        "# and an NVIDIA RTX 5000 Ada. Throughput at the target recall is the linear\n"
        "# interpolation between bracketing nprobe values along (recall, qps).\n"
    )
    df = pd.DataFrame(rows, columns=["scheme", "cpu_qps", "gpu_qps", "speedup"])
    body = df.to_csv(sep="\t", index=False, float_format="%.2f")
    dst.write_text(header + body)


def git_sha(repo: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def write_manifest_cdf(
    dst: pathlib.Path,
    artefact: pathlib.Path,
    sha: str,
    machine: str,
    nprobe: int,
    recall_target: float,
    cpu_runs: dict[str, str],
    gpu_runs: dict[str, str],
    tsv_files: list[str],
) -> None:
    lines = [
        f'slug             = "cpu-vs-gpu-latency-cdf"',
        f'source-repo      = "{artefact}"',
        f'source-repo-sha  = "{sha}"',
        f'machine-id       = "{machine}"',
        f'ported-at        = "{dt.datetime.now().astimezone().isoformat(timespec="seconds")}"',
        f'operating-point  = "nprobe={nprobe} (recall@10 ≈ {recall_target} across all four schemes)"',
        "",
        "# All eight runs are on the same host (sacs006), 64 parallel threads,",
        "# 8.8 M MS MARCO passages. Companion to plots/cpu-vs-gpu/.",
        "[runs.cpu]",
    ]
    for s in SCHEME_ORDER:
        lines.append(f'{s:<9} = "{cpu_runs[s]}"')
    lines += ["", "[runs.gpu]"]
    for s in SCHEME_ORDER:
        lines.append(f'{s:<9} = "{gpu_runs[s]}"')
    lines += ["", "[data]", "files = ["]
    for tsv in tsv_files:
        lines.append(f'  "{tsv}",')
    lines.append("]")
    dst.write_text("\n".join(lines) + "\n")


def write_manifest_table(
    dst: pathlib.Path,
    artefact: pathlib.Path,
    sha: str,
    machine: str,
    cpu_runs: dict[str, str],
    gpu_runs: dict[str, str],
) -> None:
    lines = [
        f'slug             = "cpu-vs-gpu"',
        f'source-repo      = "{artefact}"',
        f'source-repo-sha  = "{sha}"',
        f'machine-id       = "{machine}"',
        f'ported-at        = "{dt.datetime.now().astimezone().isoformat(timespec="seconds")}"',
        "",
        "# All eight runs are on the same host (sacs006), 64 parallel threads,",
        "# 8.8 M MS MARCO passages.",
        "[runs.cpu]",
    ]
    for s in SCHEME_ORDER:
        lines.append(f'{s:<9} = "{cpu_runs[s]}"')
    lines += ["", "[runs.gpu]"]
    for s in SCHEME_ORDER:
        lines.append(f'{s:<9} = "{gpu_runs[s]}"')
    lines += ["", "[data]", 'files = ["cpu-vs-gpu.tsv"]']
    dst.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--paper",
        type=pathlib.Path,
        default=pathlib.Path("../paper-middleware"),
        help="paper repo root (default: ../paper-middleware)",
    )
    ap.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path("results"),
        help="artefact results root (default: ./results)",
    )
    ap.add_argument("--machine", required=True, help="machine-id, e.g. d860be76")
    ap.add_argument("--cpu", required=True, help="scheme=run-id,... for CPU runs")
    ap.add_argument("--gpu", required=True, help="scheme=run-id,... for GPU runs")
    ap.add_argument("--nprobe", type=int, default=64, help="CDF operating point")
    ap.add_argument(
        "--recall-target", type=float, default=0.9, help="Table 1 interp target"
    )
    ap.add_argument(
        "--write-tsvs",
        default="all",
        help=(
            "comma-separated scheme names whose latency-cdf-*.tsv files to "
            "(over)write; default 'all'. Use e.g. --write-tsvs sap-ivf to "
            "refresh only the SAP-IVF CDFs while preserving the other six "
            "TSVs verbatim. Table 1 + manifests are always recomputed."
        ),
    )
    args = ap.parse_args()
    if args.write_tsvs == "all":
        write_schemes = set(SCHEME_ORDER)
    else:
        write_schemes = {s.strip() for s in args.write_tsvs.split(",") if s.strip()}
        unknown = write_schemes - set(SCHEME_ORDER)
        if unknown:
            sys.exit(f"--write-tsvs: unknown schemes {sorted(unknown)}")

    cpu_runs = parse_scheme_runs(args.cpu)
    gpu_runs = parse_scheme_runs(args.gpu)

    artefact = pathlib.Path.cwd().resolve()
    paper = args.paper.expanduser().resolve()
    results = args.results.expanduser().resolve()
    sha = git_sha(artefact)

    cdf_dir = paper / "plots" / "cpu-vs-gpu-latency-cdf"
    tab_dir = paper / "plots" / "cpu-vs-gpu"
    if not cdf_dir.exists() or not tab_dir.exists():
        sys.exit(f"paper plot dirs missing under {paper}/plots/")

    # Load all 8 raw.csvs once per (scheme, device).
    raw: dict[tuple[str, str], pd.DataFrame] = {}
    for device, runs in (("cpu", cpu_runs), ("gpu", gpu_runs)):
        for scheme, run_id in runs.items():
            rd = find_run_dir(results, args.machine, run_id)
            df = load_raw(rd)
            # Schema check.
            for col in ("scheme", "device", "quality_param_name", "quality_param",
                        "batch_size", "latency_us", "recall_at_k"):
                if col not in df.columns:
                    sys.exit(f"{rd}/raw.csv missing column {col}")
            raw[(scheme, device)] = df
            print(f"loaded {scheme:<10} {device}  run={run_id}  rows={len(df)}  from {rd}")

    # 1. Fig 6 — eight latency-cdf TSVs. The `--write-tsvs` filter lets us
    # refresh only the schemes whose run-id actually changed, preserving the
    # others verbatim (avoids spurious diff noise from sampling-algorithm
    # variation across regenerations).
    tsv_files: list[str] = []
    for s in SCHEME_ORDER:
        for d in ("cpu", "gpu"):
            tsv = f"latency-cdf-{s}-{d}.tsv"
            tsv_files.append(tsv)
            if s in write_schemes:
                n = emit_latency_cdf_tsv(raw[(s, d)], s, d, args.nprobe, cdf_dir / tsv)
                print(f"  wrote {tsv}  (sourced from {n} rows at nprobe={args.nprobe} B=1)")
            else:
                print(f"  kept {tsv}  (--write-tsvs filter)")

    # 2. Table 1 — cpu-vs-gpu.tsv via recall-target interpolation.
    table_rows: list[dict] = []
    for s in SCHEME_ORDER:
        cpu_qps = interp_qps_at_recall(raw[(s, "cpu")], s, "cpu", args.recall_target)
        gpu_qps = interp_qps_at_recall(raw[(s, "gpu")], s, "gpu", args.recall_target)
        speedup = gpu_qps / cpu_qps if cpu_qps > 0 else float("nan")
        table_rows.append(
            {
                "scheme":   TABLE_SCHEME_LABEL[s],
                "cpu_qps":  cpu_qps,
                "gpu_qps":  gpu_qps,
                "speedup":  speedup,
            }
        )
        print(f"  table1 {s:<10}  cpu={cpu_qps:6.2f}  gpu={gpu_qps:6.2f}  speedup={speedup:.2f}")
    emit_table1_tsv(table_rows, args.recall_target, args.machine, tab_dir / "cpu-vs-gpu.tsv")
    print(f"  wrote cpu-vs-gpu.tsv")

    # 3. Manifests.
    write_manifest_cdf(
        cdf_dir / "cpu-vs-gpu-latency-cdf.manifest.toml",
        artefact, sha, args.machine, args.nprobe, args.recall_target,
        cpu_runs, gpu_runs, tsv_files,
    )
    write_manifest_table(
        tab_dir / "cpu-vs-gpu.manifest.toml",
        artefact, sha, args.machine, cpu_runs, gpu_runs,
    )
    print(f"  refreshed both manifests")

    print()
    print(f"Done. Paper update area: {paper}/plots/")
    print("Next: re-render the figure / re-typeset the table from the paper's Makefile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
