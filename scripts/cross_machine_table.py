#!/usr/bin/env python3
"""Cross-machine eval comparison table.

Walks ``results/runs/**/raw.csv`` for every machine, joins with
``machines.csv`` and each run's ``run-metadata.toml``, and prints
per-scheme markdown tables comparing throughput and recall across
machines. Designed for *exploratory* curiosity, not paper data —
absolute timings across machines are confounded by hardware,
cgroup-imposed limits in RunAI pods, and which device actually
ran the eval. The output surfaces those gaps as columns so any
disagreement is visible without averaging it away.

Provenance columns rendered per row:
- ``host-cores`` — kernel's view from ``sysinfo`` (machines.csv).
- ``threads`` — rayon-realised pool size from
  ``run-metadata.toml``. Disagreement with ``host-cores``
  typically means a cgroup-limited container — common in RunAI
  pods when ``--cpu N`` is enforced as a hard cap.
- ``gpu-kind`` — host-available accelerator (machines.csv).
- ``dev`` — what the eval *actually used* per row, from raw.csv's
  ``device`` column. ``dev=cpu`` on a ``gpu-kind=cuda``
  host = GPU was available but this scheme/run didn't use it
  (not every scorer has a GPU path yet).

Usage::

    scripts/cross_machine_table.py
    scripts/cross_machine_table.py --scheme plaintext
    scripts/cross_machine_table.py --results /path/to/results
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results", default=Path("results"), type=Path,
        help="Local results dir (default: %(default)s)",
    )
    p.add_argument(
        "--scheme", default=None,
        help="Only emit table for one scheme (e.g. plaintext, sap, emvp-ivf)",
    )
    return p.parse_args()


def load_machines(results_dir: Path) -> dict[str, dict[str, str]]:
    p = results_dir / "machines.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        return {row["machine-id"]: row for row in csv.DictReader(f)}


def load_run_metadata(run_dir: Path) -> dict:
    p = run_dir / "run-metadata.toml"
    if not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError:
        return {}


def display_machine(mid: str, machines: dict[str, dict[str, str]]) -> str:
    name = (machines.get(mid, {}).get("machine-name") or "").strip()
    return f"{name} ({mid})" if name else mid


def collect_samples(
    results_dir: Path, scheme_filter: str | None,
) -> dict[str, dict[str, list[dict]]]:
    """Walk every raw.csv. Returns ``{scheme: {config_label: [samples]}}``."""
    by_scheme: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list),
    )
    runs_root = results_dir / "runs"
    if not runs_root.exists():
        return by_scheme
    for raw_csv in runs_root.rglob("raw.csv"):
        run_dir = raw_csv.parent
        meta = load_run_metadata(run_dir)
        if not meta:
            continue
        # --breakdown runs leave raw.csv header-only; skip them so
        # they don't pollute throughput stats.
        if meta.get("breakdown"):
            continue
        with raw_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        scheme = rows[0].get("scheme", "")
        if scheme_filter and scheme != scheme_filter:
            continue
        threads = meta.get("parallel-threads")
        for row in rows:
            try:
                lat = float(row["latency-us"])
                rec = float(row["recall-at-k"])
            except (KeyError, ValueError):
                continue
            by_scheme[row["scheme"]][row["config-label"]].append({
                "machine_id": row.get("machine-id", ""),
                "device": row.get("device", "cpu"),
                "parallel_threads": threads,
                "latency_us": lat,
                "recall": rec,
            })
    return by_scheme


def group_by_machine(samples: list[dict]) -> list[dict]:
    """Aggregate samples by (machine, device, threads). Distinct (threads)
    on the same machine ⇒ distinct rows, not averaged together — different
    cgroup shapes."""
    bucket: dict[tuple, list[dict]] = defaultdict(list)
    for s in samples:
        key = (s["machine_id"], s["device"], s["parallel_threads"])
        bucket[key].append(s)
    out = []
    for (mid, dev, threads), items in bucket.items():
        out.append({
            "machine_id": mid,
            "device": dev,
            "parallel_threads": threads,
            "n_samples": len(items),
            "mean_latency_us": mean(s["latency_us"] for s in items),
            "mean_recall": mean(s["recall"] for s in items),
        })
    return out


def fmt_latency(us: float) -> str:
    if us >= 1_000_000:
        return f"{us / 1_000_000:.2f} s"
    if us >= 1_000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.0f} µs"


def render_scheme(
    scheme: str, configs: dict[str, list[dict]],
    machines: dict[str, dict[str, str]],
) -> None:
    print(f"\n## `{scheme}`\n")
    cols = ["config-label", "machine", "cores", "threads",
            "gpu-kind", "dev", "latency", "recall@k", "n"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("-" * (len(c) + 2) for c in cols) + "|")
    for config_label in sorted(configs.keys()):
        rows = group_by_machine(configs[config_label])
        rows.sort(key=lambda r: (
            machines.get(r["machine_id"], {}).get("machine-name", ""),
            r["machine_id"],
            r["device"],
        ))
        for r in rows:
            mrow = machines.get(r["machine_id"], {})
            cores = mrow.get("cores") or "—"
            gpu_kind = mrow.get("gpu-kind") or "—"
            threads = r["parallel_threads"]
            threads_str = "—" if threads is None else str(threads)
            mismatch = (
                threads is not None
                and cores not in ("—", "")
                and str(threads) != str(cores)
            )
            if mismatch:
                threads_str = f"{threads_str}*"
            print(
                f"| {config_label} | {display_machine(r['machine_id'], machines)} "
                f"| {cores} | {threads_str} | {gpu_kind} | {r['device']} "
                f"| {fmt_latency(r['mean_latency_us'])} "
                f"| {r['mean_recall']:.3f} | {r['n_samples']} |"
            )


def main() -> None:
    args = parse_args()
    machines = load_machines(args.results)
    by_scheme = collect_samples(args.results, args.scheme)
    if not by_scheme:
        sys.exit("no runs matched (results dir empty? scheme filter too narrow?)")

    print("# Cross-machine eval comparison")
    print()
    print(f"Source: `{args.results}/runs/**/raw.csv`")
    n_tuples = sum(len(c) for c in by_scheme.values())
    print(
        f"Scope: {len(by_scheme)} scheme(s), {n_tuples} (scheme, config) tuples, "
        f"{len(machines)} machine(s) on file."
    )
    print()
    print("**Caveats** (read before drawing conclusions):")
    print()
    print("- `cores` (machines.csv) vs `threads` (run-metadata.toml's "
          "rayon-realised pool, Plan 13). A `*` after `threads` flags a "
          "mismatch, typically a cgroup-shaped RunAI pod.")
    print("- `gpu-kind` is host-available accelerator; `dev` is what the "
          "eval actually used (Plan 17 `--device` axis). `dev=cpu` on a "
          "`gpu-kind=cuda` host means GPU was present but the scorer ran "
          "on CPU — no Step-1+ feature, or scorer doesn't have a GPU path.")
    print("- Cross-machine *latency* is meaningful only when `cores`, "
          "`threads`, `dev`, and CPU class agree across the rows being "
          "compared. Recall is hardware-independent and always cross-comparable.")
    print()

    for scheme in sorted(by_scheme.keys()):
        render_scheme(scheme, by_scheme[scheme], machines)


if __name__ == "__main__":
    main()
