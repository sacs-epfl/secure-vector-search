#!/usr/bin/env python3
"""Validation gate for the Tiptoe Rust port vs. the Go reference.

Reads results/index.csv, picks the most-recent `tiptoe` and `tiptoe-go`
runs on the current machine, computes per-config recall@10 difference
and per-query top-10 ID overlap, emits a human-readable report, and
returns non-zero on failure.

Pass criteria:
  - |recall@10(rust) − recall@10(go)| ≤ 2 pp per common config
  - mean per-query top-10 ID overlap ≥ 80% per common config

Usage: see `make tiptoe-diff`.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

RECALL_TOLERANCE = 0.02   # 2 percentage points
OVERLAP_FLOOR = 0.80      # 80%


def latest_run(index_rows: list[dict], scheme: str, machine_id: str) -> dict | None:
    """Most-recent successful run of `scheme` on `machine_id`. Picks by
    started-at ISO 8601 (lexically sortable) then run-id (Unix seconds
    string) as tiebreaker."""
    candidates = [
        r
        for r in index_rows
        if r["scheme"] == scheme
        and r["machine-id"] == machine_id
        and r["status"] == "complete"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: (r["started-at"], r["run-id"]))


def load_csv_dicts(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def per_config_mean_recall(raw_rows: Iterable[dict]) -> dict[str, float]:
    """{config-label → mean recall@10 across queries}."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in raw_rows:
        by[r["config-label"]].append(float(r["recall-at-k"]))
    return {label: sum(xs) / len(xs) for label, xs in by.items()}


def per_config_query_topk(top_k_rows: Iterable[dict]) -> dict[str, dict[int, list[int]]]:
    """{config-label → {query_id → top-k doc IDs in rank order}}."""
    by: dict[str, dict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for r in top_k_rows:
        label = r["config-label"]
        qid = int(r["query-id"])
        rank = int(r["rank"])
        doc = int(r["doc-id"])
        by[label][qid][rank] = doc
    out: dict[str, dict[int, list[int]]] = {}
    for label, qmap in by.items():
        out[label] = {qid: [m[r] for r in sorted(m)] for qid, m in qmap.items()}
    return out


def mean_overlap(
    rust_topk: dict[int, list[int]],
    go_topk: dict[int, list[int]],
    k: int,
) -> tuple[float, int]:
    """Mean per-query Jaccard-like overlap |A∩B| / k over the union of qids,
    plus the number of queries compared. Queries missing from either side
    are skipped."""
    common_qids = sorted(set(rust_topk) & set(go_topk))
    if not common_qids:
        return 0.0, 0
    total = 0.0
    for qid in common_qids:
        a = set(rust_topk[qid][:k])
        b = set(go_topk[qid][:k])
        total += len(a & b) / k
    return total / len(common_qids), len(common_qids)


def write_report(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--machine-id",
        help="Override machine-id selector. Default: most-recent tiptoe run's machine.",
    )
    p.add_argument("--k", type=int, default=10)
    p.add_argument(
        "--require-data",
        action="store_true",
        help=(
            "Treat 'no paired data available' as a hard failure (exit 1) "
            "instead of a clean skip (exit 2). Use this once the paired "
            "tiptoe/tiptoe-go raw.csv + top_k.csv are present (e.g. after "
            "hydrating from the bulk store) so the gate cannot pass by "
            "skipping — see `make tiptoe-diff`."
        ),
    )
    args = p.parse_args()

    # Exit code used when the comparison cannot run because the paired
    # data is absent. Default 2 = clean skip (raw CSVs are bulk-stored
    # off-checkout, so a fresh CI checkout legitimately has nothing to compare).
    # --require-data flips it to 1 so a hydrated run enforces the gate
    # rather than silently passing.
    no_data_rc = 1 if args.require_data else 2

    index_path = args.results_dir / "index.csv"
    if not index_path.exists():
        print(f"error: {index_path} not found", file=sys.stderr)
        return no_data_rc

    index_rows = load_csv_dicts(index_path)

    if args.machine_id:
        machine_id = args.machine_id
    else:
        tiptoe_runs = [r for r in index_rows if r["scheme"] == "tiptoe"]
        if not tiptoe_runs:
            print("error: no tiptoe runs in results/index.csv", file=sys.stderr)
            return no_data_rc
        machine_id = max(
            tiptoe_runs, key=lambda r: (r["started-at"], r["run-id"])
        )["machine-id"]

    rust = latest_run(index_rows, "tiptoe", machine_id)
    go = latest_run(index_rows, "tiptoe-go", machine_id)
    if rust is None or go is None:
        print(
            f"error: need both `tiptoe` and `tiptoe-go` runs on machine {machine_id}\n"
            f"       found tiptoe={rust is not None}, tiptoe-go={go is not None}",
            file=sys.stderr,
        )
        return no_data_rc

    rust_dir = args.results_dir / rust["path"]
    go_dir = args.results_dir / go["path"]

    # The paired runs are registered in index.csv but their raw.csv /
    # top_k.csv may be bulk-stored and absent from the checkout.
    # Treat that as "no data" (skip / --require-data fail), not a crash.
    needed = [
        rust_dir / "raw.csv", go_dir / "raw.csv",
        rust_dir / "top_k.csv", go_dir / "top_k.csv",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print(
            "error: paired runs are registered but their data files are not "
            "present (bulk-stored?). Hydrate before validating:\n"
            + "\n".join(f"       missing {p}" for p in missing),
            file=sys.stderr,
        )
        return no_data_rc

    rust_raw = load_csv_dicts(rust_dir / "raw.csv")
    go_raw = load_csv_dicts(go_dir / "raw.csv")
    rust_topk = load_csv_dicts(rust_dir / "top_k.csv")
    go_topk = load_csv_dicts(go_dir / "top_k.csv")

    rust_recall = per_config_mean_recall(rust_raw)
    go_recall = per_config_mean_recall(go_raw)
    rust_topk_by_cfg = per_config_query_topk(rust_topk)
    go_topk_by_cfg = per_config_query_topk(go_topk)

    common_labels = sorted(set(rust_recall) & set(go_recall))

    lines: list[str] = []
    lines.append("Tiptoe validation gate — Rust port vs. Go reference")
    lines.append("=" * 60)
    lines.append(f"machine-id : {machine_id}")
    lines.append(f"tiptoe     : run-id={rust['run-id']}  git={rust['git-sha'][:8]}  path={rust['path']}")
    lines.append(f"tiptoe-go  : run-id={go['run-id']}  git={go['git-sha'][:8]}  path={go['path']}")
    lines.append("")
    if not common_labels:
        lines.append("FAIL: no shared config-label between the two runs.")
        lines.append(f"  tiptoe configs:    {sorted(rust_recall)}")
        lines.append(f"  tiptoe-go configs: {sorted(go_recall)}")
        write_report(go_dir / "tiptoe-diff.txt", lines)
        for line in lines:
            print(line)
        return 1

    overall_pass = True
    for label in common_labels:
        rr = rust_recall[label]
        gr = go_recall[label]
        diff = abs(rr - gr)
        recall_pass = diff <= RECALL_TOLERANCE

        overlap, n_q = mean_overlap(
            rust_topk_by_cfg.get(label, {}),
            go_topk_by_cfg.get(label, {}),
            args.k,
        )
        overlap_pass = overlap >= OVERLAP_FLOOR

        cfg_pass = recall_pass and overlap_pass
        overall_pass &= cfg_pass

        lines.append(f"config: {label}")
        lines.append(
            f"  recall@{args.k}: rust={rr:.4f} go={gr:.4f} "
            f"Δ={diff:.4f} ({'≤' if recall_pass else '>'} {RECALL_TOLERANCE:.2f}) "
            f"{'PASS' if recall_pass else 'FAIL'}"
        )
        lines.append(
            f"  top-{args.k} overlap: {overlap:.4f} over {n_q} queries "
            f"({'≥' if overlap_pass else '<'} {OVERLAP_FLOOR:.2f}) "
            f"{'PASS' if overlap_pass else 'FAIL'}"
        )
        lines.append(f"  → {'PASS' if cfg_pass else 'FAIL'}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    write_report(go_dir / "tiptoe-diff.txt", lines)
    for line in lines:
        print(line)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
