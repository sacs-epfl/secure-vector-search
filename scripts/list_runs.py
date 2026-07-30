#!/usr/bin/env python3
"""Inventory of every complete run under ``results/runs/``.

Quick "what's in the bag" overview for each machine: how many runs
per scheme, the configs covered, time span, breakdown / no-cache /
device flags. Reads ``run-metadata.toml`` per run; ignores
incomplete and breakdown-mode runs by default unless ``--all`` is
given.

Output groups by machine, schemes sorted by run count:

    sacs006 (4d874634) — Intel(R) Xeon(R) Gold 6426Y, 64c, 126 GB
      53 runs  2026-05-04 16:14 → 2026-05-08 11:54 CEST
      plaintext        8 runs  nprobe={1,2,4,8,16,32,64,128}
      sap-ivf          9 runs  beta=0 | nprobe={1,…,128}
      …

Run with ``--scheme <name>`` to drill into a single scheme's runs
(prints one line per run-id with its config-label).
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", default=Path("results"), type=Path)
    p.add_argument(
        "--all", action="store_true",
        help="Include incomplete and --breakdown runs (default: skip).",
    )
    p.add_argument(
        "--scheme", default=None,
        help="Restrict to one scheme; prints per-run details.",
    )
    p.add_argument(
        "--machine", default=None,
        help="Restrict to one machine-id (matches against the path "
             "under results/runs/).",
    )
    return p.parse_args()


def load_machines(results: Path) -> dict[str, dict[str, str]]:
    p = results / "machines.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        return {row["machine-id"]: row for row in csv.DictReader(f)}


def load_meta(run_dir: Path) -> dict | None:
    p = run_dir / "run-metadata.toml"
    if not p.exists():
        return None
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError:
        return None


def utc_to_local(iso_utc: str) -> str:
    if not iso_utc:
        return ""
    s = iso_utc.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return iso_utc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def summarise_configs(configs: list[str]) -> str:
    """Compact rendering of a config-label list."""
    if not configs:
        return ""
    nprobes: set[int] = set()
    betas: set[str] = set()
    quants: set[int] = set()
    other: set[str] = set()
    for c in configs:
        if c.startswith("nprobe="):
            try:
                nprobes.add(int(c.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        if "beta=" in c and "nprobe=" in c:
            try:
                # e.g. "beta=0.0000|nprobe=32"
                parts = dict(p.split("=", 1) for p in c.split("|"))
                betas.add(parts["beta"])
                nprobes.add(int(parts["nprobe"]))
                continue
            except (KeyError, ValueError):
                pass
        if c.startswith("beta="):
            betas.add(c.split("=", 1)[1])
            continue
        if c.startswith("quantisation-bits="):
            try:
                quants.add(int(c.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        other.add(c)
    bits = []
    if nprobes:
        bits.append(f"nprobe={{{','.join(str(n) for n in sorted(nprobes))}}}")
    if betas:
        # Strip trailing zeros for readability.
        cleaned = sorted({b.rstrip("0").rstrip(".") or "0" for b in betas})
        bits.append(f"beta={{{','.join(cleaned)}}}")
    if quants:
        bits.append(f"q={{{','.join(str(q) for q in sorted(quants))}}}")
    if other:
        bits.extend(sorted(other))
    return "  ".join(bits)


def collect(results: Path, include_all: bool):
    """``{machine_id: {scheme: [(run_id, config_label, started_at, meta)]}}``."""
    runs_root = results / "runs"
    if not runs_root.exists():
        return {}
    out: dict = defaultdict(lambda: defaultdict(list))
    for raw in runs_root.rglob("raw.csv"):
        run_dir = raw.parent
        meta = load_meta(run_dir)
        if meta is None:
            continue
        if not include_all:
            if meta.get("status") not in (None, "complete"):
                continue
            if meta.get("breakdown"):
                continue
        try:
            rel = run_dir.relative_to(runs_root)
        except ValueError:
            continue
        if len(rel.parts) < 3:
            continue
        machine_id = rel.parts[0]
        run_id = rel.parts[2]
        # Pull a single config-label from the first raw.csv row. Most runs
        # cover one (scheme, config) tuple; the harness writes the config
        # column on every row.
        try:
            with raw.open(newline="") as f:
                reader = csv.DictReader(f)
                first = next(reader, None)
        except (OSError, csv.Error):
            continue
        if first is None:
            continue
        scheme = first.get("scheme", "")
        config = first.get("config-label", "")
        started = meta.get("started-at", "")
        out[machine_id][scheme].append((run_id, config, started, meta))
    return out


def render_summary(
    inventory: dict, machines: dict, scheme_filter: str | None,
) -> None:
    if not inventory:
        print("(no runs found)")
        return

    total_runs = sum(
        sum(len(v) for v in by_scheme.values())
        for by_scheme in inventory.values()
    )
    schemes_seen = sorted({s for by_scheme in inventory.values() for s in by_scheme})
    print(
        f"results/runs/ — {len(inventory)} machine(s), {total_runs} run(s) "
        f"across {len(schemes_seen)} scheme(s)"
    )
    print()

    machine_ids = sorted(
        inventory,
        key=lambda m: ((machines.get(m, {}).get("machine-name") or "").lower(), m),
    )
    for mid in machine_ids:
        info = machines.get(mid, {})
        name = (info.get("machine-name") or "").strip()
        cpu = (info.get("cpu-model") or "?").strip()
        cores = info.get("cores", "?")
        ram_b = info.get("ram-bytes", "0")
        try:
            ram = f"{int(ram_b) / 1024**3:.0f} GB"
        except ValueError:
            ram = "?"
        gpu = (info.get("gpu-kind") or "none").strip()
        gpu_str = f", gpu={gpu}" if gpu and gpu != "none" else ""
        header = f"{name + ' (' + mid + ')' if name else mid} — {cpu}, {cores}c, {ram}{gpu_str}"
        print(header)

        # Per-machine totals + time span.
        by_scheme = inventory[mid]
        all_runs = [r for runs in by_scheme.values() for r in runs]
        n_runs = len(all_runs)
        starts = sorted(r[2] for r in all_runs if r[2])
        span = ""
        if starts:
            span = f"  {utc_to_local(starts[0])} → {utc_to_local(starts[-1])}"
        print(f"  {n_runs} runs{span}")

        scheme_order = sorted(by_scheme, key=lambda s: -len(by_scheme[s]))
        for scheme in scheme_order:
            if scheme_filter and scheme != scheme_filter:
                continue
            runs = by_scheme[scheme]
            configs = sorted({r[1] for r in runs if r[1]})
            cfg_str = summarise_configs(configs)
            print(f"    {scheme:<14} {len(runs):>3} runs  {cfg_str}")
            if scheme_filter:
                # Drill-down: one line per run.
                for run_id, cfg, started, meta in sorted(runs, key=lambda r: r[0]):
                    flags = []
                    if meta.get("no-cache"):
                        flags.append("no-cache")
                    if meta.get("breakdown"):
                        flags.append("breakdown")
                    if meta.get("device") and meta["device"] != "cpu":
                        flags.append(f"device={meta['device']}")
                    flag_str = f"  [{', '.join(flags)}]" if flags else ""
                    print(
                        f"      {run_id}  {utc_to_local(started)}  "
                        f"{cfg}{flag_str}"
                    )
        print()


def main() -> None:
    args = parse_args()
    machines = load_machines(args.results)
    inventory = collect(args.results, args.all)
    if args.machine is not None:
        if args.machine not in inventory:
            print(
                f"warning: machine-id {args.machine!r} has no runs on file "
                f"({len(inventory)} machine(s) present: "
                f"{', '.join(sorted(inventory)) or '∅'}).",
                file=sys.stderr,
            )
        inventory = {
            m: by_scheme for m, by_scheme in inventory.items()
            if m == args.machine
        }
    render_summary(inventory, machines, args.scheme)


if __name__ == "__main__":
    main()
