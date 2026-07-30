#!/usr/bin/env python3
"""Inventory of every report snapshot under ``results/runs/*/reports/``.

For each machine, lists the report snapshots that have been rendered
(``results/runs/<machine-id>/reports/<snapshot>/``), most-recent first.
Each snapshot's ``manifest.toml`` is the source of truth for the
``(scheme, run-id)`` tuples that fed it; the snapshot dir's ``report.pdf``
mtime is the generation timestamp. The ``latest`` symlink target is
marked ``← latest``.

Snapshots predating the manifest format show ``(no manifest)`` and only
expose mtime + dir name. Pass ``--scheme <name>`` to drill into a single
scheme: prints, for each snapshot, the run-id wired in plus its
config-label and start time pulled from the run's
``run-metadata.toml``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", default=Path("results"), type=Path)
    p.add_argument(
        "--machine", default=None,
        help="Restrict to a machine-id (dir under results/runs/) or a "
             "machine-name from results/machines.csv. A name like 'sacs006' "
             "that maps to multiple ids matches all of them.",
    )
    p.add_argument(
        "--scheme", default=None,
        help="Drill into a single scheme: print each snapshot's run-id, "
             "config-label, and start time for that scheme.",
    )
    return p.parse_args()


def load_machines(results: Path) -> dict[str, dict[str, str]]:
    p = results / "machines.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        return {row["machine-id"]: row for row in csv.DictReader(f)}


def load_toml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError:
        return None


def fmt_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime(
        "%Y-%m-%d %H:%M %Z"
    )


def fmt_started(iso_utc: str) -> str:
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


def find_run_dir(runs_root: Path, machine_id: str, run_id: str) -> Path | None:
    """Locate ``runs/<machine-id>/<git-sha>/<run-id>/`` for a run-id."""
    machine_root = runs_root / machine_id
    if not machine_root.is_dir():
        return None
    for sha_dir in machine_root.iterdir():
        if not sha_dir.is_dir() or sha_dir.name == "reports":
            continue
        candidate = sha_dir / run_id
        if candidate.is_dir():
            return candidate
    return None


def run_summary(run_dir: Path | None) -> tuple[str, str]:
    """Return ``(config-label, started-at)`` for a run dir, blanks on miss."""
    if run_dir is None:
        return ("", "")
    meta = load_toml(run_dir / "run-metadata.toml") or {}
    started = meta.get("started-at", "")
    config = ""
    raw = run_dir / "raw.csv"
    if raw.exists():
        try:
            with raw.open(newline="") as f:
                first = next(csv.DictReader(f), None)
            if first is not None:
                config = first.get("config-label", "")
        except (OSError, csv.Error):
            pass
    return (config, started)


def collect(results: Path) -> dict[str, list[dict]]:
    """``{machine_id: [snapshot_record, ...]}`` sorted newest first."""
    runs_root = results / "runs"
    if not runs_root.is_dir():
        return {}
    out: dict[str, list[dict]] = {}
    for machine_dir in sorted(runs_root.iterdir()):
        reports_dir = machine_dir / "reports"
        if not reports_dir.is_dir():
            continue
        latest_target = None
        latest_link = reports_dir / "latest"
        if latest_link.is_symlink():
            latest_target = Path(latest_link.readlink()).name
        snapshots: list[dict] = []
        for snap in reports_dir.iterdir():
            if snap.is_symlink() or not snap.is_dir():
                continue
            pdf = snap / "report.pdf"
            mtime = pdf.stat().st_mtime if pdf.exists() else snap.stat().st_mtime
            manifest = load_toml(snap / "manifest.toml")
            snapshots.append({
                "name": snap.name,
                "path": snap,
                "pdf": pdf,
                "mtime": mtime,
                "manifest": manifest,
                "has_pdf": pdf.exists(),
                "is_latest": snap.name == latest_target,
            })
        snapshots.sort(key=lambda s: s["mtime"], reverse=True)
        if snapshots:
            out[machine_dir.name] = snapshots
    return out


def machine_header(mid: str, info: dict) -> str:
    name = (info.get("machine-name") or "").strip()
    cpu = (info.get("cpu-model") or "?").strip()
    cores = info.get("cores", "?")
    ram_b = info.get("ram-bytes", "0")
    try:
        ram = f"{int(ram_b) / 1024**3:.0f} GB"
    except ValueError:
        ram = "?"
    gpu = (info.get("gpu-kind") or "none").strip()
    gpu_str = f", gpu={gpu}" if gpu and gpu != "none" else ""
    lhs = f"{name} ({mid})" if name else mid
    return f"{lhs} — {cpu}, {cores}c, {ram}{gpu_str}"


def render(
    inventory: dict[str, list[dict]],
    machines: dict[str, dict[str, str]],
    runs_root: Path,
    scheme_filter: str | None,
) -> None:
    if not inventory:
        print("(no reports found)")
        return
    total = sum(len(v) for v in inventory.values())
    print(f"results/runs/*/reports/ — {len(inventory)} machine(s), {total} snapshot(s)")
    print()
    for mid in sorted(
        inventory,
        key=lambda m: ((machines.get(m, {}).get("machine-name") or "").lower(), m),
    ):
        print(machine_header(mid, machines.get(mid, {})))
        for snap in inventory[mid]:
            tag = "  ← latest" if snap["is_latest"] else ""
            stamp = fmt_mtime(snap["mtime"])
            pdf_line = (
                f"    pdf:   {snap['pdf']}" if snap["has_pdf"]
                else "    pdf:   (missing)"
            )
            manifest = snap["manifest"]
            if manifest is None:
                print(f"  {snap['name']:<32} {stamp}{tag}  (no manifest)")
                print(pdf_line)
                continue
            runs = manifest.get("runs", {}) or {}
            if scheme_filter:
                if scheme_filter not in runs:
                    continue
                run_id = str(runs[scheme_filter])
                run_dir = find_run_dir(runs_root, mid, run_id)
                cfg, started = run_summary(run_dir)
                started_str = f"  ({fmt_started(started)})" if started else ""
                cfg_str = f"  {cfg}" if cfg else ""
                print(f"  {snap['name']:<32} {stamp}{tag}")
                print(pdf_line)
                print(f"    {scheme_filter:<12} run {run_id}{cfg_str}{started_str}")
            else:
                schemes = ", ".join(f"{s}={r}" for s, r in sorted(runs.items()))
                print(f"  {snap['name']:<32} {stamp}{tag}")
                print(pdf_line)
                print(f"    runs:  {schemes}")
        print()


def main() -> None:
    args = parse_args()
    machines = load_machines(args.results)
    inventory = collect(args.results)
    if args.machine is not None:
        name_to_ids: dict[str, set[str]] = {}
        for mid, info in machines.items():
            name = (info.get("machine-name") or "").strip()
            if name:
                name_to_ids.setdefault(name, set()).add(mid)
        if args.machine in name_to_ids:
            wanted = name_to_ids[args.machine]
        else:
            wanted = {args.machine}
        missing = wanted - inventory.keys()
        if missing == wanted:
            print(
                f"warning: {args.machine!r} has no reports on file "
                f"({len(inventory)} machine(s) present: "
                f"{', '.join(sorted(inventory)) or '∅'}).",
                file=sys.stderr,
            )
        inventory = {m: snaps for m, snaps in inventory.items() if m in wanted}
    render(inventory, machines, args.results / "runs", args.scheme)


if __name__ == "__main__":
    main()
