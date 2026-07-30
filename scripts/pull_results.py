#!/usr/bin/env python3
"""Pull eval results from a remote host into the local results/ tree.

Fetches remote ``index.csv`` + ``machines.csv``, groups runs by
``machine-id`` so you can pick one (or all), then ``rsync``s the
selected run directories down. ``index.csv`` and ``machines.csv``
are merged only with rows for runs that were actually copied — local
files don't accumulate references to remote-only data.

Default remote points at the EPFL RCP jumphost staging tree:
``jumphost.rcp.epfl.ch:/mnt/sacs/scratch/secure-vector-search/results``.
Override with ``--remote`` for other hosts (e.g. sacs006 once a sweep
finishes there).

Usage::

    scripts/pull_results.py                    # interactive
    scripts/pull_results.py --machine ee8fb9ad # pre-select one machine
    scripts/pull_results.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REMOTE = (
    "rpereira@jumphost.rcp.epfl.ch:/mnt/sacs/scratch/secure-vector-search/results"
)
DEFAULT_LOCAL = "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help="Remote results dir as <host>:<path> (default: %(default)s)",
    )
    p.add_argument(
        "--local",
        default=DEFAULT_LOCAL,
        help="Local results dir (default: %(default)s)",
    )
    p.add_argument(
        "--machine",
        default=None,
        help="Pre-select a machine-id; skip the interactive prompt",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied/merged without touching local files",
    )
    return p.parse_args()


def split_remote(remote: str) -> tuple[str, str]:
    if ":" not in remote:
        sys.exit(f"--remote must be host:path, got {remote!r}")
    host, _, path = remote.partition(":")
    return host, path.rstrip("/")


def ssh_cat(host: str, path: str) -> str:
    r = subprocess.run(
        ["ssh", host, "cat", path],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        sys.exit(f"ssh {host} cat {path} failed:\n{r.stderr.strip()}")
    return r.stdout


def read_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def utc_to_local(iso_utc: str) -> str:
    """Render an ISO 8601 UTC timestamp in the system's local timezone.

    The CSVs always store UTC for stable cross-machine analysis; the listing
    only is reformatted for human readability. Falls back to the input
    string verbatim if parsing fails."""
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


def summarise(rows: list[dict[str, str]]) -> dict[str, dict]:
    by_machine: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "min_ts": "", "max_ts": "", "schemes": set()},
    )
    for r in rows:
        m = r["machine-id"]
        s = by_machine[m]
        s["count"] += 1
        ts = r.get("started-at", "")
        if not s["min_ts"] or ts < s["min_ts"]:
            s["min_ts"] = ts
        if not s["max_ts"] or ts > s["max_ts"]:
            s["max_ts"] = ts
        s["schemes"].add(r.get("scheme", ""))
    return by_machine


def print_listing(summary: dict[str, dict],
                  machine_names: dict[str, str]) -> list[str]:
    machine_ids = sorted(summary.keys())
    print()
    print(
        f"  {'#':>2}  {'machine-id':<10}  {'name':<14}  {'runs':>5}  "
        f"{'started-at range':<46}  schemes"
    )
    print(
        f"  {'-' * 2}  {'-' * 10}  {'-' * 14}  {'-' * 5}  "
        f"{'-' * 46}  {'-' * 30}"
    )
    for i, m in enumerate(machine_ids, 1):
        s = summary[m]
        name = machine_names.get(m, "")
        if s["min_ts"]:
            rng = f"{utc_to_local(s['min_ts'])} → {utc_to_local(s['max_ts'])}"
        else:
            rng = "(no timestamps)"
        schemes = ",".join(sorted(s["schemes"]))
        print(
            f"  {i:>2}  {m:<10}  {name:<14}  {s['count']:>5}  "
            f"{rng:<46}  {schemes}"
        )
    return machine_ids


def prompt_selection(machine_ids: list[str]) -> str | None:
    """Return chosen machine-id, or None for "all", or sys.exit on abort."""
    n_all = len(machine_ids) + 1
    print(f"  {n_all:>2}  all (every machine above)")
    while True:
        choice = input("\nPick a machine number (or Enter to abort): ").strip()
        if not choice:
            sys.exit("aborted")
        if choice in ("a", "all"):
            return None
        try:
            n = int(choice)
        except ValueError:
            print(f"  not a number: {choice!r}")
            continue
        if 1 <= n <= len(machine_ids):
            return machine_ids[n - 1]
        if n == n_all:
            return None
        print(f"  out of range: {n}")


def rsync(host: str, remote_path: str, machine_id: str, local_root: Path,
          dry_run: bool) -> bool:
    """Returns True if synced, False if the remote dir is missing (skipped)."""
    remote_dir = f"{remote_path}/runs/{machine_id}"
    probe = subprocess.run(
        ["ssh", host, "test", "-d", remote_dir],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        print(
            f"\nskip {machine_id}: {host}:{remote_dir} not present "
            f"(stale index.csv row?)"
        )
        return False

    src = f"{host}:{remote_dir}/"
    dst = local_root / "runs" / machine_id
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av", "--update"]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src, str(dst) + "/"]
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)
    return True


def merge_csv(local_path: Path, remote_header: list[str],
              new_rows: list[dict[str, str]], key: str,
              dry_run: bool) -> None:
    """Append `new_rows` to `local_path`, skipping any whose `key` is already present.

    Preserves existing rows and their order; never overwrites locally-present
    rows. If `local_path` exists with a different schema, the union of headers
    is written (extra fields default to empty)."""
    existing_keys: set[str] = set()
    existing_rows: list[dict[str, str]] = []
    local_header = remote_header
    if local_path.exists():
        with local_path.open(newline="") as f:
            r = csv.DictReader(f)
            local_header = list(r.fieldnames or remote_header)
            for row in r:
                existing_keys.add(row[key])
                existing_rows.append(row)

    to_add = [row for row in new_rows if row[key] not in existing_keys]

    if dry_run:
        print(f"would append {len(to_add)} new row(s) to {local_path}")
        return

    out_header = list(local_header)
    for col in remote_header:
        if col not in out_header:
            out_header.append(col)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_header, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing_rows)
        w.writerows(to_add)
    total = len(existing_rows) + len(to_add)
    print(f"appended {len(to_add)} new row(s) to {local_path} ({total} total)")


def main() -> None:
    args = parse_args()
    host, remote_path = split_remote(args.remote)
    local_root = Path(args.local)

    print(f"Fetching {remote_path}/index.csv from {host}…")
    idx_header, idx_rows = read_csv(ssh_cat(host, f"{remote_path}/index.csv"))
    if not idx_rows:
        sys.exit("remote index.csv has no rows")

    print(f"Fetching {remote_path}/machines.csv from {host}…")
    m_header, m_rows = read_csv(ssh_cat(host, f"{remote_path}/machines.csv"))
    machine_names = {r["machine-id"]: r.get("machine-name", "") for r in m_rows}

    summary = summarise(idx_rows)
    machine_ids = print_listing(summary, machine_names)

    if args.machine is not None:
        if args.machine not in summary:
            sys.exit(f"--machine {args.machine!r} not present remotely")
        selected_machines = [args.machine]
    else:
        sel = prompt_selection(machine_ids)
        selected_machines = machine_ids if sel is None else [sel]

    selected_rows = [
        r for r in idx_rows if r["machine-id"] in selected_machines
    ]
    print(
        f"\nSelected: {len(selected_rows)} runs across "
        f"{len(selected_machines)} machine(s)"
    )
    for m in selected_machines:
        c = sum(1 for r in selected_rows if r["machine-id"] == m)
        print(f"  {m}  {c} runs")

    synced_machines = [
        m for m in selected_machines
        if rsync(host, remote_path, m, local_root, args.dry_run)
    ]

    synced_rows = [
        r for r in selected_rows if r["machine-id"] in synced_machines
    ]
    merge_csv(
        local_root / "index.csv", idx_header, synced_rows,
        key="run-id", dry_run=args.dry_run,
    )

    synced_machine_rows = [
        r for r in m_rows if r["machine-id"] in synced_machines
    ]
    merge_csv(
        local_root / "machines.csv", m_header, synced_machine_rows,
        key="machine-id", dry_run=args.dry_run,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
