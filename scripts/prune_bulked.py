#!/usr/bin/env python3
"""Delete local copies of files already verified-uploaded to the bulk store.

For each run-dir with a ``[bulk]`` block in ``run-metadata.toml``,
re-hash every listed file's local copy and compare to the recorded
sha256. On match, the local file is deleted (the bulk-store copy
remains the source of truth). Mismatches are skipped with a loud
warning — the local copy diverged from the bulk record and may
contain unique data.

Idempotent: files already absent locally are no-ops; runs without
``[bulk]`` blocks are silently skipped (they're unarchived and not
safe to delete).

Usage:

    # single run
    scripts/prune_bulked.py results/runs/<machine-id>/<sha>/<run-id>/

    # every run under one machine
    scripts/prune_bulked.py --machine <machine-id>

    # everything under results/runs/
    scripts/prune_bulked.py --all

Mirror of ``scripts/hydrate_bulk.py``; the pair is meant to be used
together: hydrate to bring data in for analysis, prune to send it
back to the bulk store when local disk pressure matters.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class PruneError(Exception):
    """User-visible failure that should print without a stack trace."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
            n += len(buf)
    return h.hexdigest(), n


@dataclass
class BulkFile:
    name: str
    sha256: str
    bytes: int


def read_bulk_files(run_dir: Path) -> list[BulkFile]:
    meta_path = run_dir / "run-metadata.toml"
    if not meta_path.is_file():
        raise PruneError(f"no run-metadata.toml at {meta_path}")
    parsed = tomllib.loads(meta_path.read_text(encoding="utf-8"))
    bulk = parsed.get("bulk")
    if not bulk:
        raise PruneError(f"no [bulk] block in {meta_path}")
    try:
        files = [
            BulkFile(name=str(f["name"]), sha256=str(f["sha256"]), bytes=int(f["bytes"]))
            for f in bulk.get("files", [])
        ]
    except KeyError as e:
        raise PruneError(f"[bulk] missing required key {e.args[0]!r}") from None
    if not files:
        raise PruneError(f"[bulk.files] empty in {meta_path}")
    return files


def prune_run(
    run_dir: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Return (removed, mismatched, absent) file counts for `run_dir`."""
    files = read_bulk_files(run_dir)
    removed = mismatched = absent = 0
    for f in files:
        local = run_dir / f.name
        if not local.is_file():
            absent += 1
            continue
        local_sha, local_bytes = sha256_file(local)
        if local_sha != f.sha256 or local_bytes != f.bytes:
            print(
                f"  MISMATCH {local}: local sha={local_sha[:12]}…/{local_bytes}B "
                f"vs bulk sha={f.sha256[:12]}…/{f.bytes}B — keeping local",
                file=sys.stderr,
            )
            mismatched += 1
            continue
        if dry_run:
            print(f"  would remove {f.name} ({f.bytes} bytes)")
        else:
            local.unlink()
        removed += 1
    return removed, mismatched, absent


def find_run_dirs(
    results_root: Path,
    machine_id: Optional[str] = None,
) -> list[Path]:
    runs_root = results_root / "runs"
    if not runs_root.is_dir():
        raise PruneError(f"no runs directory at {runs_root}")
    machines = [runs_root / machine_id] if machine_id else sorted(runs_root.iterdir())
    out: list[Path] = []
    for m in machines:
        if not m.is_dir():
            continue
        for sha_dir in sorted(m.iterdir()):
            if not sha_dir.is_dir():
                continue
            for run_dir in sorted(sha_dir.iterdir()):
                if (run_dir / "run-metadata.toml").is_file():
                    out.append(run_dir)
    return out


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="single run directory (containing run-metadata.toml)",
    )
    p.add_argument(
        "--machine",
        help="prune every run under results/runs/<machine>/",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="prune every run under results/runs/ that has a [bulk] block",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="results tree root (default: ./results)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be removed without deleting anything",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    targets: list[Path]
    if args.run_dir is not None:
        if args.machine or args.all:
            print(
                "error: --machine/--all are mutually exclusive with a positional run_dir",
                file=sys.stderr,
            )
            return 2
        targets = [args.run_dir]
    elif args.machine or args.all:
        try:
            targets = find_run_dirs(
                args.results_root,
                machine_id=args.machine if args.machine else None,
            )
        except PruneError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        print("error: pass a run dir, --machine <id>, or --all", file=sys.stderr)
        return 2

    total_removed = total_mismatched = total_absent = 0
    total_runs = total_no_bulk = 0
    for run_dir in targets:
        try:
            removed, mismatched, absent = prune_run(run_dir, dry_run=args.dry_run)
        except PruneError as e:
            msg = str(e)
            if "no [bulk] block" in msg:
                total_no_bulk += 1
                continue
            print(f"  ===> {run_dir}")
            print(f"  error: {e}", file=sys.stderr)
            return 1
        total_runs += 1
        total_removed += removed
        total_mismatched += mismatched
        total_absent += absent
        verb = "would remove" if args.dry_run else "removed"
        print(
            f"  ===> {run_dir}: {verb} {removed}, mismatched {mismatched}, "
            f"already-absent {absent}"
        )

    print(
        f"done: {total_runs} run(s), "
        f"{total_removed} file(s) {'would-remove' if args.dry_run else 'removed'}, "
        f"{total_mismatched} mismatched, "
        f"{total_absent} already-absent, "
        f"{total_no_bulk} run(s) without [bulk] block"
    )
    return 1 if total_mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
