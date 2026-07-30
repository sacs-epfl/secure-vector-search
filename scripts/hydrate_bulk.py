#!/usr/bin/env python3
"""Hydrate an eval run's raw CSVs from the bulk store back into the run dir.

Reverse of ``scripts/upload_bulk.py``. Reads the ``[bulk]`` block in
``run-metadata.toml`` to discover the canonical URI and per-file
checksums, fetches each listed file back into the run dir, and
verifies sha256 against the recorded value.

Usage:

    # single run
    scripts/hydrate_bulk.py results/runs/<machine-id>/<sha>/<run-id>/

    # every run for a machine
    scripts/hydrate_bulk.py --machine <machine-id>

    # everything under results/runs/ that has a [bulk] block and is
    # missing one or more of its bulk files locally
    scripts/hydrate_bulk.py --all

Backend resolution matches upload_bulk.py: an ``rcp-scratch://`` URI
copies directly when ``/mnt/sacs/scratch/shared/secure-vsearch`` is
mounted and readable; otherwise rsync over SSH through the
``jumphost`` alias.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


RCP_SCRATCH_LOCAL_PROBE = "/mnt/sacs/scratch/shared/secure-vsearch"
JUMPHOST_ALIAS = "jumphost"


class HydrateError(Exception):
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


@dataclass
class BulkBlock:
    uri: str
    files: list[BulkFile]


def read_bulk_block(run_dir: Path) -> BulkBlock:
    meta_path = run_dir / "run-metadata.toml"
    if not meta_path.is_file():
        raise HydrateError(f"no run-metadata.toml at {meta_path}")
    parsed = tomllib.loads(meta_path.read_text(encoding="utf-8"))
    bulk = parsed.get("bulk")
    if not bulk:
        raise HydrateError(f"no [bulk] block in {meta_path}")
    try:
        uri = str(bulk["uri"])
        files = [
            BulkFile(name=str(f["name"]), sha256=str(f["sha256"]), bytes=int(f["bytes"]))
            for f in bulk.get("files", [])
        ]
    except KeyError as e:
        raise HydrateError(f"[bulk] missing required key {e.args[0]!r}") from None
    if not files:
        raise HydrateError(f"[bulk.files] empty in {meta_path}")
    return BulkBlock(uri=uri, files=files)


@dataclass
class Source:
    local_dir: Optional[Path]
    rsync_source_prefix: Optional[str]
    ssh_control_path: Optional[str] = None


def _ssh_cmd(ssh_control_path: Optional[str]) -> list[str]:
    cmd = ["ssh"]
    if ssh_control_path:
        cmd += ["-o", f"ControlPath={ssh_control_path}"]
    return cmd


def _rsync_remote_shell(ssh_control_path: Optional[str]) -> list[str]:
    if not ssh_control_path:
        return []
    return ["-e", f"ssh -o ControlPath={ssh_control_path}"]


def resolve_source(uri: str, ssh_control_path: Optional[str] = None) -> Source:
    parsed = urlparse(uri)
    scheme = parsed.scheme
    base_path = parsed.path

    if scheme == "rcp-scratch":
        probe = Path(RCP_SCRATCH_LOCAL_PROBE)
        if probe.is_dir() and os.access(probe, os.R_OK):
            return Source(local_dir=Path(base_path), rsync_source_prefix=None)
        return Source(
            local_dir=None,
            rsync_source_prefix=f"{JUMPHOST_ALIAS}:{base_path}",
            ssh_control_path=ssh_control_path,
        )

    if scheme in ("file", ""):
        return Source(local_dir=Path(base_path), rsync_source_prefix=None)

    raise HydrateError(
        f"unsupported bulk URI scheme {scheme!r}; "
        f"recognised: rcp-scratch://, file://, bare path"
    )


def fetch_file(source: Source, name: str, dest: Path) -> None:
    if source.local_dir is not None:
        src = source.local_dir / name
        if not src.is_file():
            raise HydrateError(f"missing in bulk store: {src}")
        shutil.copy2(src, dest)
        return

    assert source.rsync_source_prefix is not None
    remote = f"{source.rsync_source_prefix}/{name}"
    cmd = [
        "rsync",
        "-a",
        *_rsync_remote_shell(source.ssh_control_path),
        remote,
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HydrateError(f"rsync failed for {remote}: {result.stderr.strip()}")


def hydrate_run(
    run_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
    ssh_control_path: Optional[str] = None,
) -> tuple[int, int]:
    """Return (fetched, skipped) file counts for `run_dir`."""
    block = read_bulk_block(run_dir)
    source = resolve_source(block.uri, ssh_control_path=ssh_control_path)

    fetched = 0
    skipped = 0
    for f in block.files:
        dest = run_dir / f.name
        if dest.is_file() and not force:
            local_sha, local_bytes = sha256_file(dest)
            if local_sha == f.sha256 and local_bytes == f.bytes:
                skipped += 1
                continue

        if dry_run:
            print(f"  would fetch {f.name} ({f.bytes} bytes)")
            fetched += 1
            continue

        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            fetch_file(source, f.name, tmp)
            actual_sha, actual_bytes = sha256_file(tmp)
            if actual_sha != f.sha256 or actual_bytes != f.bytes:
                raise HydrateError(
                    f"checksum mismatch for {f.name}: "
                    f"expected {f.sha256[:12]}…/{f.bytes}B, "
                    f"got {actual_sha[:12]}…/{actual_bytes}B"
                )
            os.replace(tmp, dest)
            fetched += 1
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    return fetched, skipped


def find_run_dirs(
    results_root: Path,
    machine_id: Optional[str] = None,
) -> list[Path]:
    runs_root = results_root / "runs"
    if not runs_root.is_dir():
        raise HydrateError(f"no runs directory at {runs_root}")
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
        help="hydrate every run under results/runs/<machine>/",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="hydrate every run under results/runs/ that has a [bulk] block",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="results tree root (default: ./results)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the local file already verifies",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be fetched without writing anything",
    )
    p.add_argument(
        "--ssh-control-path",
        help="reuse an existing ssh ControlMaster socket for the jumphost",
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
        except HydrateError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        print("error: pass a run dir, --machine <id>, or --all", file=sys.stderr)
        return 2

    total_fetched = 0
    total_skipped = 0
    total_runs = 0
    total_no_bulk = 0
    for run_dir in targets:
        try:
            fetched, skipped = hydrate_run(
                run_dir,
                force=args.force,
                dry_run=args.dry_run,
                ssh_control_path=args.ssh_control_path,
            )
        except HydrateError as e:
            msg = str(e)
            if "no [bulk] block" in msg:
                total_no_bulk += 1
                continue
            print(f"  ===> {run_dir}")
            print(f"  error: {e}", file=sys.stderr)
            return 1
        total_runs += 1
        total_fetched += fetched
        total_skipped += skipped
        verb = "would fetch" if args.dry_run else "fetched"
        print(
            f"  ===> {run_dir}: {verb} {fetched}, skipped {skipped} (already local)"
        )

    print(
        f"done: {total_runs} run(s), "
        f"{total_fetched} file(s) {'would-fetch' if args.dry_run else 'fetched'}, "
        f"{total_skipped} skipped, "
        f"{total_no_bulk} run(s) without [bulk] block"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
