#!/usr/bin/env python3
"""Delete run-dirs (and stale report snapshots) from failed eval attempts.

Four detection reasons, all enabled by default — narrow with ``--reasons``:

  1. ``incomplete-status``    — ``status`` in ``{"partial", "failed",
                                "incomplete"}`` in ``run-metadata.toml``
  2. ``missing-raw-csv``      — run dir has no ``raw.csv``
  3. ``empty-raw-csv``        — header-only ``raw.csv`` on a non-breakdown
                                run (breakdown runs legitimately emit a
                                header-only file by design)
  4. ``pre-manifest-report``  — ``runs/<machine>/reports/<snap>/`` dirs
                                that predate the ``manifest.toml`` convention

Reasons 1–3 apply to run-dirs and respect the ``[bulk]`` guard: a run
that has been uploaded to the bulk store is skipped by default. Use
``--include-archived`` to override (e.g. an interrupted sweep got
partway through, you reran it cleanly, and you don't need the partial
uploaded artifact — the bntm-ivf-suspended pattern from RunAI's 2h
idle-GPU reclaim). Each run-dir is classified under the first matching
reason; the same dir is never listed twice.

Reason 4 lives in a separate subtree (``reports/`` rather than
``<sha>/<run-id>/``) and has no remote counterpart — the ``latest``
report symlink is never followed or deleted.

Locations cleaned per run-dir candidate:

- Local: ``results/runs/<machine>/<git-sha>/<run-id>/`` (always).
- Remote (with ``--remote``, run-dir reasons only):
  - ``/mnt/sacs/scratch/secure-vector-search/results/runs/<machine>/<git-sha>/<run-id>/``
    (the producer's working dir on NAS3, where pods write before upload).
  - ``/mnt/sacs/scratch/shared/secure-vsearch/bulk-store/<machine>/<run-id>/``
    (the bulk-store entry; only if the run carries a ``[bulk]`` block).

Default mode is dry-run. Pass ``--apply`` to actually delete.
Mirror of ``hydrate_bulk.py`` / ``prune_bulked.py`` in shape: per-run,
``--machine``, and ``--all`` selection modes; SSH transport via the
``jumphost`` alias for the remote cleanup.

Usage::

    scripts/cleanup_runs.py --machine <id>                  # preview
    scripts/cleanup_runs.py --machine <id> --apply          # delete locally
    scripts/cleanup_runs.py --machine <id> --apply --remote # + jumphost
    scripts/cleanup_runs.py --all --apply --include-archived
    scripts/cleanup_runs.py --all --reasons pre-manifest-report --apply
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


JUMPHOST_ALIAS = "jumphost"
REMOTE_RESULTS_ROOT = "/mnt/sacs/scratch/secure-vector-search/results/runs"
REMOTE_BULK_ROOT = "/mnt/sacs/scratch/shared/secure-vsearch/bulk-store"
INCOMPLETE_STATUSES = frozenset({"partial", "failed", "incomplete"})

REASONS = [
    "incomplete-status",
    "missing-raw-csv",
    "empty-raw-csv",
    "pre-manifest-report",
]
RUN_DIR_REASONS = frozenset(REASONS[:3])
REASON_LABEL = {
    "incomplete-status": "incomplete status",
    "missing-raw-csv": "missing raw.csv",
    "empty-raw-csv": "empty raw.csv (header-only, non-breakdown)",
    "pre-manifest-report": "pre-manifest report snapshot",
}


class CleanupError(Exception):
    """User-visible failure that should print without a stack trace."""


@dataclass
class Candidate:
    run_dir: Path
    machine_id: str
    git_sha: str           # "" for pre-manifest-report
    run_id: str            # snap dir name for pre-manifest-report
    status: str            # "n/a" for pre-manifest-report
    has_bulk: bool         # always False for pre-manifest-report
    reason: str


@dataclass
class Meta:
    status: str
    has_bulk: bool
    breakdown: bool


def read_meta(run_dir: Path) -> Meta:
    meta_path = run_dir / "run-metadata.toml"
    if not meta_path.is_file():
        return Meta(status="unknown", has_bulk=False, breakdown=False)
    parsed = tomllib.loads(meta_path.read_text(encoding="utf-8"))
    return Meta(
        status=str(parsed.get("status", "unknown")),
        has_bulk="bulk" in parsed,
        breakdown=bool(parsed.get("breakdown", False)),
    )


def raw_csv_is_header_only(path: Path) -> bool:
    """True iff raw.csv exists, has at most a header row, and no data rows."""
    try:
        with path.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return True
            return next(reader, None) is None
    except (OSError, csv.Error):
        return False


def detect_reason(run_dir: Path, meta: Meta) -> Optional[str]:
    """Return the first matching reason for `run_dir`, or None."""
    if meta.status in INCOMPLETE_STATUSES:
        return "incomplete-status"
    raw = run_dir / "raw.csv"
    if not raw.exists():
        return "missing-raw-csv"
    if not meta.breakdown and raw_csv_is_header_only(raw):
        return "empty-raw-csv"
    return None


def is_candidate(
    reason: Optional[str],
    has_bulk: bool,
    include_archived: bool,
    allowed_reasons: frozenset[str],
) -> bool:
    if reason is None or reason not in allowed_reasons:
        return False
    if has_bulk and not include_archived:
        return False
    return True


def collect_run_candidates(
    results_root: Path,
    *,
    machine_id: Optional[str],
    include_archived: bool,
    allowed_reasons: frozenset[str],
) -> list[Candidate]:
    runs_root = results_root / "runs"
    if not runs_root.is_dir():
        raise CleanupError(f"no runs directory at {runs_root}")
    machines = [runs_root / machine_id] if machine_id else sorted(runs_root.iterdir())
    out: list[Candidate] = []
    for m in machines:
        if not m.is_dir():
            continue
        for sha_dir in sorted(m.iterdir()):
            if not sha_dir.is_dir() or sha_dir.name == "reports":
                continue
            for run_dir in sorted(sha_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                meta = read_meta(run_dir)
                reason = detect_reason(run_dir, meta)
                if is_candidate(reason, meta.has_bulk, include_archived, allowed_reasons):
                    out.append(Candidate(
                        run_dir=run_dir,
                        machine_id=m.name,
                        git_sha=sha_dir.name,
                        run_id=run_dir.name,
                        status=meta.status,
                        has_bulk=meta.has_bulk,
                        reason=reason,  # type: ignore[arg-type]
                    ))
    return out


def collect_pre_manifest_reports(
    results_root: Path,
    *,
    machine_id: Optional[str],
) -> list[Candidate]:
    runs_root = results_root / "runs"
    if not runs_root.is_dir():
        return []
    machines = [runs_root / machine_id] if machine_id else sorted(runs_root.iterdir())
    out: list[Candidate] = []
    for m in machines:
        if not m.is_dir():
            continue
        reports_dir = m / "reports"
        if not reports_dir.is_dir():
            continue
        for snap in sorted(reports_dir.iterdir()):
            if snap.is_symlink() or not snap.is_dir():
                continue
            if (snap / "manifest.toml").exists():
                continue
            out.append(Candidate(
                run_dir=snap,
                machine_id=m.name,
                git_sha="",
                run_id=snap.name,
                status="n/a",
                has_bulk=False,
                reason="pre-manifest-report",
            ))
    return out


def collect_candidate_from_path(
    run_dir: Path,
    *,
    include_archived: bool,
    allowed_reasons: frozenset[str],
) -> Optional[Candidate]:
    if not run_dir.is_dir():
        raise CleanupError(f"not a directory: {run_dir}")
    parts = run_dir.resolve().parts
    if len(parts) < 3:
        raise CleanupError(f"can't infer machine/sha/run-id from {run_dir}")
    # Pre-manifest report case: <…>/<machine>/reports/<snap>/
    if parts[-2] == "reports":
        if (run_dir / "manifest.toml").exists():
            return None
        if "pre-manifest-report" not in allowed_reasons:
            return None
        return Candidate(
            run_dir=run_dir,
            machine_id=parts[-3],
            git_sha="",
            run_id=parts[-1],
            status="n/a",
            has_bulk=False,
            reason="pre-manifest-report",
        )
    if not (run_dir / "run-metadata.toml").is_file() and not (run_dir / "raw.csv").exists():
        raise CleanupError(f"no run-metadata.toml or raw.csv at {run_dir}")
    meta = read_meta(run_dir)
    reason = detect_reason(run_dir, meta)
    if not is_candidate(reason, meta.has_bulk, include_archived, allowed_reasons):
        return None
    return Candidate(
        run_dir=run_dir,
        machine_id=parts[-3],
        git_sha=parts[-2],
        run_id=parts[-1],
        status=meta.status,
        has_bulk=meta.has_bulk,
        reason=reason,  # type: ignore[arg-type]
    )


def remote_rm(
    paths: list[str], ssh_control_path: Optional[str]
) -> tuple[int, str]:
    if not paths:
        return 0, ""
    ssh = ["ssh"]
    if ssh_control_path:
        ssh += ["-o", f"ControlPath={ssh_control_path}"]
    cmd = [*ssh, JUMPHOST_ALIAS, "rm", "-rf", "--", *paths]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stderr.strip()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="single run directory (run-id dir or pre-manifest reports/<snap>/ dir)",
    )
    p.add_argument(
        "--machine",
        help="scope cleanup to results/runs/<machine>/",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="scan every machine under results/runs/",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="results tree root (default: ./results)",
    )
    p.add_argument(
        "--reasons",
        nargs="+",
        choices=REASONS,
        default=None,
        help=f"restrict to one or more reasons (default: all four: {', '.join(REASONS)})",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; default is dry-run preview",
    )
    p.add_argument(
        "--remote",
        action="store_true",
        help=(
            "also delete the matching dirs on jumphost (run-dir reasons only): "
            "the producer's working tree under "
            "/mnt/sacs/scratch/secure-vector-search/results/runs/ "
            "and any bulk-store entry under "
            "/mnt/sacs/scratch/shared/secure-vsearch/bulk-store/"
        ),
    )
    p.add_argument(
        "--include-archived",
        "--include-archived-partial",
        dest="include_archived",
        action="store_true",
        help=(
            "also delete run-dirs that have a [bulk] block "
            "(default: skip them — the data is archived). "
            "Old name --include-archived-partial still accepted."
        ),
    )
    p.add_argument(
        "--ssh-control-path",
        help="reuse an existing ssh ControlMaster socket for the jumphost",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    allowed_reasons = frozenset(args.reasons) if args.reasons else frozenset(REASONS)

    candidates: list[Candidate]
    if args.run_dir is not None:
        if args.machine or args.all:
            print(
                "error: --machine/--all are mutually exclusive with a positional run_dir",
                file=sys.stderr,
            )
            return 2
        try:
            c = collect_candidate_from_path(
                args.run_dir,
                include_archived=args.include_archived,
                allowed_reasons=allowed_reasons,
            )
        except CleanupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        candidates = [c] if c is not None else []
    elif args.machine or args.all:
        try:
            candidates = collect_run_candidates(
                args.results_root,
                machine_id=args.machine if args.machine else None,
                include_archived=args.include_archived,
                allowed_reasons=allowed_reasons,
            )
            if "pre-manifest-report" in allowed_reasons:
                candidates.extend(collect_pre_manifest_reports(
                    args.results_root,
                    machine_id=args.machine if args.machine else None,
                ))
        except CleanupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        print(
            "error: pass a run dir, --machine <id>, or --all",
            file=sys.stderr,
        )
        return 2

    if not candidates:
        print("no candidates to delete")
        return 0

    by_reason: dict[str, list[Candidate]] = {r: [] for r in REASONS}
    for c in candidates:
        by_reason[c.reason].append(c)

    for reason in REASONS:
        cs = by_reason[reason]
        if not cs:
            continue
        print(f"Reason: {REASON_LABEL[reason]}  ({len(cs)})")
        for c in cs:
            bulk_note = " [archived]" if c.has_bulk else ""
            if reason == "pre-manifest-report":
                print(f"  {c.machine_id}/reports/{c.run_id}")
            else:
                print(
                    f"  {c.machine_id}/{c.git_sha[:8]}/{c.run_id}  "
                    f"status={c.status}{bulk_note}"
                )
        print()

    n = len(candidates)
    verb = "delete" if args.apply else "would delete"
    n_remote = sum(1 for c in candidates if c.reason in RUN_DIR_REASONS)
    print(
        f"{verb} {n} dir(s) locally"
        + (f" + remote jumphost paths for {n_remote} run-dir(s)" if args.remote else "")
    )

    if not args.apply:
        print("re-run with --apply to act")
        return 0

    failures = 0
    for c in candidates:
        try:
            shutil.rmtree(c.run_dir)
        except OSError as e:
            print(f"  warning: local rm failed for {c.run_dir}: {e}", file=sys.stderr)
            failures += 1

    if args.remote:
        paths: list[str] = []
        for c in candidates:
            if c.reason not in RUN_DIR_REASONS:
                continue
            paths.append(f"{REMOTE_RESULTS_ROOT}/{c.machine_id}/{c.git_sha}/{c.run_id}")
            if c.has_bulk:
                paths.append(f"{REMOTE_BULK_ROOT}/{c.machine_id}/{c.run_id}")
        rc, err = remote_rm(paths, args.ssh_control_path)
        if rc != 0:
            print(f"warning: remote rm exited {rc}: {err}", file=sys.stderr)
            failures += 1

    print(
        f"done: {n - failures} cleaned, {failures} failed"
        if failures
        else f"done: {n} dir(s) cleaned"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
