#!/usr/bin/env python3
"""Upload an eval run's raw CSVs to the bulk store; write `[bulk]` to metadata.

Post-run mode uploads a single run dir; migration mode
(`--migrate`) walks the results tree. Per-run invocation:

    scripts/upload_bulk.py <run-dir> [--bulk-base <uri>] [--keep-source]

`<run-dir>` is the directory containing `run-metadata.toml`. The
tool reads `machine-id` + `run-id` from that file and computes
the bulk destination as ``<bulk-base>/<machine-id>/<run-id>/``.

Files uploaded (if present on disk): ``raw.csv``, ``top_k.csv``,
``substep-breakdown.csv``. Each upload is verified by re-hashing
at the destination; the `[bulk]` block is written atomically to
``run-metadata.toml`` only after every hash verifies. Local raw
CSVs are deleted after the verified upload by default; pass
``--keep-source`` to retain them.

Backend resolution: if the bulk-base path is
``rcp-scratch://`` *and* the local mount directory is present +
writable, write directly via filesystem copy; otherwise fall
back to ``rsync`` over SSH through the alias ``jumphost``
(configured in ``~/.ssh/config``).

Already-tracked content stays in git: migration mode enforces this
via a ``git ls-files`` skip rule; in post-run mode the operator is
in control of the run-dir argument and we assume they pass a fresh
run dir.
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
from typing import Iterable, Optional
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BULK_BASE = (
    "rcp-scratch:///mnt/sacs/scratch/shared/secure-vsearch/bulk-store"
)
DEFAULT_RETENTION = "60d"

# Files we attempt to upload; absent files are skipped silently.
RAW_FILES = ("raw.csv", "top_k.csv", "substep-breakdown.csv")

# Path checked for direct-mount detection on rcp-scratch:// URIs.
# The project subdir under /mnt/sacs/scratch/shared/. We probe this
# path directly rather than calling ismount() because the subdir is
# several levels below the actual NFS mount root.
RCP_SCRATCH_LOCAL_PROBE = "/mnt/sacs/scratch/shared/secure-vsearch"

# SSH alias the rsync fallback uses; operator must have it in
# ~/.ssh/config.
JUMPHOST_ALIAS = "jumphost"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UploadError(Exception):
    """User-visible failure that should print without a stack trace."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int]:
    """Return (hex sha256, bytes) of `path`. One-shot read; no decompression."""
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


# ---------------------------------------------------------------------------
# Metadata reads + atomic write of the [bulk] block
# ---------------------------------------------------------------------------


@dataclass
class RunMeta:
    machine_id: str
    run_id: str
    has_bulk_block: bool
    status: str  # "complete", "partial", "" if absent
    raw_text: str  # full file content; we append to it


def read_run_metadata(run_dir: Path) -> RunMeta:
    meta_path = run_dir / "run-metadata.toml"
    if not meta_path.is_file():
        raise UploadError(f"no run-metadata.toml at {meta_path}")
    raw_text = meta_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(raw_text)
    try:
        machine_id = parsed["machine-id"]
        run_id = parsed["run-id"]
    except KeyError as e:
        raise UploadError(
            f"run-metadata.toml missing required key {e.args[0]!r}"
        ) from None
    has_bulk_block = "bulk" in parsed
    return RunMeta(
        machine_id=str(machine_id),
        run_id=str(run_id),
        has_bulk_block=has_bulk_block,
        status=str(parsed.get("status", "")),
        raw_text=raw_text,
    )


def render_bulk_block(
    uri: str,
    retention: str,
    files: list[tuple[str, str, int]],
) -> str:
    """Emit a `[bulk]` block as kebab-case TOML text.

    `files` is a list of (name, sha256_hex, bytes). The block is
    self-contained; the caller is responsible for prepending a
    blank line separator when appending to existing content.
    """
    lines = []
    lines.append("[bulk]")
    lines.append(f'uri       = "{uri}"')
    lines.append(f'retention = "{retention}"')
    for name, sha, size in files:
        lines.append("")
        lines.append("  [[bulk.files]]")
        lines.append(f'  name   = "{name}"')
        lines.append(f'  sha256 = "{sha}"')
        lines.append(f"  bytes  = {size}")
    lines.append("")
    return "\n".join(lines)


def write_metadata_atomic(meta_path: Path, new_text: str) -> None:
    """Atomic-replace via sibling tempfile + fsync + rename.

    The fsync is on the tempfile only; we rely on the rename being
    atomic on the same filesystem (POSIX) and the parent dir's
    durability semantics being sufficient for a results tree.
    """
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(new_text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, meta_path)


# ---------------------------------------------------------------------------
# Backend resolution + upload
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """Resolved upload backend.

    `local_path` is the filesystem path to write directly to when
    available (mount up + writable). `rsync_target` is the
    `<host>:<path>` form for `rsync` fallback. Exactly one is set
    depending on resolution.

    `ssh_control_path`, when set, is forwarded to every `ssh` and
    `rsync` subprocess via `-o ControlPath=<path>` / `-e 'ssh -o
    ControlPath=<path>'`. Lets the operator authenticate to a
    password-only bastion once interactively
    (``ssh -fNM -S <path> jumphost``) and have all of the tool's
    subsequent subprocesses inherit the multiplexed connection
    non-interactively.
    """

    uri: str  # canonical URI to record in [bulk]
    local_path: Optional[Path]
    rsync_target: Optional[str]
    ssh_control_path: Optional[str] = None


def _ssh_cmd(ssh_control_path: Optional[str]) -> list[str]:
    """Base `ssh` command; threads ControlPath when set."""
    cmd = ["ssh"]
    if ssh_control_path:
        cmd += ["-o", f"ControlPath={ssh_control_path}"]
    return cmd


def _rsync_remote_shell(ssh_control_path: Optional[str]) -> list[str]:
    """`-e <shell>` segment for rsync. Empty list when no override
    needed (rsync's default ssh works fine when there's no auth
    requirement).
    """
    if not ssh_control_path:
        return []
    # rsync's `-e` takes a single string, not separate tokens.
    return ["-e", f"ssh -o ControlPath={ssh_control_path}"]


def resolve_backend(
    bulk_base: str,
    machine_id: str,
    run_id: str,
    ssh_control_path: Optional[str] = None,
) -> Backend:
    """Decide between direct-mount and rsync-fallback for this run.

    `bulk_base` is the operator-provided base, e.g.
    ``rcp-scratch:///mnt/sacs/scratch/shared/secure-vsearch/bulk-store``.
    The canonical URI stored in `[bulk]` is the same in either
    mode — the transport is an implementation detail of the
    producer.
    """
    parsed = urlparse(bulk_base)
    scheme = parsed.scheme
    base_path = parsed.path  # absolute path component
    run_uri = f"{bulk_base.rstrip('/')}/{machine_id}/{run_id}"

    if scheme == "rcp-scratch":
        probe = Path(RCP_SCRATCH_LOCAL_PROBE)
        if probe.is_dir() and os.access(probe, os.W_OK):
            local = Path(base_path) / machine_id / run_id
            return Backend(
                uri=run_uri,
                local_path=local,
                rsync_target=None,
                ssh_control_path=ssh_control_path,
            )
        # Fall back to rsync via jumphost. The destination path is
        # the same absolute filesystem path; rsync over SSH delivers
        # the bytes to the same /mnt/... location on the remote.
        rsync_dest_dir = f"{JUMPHOST_ALIAS}:{base_path}/{machine_id}/{run_id}"
        return Backend(
            uri=run_uri,
            local_path=None,
            rsync_target=rsync_dest_dir,
            ssh_control_path=ssh_control_path,
        )

    if scheme in ("file", ""):
        # `file:///abs/path` or bare path — direct write only.
        local = Path(base_path) / machine_id / run_id
        return Backend(uri=run_uri, local_path=local, rsync_target=None)

    raise UploadError(
        f"unsupported bulk-base scheme {scheme!r}; "
        f"recognised: rcp-scratch://, file://, bare path"
    )


def upload_files(
    backend: Backend, sources: Iterable[Path], dest_subpath: str
) -> dict[str, tuple[str, int]]:
    """Copy `sources` to `<backend>/<dest_subpath>/<name>` and re-hash.

    Returns a dict mapping each source's basename to (sha256_hex,
    bytes) computed from the destination after upload. Pre-flight
    source hashes are compared against post-flight destination
    hashes by the caller; any mismatch must trigger a rollback.
    """
    if backend.local_path is not None:
        backend.local_path.mkdir(parents=True, exist_ok=True)
        for src in sources:
            shutil.copy2(src, backend.local_path / src.name)
        # Re-hash from destination.
        return {
            src.name: sha256_file(backend.local_path / src.name)
            for src in sources
        }

    assert backend.rsync_target is not None
    host, _, dest_dir = backend.rsync_target.partition(":")
    # Pre-create the remote destination dir. macOS ships rsync 2.6.9
    # which predates `--mkpath` (added in 3.2.3), so we can't lean on
    # rsync for path creation portably.
    mk = subprocess.run(
        [*_ssh_cmd(backend.ssh_control_path), host, "mkdir", "-p", "--", dest_dir],
        capture_output=True, text=True,
    )
    if mk.returncode != 0:
        raise UploadError(
            f"remote mkdir failed: {mk.stderr.strip()}"
        )
    cmd = [
        "rsync",
        "-az",
        *_rsync_remote_shell(backend.ssh_control_path),
        *[str(s) for s in sources],
        f"{backend.rsync_target}/",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise UploadError(
            f"rsync failed: {result.stderr.strip()}"
        )
    # Re-hash from destination by streaming each file back through
    # ssh. Slower than a local re-hash; acceptable cost for the
    # rsync-fallback case (every operator using jumphost is on a
    # high-bandwidth link already).
    ssh_base = _ssh_cmd(backend.ssh_control_path)
    out = {}
    for src in sources:
        remote = f"{backend.rsync_target}/{src.name}"
        # Split <host>:<path>; use `ssh host sha256sum path`.
        host, _, path = remote.partition(":")
        rh = subprocess.run(
            [*ssh_base, host, "sha256sum", "--", path],
            capture_output=True, text=True,
        )
        if rh.returncode != 0:
            raise UploadError(
                f"remote sha256 failed for {remote}: {rh.stderr.strip()}"
            )
        # sha256sum output: "<hex>  <path>"
        hex_part = rh.stdout.split(maxsplit=1)[0]
        sz = subprocess.run(
            [*ssh_base, host, "stat", "-c", "%s", path],
            capture_output=True, text=True,
        )
        bytes_part = int(sz.stdout.strip())
        out[src.name] = (hex_part, bytes_part)
    return out


def rollback_upload(backend: Backend, dest_subpath: str) -> None:
    """Remove the destination subdir to restore pre-upload state.

    Best-effort; logs but does not raise on cleanup failure — the
    operator can clean up manually from the canonical URI in any
    case.
    """
    if backend.local_path is not None:
        try:
            shutil.rmtree(backend.local_path)
        except OSError as e:
            print(f"warning: rollback rmtree failed: {e}", file=sys.stderr)
        return
    assert backend.rsync_target is not None
    host, _, path = backend.rsync_target.partition(":")
    rr = subprocess.run(
        [*_ssh_cmd(backend.ssh_control_path), host, "rm", "-rf", "--", path],
        capture_output=True, text=True,
    )
    if rr.returncode != 0:
        print(
            f"warning: rollback ssh rm failed: {rr.stderr.strip()}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Insert [bulk] into existing run-metadata.toml text
# ---------------------------------------------------------------------------


def insert_bulk_block(existing: str, bulk_block: str) -> str:
    """Place `bulk_block` inside `existing` TOML text.

    Strategy: if a `[campaign]` block exists, insert immediately
    after it ends (i.e. before the next `[...]` heading). Else,
    insert before the first `[...]` heading. Else, append at end.

    Idempotent against re-running on a file that already has
    [bulk] — caller should check `RunMeta.has_bulk_block` first
    and skip; this function assumes a fresh insert.
    """
    lines = existing.splitlines(keepends=True)
    # Find boundaries.
    campaign_start: Optional[int] = None
    next_section_after_campaign: Optional[int] = None
    first_section: Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("["):
            continue
        # Skip array-of-tables entries `[[...]]` — they are content
        # of an enclosing table, not new top-level sections for
        # placement purposes.
        if stripped.startswith("[["):
            continue
        # Top-level heading like `[campaign]`, `[ivf]`, `[gpu]`,
        # `[dataset]`, etc. Sub-tables like `[gpu.cloud]` also
        # match but we want the first top-level boundary.
        if first_section is None:
            first_section = i
        if stripped.startswith("[campaign]"):
            campaign_start = i
        elif campaign_start is not None and next_section_after_campaign is None:
            next_section_after_campaign = i

    if campaign_start is not None:
        insert_at = next_section_after_campaign or len(lines)
    elif first_section is not None:
        insert_at = first_section
    else:
        insert_at = len(lines)

    # Ensure exactly one blank line of separation on each side.
    block_text = bulk_block
    if not block_text.endswith("\n"):
        block_text += "\n"
    # Prepend separator if the preceding char isn't already a blank line.
    if insert_at > 0 and lines[insert_at - 1].rstrip() != "":
        block_text = "\n" + block_text
    # Append a trailing blank if the next line isn't blank/EOF.
    if insert_at < len(lines) and lines[insert_at].rstrip() != "":
        block_text += "\n"

    new_lines = lines[:insert_at] + [block_text] + lines[insert_at:]
    return "".join(new_lines)


# ---------------------------------------------------------------------------
# Git-tracked guard (migration-mode skip rule)
# ---------------------------------------------------------------------------


def is_path_tracked_in_git(path: Path, repo_root: Optional[Path] = None) -> bool:
    """Return True if `path` is currently tracked by git.

    Uses ``git ls-files --error-unmatch``; runs from ``repo_root``
    if provided, else from the path's parent. Returns False if git
    is unavailable or `path` is outside a git repo entirely. The
    function is intentionally conservative: any non-zero exit (not
    tracked, outside repo, no git installed) reads as "not
    tracked", which permits migration to proceed.
    """
    cmd = ["git", "ls-files", "--error-unmatch", "--", str(path)]
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root or path.parent,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        # git binary not on PATH; treat as untracked (the
        # operator-facing invocation is responsible for running
        # from inside a clone).
        return False


def run_has_git_tracked_raw_csv(run_dir: Path, repo_root: Optional[Path] = None) -> bool:
    """Guard for migration mode: skip runs whose CSVs are still in git.

    Returns True if any of the run's raw CSVs is tracked by git
    (regardless of whether it's still on disk). Such runs are
    pre-policy historical record and must not be mutated by
    migration; they stay in git as-is.
    """
    for name in RAW_FILES:
        if is_path_tracked_in_git(run_dir / name, repo_root):
            return True
    return False


# ---------------------------------------------------------------------------
# Migration walk
# ---------------------------------------------------------------------------


def _resolve_self_machine_id() -> Optional[str]:
    """Compute the current host's machine-id by invoking the same
    hashing logic the harness uses (eval-harness meta.rs). We
    approximate by reading any locally-emitted
    run-metadata.toml: the harness embeds machine-id consistently,
    so the first one wins. Returns None if no such file is
    findable — operator must pass an explicit id.
    """
    here = Path("results/runs")
    if not here.is_dir():
        return None
    # results/runs/<machine-id>/... → first child dir name (if exactly
    # one) is a strong signal. If multiple machine-ids are present,
    # the operator is migrating others' runs and should pass --machine
    # explicitly.
    machine_dirs = [d for d in here.iterdir() if d.is_dir()]
    if len(machine_dirs) == 1:
        return machine_dirs[0].name
    return None


def find_uploadable_runs(
    roots: list[Path],
    repo_root: Optional[Path] = None,
    machine_filter: Optional[str] = None,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Walk each root for run dirs eligible for migration.

    Returns ``(uploadable, skipped)``:
      - ``uploadable``: run dirs to upload, in deterministic
        (sorted) order.
      - ``skipped``: list of ``(run_dir, reason)`` for runs that
        matched a directory but failed an admission gate.

    A run dir is eligible when:
      1. It contains a ``run-metadata.toml``.
      2. It contains at least one of the raw CSVs.
      3. The metadata's ``machine-id`` matches ``machine_filter``
         (if provided).
      4. The metadata has no ``[bulk]`` block.
      5. None of its raw CSVs is tracked in git (the git-tracked
         guard).
    """
    uploadable: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            skipped.append((root, "root is not a directory"))
            continue
        for meta_path in sorted(root.rglob("run-metadata.toml")):
            run_dir = meta_path.parent.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            try:
                meta = read_run_metadata(run_dir)
            except UploadError as e:
                skipped.append((run_dir, f"unreadable metadata: {e}"))
                continue
            # 1. machine filter (cheap; do early so machine-mismatch
            # noise doesn't drown the more informative skips).
            if machine_filter and meta.machine_id != machine_filter:
                skipped.append(
                    (run_dir, f"machine-id={meta.machine_id} (filter={machine_filter})")
                )
                continue
            # 2. already has [bulk] — checked BEFORE the raw-CSV
            # presence gate so an uploaded-then-deleted run (the
            # post-upload steady state) skips with the informative
            # reason rather than the misleading "no raw CSVs".
            if meta.has_bulk_block:
                skipped.append((run_dir, "already has [bulk] block"))
                continue
            # 3. git-tracked guard
            if run_has_git_tracked_raw_csv(run_dir, repo_root):
                skipped.append(
                    (run_dir, "raw CSVs tracked in git")
                )
                continue
            # 4. at least one raw CSV must be present locally — the
            # admission gate for upload work.
            if not any((run_dir / n).is_file() for n in RAW_FILES):
                skipped.append((run_dir, "no raw CSVs on disk"))
                continue
            uploadable.append(run_dir)
    return uploadable, skipped


# ---------------------------------------------------------------------------
# Precondition: refuse to upload when results/aggregated/<m>/ is absent.
# ---------------------------------------------------------------------------


def _aggregated_dir_for_run(run_dir: Path) -> Optional[Path]:
    """Derive the canonical TSV location for the machine that owns
    `run_dir`. Returns None when the path layout isn't the expected
    ``.../results/runs/<machine-id>/<git-sha>/<run-id>/`` — in that
    case the check is skipped with a warning.
    """
    parts = run_dir.resolve().parts
    # Find the `results` segment, walking from the end.
    try:
        results_idx = len(parts) - 1 - parts[::-1].index("results")
    except ValueError:
        return None
    # Standard layout: results / runs / <machine-id> / <git-sha> / <run-id>
    if results_idx + 2 >= len(parts) or parts[results_idx + 1] != "runs":
        return None
    machine_id = parts[results_idx + 2]
    return Path(*parts[: results_idx + 1]) / "aggregated" / machine_id


def check_aggregated_precondition(run_dir: Path) -> None:
    """Precondition: the canonical TSV emission path
    must exist before raw CSVs can leave the local filesystem.

    Enforces the producer-side workflow invariant
    (harness → preprocess.py → upload_bulk.py): without
    ``preprocess.py`` having run on the producer first, the
    machine's TSVs aren't yet in
    ``results/aggregated/<machine-id>/`` and a subsequent figure
    rebuild on the laptop would require a bulk-store round-trip.
    Refusing here keeps that footgun out of operators' way.
    """
    agg = _aggregated_dir_for_run(run_dir)
    if agg is None:
        # Non-standard layout (e.g. test fixture, ad-hoc operator
        # path). The precondition can't be evaluated; warn but
        # don't block.
        print(
            f"warning: cannot derive aggregated path from {run_dir} — "
            "precondition check skipped",
            file=sys.stderr,
        )
        return
    if not agg.is_dir():
        raise UploadError(
            f"precondition failed: {agg} is absent. Run `make preprocess "
            f"MACHINE=<id>` (or `python analysis/preprocess.py --results "
            f"<results> --machine <id> --outdir {agg}`) on this host "
            "before uploading. This enforces the producer-side workflow "
            "invariant."
        )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def upload_run(
    run_dir: Path,
    bulk_base: str,
    retention: str,
    keep_source: bool,
    skip_precondition: bool = False,
    ssh_control_path: Optional[str] = None,
) -> None:
    """Upload one run dir, writing the [bulk] block only after every hash verifies."""
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise UploadError(f"not a directory: {run_dir}")

    meta = read_run_metadata(run_dir)
    if meta.has_bulk_block:
        print(
            f"skip: {run_dir} already has a [bulk] block — already uploaded",
            file=sys.stderr,
        )
        return

    # Skip in-flight runs. Uploading + `src.unlink()`-ing the raw
    # CSVs while the eval harness still has them open for append
    # leaves the writer's FD pointing at an orphaned inode — every
    # row written after the unlink is silently lost when the
    # process exits. Discovered 2026-05-20 during an 8.8M
    # bntm-ivf sweep: a `make finalize` while the eval was running
    # uploaded a header-only raw.csv and unlinked the live file;
    # only the open-FD recovery dance via /proc/<pid>/fd/<n> saved
    # the data. Status is set to "complete" by the harness on
    # normal exit; "partial" means the run is still in flight or
    # exited abnormally.
    if meta.status == "partial":
        print(
            f"skip: {run_dir} has status=\"partial\" — not uploading "
            f"in-flight runs (would orphan the eval harness's open "
            f"FDs on raw.csv / top_k.csv). Re-run upload after the "
            f"harness has set status=\"complete\".",
            file=sys.stderr,
        )
        return

    # Refuse to upload if the canonical aggregated TSVs are
    # absent. Bypass with `--skip-precondition` for emergency
    # situations (e.g. a machine where preprocess.py is permanently
    # incompatible).
    if not skip_precondition:
        check_aggregated_precondition(run_dir)

    # Step 0: enumerate present source files + pre-flight hashes.
    sources: list[Path] = []
    pre_hashes: dict[str, tuple[str, int]] = {}
    for name in RAW_FILES:
        p = run_dir / name
        if p.is_file():
            sources.append(p)
            pre_hashes[name] = sha256_file(p)
    if not sources:
        raise UploadError(
            f"no raw CSVs found under {run_dir}; nothing to upload"
        )

    backend = resolve_backend(
        bulk_base, meta.machine_id, meta.run_id, ssh_control_path=ssh_control_path
    )
    dest_subpath = f"{meta.machine_id}/{meta.run_id}"
    print(f"upload: {len(sources)} file(s) → {backend.uri}", file=sys.stderr)
    if backend.local_path is not None:
        print(f"  transport: direct write ({backend.local_path})", file=sys.stderr)
    else:
        print(f"  transport: rsync via {backend.rsync_target}", file=sys.stderr)

    # Step 1: copy to bulk.
    try:
        dest_hashes = upload_files(backend, sources, dest_subpath)
    except UploadError:
        raise
    except Exception as e:
        raise UploadError(f"upload failed: {e}") from e

    # Step 2: re-hash + compare. Any mismatch → rollback + abort.
    for name, (pre_sha, pre_bytes) in pre_hashes.items():
        post_sha, post_bytes = dest_hashes[name]
        if pre_sha != post_sha or pre_bytes != post_bytes:
            rollback_upload(backend, dest_subpath)
            raise UploadError(
                f"hash mismatch on {name}: "
                f"pre={pre_sha[:12]}({pre_bytes}) "
                f"post={post_sha[:12]}({post_bytes}) — rolled back"
            )

    # Step 3: build + atomic-write the [bulk] block.
    files_list = sorted(
        (name, dest_hashes[name][0], dest_hashes[name][1])
        for name in pre_hashes
    )
    bulk_block = render_bulk_block(backend.uri, retention, files_list)
    new_text = insert_bulk_block(meta.raw_text, bulk_block)
    write_metadata_atomic(run_dir / "run-metadata.toml", new_text)

    # Step 4: delete local raw CSVs unless --keep-source.
    if not keep_source:
        for src in sources:
            src.unlink()
        print(f"  deleted local source CSVs ({len(sources)})", file=sys.stderr)
    else:
        print(f"  kept local source CSVs (--keep-source)", file=sys.stderr)

    print(f"ok: {run_dir} → {backend.uri}", file=sys.stderr)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "paths",
        type=Path,
        nargs="*",
        help=(
            "Post-run mode: one run directory containing run-metadata.toml. "
            "Migration mode (--migrate): zero or more root directories to "
            "walk; default `results/runs/` when no path given."
        ),
    )
    p.add_argument(
        "--migrate",
        action="store_true",
        help=(
            "Walk each given path for run dirs and upload all eligible "
            "ones. Idempotent: runs that already have a [bulk] block "
            "are skipped, and runs whose raw CSVs are tracked in git "
            "are skipped."
        ),
    )
    p.add_argument(
        "--machine",
        default=None,
        help=(
            "Migration mode only: restrict the walk to runs with this "
            "machine-id. Literal `self` resolves to the current host's "
            "machine-id (inferred from the local results/runs/ tree)."
        ),
    )
    p.add_argument(
        "--bulk-base",
        default=DEFAULT_BULK_BASE,
        help="Bulk-store base URI (default: %(default)s)",
    )
    p.add_argument(
        "--retention",
        default=DEFAULT_RETENTION,
        help='Retention string written into [bulk] (default: %(default)s; "durable" for archival).',
    )
    p.add_argument(
        "--keep-source",
        action="store_true",
        help="Retain local raw CSVs after upload (default: delete).",
    )
    p.add_argument(
        "--skip-precondition",
        action="store_true",
        help=(
            "Bypass the `results/aggregated/<machine-id>/` precondition "
            "check. Use only when preprocess.py is "
            "permanently unable to run on this host."
        ),
    )
    p.add_argument(
        "--ssh-control-path",
        default=None,
        help=(
            "SSH ControlPath socket to reuse for all ssh/rsync "
            "subprocesses. Lets the operator authenticate once "
            "interactively (e.g. `ssh -fNM -S /tmp/jumphost-master "
            "jumphost`) when the bastion is password-only; the tool's "
            "subsequent ssh/rsync calls then inherit the multiplexed "
            "connection non-interactively. Ignored when the bulk "
            "backend resolves to a local-mount direct write."
        ),
    )
    args = p.parse_args(argv)
    if not args.migrate and not args.paths:
        p.error("post-run mode requires exactly one run directory")
    if not args.migrate and len(args.paths) != 1:
        p.error("post-run mode takes exactly one run directory")
    if not args.migrate and args.machine:
        p.error("--machine is only meaningful with --migrate")
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.migrate:
            roots = args.paths or [Path("results/runs")]
            machine_filter: Optional[str] = None
            if args.machine == "self":
                machine_filter = _resolve_self_machine_id()
                if machine_filter is None:
                    raise UploadError(
                        "--machine self could not infer current host's "
                        "machine-id (results/runs/ absent or multi-machine)"
                    )
            elif args.machine:
                machine_filter = args.machine

            uploadable, skipped = find_uploadable_runs(
                roots=[p.resolve() for p in roots],
                machine_filter=machine_filter,
            )
            print(
                f"migration: {len(uploadable)} uploadable, "
                f"{len(skipped)} skipped",
                file=sys.stderr,
            )
            for run_dir, reason in skipped:
                print(f"  skip: {run_dir} — {reason}", file=sys.stderr)
            failed = 0
            for run_dir in uploadable:
                try:
                    upload_run(
                        run_dir=run_dir,
                        bulk_base=args.bulk_base,
                        retention=args.retention,
                        keep_source=args.keep_source,
                        skip_precondition=args.skip_precondition,
                        ssh_control_path=args.ssh_control_path,
                    )
                except UploadError as e:
                    failed += 1
                    print(f"  fail: {run_dir} — {e}", file=sys.stderr)
            print(
                f"migration summary: {len(uploadable) - failed} uploaded, "
                f"{failed} failed, {len(skipped)} skipped",
                file=sys.stderr,
            )
            return 1 if failed else 0
        else:
            upload_run(
                run_dir=args.paths[0],
                bulk_base=args.bulk_base,
                retention=args.retention,
                keep_source=args.keep_source,
                skip_precondition=args.skip_precondition,
                ssh_control_path=args.ssh_control_path,
            )
    except UploadError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
