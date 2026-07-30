#!/usr/bin/env python3
"""Insert a `[campaign]` block into existing run-metadata.toml files.

The campaign-tag mechanism groups related eval runs. Runs launched
before the operator's env
carried `CAMPAIGN_ID` / `CAMPAIGN_TITLE` (or before
`deploy/runai-submit.sh` learned to forward those env vars) end up
with no `[campaign]` block in their `run-metadata.toml`. This script
backfills the block in place, matching the exact TOML shape that
`eval_harness::meta::Campaign` serialises (kebab-case section name,
`id` + `title` required, `note` optional).

CLI:

    scripts/backfill_campaign.py \\
        --id fig14-sweep-cpu-2026-05-12 \\
        --title "figure 14 production sweep" \\
        --note "sacs006 local sweep" \\
        results/runs/d860be76/<sha>/<run-id>/run-metadata.toml

Path args may be either individual `run-metadata.toml` files or
directories containing them (scanned recursively).

Idempotence: files that already carry a `[campaign]` block are
skipped (with a message). Use `--force` to overwrite — the existing
block is replaced wholesale, including any operator edits.

Validation: `--id` must match the campaign-id character set
(`[A-Za-z0-9._:-]{1,128}`); `--title` must be non-empty; `--note`
is optional and trimmed.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _escape_toml(s: str) -> str:
    """Escape a string for embedding inside a TOML double-quoted value.

    Matches the subset of escapes the `toml` crate emits when round-
    tripping a String through `toml::to_string`.
    """
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return "".join(out)


def build_block(campaign_id: str, title: str, note: str | None) -> str:
    """Render the `[campaign]` block. Trailing newline so the insertion
    leaves a blank line before the next section.
    """
    lines = ["[campaign]"]
    lines.append(f'id = "{_escape_toml(campaign_id)}"')
    lines.append(f'title = "{_escape_toml(title)}"')
    if note and note.strip():
        lines.append(f'note = "{_escape_toml(note.strip())}"')
    return "\n".join(lines) + "\n"


def has_campaign_block(text: str) -> bool:
    return re.search(r"^\[campaign\]\s*$", text, flags=re.MULTILINE) is not None


def strip_existing_block(text: str) -> str:
    """Remove an existing `[campaign]` block + its key=value lines.

    A TOML table runs from its `[section]` header to the next `[`
    header (or EOF). We split on that boundary and drop the matching
    range. Returns text without the campaign block.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[campaign]":
            skipping = True
            continue
        if skipping and stripped.startswith("[") and stripped.endswith("]"):
            # Reached next table — stop skipping, include this line.
            skipping = False
            out.append(line)
            continue
        if skipping:
            continue
        out.append(line)
    return "".join(out)


def insert_block(text: str, block: str) -> str:
    """Insert `block` immediately before the first `[section]` line.

    Falls back to appending at EOF if the file has no table sections
    (shouldn't happen for harness-produced files — every
    run-metadata.toml has at least `[ivf]` + `[scheme-config]` +
    `[dataset]`).
    """
    m = re.search(r"^\[[A-Za-z_][\w.-]*\]\s*$", text, flags=re.MULTILINE)
    if m is None:
        # No table section — append at EOF with a separating newline.
        suffix = "" if text.endswith("\n") else "\n"
        return text + suffix + "\n" + block
    insert_at = m.start()
    # Ensure exactly one blank line separates the preceding key=value
    # block from the new [campaign] header. Trim trailing newlines on
    # the prefix, then re-add exactly one before the block.
    prefix = text[:insert_at].rstrip("\n")
    suffix = text[insert_at:]
    return f"{prefix}\n\n{block}\n{suffix}"


def discover_paths(args_paths: list[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for arg in args_paths:
        p = pathlib.Path(arg)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.rglob("run-metadata.toml")))
        else:
            print(f"warning: skipping {arg} (not a file or dir)", file=sys.stderr)
    # De-duplicate while preserving order.
    seen: set[pathlib.Path] = set()
    out: list[pathlib.Path] = []
    for p in paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Backfill [campaign] into run-metadata.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--id", required=True, help="campaign-id (ASCII alnum + -_.:, ≤128 chars)")
    ap.add_argument("--title", required=True, help="campaign title (human-readable)")
    ap.add_argument("--note", default=None, help="optional free-form note")
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing [campaign] blocks (default: skip them)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print would-be edits to stdout without writing",
    )
    ap.add_argument(
        "paths",
        nargs="+",
        help="run-metadata.toml files or directories to scan recursively",
    )
    args = ap.parse_args()

    if not _ID_RE.match(args.id):
        ap.error(
            f"--id {args.id!r} doesn't match Plan 22 charset "
            f"[A-Za-z0-9._:-]{{1,128}}"
        )
    if not args.title.strip():
        ap.error("--title must be non-empty")

    block = build_block(args.id, args.title, args.note)

    paths = discover_paths(args.paths)
    if not paths:
        print("no run-metadata.toml files found", file=sys.stderr)
        return 2

    n_written = 0
    n_skipped = 0
    n_overwritten = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if has_campaign_block(text):
            if not args.force:
                print(f"skip (has [campaign] already): {path}")
                n_skipped += 1
                continue
            text = strip_existing_block(text)
            n_overwritten += 1
        new_text = insert_block(text, block)
        if args.dry_run:
            print(f"--- would write: {path} ---")
            print(new_text)
        else:
            path.write_text(new_text, encoding="utf-8")
            print(f"wrote [campaign] into {path}")
            n_written += 1

    print(
        f"\nbackfill summary: wrote={n_written}, "
        f"overwritten={n_overwritten}, skipped={n_skipped}, total={len(paths)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
