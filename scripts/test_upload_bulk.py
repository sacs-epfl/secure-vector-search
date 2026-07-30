#!/usr/bin/env python3
"""Unit tests for scripts/upload_bulk.py.

Runs under stdlib unittest because pytest isn't required by the
project's venv. Invoke via::

    venv/bin/python scripts/test_upload_bulk.py

Covers four acceptance gates:

1. Hash matches stdlib hashlib on a fixture.
2. insert_bulk_block places after [campaign] if present, else
   before the first table.
3. Backend resolution picks file:// direct write when the URI
   scheme is `file://` or bare path.
4. End-to-end with a `file://` backend writes the [bulk] block,
   re-hashes, and deletes (or retains) the source.

The atomicity SIGTERM gate is non-deterministic
to reproduce in CI; we substitute a tempfile-rename check that
asserts no `.tmp` sibling lingers after a successful write.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

# Make `upload_bulk` importable when invoked as `python scripts/test_upload_bulk.py`.
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import upload_bulk as ub  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_run_dir(
    tmp: Path,
    *,
    with_campaign: bool = True,
    with_bulk: bool = False,
    with_aggregated: bool = True,
    machine_id: str = "abc12345",
    files: tuple[str, ...] = ("raw.csv", "top_k.csv"),
) -> Path:
    """Create a synthetic run dir under `tmp`.

    Layout: ``<tmp>/results/runs/<machine-id>/<git-sha>/<run-id>/``
    (matches production so the aggregated-dir precondition derivation works).
    When `with_aggregated=True` (the default) also creates
    ``<tmp>/results/aggregated/<machine-id>/`` so the precondition
    check passes; pass False to exercise the failure path.
    """
    repo_root = tmp.resolve()
    sha = "deadbeefcafef00d"
    run = repo_root / "results" / "runs" / machine_id / sha / "1700000000"
    run.mkdir(parents=True)
    if with_aggregated:
        (repo_root / "results" / "aggregated" / machine_id).mkdir(parents=True)
    parts = [
        'run-id = "1700000000"',
        f'machine-id = "{machine_id}"',
        'git-sha = "deadbeefcafef00d"',
        'git-dirty = false',
        'git-branch = "main"',
        'started-at = "2026-05-12T00:00:00Z"',
        'status = "complete"',
        'harness-version = "0.1.0"',
        'rust-toolchain = "rustc 1.95.0"',
        'kernel-version = "unknown"',
        'cpu-governor = "unknown"',
        'notes = ""',
        'no-cache = false',
        'breakdown = false',
        'parallel-threads = 1',
        'numactl-binding = "none"',
        'device = "cpu"',
        "",
    ]
    if with_campaign:
        parts += [
            "[campaign]",
            'id    = "test-campaign-2026-05-12"',
            'title = "test campaign"',
            "",
        ]
    if with_bulk:
        parts += [
            "[bulk]",
            'uri       = "file:///already/uploaded"',
            'retention = "60d"',
            "",
        ]
    # An [ivf] block to test the "before first table" placement
    # path when [campaign] is absent.
    parts += [
        "[ivf]",
        "n-centroids = 317",
        "train-seed = 42",
        "max-iter = 25",
        "",
        "[scheme-config]",
        'scheme = "plaintext"',
        "",
        "[dataset]",
        'path = "data/msmarco"',
        'corpus-file = "passages.fvecs"',
        'query-file = "queries.fvecs"',
        'ground-truth = "ground_truth.ivecs"',
        "n-passages = 100",
        "n-queries = 10",
        'embedding-model = "test"',
        "dimension = 768",
        "",
    ]
    (run / "run-metadata.toml").write_text("\n".join(parts), encoding="utf-8")
    for name in files:
        (run / name).write_bytes(name.encode("utf-8") * 7)  # small + deterministic
    return run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class HashTests(unittest.TestCase):
    def test_sha256_matches_stdlib(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "fixture.bin"
            blob = b"the quick brown fox\n" * 1024
            p.write_bytes(blob)
            digest, size = ub.sha256_file(p)
            self.assertEqual(digest, hashlib.sha256(blob).hexdigest())
            self.assertEqual(size, len(blob))


class InsertBulkTests(unittest.TestCase):
    BULK = ub.render_bulk_block(
        "file:///tmp/x", "60d", [("raw.csv", "a" * 64, 7)]
    )

    def test_insert_after_campaign(self):
        existing = (
            'run-id = "x"\n'
            'machine-id = "y"\n'
            "\n"
            "[campaign]\n"
            'id = "c"\n'
            'title = "t"\n'
            "\n"
            "[ivf]\n"
            "n-centroids = 1\n"
        )
        out = ub.insert_bulk_block(existing, self.BULK)
        # [bulk] header must appear after [campaign] and before [ivf].
        i_camp = out.index("[campaign]")
        i_bulk = out.index("[bulk]")
        i_ivf = out.index("[ivf]")
        self.assertLess(i_camp, i_bulk)
        self.assertLess(i_bulk, i_ivf)

    def test_insert_before_first_table_when_no_campaign(self):
        existing = (
            'run-id = "x"\n'
            'machine-id = "y"\n'
            "\n"
            "[ivf]\n"
            "n-centroids = 1\n"
        )
        out = ub.insert_bulk_block(existing, self.BULK)
        i_bulk = out.index("[bulk]")
        i_ivf = out.index("[ivf]")
        self.assertLess(i_bulk, i_ivf)
        # And the bulk block is still a valid block (tomllib parses).
        parsed = tomllib.loads(out)
        self.assertIn("bulk", parsed)
        self.assertEqual(parsed["bulk"]["uri"], "file:///tmp/x")

    def test_insert_appends_when_no_tables(self):
        existing = 'run-id = "x"\n'
        out = ub.insert_bulk_block(existing, self.BULK)
        self.assertIn("[bulk]", out)
        # tomllib should still parse the merged content.
        parsed = tomllib.loads(out)
        self.assertIn("bulk", parsed)


class BackendResolutionTests(unittest.TestCase):
    def test_file_scheme_picks_local(self):
        b = ub.resolve_backend("file:///tmp/bulk", "abc", "1700000000")
        self.assertEqual(b.local_path, Path("/tmp/bulk/abc/1700000000"))
        self.assertIsNone(b.rsync_target)
        self.assertEqual(b.uri, "file:///tmp/bulk/abc/1700000000")

    def test_bare_path_picks_local(self):
        b = ub.resolve_backend("/some/abs/path", "abc", "1700000000")
        # urlparse on a bare path gives empty scheme; treated as file://.
        self.assertEqual(b.local_path, Path("/some/abs/path/abc/1700000000"))
        self.assertIsNone(b.rsync_target)

    def test_rcp_scratch_falls_back_to_rsync_when_no_mount(self):
        # Probe directory probably doesn't exist on the test host
        # (mini's /mnt/sacs/scratch/shared/secure-vsearch is absent).
        # If it does exist somehow, this test is a no-op for the
        # rsync branch — skip rather than mis-assert.
        if Path(ub.RCP_SCRATCH_LOCAL_PROBE).is_dir():
            self.skipTest(
                "RCP scratch probe dir exists on this host; rsync-fallback "
                "branch not exercised"
            )
        b = ub.resolve_backend(
            "rcp-scratch:///mnt/sacs/scratch/shared/secure-vsearch/bulk-store",
            "abc",
            "1700000000",
        )
        self.assertIsNone(b.local_path)
        self.assertIsNotNone(b.rsync_target)
        self.assertIn("jumphost:", b.rsync_target)
        # Canonical URI is the same in both branches.
        self.assertEqual(
            b.uri,
            "rcp-scratch:///mnt/sacs/scratch/shared/secure-vsearch/bulk-store/abc/1700000000",
        )

    def test_unsupported_scheme_rejected(self):
        with self.assertRaises(ub.UploadError):
            ub.resolve_backend("s3://bucket/path", "abc", "x")


class EndToEndFileBackendTests(unittest.TestCase):
    def test_upload_writes_bulk_block_and_deletes_source(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo))
            ub.upload_run(
                run_dir=run,
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=False,
            )
            # Source CSVs gone.
            self.assertFalse((run / "raw.csv").exists())
            self.assertFalse((run / "top_k.csv").exists())
            # Metadata has [bulk] block.
            parsed = tomllib.loads((run / "run-metadata.toml").read_text())
            self.assertIn("bulk", parsed)
            self.assertEqual(parsed["bulk"]["retention"], "60d")
            self.assertTrue(
                parsed["bulk"]["uri"].endswith("/abc12345/1700000000"),
                f"unexpected bulk uri: {parsed['bulk']['uri']}",
            )
            files = parsed["bulk"]["files"]
            names = {f["name"] for f in files}
            self.assertEqual(names, {"raw.csv", "top_k.csv"})
            # Files are at the bulk destination with matching hashes.
            for f in files:
                dest = Path(f"{tmp_bulk}/abc12345/1700000000/{f['name']}")
                self.assertTrue(dest.is_file())
                self.assertEqual(
                    hashlib.sha256(dest.read_bytes()).hexdigest(),
                    f["sha256"],
                )
                self.assertEqual(dest.stat().st_size, f["bytes"])

    def test_upload_keep_source_retains_csvs(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo))
            ub.upload_run(
                run_dir=run,
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=True,
            )
            self.assertTrue((run / "raw.csv").exists())
            self.assertTrue((run / "top_k.csv").exists())

    def test_upload_skips_when_bulk_block_present(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo), with_bulk=True)
            # Should not overwrite the existing [bulk] block; raw
            # CSVs should remain on disk (no destructive action).
            ub.upload_run(
                run_dir=run,
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=False,
            )
            parsed = tomllib.loads((run / "run-metadata.toml").read_text())
            # Untouched pre-existing URI.
            self.assertEqual(parsed["bulk"]["uri"], "file:///already/uploaded")
            # Source CSVs still present (the skip path is read-only).
            self.assertTrue((run / "raw.csv").exists())

    def test_no_tmp_file_lingers_after_atomic_write(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo))
            ub.upload_run(
                run_dir=run,
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=True,
            )
            # The atomic-write sentinel must not be left behind.
            tmp_sentinel = run / "run-metadata.toml.tmp"
            self.assertFalse(tmp_sentinel.exists())


class MigrationModeTests(unittest.TestCase):
    """End-to-end tests for `--migrate`.

    Builds a synthetic mixed tree under a tempdir:
      - one run with no [bulk] (uploadable)
      - one run already-uploaded (skipped by [bulk] guard)
      - one run with raw CSVs `git ls-files`-tracked (skipped by
        the git-tracked guard) — simulated by initialising a real
        git repo in the tempdir and adding the relevant files
      - one run for a different machine-id (skipped by --machine
        filter)
    """

    def _git(self, cwd, *args, check=True):
        import subprocess
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {result.stderr}"
            )
        return result

    def test_migration_admits_correct_subset(self):
        with tempfile.TemporaryDirectory() as tmp_repo_str, tempfile.TemporaryDirectory() as tmp_bulk:
            tmp_repo = Path(tmp_repo_str)
            # Init a real git repo so `git ls-files` returns
            # meaningful results.
            self._git(tmp_repo, "init", "-q", "-b", "main")
            self._git(tmp_repo, "config", "user.email", "test@example.com")
            self._git(tmp_repo, "config", "user.name", "Test")

            # Run A — uploadable (no [bulk], not tracked).
            run_a = _write_run_dir(tmp_repo, machine_id="abc12345", with_campaign=False)
            # Run B — already has [bulk] (skipped).
            run_b = _write_run_dir(tmp_repo, machine_id="def67890", with_bulk=True)
            # Run C — raw CSVs tracked in git (skipped by the git-tracked guard).
            run_c = _write_run_dir(tmp_repo, machine_id="aaaaaaaa")
            # Stage + commit run_c's raw CSV so `git ls-files` reports it.
            self._git(tmp_repo, "add", str(run_c / "raw.csv"))
            self._git(tmp_repo, "commit", "-q", "-m", "track run C raw")
            # Run D — different machine-id, gets filtered out by --machine.
            run_d = _write_run_dir(tmp_repo, machine_id="bbbbbbbb", with_campaign=False)

            # Migration: no machine filter — admits A and D (B has
            # [bulk]; C is git-tracked).
            uploadable, skipped = ub.find_uploadable_runs(
                roots=[tmp_repo / "results/runs"],
                repo_root=tmp_repo,
            )
            # Run dirs share the run-id "1700000000" because the
            # fixture builder doesn't vary it; identify by the
            # parent-of-parent machine-id (run / git-sha / machine).
            uploadable_machines = sorted(p.parents[1].name for p in uploadable)
            self.assertEqual(uploadable_machines, ["abc12345", "bbbbbbbb"])
            skipped_machines = sorted(
                p.parents[1].name for p, _ in skipped
            )
            self.assertEqual(skipped_machines, ["aaaaaaaa", "def67890"])

    def test_migration_with_machine_filter(self):
        with tempfile.TemporaryDirectory() as tmp_repo_str:
            tmp_repo = Path(tmp_repo_str)
            self._git(tmp_repo, "init", "-q", "-b", "main")
            self._git(tmp_repo, "config", "user.email", "t@e")
            self._git(tmp_repo, "config", "user.name", "T")

            # Two runs, two different machines.
            run_a = _write_run_dir(tmp_repo, machine_id="abc12345", with_campaign=False)
            run_b = _write_run_dir(tmp_repo, machine_id="bbbbbbbb", with_campaign=False)

            uploadable, skipped = ub.find_uploadable_runs(
                roots=[tmp_repo / "results/runs"],
                repo_root=tmp_repo,
                machine_filter="abc12345",
            )
            self.assertEqual(
                [p.parents[1].name for p in uploadable], ["abc12345"]
            )
            # The other machine was skipped for the right reason.
            self.assertTrue(
                any("machine-id=bbbbbbbb" in r for _, r in skipped),
                f"expected machine-filter skip, got skipped={skipped}",
            )

    def test_migration_uploads_to_bulk(self):
        with tempfile.TemporaryDirectory() as tmp_repo_str, tempfile.TemporaryDirectory() as tmp_bulk:
            tmp_repo = Path(tmp_repo_str)
            self._git(tmp_repo, "init", "-q", "-b", "main")
            self._git(tmp_repo, "config", "user.email", "t@e")
            self._git(tmp_repo, "config", "user.name", "T")

            run_a = _write_run_dir(tmp_repo, machine_id="abc12345", with_campaign=False)

            uploadable, _ = ub.find_uploadable_runs(
                roots=[tmp_repo / "results/runs"],
                repo_root=tmp_repo,
            )
            self.assertEqual(len(uploadable), 1)
            ub.upload_run(
                run_dir=uploadable[0],
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=False,
            )
            # Verify the [bulk] block was inserted.
            parsed = tomllib.loads(
                (uploadable[0] / "run-metadata.toml").read_text()
            )
            self.assertIn("bulk", parsed)

            # Second migration walk: the run is now skipped (already has [bulk]).
            uploadable2, skipped2 = ub.find_uploadable_runs(
                roots=[tmp_repo / "results/runs"],
                repo_root=tmp_repo,
            )
            self.assertEqual(uploadable2, [])
            self.assertTrue(
                any("already has [bulk] block" in r for _, r in skipped2),
                f"second walk should skip uploaded run, got {skipped2}",
            )


class PreconditionTests(unittest.TestCase):
    """Refuse to upload when results/aggregated/<m>/ is absent."""

    def test_precondition_blocks_when_aggregated_dir_absent(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo), with_aggregated=False)
            with self.assertRaises(ub.UploadError) as cm:
                ub.upload_run(
                    run_dir=run,
                    bulk_base=f"file://{tmp_bulk}",
                    retention="60d",
                    keep_source=False,
                )
            self.assertIn("precondition failed", str(cm.exception))

    def test_precondition_passes_when_aggregated_dir_present(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo), with_aggregated=True)
            ub.upload_run(
                run_dir=run,
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=False,
            )
            parsed = tomllib.loads((run / "run-metadata.toml").read_text())
            self.assertIn("bulk", parsed)

    def test_skip_precondition_flag_bypasses(self):
        with tempfile.TemporaryDirectory() as tmp_repo, tempfile.TemporaryDirectory() as tmp_bulk:
            run = _write_run_dir(Path(tmp_repo), with_aggregated=False)
            # Should not raise: bypass exists for genuine edge cases.
            ub.upload_run(
                run_dir=run,
                bulk_base=f"file://{tmp_bulk}",
                retention="60d",
                keep_source=False,
                skip_precondition=True,
            )
            parsed = tomllib.loads((run / "run-metadata.toml").read_text())
            self.assertIn("bulk", parsed)


if __name__ == "__main__":
    unittest.main()
