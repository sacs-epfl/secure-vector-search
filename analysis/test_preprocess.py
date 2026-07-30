#!/usr/bin/env python3
"""Synthetic-fixture tests for analysis/preprocess.py.

No pytest in the project's venv — run directly:
    python analysis/test_preprocess.py
"""
import pathlib
import sys
import tempfile
import textwrap

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import preprocess  # noqa: E402


_RAW_HEADER = (
    "run-id,scheme,quality-param-name,quality-param,config-label,k,"
    "query-id,recall-at-k,latency-us,query-bytes,response-bytes,"
    "cluster-response-bytes,setup-bytes,pre-query-offline-bytes,"
    "verification-overhead-us,machine-id"
)


def _write_run(
    runs_dir: pathlib.Path,
    run_id: str,
    scheme: str,
    threads: int,
    binding: str,
    latency_us_per_query: list[int],
) -> None:
    """Materialise one run directory with a non-breakdown raw.csv and
    minimal run-metadata.toml carrying the threading fields we test."""
    run_dir = runs_dir / "machineX" / "shaX" / run_id
    run_dir.mkdir(parents=True)

    rows = [
        f"{run_id},{scheme},nprobe,32,nprobe=32,10,{qid},1.0,{us},0,0,0,0,0,0,machineX"
        for qid, us in enumerate(latency_us_per_query)
    ]
    (run_dir / "raw.csv").write_text(_RAW_HEADER + "\n" + "\n".join(rows) + "\n")

    meta = textwrap.dedent(
        f"""\
        run-id = "{run_id}"
        machine-id = "machineX"
        git-sha = "shaX"
        git-dirty = false
        git-branch = "main"
        started-at = "2026-05-08T00:00:00Z"
        status = "complete"
        harness-version = "0.1.0"
        rust-toolchain = "1.95.0"
        kernel-version = "test"
        cpu-governor = "performance"
        notes = ""
        no-cache = false
        breakdown = false
        parallel-threads = {threads}
        numactl-binding = "{binding}"

        [ivf]
        n-centroids = 317
        train-seed = 42
        max-iter = 25

        [scheme-config]
        scheme = "{scheme}"

        [dataset]
        path = "data/test"
        corpus-file = "passages.fvecs"
        query-file = "queries.fvecs"
        ground-truth = "ground_truth.ivecs"
        n-passages = 100
        n-queries = 5
        embedding-model = "test"
        dimension = 768
        """
    )
    (run_dir / "run-metadata.toml").write_text(meta)


def test_parallel_scaling_two_schemes_two_anchors() -> None:
    """Three runs per scheme: threads=1 pinned, threads=16 pinned,
    threads=16 unpinned. Output TSV must:
      - have 3 rows per scheme
      - distinguish the two threads=16 rows by `binding`
      - show monotonic-decreasing latency on the pinned subset
        (1 → 16-pinned)
      - report parallel_efficiency in [0, 1.5] for both pinned points
    """
    pinned = "physcpubind=0-15,membind=0"
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        runs.mkdir()
        outdir = results / "out"
        outdir.mkdir()

        # plaintext: clean ~12× speedup at N=16 pinned, ~8× at N=16
        # unpinned (to model a pinning gap).
        _write_run(runs, "1100000001", "plaintext", 1, pinned, [10_000] * 5)
        _write_run(runs, "1100000002", "plaintext", 16, pinned, [833] * 5)
        _write_run(runs, "1100000003", "plaintext", 16, "none", [1_250] * 5)

        # bntm-ivf: same shape, different absolute latencies.
        _write_run(runs, "1100000011", "bntm-ivf", 1, pinned, [50_000] * 5)
        _write_run(runs, "1100000012", "bntm-ivf", 16, pinned, [4_000] * 5)
        _write_run(runs, "1100000013", "bntm-ivf", 16, "none", [6_000] * 5)

        preprocess.write_parallel_scaling(results, outdir)

        for scheme in ("plaintext", "bntm-ivf"):
            tsv = outdir / f"parallel-scaling-{scheme}.tsv"
            assert tsv.exists(), f"missing {tsv}"
            df = pd.read_csv(tsv, sep="\t")
            assert len(df) == 3, f"{scheme}: expected 3 rows, got {len(df)}\n{df}"
            assert set(df["threads"]) == {1, 16}
            sixteen = df[df["threads"] == 16]
            assert len(sixteen) == 2, f"{scheme}: two N=16 rows expected"
            assert set(sixteen["binding"]) == {pinned, "none"}, (
                f"{scheme}: bindings on N=16 must distinguish the rows"
            )

            pinned_subset = df[df["binding"] == pinned].sort_values("threads")
            assert pinned_subset["latency_ms_mean"].is_monotonic_decreasing, (
                f"{scheme}: pinned latency must be monotonic-decreasing\n"
                f"{pinned_subset}"
            )

            # T(1) pinned exists → efficiency populated for every row.
            eff = df["parallel_efficiency"].dropna()
            assert len(eff) == len(df), f"{scheme}: efficiency must populate"
            assert ((eff >= 0) & (eff <= 1.5)).all(), (
                f"{scheme}: efficiency must be in [0, 1.5], got {eff.tolist()}"
            )

    print("test_parallel_scaling_two_schemes_two_anchors: ok")


def test_parallel_scaling_missing_t1_leaves_efficiency_nan() -> None:
    """Tiptoe-shape sweep that omits N=1: latency curve still plots,
    parallel_efficiency is NaN for every row."""
    pinned = "physcpubind=0-15,membind=0"
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        runs.mkdir()
        outdir = results / "out"
        outdir.mkdir()

        _write_run(runs, "1100000021", "tiptoe", 8, pinned, [9_000] * 5)
        _write_run(runs, "1100000022", "tiptoe", 16, pinned, [5_000] * 5)
        _write_run(runs, "1100000023", "tiptoe", 16, "none", [6_500] * 5)

        preprocess.write_parallel_scaling(results, outdir)

        df = pd.read_csv(outdir / "parallel-scaling-tiptoe.tsv", sep="\t")
        assert len(df) == 3
        # parallel_efficiency must be NaN-only for every row when T(1) is missing.
        assert df["parallel_efficiency"].isna().all(), (
            f"expected NaN-only efficiency without T(1), got {df}"
        )
        # Latency / qps still populated.
        assert df["latency_ms_mean"].notna().all()
        assert df["qps"].notna().all()

    print("test_parallel_scaling_missing_t1_leaves_efficiency_nan: ok")


_RAW_HEADER_PLAN23 = (
    "run-id,scheme,quality-param-name,quality-param,config-label,k,"
    "query-id,recall-at-k,latency-us,query-bytes,response-bytes,"
    "cluster-response-bytes,setup-bytes,"
    "pre-query-offline-up-bytes,pre-query-offline-down-bytes,"
    "verification-overhead-us,machine-id,device,effective-bytes-per-query,"
    "batch-size,wallclock-us,amortised-latency-us"
)


def _write_minimal_metadata(
    run_dir: pathlib.Path,
    run_id: str,
    scheme: str,
    *,
    threads: int = 1,
    binding: str = "none",
) -> None:
    meta = textwrap.dedent(
        f"""\
        run-id = "{run_id}"
        machine-id = "machineX"
        git-sha = "shaX"
        git-dirty = false
        git-branch = "main"
        started-at = "2026-05-12T00:00:00Z"
        status = "complete"
        harness-version = "0.1.0"
        rust-toolchain = "1.95.0"
        kernel-version = "test"
        cpu-governor = "performance"
        notes = ""
        no-cache = false
        breakdown = false
        parallel-threads = {threads}
        numactl-binding = "{binding}"

        [ivf]
        n-centroids = 317
        train-seed = 42
        max-iter = 25

        [scheme-config]
        scheme = "{scheme}"

        [dataset]
        path = "data/test"
        corpus-file = "passages.fvecs"
        query-file = "queries.fvecs"
        ground-truth = "ground_truth.ivecs"
        n-passages = 100
        n-queries = 5
        embedding-model = "test"
        dimension = 768
        """
    )
    (run_dir / "run-metadata.toml").write_text(meta)


def test_legacy_csv_backfills_batch_size_columns() -> None:
    """An older raw.csv (lacking batch-size /
    wallclock-us / amortised-latency-us) must load with the three
    columns defaulted such that B=1 single-query semantics hold:
        batch_size = 1
        wallclock_us = latency_us
        amortised_latency_us = latency_us
    The legacy _write_run helper writes the oldest CSV shape, so
    this test exercises both the device / effective-bytes
    backfill AND the batch-size backfill in one load.
    """
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        runs.mkdir()
        _write_run(runs, "1100000031", "plaintext", 4, "none", [1000, 2000, 3000])

        df = preprocess.load_results(results, all_runs=False)
        assert not df.empty, "load_results returned empty on a synthetic legacy run"
        assert (df["batch_size"] == 1).all(), (
            f"legacy CSV must backfill batch_size = 1, got {df['batch_size'].tolist()}"
        )
        # wallclock_us and amortised_latency_us collapse to latency_us at B=1.
        assert (df["wallclock_us"] == df["latency_us"]).all(), (
            f"legacy wallclock_us must equal latency_us, got "
            f"{(df['wallclock_us'] - df['latency_us']).tolist()}"
        )
        assert (df["amortised_latency_us"] == df["latency_us"]).all(), (
            "legacy amortised_latency_us must equal latency_us"
        )
        # Derived ms columns populated.
        assert df["wallclock_ms"].notna().all()
        assert df["amortised_latency_ms"].notna().all()

    print("test_legacy_csv_backfills_batch_size_columns: ok")


def test_plan23_csv_preserves_batched_rows() -> None:
    """A raw.csv with the batch-size columns + B>1 rows must
    load with latency_us empty (NaN) on batched rows and wallclock_us
    / amortised_latency_us populated. Backfill is a no-op on these
    rows (columns are already present)."""
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        run_dir = runs / "machineX" / "shaX" / "1100000041"
        run_dir.mkdir(parents=True)

        # 2 B=1 rows (latency populated) + 2 B=4 rows (latency empty).
        rows = [
            "1100000041,plaintext,nprobe,32,nprobe=32,10,0,1.0,1000,0,0,0,0,0,0,0,machineX,cpu,0,1,1000,1000",
            "1100000041,plaintext,nprobe,32,nprobe=32,10,1,1.0,1200,0,0,0,0,0,0,0,machineX,cpu,0,1,1200,1200",
            "1100000041,plaintext,nprobe,32,nprobe=32,10,0,1.0,,0,0,0,0,0,0,0,machineX,cpu,0,4,8000,2000",
            "1100000041,plaintext,nprobe,32,nprobe=32,10,1,1.0,,0,0,0,0,0,0,0,machineX,cpu,0,4,8000,2000",
        ]
        (run_dir / "raw.csv").write_text(_RAW_HEADER_PLAN23 + "\n" + "\n".join(rows) + "\n")
        _write_minimal_metadata(run_dir, "1100000041", "plaintext")

        df = preprocess.load_results(results, all_runs=False)
        assert len(df) == 4, f"expected 4 rows, got {len(df)}"

        b1 = df[df["batch_size"] == 1]
        b4 = df[df["batch_size"] == 4]
        assert len(b1) == 2 and len(b4) == 2, (
            f"batch_size partition: B=1→{len(b1)}, B=4→{len(b4)}"
        )

        # B=1: latency_us populated, equal to wallclock/amortised.
        assert b1["latency_us"].notna().all()
        assert (b1["wallclock_us"] == b1["latency_us"]).all()
        assert (b1["amortised_latency_us"] == b1["latency_us"]).all()

        # B=4: latency_us is NaN (empty cell in CSV), wallclock_us
        # populated, amortised = wallclock / 4 in the data we wrote.
        assert b4["latency_us"].isna().all(), (
            f"B>1 rows must have latency_us NaN, got {b4['latency_us'].tolist()}"
        )
        assert (b4["wallclock_us"] == 8000).all()
        assert (b4["amortised_latency_us"] == 2000).all()
        # latency_ms is NaN for B>1 rows (it's derived from latency_us);
        # amortised_latency_ms is the canonical figure-14 x-axis.
        assert b4["latency_ms"].isna().all()
        assert (b4["amortised_latency_ms"] == 2.0).all()

    print("test_plan23_csv_preserves_batched_rows: ok")


def test_throughput_vs_latency_batch_picks_qp_closest_to_target_recall() -> None:
    """Figure 14 emitter: per scheme pick the qp* whose
    B=1 recall is closest to target_recall (0.9 default), then emit
    one row per batch_size with mean amortised latency / qps / CI95 /
    recall. The fixture has two qp values for plaintext (nprobe=4 →
    recall 0.7, nprobe=32 → recall 0.95) — qp* must be 32 (|0.95 −
    0.9| = 0.05 < |0.7 − 0.9| = 0.20). At nprobe=32, two batch sizes
    (B=1 and B=8) yield a 2-row TSV with batch_size sorted ascending."""
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        outdir = results / "out"
        outdir.mkdir(parents=True)
        run_dir = runs / "machineX" / "shaX" / "1100000051"
        run_dir.mkdir(parents=True)

        rows: list[str] = []
        run = "1100000051"
        # nprobe=4: B=1 only, recall 0.70 (off target).
        for qid, rec in enumerate([1.0, 1.0, 0.0, 1.0, 0.5]):
            rows.append(
                f"{run},plaintext,nprobe,4,nprobe=4,10,{qid},{rec},2000,0,0,0,0,0,0,0,machineX,cpu,0,1,2000,2000"
            )
        # nprobe=32: B=1 (recall mean 0.95) + B=8 chunk (1 chunk × 8 queries).
        for qid, rec in enumerate([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6]):
            rows.append(
                f"{run},plaintext,nprobe,32,nprobe=32,10,{qid},{rec},5000,0,0,0,0,0,0,0,machineX,cpu,0,1,5000,5000"
            )
        # B=8 chunk: wallclock 8000us, amortised 1000us, latency-us empty.
        for qid, rec in enumerate([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6]):
            rows.append(
                f"{run},plaintext,nprobe,32,nprobe=32,10,{qid},{rec},,0,0,0,0,0,0,0,machineX,cpu,0,8,8000,1000"
            )
        (run_dir / "raw.csv").write_text(_RAW_HEADER_PLAN23 + "\n" + "\n".join(rows) + "\n")
        _write_minimal_metadata(run_dir, run, "plaintext")

        df = preprocess.load_results(results, all_runs=False)
        preprocess.write_throughput_vs_latency_batch(df, outdir)

        out_path = outdir / "throughput-vs-latency-batch-plaintext.tsv"
        assert out_path.exists(), f"expected emitter to write {out_path}"
        out = pd.read_csv(out_path, sep="\t")
        assert list(out.columns) == [
            "batch_size",
            "latency_ms_mean",
            "qps",
            "qps_rep_ci95",
            "recall_mean",
        ], f"column shape: {list(out.columns)}"
        assert len(out) == 2, f"expected 2 rows (B=1, B=8), got {len(out)}"
        assert out["batch_size"].tolist() == [1, 8], (
            f"batch_size order: {out['batch_size'].tolist()}"
        )
        # qp* = 32 (recall ≈ 0.95); nprobe=4 rows ignored.
        # At B=1: amortised_latency_us = 5000 → 5.0 ms → 200 qps.
        # At B=8: amortised_latency_us = 1000 → 1.0 ms → 1000 qps.
        assert abs(out.loc[0, "latency_ms_mean"] - 5.0) < 1e-6
        assert abs(out.loc[0, "qps"] - 200.0) < 1e-3
        assert abs(out.loc[1, "latency_ms_mean"] - 1.0) < 1e-6
        assert abs(out.loc[1, "qps"] - 1000.0) < 1e-3
        # recall_mean reflects the qp=32 rows, not qp=4.
        assert abs(out.loc[0, "recall_mean"] - 0.95) < 1e-6

    print("test_throughput_vs_latency_batch_picks_qp_closest_to_target_recall: ok")


def test_throughput_vs_latency_batch_skips_tiptoe() -> None:
    """Tiptoe is cost-floor-excluded from batched
    scope. The emitter must not produce a Tiptoe TSV even when given
    a fixture that has Tiptoe rows."""
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        outdir = results / "out"
        outdir.mkdir(parents=True)
        run_dir = runs / "machineX" / "shaX" / "1100000061"
        run_dir.mkdir(parents=True)
        rows = [
            f"1100000061,tiptoe,quantisation-bits,3,quantisation-bits=3,10,{qid},1.0,1000,0,0,0,0,0,0,0,machineX,cpu,0,1,1000,1000"
            for qid in range(3)
        ]
        (run_dir / "raw.csv").write_text(_RAW_HEADER_PLAN23 + "\n" + "\n".join(rows) + "\n")
        _write_minimal_metadata(run_dir, "1100000061", "tiptoe")

        df = preprocess.load_results(results, all_runs=False)
        preprocess.write_throughput_vs_latency_batch(df, outdir)

        out_path = outdir / "throughput-vs-latency-batch-tiptoe.tsv"
        assert not out_path.exists(), "tiptoe must be excluded from figure 14"

    print("test_throughput_vs_latency_batch_skips_tiptoe: ok")


def test_bntm_verification_groups_filters_and_no_cross_state_avg() -> None:
    """Figure 13 fixture test. Three runs on `machineBN`, all
    `scheme=bntm-ivf`, two nprobes each (8, 32):

      run-id 2001 — verification=on,  records verify_us (verify_us = 300)
      run-id 2002 — verification=off, verify_us = 0
      run-id 2003 — verification=on,  predates verify_us recording
                    (verify_us = 0 across every row, latency 100× higher)

    2003 is later than 2001; without the stale-run filter,
    `_select_bntm_runs` picks 2003 (max run-id) for
    (machineBN, bntm-ivf, on) and Panel B's verify_us_mean lands at 0,
    Panel A's on-TSV shows the absurd ~100ms latencies. With the filter,
    2003 is dropped (max verification-overhead-us = 0 across the head),
    2001 wins, and Panel B's on-row carries verify_us_mean = 300.

    Pins three invariants:
      (1) Panel A state-separation: distinct `-bntm-ivf-on.tsv` and
          `-bntm-ivf-off.tsv` files emitted.
      (2) Stale-run filter inside `_select_bntm_runs`: on-state data
          reflects the run that records verify_us, not the higher-run-id
          run that predates verify_us recording.
      (3) No cross-state averaging in Panel B: on-row verify_us_mean is
          the on-run's value (300), off-row is the off-run's (0); a
          wrong groupby key (collapsing state) would average to ~150.
    """
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        outdir = results / "out"
        outdir.mkdir(parents=True)

        # (run-id, verify_enabled, per-row verify_us, base latency_us)
        runs_spec = [
            ("2001", True, 300, 1000),     # post-Plan-18 on
            ("2002", False, 0, 700),       # off
            ("2003", True, 0, 100000),     # pre-Plan-18 on (filtered)
        ]
        for run_id, verify_enabled, verify_us, base_latency in runs_spec:
            run_dir = runs / "machineBN" / "shaBN" / run_id
            run_dir.mkdir(parents=True)
            rows = []
            for nprobe in (8, 32):
                latency = base_latency + nprobe * 10
                rows.append(
                    f"{run_id},bntm-ivf,nprobe,{nprobe},nprobe={nprobe},"
                    f"10,0,1.0,{latency},0,0,0,0,0,0,{verify_us},"
                    f"machineBN,cpu,0,1,{latency},{latency}"
                )
            (run_dir / "raw.csv").write_text(
                _RAW_HEADER_PLAN23 + "\n" + "\n".join(rows) + "\n"
            )
            (run_dir / "run-metadata.toml").write_text(
                textwrap.dedent(f"""\
                    status = "complete"
                    breakdown = false

                    [scheme-config]
                    scheme = "bntm-ivf"
                    verification-enabled = {str(verify_enabled).lower()}
                """)
            )

        preprocess.write_bntm_verification(
            pd.DataFrame(),
            outdir,
            results_dir=results,
            machine_id="machineBN",
        )

        # (1) Panel A state-separation: both TSVs emit.
        on_a = outdir / "bntm-verification-recall-latency-bntm-ivf-on.tsv"
        off_a = outdir / "bntm-verification-recall-latency-bntm-ivf-off.tsv"
        assert on_a.exists(), f"missing {on_a}"
        assert off_a.exists(), f"missing {off_a}"

        # (2) Filter correctness: on-TSV reflects 2001's latency
        # (~1 ms), not 2003's (~100 ms). The 10 ms threshold is two
        # orders of magnitude below 2003's data and an order above
        # 2001's, so a missed filter shows up unambiguously.
        on_df = pd.read_csv(on_a, sep="\t")
        assert len(on_df) == 2, f"on-TSV: expected 2 rows (one per nprobe), got\n{on_df}"
        assert on_df["latency_ms_mean"].max() < 10.0, (
            f"on-TSV's max latency_ms_mean = {on_df['latency_ms_mean'].max()}; "
            f"> 10 ms means `_select_bntm_runs` failed to filter the "
            f"pre-Plan-18 on-run (2003)\n{on_df}"
        )

        # (3) No cross-state averaging in Panel B: each state's verify
        # mean reflects only that state's runs.
        panel_b = pd.read_csv(outdir / "bntm-verification-summary.tsv", sep="\t")
        on_row = panel_b[panel_b["state"] == "on"].iloc[0]
        off_row = panel_b[panel_b["state"] == "off"].iloc[0]
        assert int(on_row["verify_us_mean"]) == 300, (
            f"on-row verify_us_mean = {on_row['verify_us_mean']}, expected 300. "
            f"0 means filter missed 2003; ~150 means state grouping collapsed "
            f"on+off into one bucket"
        )
        assert int(off_row["verify_us_mean"]) == 0, (
            f"off-row verify_us_mean = {off_row['verify_us_mean']}, expected 0"
        )

    print("test_bntm_verification_groups_filters_and_no_cross_state_avg: ok")


def test_isa_tag_classifies_avx512f_as_simd() -> None:
    """`_isa_tag` classifier contract.

    Empty tuple (= older runs with the target-features field absent)
    must classify as `"scalar"` so backwards-compat works without
    changing every old run's metadata. Tuples containing `"avx512f"`
    classify as `"simd"` (the only project-internal SIMD cfg gate
    today; if a future gate adds an `avx2`-only path, the
    classifier extends to a 3-tier scheme and this test grows)."""
    assert preprocess._isa_tag(()) == "scalar"
    assert preprocess._isa_tag(("sse2",)) == "scalar"
    assert preprocess._isa_tag(("avx2", "fma", "sse2")) == "scalar"
    assert preprocess._isa_tag(("avx2", "avx512f", "fma", "sse2")) == "simd"
    assert preprocess._isa_tag(("avx512f",)) == "simd"
    print("test_isa_tag_classifies_avx512f_as_simd: ok")


def _write_run_with_target_features(
    runs_dir: pathlib.Path,
    run_id: str,
    scheme: str,
    target_features: list[str],
) -> None:
    """Minimal raw.csv + run-metadata.toml fixture for the
    target-features selector tests. Status="complete" so the
    selector takes the fast path (the no-complete-fallback branch
    is exercised by other tests indirectly)."""
    run_dir = runs_dir / "machineX" / "shaX" / run_id
    run_dir.mkdir(parents=True)

    # One row; the selector only reads the header + first data row
    # to peek (scheme, qpn). Latency value doesn't matter here.
    rows = [
        f"{run_id},{scheme},nprobe,32,nprobe=32,10,0,1.0,1000,0,0,0,0,0,0,machineX"
    ]
    (run_dir / "raw.csv").write_text(_RAW_HEADER + "\n" + "\n".join(rows) + "\n")

    tf_list = ", ".join(f'"{f}"' for f in target_features)
    meta = textwrap.dedent(
        f"""\
        run-id = "{run_id}"
        machine-id = "machineX"
        git-sha = "shaX"
        git-dirty = false
        git-branch = "main"
        started-at = "2026-05-21T00:00:00Z"
        status = "complete"
        harness-version = "0.1.0"
        rust-toolchain = "rustc 1.95.0"
        target-features = [{tf_list}]
        kernel-version = "test"
        cpu-governor = "performance"
        notes = ""
        no-cache = false
        breakdown = false
        parallel-threads = 1
        numactl-binding = "none"
        batch-sizes = [1]
        device = "cpu"

        [ivf]
        n-centroids = 317
        train-seed = 42
        max-iter = 25

        [scheme-config]
        scheme = "{scheme}"

        [dataset]
        path = "data/test"
        corpus-file = "passages.fvecs"
        query-file = "queries.fvecs"
        ground-truth = "ground_truth.ivecs"
        n-passages = 100
        n-queries = 10
        embedding-model = "test"
        dimension = 8
        """
    )
    (run_dir / "run-metadata.toml").write_text(meta)


def test_target_features_selector_keeps_scalar_and_simd_runs() -> None:
    """`_select_runs_by_target_features` partition
    contract. Two runs that differ only on target-features must
    both survive selection; the existing `_select_runs` selector
    silently drops the older one, which is the bug this figure
    fixes. The classifier output is attached to each path so the
    downstream emitter can group by ISA without re-reading
    metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        results_dir = tmp / "results"
        runs_dir = results_dir / "runs"
        runs_dir.mkdir(parents=True)

        # Scalar baseline: earlier run-id, no avx512f.
        _write_run_with_target_features(
            runs_dir,
            run_id="1000",
            scheme="bntm-ivf",
            target_features=["sse2", "sse4.2"],
        )
        # SIMD native: later run-id, includes avx512f.
        _write_run_with_target_features(
            runs_dir,
            run_id="2000",
            scheme="bntm-ivf",
            target_features=["avx2", "avx512f", "fma", "sse2", "sse4.2"],
        )

        # Sanity: the existing `_select_runs` selector silently
        # drops the scalar baseline. This test pins that the new
        # selector preserves both.
        legacy = preprocess._select_runs(results_dir, all_runs=False, machine="machineX")
        assert len(legacy) == 1, (
            f"_select_runs should pick latest only, got {len(legacy)} paths"
        )

        # New selector: both partitions survive.
        partitioned = preprocess._select_runs_by_target_features(
            results_dir, machine="machineX"
        )
        assert len(partitioned) == 2, (
            f"expected 2 partitions (scalar + simd), got {len(partitioned)}"
        )
        tags = sorted(tag for _path, tag in partitioned)
        assert tags == ["scalar", "simd"], f"expected both isa tags, got {tags}"

    print("test_target_features_selector_keeps_scalar_and_simd_runs: ok")


def test_target_features_selector_handles_pre_plan25_absent_field() -> None:
    """Older runs (target-features field
    absent from run-metadata.toml) must classify as `"scalar"` and
    participate in the new selector. Without this the older runs
    silently disappear from figure 15 instead of feeding the
    baseline curve."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        results_dir = tmp / "results"
        runs_dir = results_dir / "runs"
        runs_dir.mkdir(parents=True)

        run_dir = runs_dir / "machineX" / "shaX" / "1000"
        run_dir.mkdir(parents=True)
        # Same minimal csv shape as the helper above…
        rows = [
            "1000,bntm-ivf,nprobe,32,nprobe=32,10,0,1.0,1000,0,0,0,0,0,0,machineX"
        ]
        (run_dir / "raw.csv").write_text(_RAW_HEADER + "\n" + "\n".join(rows) + "\n")
        # …but the metadata deliberately omits target-features.
        (run_dir / "run-metadata.toml").write_text(textwrap.dedent(
            """\
            run-id = "1000"
            machine-id = "machineX"
            git-sha = "shaX"
            git-dirty = false
            git-branch = "main"
            started-at = "2026-05-19T00:00:00Z"
            status = "complete"
            harness-version = "0.1.0"
            rust-toolchain = "rustc 1.95.0"
            kernel-version = "test"
            cpu-governor = "performance"
            notes = ""
            no-cache = false
            breakdown = false
            parallel-threads = 1
            numactl-binding = "none"
            batch-sizes = [1]
            device = "cpu"

            [ivf]
            n-centroids = 317
            train-seed = 42
            max-iter = 25

            [scheme-config]
            scheme = "bntm-ivf"

            [dataset]
            path = "data/test"
            corpus-file = "passages.fvecs"
            query-file = "queries.fvecs"
            ground-truth = "ground_truth.ivecs"
            n-passages = 100
            n-queries = 10
            embedding-model = "test"
            dimension = 8
            """
        ))

        partitioned = preprocess._select_runs_by_target_features(
            results_dir, machine="machineX"
        )
        assert len(partitioned) == 1, f"expected 1 partition, got {len(partitioned)}"
        _path, tag = partitioned[0]
        assert tag == "scalar", (
            f"pre-Plan-25 run (no target-features) must classify as scalar, got {tag}"
        )

    print("test_target_features_selector_handles_pre_plan25_absent_field: ok")


def _write_run_with_n_passages(
    runs_dir: pathlib.Path,
    run_id: str,
    scheme: str,
    n_passages: int,
    n_centroids: int,
) -> None:
    """Fixture mirroring the production run-metadata.toml shape with
    `n_passages` + `n_centroids` parameterised. Stages mixed-corpus
    runs under one machine to exercise the `--n-passages` filter."""
    run_dir = runs_dir / "machineX" / "shaX" / run_id
    run_dir.mkdir(parents=True)
    rows = [
        f"{run_id},{scheme},nprobe,32,nprobe=32,10,{qid},1.0,1000,0,0,0,0,0,0,machineX"
        for qid in range(3)
    ]
    (run_dir / "raw.csv").write_text(_RAW_HEADER + "\n" + "\n".join(rows) + "\n")
    meta = textwrap.dedent(
        f"""\
        run-id = "{run_id}"
        machine-id = "machineX"
        git-sha = "shaX"
        git-dirty = false
        git-branch = "main"
        started-at = "2026-05-21T00:00:00Z"
        status = "complete"
        breakdown = false

        [ivf]
        n-centroids = {n_centroids}
        train-seed = 42
        max-iter = 25

        [scheme-config]
        scheme = "{scheme}"

        [dataset]
        n-passages = {n_passages}
        """
    )
    (run_dir / "run-metadata.toml").write_text(meta)


def test_n_passages_filter_keeps_one_corpus_drops_the_other() -> None:
    """Stage four runs across two corpora (100 and 8_800_000) on one
    machine; without a filter the cross-IVF parity guard would fire.

    With `n_passages=8_800_000` the selector drops the two 100-passage
    runs entirely and `load_results` returns only the 8.8M rows — the
    parity guard then sees a single IVF key and passes."""
    with tempfile.TemporaryDirectory() as tmp:
        results = pathlib.Path(tmp)
        runs = results / "runs"
        runs.mkdir()
        _write_run_with_n_passages(runs, "1100000001", "sap", n_passages=100, n_centroids=317)
        _write_run_with_n_passages(runs, "1100000002", "emvp", n_passages=100, n_centroids=317)
        _write_run_with_n_passages(runs, "1100000011", "plaintext", n_passages=8_800_000, n_centroids=2967)
        _write_run_with_n_passages(runs, "1100000012", "sap-ivf", n_passages=8_800_000, n_centroids=2967)

        # Unfiltered: parity guard fires (sys.exit) — expect SystemExit.
        try:
            preprocess.load_results(results, machine="machineX")
        except SystemExit as e:
            assert "IVF defaults differ" in str(e), f"unexpected exit message: {e}"
        else:
            raise AssertionError("expected SystemExit from cross-IVF parity guard")

        # Filtered to 8.8M: returns only the two 8.8M schemes, no exit.
        df = preprocess.load_results(results, machine="machineX", n_passages=8_800_000)
        assert not df.empty, "filter should retain the 8.8M runs"
        schemes = set(df["scheme"].unique())
        assert schemes == {"plaintext", "sap-ivf"}, (
            f"expected only 8.8M schemes after filter, got {schemes}"
        )

        # Filtered to 100: returns only the two 100-passage schemes.
        df_small = preprocess.load_results(results, machine="machineX", n_passages=100)
        small_schemes = set(df_small["scheme"].unique())
        assert small_schemes == {"sap", "emvp"}, (
            f"expected only 100-passage schemes after filter, got {small_schemes}"
        )

        # Filtered to a value present in no run: empty DataFrame.
        df_none = preprocess.load_results(results, machine="machineX", n_passages=42)
        assert df_none.empty, "filter to absent n-passages should return empty df"

    print("test_n_passages_filter_keeps_one_corpus_drops_the_other: ok")


if __name__ == "__main__":
    test_parallel_scaling_two_schemes_two_anchors()
    test_parallel_scaling_missing_t1_leaves_efficiency_nan()
    test_legacy_csv_backfills_batch_size_columns()
    test_plan23_csv_preserves_batched_rows()
    test_throughput_vs_latency_batch_picks_qp_closest_to_target_recall()
    test_throughput_vs_latency_batch_skips_tiptoe()
    test_bntm_verification_groups_filters_and_no_cross_state_avg()
    test_isa_tag_classifies_avx512f_as_simd()
    test_target_features_selector_keeps_scalar_and_simd_runs()
    test_target_features_selector_handles_pre_plan25_absent_field()
    test_n_passages_filter_keeps_one_corpus_drops_the_other()
    print("all tests passed")
