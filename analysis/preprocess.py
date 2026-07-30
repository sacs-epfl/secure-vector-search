#!/usr/bin/env python3
"""Aggregate eval CSVs into per-figure TSVs for pgfplots."""
import argparse
import pathlib
import subprocess
import sys

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli for Python < 3.11
    except ImportError:
        tomllib = None  # type: ignore[assignment]

import numpy as np
import pandas as pd


def _read_status(run_dir: pathlib.Path) -> str:
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            return tomllib.load(f).get("status", "unknown")
    return "unknown"


def _read_breakdown_flag(run_dir: pathlib.Path) -> bool:
    """Return `[run-metadata.toml].breakdown`. Defaults to `False`
    (pre-Plan-14 runs and any run without `--breakdown` set)."""
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            return bool(tomllib.load(f).get("breakdown", False))
    return False


def _read_target_features(run_dir: pathlib.Path) -> tuple[str, ...]:
    """Return the sorted `[run-metadata.toml].target-features` list as
    a tuple (so it's hashable for `dict` keys). Empty tuple when the
    field is absent (older runs recorded before the field existed).
    The empty tuple is the canonical "unknown,
    assume baseline-x86-64" sentinel; downstream code must not
    distinguish it from an explicit empty list.

    The collected features are a curated set (see
    `eval_harness::meta::collect_target_features`); adding a new
    `cfg(target_feature = "X")` site to the workspace should be
    paired with adding "X" to that probe list."""
    meta_path = run_dir / "run-metadata.toml"
    if not (meta_path.exists() and tomllib is not None):
        return ()
    with open(meta_path, "rb") as f:
        raw = tomllib.load(f).get("target-features", [])
    return tuple(sorted(str(x) for x in raw))


def _isa_tag(features: tuple[str, ...]) -> str:
    """Classify a run as `"scalar"` or `"simd"` based on
    its target-features.

    Today the only project-internal SIMD cfg gate is
    `target_feature = "avx512f"` (BN AVX-512 matvec in
    `scorer-bntm/src/crypto.rs`, L2 dispatchers in
    `ivf-index/src/distance.rs` and `scorer-sap/src/distance.rs`).
    Adding a new gate (e.g. `avx2`-only path) would extend the
    classifier's tier set.

    Runs with no recorded target-features (empty tuple) classify as `"scalar"` — the
    sentinel maps to the "no project SIMD active" partition, which
    is the correct default for a baseline-x86-64 build."""
    if "avx512f" in features:
        return "simd"
    return "scalar"


def _read_index_block(run_dir: pathlib.Path) -> dict | None:
    """Parse the [index] block from run-metadata.toml. Returns None if
    the block is missing (older runs recorded before the block existed)."""
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            return tomllib.load(f).get("index")
    return None


def _read_scheme_name(run_dir: pathlib.Path) -> str | None:
    """Extract the scheme identifier from `[scheme-config].scheme`."""
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            cfg = tomllib.load(f).get("scheme-config", {})
            return cfg.get("scheme")
    return None


def _read_device(run_dir: pathlib.Path) -> str:
    """Return the top-level `device` field from run-metadata.toml.
    Defaults to `'cpu'` for pre-Plan-17 runs without the field —
    matches `_backfill_legacy_columns`'s treatment of the per-row
    raw.csv column. Used as part of `_select_runs`' partition key so
    a scheme's cpu and gpu runs don't cross-contaminate a
    single-device figure (figure 11's cpu-vs-gpu emitter handles
    both substrates explicitly)."""
    meta_path = run_dir / "run-metadata.toml"
    if not (meta_path.exists() and tomllib is not None):
        return "cpu"
    with open(meta_path, "rb") as f:
        return str(tomllib.load(f).get("device", "cpu"))


def _read_sweep_size(run_dir: pathlib.Path, qpn: str) -> int:
    """Number of quality-param values in this run's [scheme-config]
    sweep list. Used as the primary tiebreaker in `_select_runs` so a
    full sweep (e.g. nprobe=[1,2,4,8,16,32,64,128]) beats a
    single-point follow-up (e.g. a parallel-scaling sweep that pins
    nprobe=32 and varies thread count) when both runs share the same
    (machine, scheme, qpn) key. Without this tiebreaker the freshest
    run wins, and the scaling sweeps — which always land after the
    figure-1 sweep on the same machine — silently truncate figure 1
    to a single point.

    Returns 1 when qpn is `none`, when no metadata is readable, or
    when the field isn't a list — those all fall through to the
    secondary run-id-desc tiebreaker."""
    field = {
        "nprobe": "nprobe-values",
        "beta": "beta-values",
        "quantisation-bits": "quantisation-bits-values",
    }.get(qpn)
    if field is None:
        return 1
    meta_path = run_dir / "run-metadata.toml"
    if not (meta_path.exists() and tomllib is not None):
        return 1
    with open(meta_path, "rb") as f:
        cfg = tomllib.load(f).get("scheme-config", {})
    v = cfg.get(field)
    return len(v) if isinstance(v, list) else 1


def _read_verification_enabled(run_dir: pathlib.Path) -> bool | None:
    """Extract `[scheme-config].verification-enabled`. Returns None for
    runs without the field (every scheme except BN today)."""
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            cfg = tomllib.load(f).get("scheme-config", {})
            return cfg.get("verification-enabled")
    return None


def _read_ivf_block(run_dir: pathlib.Path) -> dict | None:
    """Parse the [ivf] block from run-metadata.toml. Returns None if the
    block is missing (older runs recorded before the block existed) — those
    skip the parity check rather than hard-fail."""
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            return tomllib.load(f).get("ivf")
    return None


def _read_n_passages(run_dir: pathlib.Path) -> int | None:
    """Parse [dataset].n-passages from run-metadata.toml. Returns None
    when the field (or the file) is missing — keeps pre-Plan-10 runs
    and any partial metadata from blowing up the `--n-passages` filter."""
    meta_path = run_dir / "run-metadata.toml"
    if not (meta_path.exists() and tomllib is not None):
        return None
    with open(meta_path, "rb") as f:
        ds = tomllib.load(f).get("dataset")
    if ds is None:
        return None
    v = ds.get("n-passages")
    return int(v) if v is not None else None


def _check_ivf_parity(csv_paths: list[pathlib.Path]) -> None:
    """Configuration parity: runs combined into one figure must
    share IVF defaults (n_centroids, train_seed, max_iter). Differences
    confound the comparison — fail loud rather than silently producing
    incomparable-but-similar-looking numbers.

    Runs without an [ivf] block are skipped with a warning.
    """
    seen: dict[tuple, list[pathlib.Path]] = {}
    skipped: list[pathlib.Path] = []
    for path in csv_paths:
        ivf = _read_ivf_block(path.parent)
        if ivf is None:
            skipped.append(path)
            continue
        key = (
            int(ivf.get("n-centroids", -1)),
            int(ivf.get("train-seed", -1)),
            int(ivf.get("max-iter", -1)),
        )
        seen.setdefault(key, []).append(path)

    if skipped:
        print(
            f"warning: {len(skipped)} run(s) have no [ivf] block "
            "(pre-Plan-10); IVF parity not checked for these.",
            file=sys.stderr,
        )

    if len(seen) > 1:
        msg = ["IVF defaults differ across runs being aggregated:"]
        for key, paths in seen.items():
            n_centroids, train_seed, max_iter = key
            msg.append(
                f"  n_centroids={n_centroids} train_seed={train_seed} "
                f"max_iter={max_iter}: {len(paths)} run(s)"
            )
            for p in paths[:3]:
                msg.append(f"    {p.parent}")
            if len(paths) > 3:
                msg.append(f"    ... and {len(paths) - 3} more")
        msg.append(
            "Combining these runs would produce comparable-looking but actually "
            "incomparable numbers. Re-run with consistent IVF defaults, filter "
            "to a single configuration before plotting, or pass "
            "--n-passages N (Makefile: N_PASSAGES=N) to restrict to one "
            "corpus size."
        )
        sys.exit("\n".join(msg))


def _peek_scheme_qpn(csv_path: pathlib.Path) -> tuple[str, str] | None:
    """Return (scheme, quality-param-name) from raw.csv's first data row.
    None when the file is header-only or unreadable."""
    try:
        with csv_path.open() as f:
            f.readline()
            row = f.readline()
    except OSError:
        return None
    if not row:
        return None
    cols = row.rstrip("\n").split(",")
    if len(cols) < 3:
        return None
    return cols[1], cols[2]


def _select_runs(
    results_dir: pathlib.Path,
    all_runs: bool,
    machine: str | None = None,
    n_passages: int | None = None,
) -> list[pathlib.Path]:
    """Return raw.csv paths to load: latest complete run per
    (machine, scheme, quality-param-name, device), or all runs.

    The cluster-bundle workflow (deploy/run-bundle*.sh) invokes
    `cargo run --bin eval` once per scheme, producing one run-dir
    per scheme under the same machine-id; SAP-IVF's nprobe and beta
    sweeps additionally produce two distinct (scheme=sap-ivf,
    quality-param-name) keys; the `device` axis keeps cpu and gpu
    runs of the same scheme from sharing a slot. Grouping at
    (machine, scheme, quality-param-name, device) ensures every
    scheme + sweep + substrate gets exactly one representative
    without silently merging dev iterations or cross-SHA data, and
    without cross-contaminating single-device figures with the
    other substrate.

    Returns an empty list when only `--breakdown` runs are present —
    those don't produce raw.csv rows by design (figure 02 throughput
    data only comes from non-breakdown runs).

    When `machine` is given, only that machine's runs
    are considered — used by the per-machine canonical emission path
    `results/aggregated/<machine-id>/`."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return []

    # Layout: runs/<machine-id>/<git-sha>/<run-id>/raw.csv
    # Breakdown runs emit substep-breakdown.csv as
    # their data source and leave raw.csv header-only — skip them when
    # selecting the data source for figures 01–07.
    by_key: dict[
        tuple[str, str, str, str],
        list[tuple[str, str, pathlib.Path]],
    ] = {}
    for csv_path in sorted(runs_dir.rglob("raw.csv")):
        if _read_breakdown_flag(csv_path.parent):
            continue
        rel = csv_path.relative_to(runs_dir)
        machine_id = rel.parts[0]
        if machine and machine_id != machine:
            continue
        if n_passages is not None and _read_n_passages(csv_path.parent) != n_passages:
            continue
        sha = rel.parts[1]
        run_id = rel.parts[2]
        header = _peek_scheme_qpn(csv_path)
        if header is None:
            continue
        scheme, qpn = header
        device = _read_device(csv_path.parent)
        by_key.setdefault((machine_id, scheme, qpn, device), []).append(
            (run_id, sha, csv_path)
        )

    if not by_key:
        return []

    if all_runs:
        return [csv for runs in by_key.values() for _, _, csv in runs]

    selected: list[tuple[str, str, str, str, str, pathlib.Path]] = []
    for (machine_id, scheme, qpn, device), runs in by_key.items():
        complete = [
            (rid, sha, p)
            for rid, sha, p in runs
            if _read_status(p.parent) == "complete"
        ]
        pool = complete or runs
        if not complete:
            print(
                f"warning: no complete run for ({machine_id}, {scheme}, "
                f"{qpn}, {device}), using latest available",
                file=sys.stderr,
            )
        # Sweep size first so a full quality-param sweep beats a
        # single-point scaling-sweep follow-up on the same key; run-id
        # desc breaks ties between equally-wide sweeps (latest wins).
        rid, sha, p = max(
            pool,
            key=lambda x: (_read_sweep_size(x[2].parent, qpn), x[0]),
        )
        selected.append((machine_id, scheme, qpn, device, sha, p))

    # Warn loudly when the selected runs for one machine span more than
    # one git SHA: cross-scheme comparisons would otherwise silently mix
    # code versions (e.g. plaintext at SHA A vs. SAP at SHA B). SHA-set
    # is computed per (machine, device) so a cpu sweep on SHA A and an
    # independent gpu sweep on SHA B don't flag — they don't share a
    # figure.
    per_md_shas: dict[tuple[str, str], set[str]] = {}
    for machine_id, _scheme, _qpn, dev, sha, _p in selected:
        per_md_shas.setdefault((machine_id, dev), set()).add(sha)
    for (machine_id, dev), shas in per_md_shas.items():
        if len(shas) > 1:
            print(
                f"warning: machine {machine_id} ({dev}) selected runs span "
                f"{len(shas)} git SHAs ({', '.join(sorted(s[:8] for s in shas))}); "
                f"cross-scheme comparisons may mix code versions — "
                f"prune older runs or re-run on the current SHA",
                file=sys.stderr,
            )

    return [p for _machine, _scheme, _qpn, _dev, _sha, p in selected]


def _select_runs_by_target_features(
    results_dir: pathlib.Path,
    machine: str | None = None,
    n_passages: int | None = None,
) -> list[tuple[pathlib.Path, str]]:
    """Variant of `_select_runs` that adds `target-features`
    as a 4th partition key. Used exclusively by the figure-15
    (scalar-vs-SIMD) TSV emitters; the main `_select_runs` selector
    stays unchanged so every existing figure keeps its current
    "latest complete per (machine, scheme, qpn)" behaviour.

    Returns a list of `(raw_csv_path, isa_tag)` pairs. The partition
    key is `(machine, scheme, qpn, target-features-tuple)`; the
    `isa_tag` string is `_isa_tag(features)`'s output, attached to
    every selected path so downstream emitters can group / colour /
    style by ISA without re-reading metadata.

    Within each partition the selection rule mirrors `_select_runs`:
    latest complete `run-id` wins; falls back to latest of any status
    with a stderr warning when no complete run is available.

    Pre-Plan-25 runs (target-features field absent) get the empty-
    tuple partition; `_isa_tag(())` is `"scalar"`. If a machine has
    both a pre-Plan-25 scalar run and a new explicit-baseline scalar
    run, they collide on the same partition — the newer one wins per
    the latest-run-id rule, which is the intended semantics
    (equivalent data, newer is better)."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return []

    by_key: dict[
        tuple[str, str, str, tuple[str, ...]],
        list[tuple[str, str, pathlib.Path]],
    ] = {}
    for csv_path in sorted(runs_dir.rglob("raw.csv")):
        if _read_breakdown_flag(csv_path.parent):
            continue
        rel = csv_path.relative_to(runs_dir)
        machine_id = rel.parts[0]
        if machine and machine_id != machine:
            continue
        if n_passages is not None and _read_n_passages(csv_path.parent) != n_passages:
            continue
        sha = rel.parts[1]
        run_id = rel.parts[2]
        header = _peek_scheme_qpn(csv_path)
        if header is None:
            continue
        scheme, qpn = header
        features = _read_target_features(csv_path.parent)
        by_key.setdefault((machine_id, scheme, qpn, features), []).append(
            (run_id, sha, csv_path)
        )

    if not by_key:
        return []

    selected: list[tuple[pathlib.Path, str]] = []
    for (machine_id, scheme, qpn, features), runs in by_key.items():
        complete = [
            (rid, sha, p)
            for rid, sha, p in runs
            if _read_status(p.parent) == "complete"
        ]
        pool = complete or runs
        if not complete:
            print(
                f"warning: no complete run for "
                f"({machine_id}, {scheme}, {qpn}, target-features={list(features)}), "
                f"using latest available",
                file=sys.stderr,
            )
        _rid, _sha, p = max(pool, key=lambda x: x[0])
        selected.append((p, _isa_tag(features)))

    return selected


def load_results(
    results_dir: pathlib.Path,
    all_runs: bool = False,
    machine: str | None = None,
    n_passages: int | None = None,
) -> pd.DataFrame:
    csv_paths = _select_runs(results_dir, all_runs, machine=machine, n_passages=n_passages)
    if not csv_paths:
        # Empty results tree (or only breakdown-mode runs). Return an
        # empty DataFrame with the expected columns so callers can
        # still iterate without `if df.empty` everywhere.
        return pd.DataFrame()
    _check_ivf_parity(csv_paths)

    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        status = _read_status(csv_path.parent)
        df["status"] = status
        # BN's verification on/off flag lives in
        # run-metadata.toml [scheme-config], not raw.csv columns.
        # Broadcast it onto every row so downstream emitters can
        # group on it. None for non-BN runs (and older runs
        # without the field).
        df["verification_enabled"] = _read_verification_enabled(csv_path.parent)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return _backfill_legacy_columns(df)


def _backfill_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names + backfill columns added across plan
    iterations. Shared by `load_results` and any emitter (today:
    `write_bntm_verification`) that does its own `pd.read_csv` for
    scheme-specific run selection — the backfill must be applied at
    every entry point so downstream functions (e.g.
    `_rep_latency_stats`) can rely on the modern column set.

    - device, effective_bytes_per_query.
    - batch_size, wallclock_us, amortised_latency_us,
      plus the derived ms columns figure 14 reads.

    Idempotent: a DataFrame already carrying the modern columns
    passes through untouched (each `fillna(...)` is a no-op when the
    source column has no nulls).
    """
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["latency_ms"] = df["latency_us"] / 1000.0
    # Backfill the device / effective-bytes columns for older CSVs.
    if "device" not in df.columns:
        df["device"] = "cpu"
    else:
        df["device"] = df["device"].fillna("cpu")
    if "effective_bytes_per_query" not in df.columns:
        df["effective_bytes_per_query"] = 0
    else:
        df["effective_bytes_per_query"] = (
            df["effective_bytes_per_query"].fillna(0).astype("int64")
        )
    # Backfill the batch-size / wallclock columns for older CSVs. Defaults
    # collapse to the B=1 single-query case the legacy harness produced.
    # For newer CSVs the columns
    # are present; B>1 rows carry an empty latency_us cell (NaN) and
    # populated wallclock_us / amortised_latency_us.
    if "batch_size" not in df.columns:
        df["batch_size"] = 1
    else:
        df["batch_size"] = df["batch_size"].fillna(1).astype("int64")
    if "wallclock_us" not in df.columns:
        df["wallclock_us"] = df["latency_us"]
    else:
        df["wallclock_us"] = df["wallclock_us"].fillna(df["latency_us"])
    if "amortised_latency_us" not in df.columns:
        df["amortised_latency_us"] = df["latency_us"]
    else:
        df["amortised_latency_us"] = df["amortised_latency_us"].fillna(df["latency_us"])
    df["wallclock_ms"] = df["wallclock_us"] / 1000.0
    df["amortised_latency_ms"] = df["amortised_latency_us"] / 1000.0
    return df


def _rep_latency_stats(sdf: pd.DataFrame) -> pd.DataFrame:
    """Per-quality_param latency stats with across-rep error bars (option A).

    Pools within-query rep variance: each query is treated as a fixed
    label, so the only variance source is run-to-run jitter on the same
    (query, quality_param) pair. The 95% CI on the grand mean is
    1.96 · s_pooled / √N, where s_pooled is the pooled within-query rep
    std (dof = N − Q) and N = R · Q. Single-rep runs (dof = 0) report
    zero std rather than NaN — the column then encodes "no across-rep
    information available" instead of breaking pgfplots.

    Filter to batch_size == 1 at the top — this aggregator
    consumes per-query latency_ms, which is NaN for batched rows by
    design. Existing callers (recall_latency, recall_throughput, the
    BN verification figure, the build-time summary) implicitly assume
    per-query data; filtering here keeps that assumption local.
    Figure 14's batched-throughput TSV does not route through this
    function — it reads amortised_latency_ms directly off all rows.
    """
    sdf = sdf[sdf["batch_size"] == 1].copy()
    sdf["_rep_residual"] = sdf["latency_ms"] - sdf.groupby(
        ["quality_param", "query_id"]
    )["latency_ms"].transform("mean")
    agg = sdf.groupby("quality_param").agg(
        n=("latency_ms", "count"),
        n_queries=("query_id", "nunique"),
        recall_mean=("recall_at_k", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        latency_ms_p50=("latency_ms", lambda x: float(np.percentile(x, 50))),
        latency_ms_p95=("latency_ms", lambda x: float(np.percentile(x, 95))),
        latency_ms_p99=("latency_ms", lambda x: float(np.percentile(x, 99))),
        _sum_sq_rep_residual=("_rep_residual", lambda x: float((x ** 2).sum())),
    ).reset_index()
    dof = agg["n"] - agg["n_queries"]
    agg["latency_ms_rep_std"] = np.where(
        dof > 0, np.sqrt(agg["_sum_sq_rep_residual"] / dof), 0.0
    )
    agg["latency_ms_rep_se"] = np.where(
        agg["n"] > 0, agg["latency_ms_rep_std"] / np.sqrt(agg["n"]), 0.0
    )
    agg["latency_ms_rep_ci95"] = 1.96 * agg["latency_ms_rep_se"]
    return agg.drop(columns=["_sum_sq_rep_residual"])


def write_recall_latency(
    df: pd.DataFrame,
    scheme: str,
    outdir: pathlib.Path,
    qp_name: str | None = None,
    suffix: str | None = None,
) -> None:
    sdf = df[df["scheme"] == scheme]
    if qp_name is not None:
        sdf = sdf[sdf["quality_param_name"] == qp_name]
    if suffix is None:
        suffix = scheme
    result = _rep_latency_stats(sdf)[
        [
            "quality_param",
            "recall_mean",
            "latency_ms_mean",
            "latency_ms_p50",
            "latency_ms_p95",
            "latency_ms_p99",
            "latency_ms_rep_std",
            "latency_ms_rep_se",
            "latency_ms_rep_ci95",
        ]
    ]
    result.to_csv(
        outdir / f"recall-latency-{suffix}.tsv", sep="\t", index=False, float_format="%.6f"
    )


def write_recall_throughput_panels(
    results_dir: pathlib.Path,
    machine: str | None,
    outdir: pathlib.Path,
) -> None:
    """Fig. 1 (paper figure 1, artefact figure 02) groupplot: three
    panels — 100k CPU (with Tiptoe + flat scorers), 8.8M CPU, 8.8M GPU.
    Each panel is loaded independently so the IVF parity guard sees one
    corpus size at a time (100k → 317 centroids; 8.8M → 2967 — not
    comparable in one set). Emits panel-prefixed TSVs alongside the
    canonical single-set ones; the figure-02 groupplot reads these via
    \\IfFileExists so absent series skip cleanly per panel.
    """
    panels = [
        ("100k-cpu", 100_000,   "cpu"),
        ("8m-cpu",   8_800_000, "cpu"),
        ("8m-gpu",   8_800_000, "gpu"),
    ]
    multi_qp_schemes = {"sap-ivf", "emvp-ivf", "bntm-ivf"}
    for panel_id, n_passages, device in panels:
        df = load_results(results_dir, machine=machine, n_passages=n_passages)
        if df.empty:
            continue
        df = df[df["device"] == device]
        if df.empty:
            continue
        for scheme in sorted(df["scheme"].unique()):
            scheme_df = df[df["scheme"] == scheme]
            qp_names = sorted(scheme_df["quality_param_name"].unique())
            if len(qp_names) > 1 or scheme in multi_qp_schemes:
                for qpn in qp_names:
                    write_recall_throughput(
                        df, scheme, outdir,
                        qp_name=qpn, suffix=f"{panel_id}-{scheme}-{qpn}",
                    )
            else:
                write_recall_throughput(
                    df, scheme, outdir, suffix=f"{panel_id}-{scheme}",
                )


def write_recall_throughput(
    df: pd.DataFrame,
    scheme: str,
    outdir: pathlib.Path,
    qp_name: str | None = None,
    suffix: str | None = None,
) -> None:
    sdf = df[df["scheme"] == scheme]
    if qp_name is not None:
        sdf = sdf[sdf["quality_param_name"] == qp_name]
    if suffix is None:
        suffix = scheme
    stats = _rep_latency_stats(sdf)
    # qps = 1 / latency_seconds. Relative SE preserved by the
    # reciprocal: SE(qps)/qps = SE(latency)/latency, so a CI in
    # ms-space converts cleanly to a CI in qps-space.
    stats["qps"] = 1000.0 / stats["latency_ms_mean"]
    rel_ci = np.where(
        stats["latency_ms_mean"] > 0,
        stats["latency_ms_rep_ci95"] / stats["latency_ms_mean"],
        0.0,
    )
    stats["qps_rep_ci95"] = stats["qps"] * rel_ci
    # Tail-latency overlay for figure 02: qps_p99 is what
    # throughput would be if every query ran at the p99 latency.
    # `qps_to_p99` is the magnitude of the negative whisker
    # (`x error minus`) for an asymmetric error bar — the
    # `x error plus` side carries the across-rep CI as before.
    # The two error sources are visually distinct (rep CI is
    # tight; tail whisker is long); plotting them together on
    # the same point answers "is the mean stable AND how far
    # does the tail reach".
    stats["qps_p99"] = np.where(
        stats["latency_ms_p99"] > 0,
        1000.0 / stats["latency_ms_p99"],
        0.0,
    )
    stats["qps_to_p99"] = np.maximum(stats["qps"] - stats["qps_p99"], 0.0)
    stats[[
        "quality_param",
        "recall_mean",
        "qps",
        "qps_rep_ci95",
        "qps_p99",
        "qps_to_p99",
    ]].to_csv(
        outdir / f"recall-throughput-{suffix}.tsv", sep="\t", index=False, float_format="%.6f"
    )


def write_recall_throughput_cpu_vs_gpu(
    df: pd.DataFrame,
    scheme: str,
    outdir: pathlib.Path,
    qp_name: str | None = None,
    suffix: str | None = None,
) -> None:
    """Figure 11: recall vs throughput, CPU and GPU per scheme.

    Splits `df` on the `device` column and emits one TSV per
    `(scheme, qp_name)` slug carrying both substrates side-by-side.
    The .tex template (`11-recall-throughput-cpu-vs-gpu.tex`) plots
    the CPU column as a solid line and the GPU column as a dashed
    line of the same colour, with `\\IfFileExists` so missing
    substrates render the half-figure cleanly. No-op when the
    scheme has zero GPU rows (saves the .tex from a broken plot).
    """
    sdf = df[df["scheme"] == scheme]
    if qp_name is not None:
        sdf = sdf[sdf["quality_param_name"] == qp_name]
    if suffix is None:
        suffix = scheme
    if sdf.empty:
        return
    # Skip when the scheme is CPU-only — figure 11's purpose is the
    # CPU/GPU comparison, so a single-substrate row would be misleading
    # (the existing recall-throughput-<scheme>.tsv already covers it).
    devices = set(sdf["device"].unique())
    if "gpu" not in devices:
        return

    cpu = _rep_latency_stats(sdf[sdf["device"] == "cpu"])
    gpu = _rep_latency_stats(sdf[sdf["device"] == "gpu"])
    for stats in (cpu, gpu):
        stats["qps"] = 1000.0 / stats["latency_ms_mean"]
        rel_ci = np.where(
            stats["latency_ms_mean"] > 0,
            stats["latency_ms_rep_ci95"] / stats["latency_ms_mean"],
            0.0,
        )
        stats["qps_rep_ci95"] = stats["qps"] * rel_ci

    cpu = cpu.rename(
        columns={
            "recall_mean": "recall_mean_cpu",
            "qps": "qps_cpu",
            "qps_rep_ci95": "qps_cpu_rep_ci95",
        }
    )[["quality_param", "recall_mean_cpu", "qps_cpu", "qps_cpu_rep_ci95"]]
    gpu = gpu.rename(
        columns={
            "recall_mean": "recall_mean_gpu",
            "qps": "qps_gpu",
            "qps_rep_ci95": "qps_gpu_rep_ci95",
        }
    )[["quality_param", "recall_mean_gpu", "qps_gpu", "qps_gpu_rep_ci95"]]

    # Outer-join so a quality_param present on only one substrate
    # (e.g., CPU sweep at finer nprobe than the GPU sweep) still
    # surfaces in the output. pgfplots tolerates NaN cells via
    # `unbounded coords=jump`.
    merged = cpu.merge(gpu, on="quality_param", how="outer").sort_values("quality_param")
    merged.to_csv(
        outdir / f"recall-throughput-cpu-vs-gpu-{suffix}.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
        na_rep="nan",
    )


def write_recall_effective_bytes(
    df: pd.DataFrame,
    scheme: str,
    outdir: pathlib.Path,
    qp_name: str | None = None,
    suffix: str | None = None,
) -> None:
    """Figure 12: recall vs effective-bytes-per-query.

    Hardware-agnostic — the analytical proxy lives in the
    `effective_bytes_per_query` column regardless of `device`. Using
    only CPU rows to stay deterministic across re-runs (GPU rows
    carry the realised proxy too but the value is identical for IVF
    schemes whose probe set is RNG-determined the same way).
    """
    sdf = df[df["scheme"] == scheme]
    if qp_name is not None:
        sdf = sdf[sdf["quality_param_name"] == qp_name]
    if suffix is None:
        suffix = scheme
    # Prefer CPU rows when both substrates are present; fall back to
    # the full set when CPU rows are missing (pure-GPU scheme tree).
    cpu_rows = sdf[sdf["device"] == "cpu"]
    src = cpu_rows if not cpu_rows.empty else sdf
    if src.empty:
        return
    result = (
        src.groupby("quality_param")
        .agg(
            recall_mean=("recall_at_k", "mean"),
            eff_bytes_mean=("effective_bytes_per_query", "mean"),
        )
        .reset_index()
    )
    # Drop rows with eff_bytes = 0 — those are pre-Plan-17 legacy
    # runs that backfilled to 0 in `load_results`. Plotting them
    # would put a phantom point at the y-axis.
    result = result[result["eff_bytes_mean"] > 0]
    if result.empty:
        return
    result.to_csv(
        outdir / f"recall-effective-bytes-{suffix}.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )


def write_throughput_vs_latency_batch(
    df: pd.DataFrame,
    outdir: pathlib.Path,
    target_recall: float = 0.9,
    qp_name_override: dict[str, str] | None = None,
) -> None:
    """Figure 14 data source.

    Per scheme, picks the quality-param value whose B=1 mean recall is
    closest to `target_recall` (a per-scheme
    operating point chosen for comparable recall, exact recall recorded
    on every row so the figure can annotate). For each batch size in
    the sweep at that quality-param, emits:

      batch_size, latency_ms_mean, qps, qps_rep_ci95, recall_mean

    `latency_ms_mean` is the mean `amortised_latency_ms` across rows
    (per-query at B=1, per-query-amortised at B>1). `qps` is aggregate
    throughput in queries/s — at B>1 it captures the batched throughput
    benefit since `1000/amortised_latency_ms = 1000·B/wallclock_ms`.

    Schemes with multiple `quality_param_name` axes (e.g. sap-ivf has
    nprobe and beta variants) pick a canonical axis via
    `qp_name_override[scheme]`; absent that, every quality-param row
    is considered candidate for qp* selection.

    No-op for a scheme lacking B=1 data — without a per-config recall
    anchor at B=1 we can't pick qp* reliably. Tiptoe is also skipped:
    its `score_batch` falls back to the sequential default (cost-floor
    exclusion), so plotting it on figure 14
    would be a baseline-shape-only line of identical-throughput dots
    — figure 02 already shows Tiptoe's per-query throughput at B=1.
    """
    qp_name_override = qp_name_override or {}
    schemes = sorted(s for s in df["scheme"].unique() if s not in {"tiptoe", "tiptoe-go"})

    for scheme in schemes:
        sdf = df[df["scheme"] == scheme]
        if scheme in qp_name_override:
            sdf = sdf[sdf["quality_param_name"] == qp_name_override[scheme]]
        if sdf.empty:
            continue

        # qp* selection: closest B=1 recall to target_recall. Recall is
        # ranking-invariant across batch sizes (batching preserves the
        # per-query ordering), so qp* picked from B=1 is the right anchor.
        b1 = sdf[sdf["batch_size"] == 1]
        if b1.empty:
            continue
        qp_recall = b1.groupby("quality_param")["recall_at_k"].mean()
        if qp_recall.empty:
            continue
        qp_star = (qp_recall - target_recall).abs().idxmin()
        ssdf = sdf[sdf["quality_param"] == qp_star]

        rows = []
        for batch_size, gdf in ssdf.groupby("batch_size"):
            n = len(gdf)
            if n == 0:
                continue
            latency_ms_mean = float(gdf["amortised_latency_ms"].mean())
            # std absorbs (rep × query) variance at B=1 and
            # (rep × chunk) variance at B>1 — both are the right
            # measurement-axis for a "where does this scheme sit on
            # figure 14" CI bar.
            latency_ms_std = (
                float(gdf["amortised_latency_ms"].std(ddof=1)) if n > 1 else 0.0
            )
            if latency_ms_mean > 0:
                qps = 1000.0 / latency_ms_mean
                # SD propagation: Var(1/x) ≈ Var(x)/x⁴ → SD(qps) =
                # 1000·SD(latency)/latency². CI95 = 1.96·SD/√n.
                qps_sd = latency_ms_std * 1000.0 / (latency_ms_mean ** 2)
                qps_rep_ci95 = 1.96 * qps_sd / (n ** 0.5) if n > 1 else 0.0
            else:
                qps = float("nan")
                qps_rep_ci95 = 0.0
            rows.append({
                "batch_size": int(batch_size),
                "latency_ms_mean": latency_ms_mean,
                "qps": qps,
                "qps_rep_ci95": qps_rep_ci95,
                "recall_mean": float(gdf["recall_at_k"].mean()),
            })
        if len(rows) < 2:
            # Figure 14 plots a line connecting per-B operating points;
            # a single point on a log-log axis is both visually
            # useless and triggers pgfplots "Dimension too large"
            # when the axis range can't be inferred. Skip until the
            # producer (step 9) runs a real batch sweep.
            continue
        out_df = pd.DataFrame(rows).sort_values("batch_size")
        out_path = outdir / f"throughput-vs-latency-batch-{scheme}.tsv"
        out_df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")


def write_recall_nprobe(
    df: pd.DataFrame,
    scheme: str,
    outdir: pathlib.Path,
) -> None:
    """Recall vs nprobe directly, with `nprobe` on the x-axis.

    Filters to rows where `quality_param_name == "nprobe"` so SAP+IVF's
    β-only sweep (and any flat scorer that has no nprobe data) is
    skipped silently. Output filename is `recall-nprobe-<scheme>.tsv`
    — `nprobe` is already in the filename so no slug suffix needed.
    """
    sdf = df[df["scheme"] == scheme]
    sdf = sdf[sdf["quality_param_name"] == "nprobe"]
    if sdf.empty:
        return
    result = (
        sdf.groupby("quality_param")
        .agg(recall_mean=("recall_at_k", "mean"))
        .reset_index()
        .rename(columns={"quality_param": "nprobe"})
    )
    result.to_csv(
        outdir / f"recall-nprobe-{scheme}.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )


def _pick_cdf_operating_point(sdf: pd.DataFrame, cdf_target: float):
    """Return the quality_param that defines the CDF operating point.

    Among qps whose mean recall ≥ target, pick the one with the lowest
    mean latency — i.e. "fastest config that hits the recall bar."
    If no qp qualifies, fall back to the highest-recall qp so the
    figure / summary row still appears (with `recall_at_op` honestly
    reporting the shortfall). The legacy `(recall - target).abs().idxmin()`
    silently picked BELOW-target configs when they were numerically
    closer (SAP β=0.5 at recall=0.857 beating β=0 at recall=1.0 for
    target=0.9), making SAP "look faster than plaintext" by virtue of
    being measured at a lower-quality operating point.
    """
    recall_by_qp = sdf.groupby("quality_param")["recall_at_k"].mean()
    if recall_by_qp.empty:
        return None
    qualifying = recall_by_qp[recall_by_qp >= cdf_target]
    if qualifying.empty:
        return recall_by_qp.idxmax()
    latency_by_qp = sdf.groupby("quality_param")["latency_ms"].mean()
    return latency_by_qp.loc[qualifying.index].idxmin()


def write_latency_cdf(
    df: pd.DataFrame,
    scheme: str,
    outdir: pathlib.Path,
    cdf_target: float,
    qp_name: str | None = None,
    suffix: str | None = None,
) -> None:
    # Drop batched rows. Batched sweeps leave `latency-us`
    # empty for B>1 rows (per-query latency inside a chunk isn't
    # separately observable); those parse to NaN in latency_ms here.
    # Without this filter, NaN rows sort to the end of latency_ms and
    # CDF cumsum reaches 1.0 at a NaN x-coordinate — pgfplots'
    # `addplot table` skips that row and the visible curve ends one
    # row earlier at cdf < 1. Same rule `_rep_latency_stats` follows.
    sdf = df[(df["scheme"] == scheme) & (df["batch_size"] == 1)]
    if qp_name is not None:
        sdf = sdf[sdf["quality_param_name"] == qp_name]
    if suffix is None:
        suffix = scheme
    best_qp = _pick_cdf_operating_point(sdf, cdf_target)
    if best_qp is None:
        return
    samples = sdf[sdf["quality_param"] == best_qp]["latency_ms"].sort_values().values
    n = len(samples)
    if n > 500:
        idx = np.linspace(0, n - 1, 500).astype(int)
        latency_pts = samples[idx]
        cdf_pts = (idx + 1) / n
    else:
        latency_pts = samples
        cdf_pts = np.arange(1, n + 1) / n
    pd.DataFrame({"latency_ms": latency_pts, "cdf": cdf_pts}).to_csv(
        outdir / f"latency-cdf-{suffix}.tsv", sep="\t", index=False, float_format="%.6f"
    )


def write_latency_cdf_summary(
    df: pd.DataFrame, outdir: pathlib.Path, cdf_target: float
) -> None:
    """Per-scheme latency summary at the CDF operating point.

    For every scheme present in `df`, picks the quality_param whose
    mean recall is closest to `cdf_target` (the same logic
    `write_latency_cdf` uses) and computes mean / median / p95 / p99
    of per-query latency at that config. Rendered as a table next to
    figure 04 so a reader can read off "what does this scheme cost
    at ~90% recall" without squinting at the CDF curves.

    For schemes that have multiple quality_param families
    (e.g. SAP+IVF sweeps both nprobe and beta) one row per family is
    emitted; the family name is glued into `scheme` as
    `<scheme>-<qp_name>`.
    """
    # Same B=1 filter as write_latency_cdf — the summary
    # consumes the same mean/median/p95/p99 from per-query latency_ms,
    # which is NaN for B>1 rows (latency-us empty in the batched schema).
    df = df[df["batch_size"] == 1]
    rows: list[dict] = []
    for scheme in sorted(df["scheme"].unique()):
        sdf_full = df[df["scheme"] == scheme]
        qp_names = sorted(sdf_full["quality_param_name"].unique())
        # If a scheme has only one quality_param family, the suffix is
        # just the scheme name; otherwise we glue the family in.
        for qp_name in qp_names:
            sdf = sdf_full[sdf_full["quality_param_name"] == qp_name]
            if sdf.empty or "recall_at_k" not in sdf.columns:
                continue
            best_qp = _pick_cdf_operating_point(sdf, cdf_target)
            if best_qp is None:
                continue
            best_recall = float(
                sdf.groupby("quality_param")["recall_at_k"].mean().loc[best_qp]
            )
            samples = sdf[sdf["quality_param"] == best_qp]["latency_ms"]
            if samples.empty:
                continue
            label = scheme if len(qp_names) == 1 else f"{scheme}-{qp_name}"
            rows.append(
                {
                    "scheme": label,
                    "config": f"{qp_name}={best_qp:g}",
                    "n": int(samples.shape[0]),
                    "recall_at_op": best_recall,
                    "mean_ms": float(samples.mean()),
                    "median_ms": float(samples.median()),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "p99_ms": float(np.percentile(samples, 99)),
                }
            )
    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        outdir / "latency-cdf-summary.tsv",
        sep="\t",
        index=False,
        float_format="%.4f",
    )


def write_scalar_vs_simd_cdf(
    results_dir: pathlib.Path,
    outdir: pathlib.Path,
    cdf_target: float,
    machine_id: str | None = None,
    n_passages: int | None = None,
) -> None:
    """Figure 15 emitter. Per scheme, write one CDF TSV per
    ISA tag (`scalar` / `simd`) at a shared operating point.

    The selector partitions on `target-features`, so both ISA
    flavours of the same sweep survive (the main `_select_runs`
    selector silently drops the older one). For each scheme that
    has at least one ISA partition, pick the shared `quality_param`
    closest to `cdf_target` across the union of rows, then emit one
    CDF per ISA. Schemes with only one ISA still emit a single TSV
    — figure 15's `\\IfFileExists{}` blocks skip the absent curve
    cleanly.

    Output paths:
        latency-cdf-scalar-vs-simd-<scheme>-<isa>.tsv
        latency-cdf-scalar-vs-simd-<scheme>-<isa>-<qpn>.tsv
            (when the scheme has multiple quality_param families
             — e.g. SAP+IVF's `nprobe` vs `beta` — to disambiguate)
    """
    selected = _select_runs_by_target_features(
        results_dir, machine=machine_id, n_passages=n_passages,
    )
    if not selected:
        return

    # Load + tag each raw.csv with its isa_tag.
    frames: list[pd.DataFrame] = []
    for csv_path, isa_tag in selected:
        df = pd.read_csv(csv_path)
        df.columns = [c.replace("-", "_") for c in df.columns]
        df = _backfill_legacy_columns(df)
        # Latency in ms for consistency with write_latency_cdf.
        df["latency_ms"] = df["latency_us"].astype(float) / 1000.0
        df["isa_tag"] = isa_tag
        frames.append(df)
    if not frames:
        return
    full = pd.concat(frames, ignore_index=True)
    # Drop batched rows (latency_ms is NaN for B>1).
    full = full[full["batch_size"] == 1]
    if full.empty:
        return

    for scheme in sorted(full["scheme"].unique()):
        sdf_full = full[full["scheme"] == scheme]
        qp_names = sorted(sdf_full["quality_param_name"].unique())
        for qp_name in qp_names:
            sdf = sdf_full[sdf_full["quality_param_name"] == qp_name]
            isas_here = sorted(sdf["isa_tag"].unique())
            # Fixed shared quality-param: pick the qp whose
            # mean recall across the union of ISA partitions is closest
            # to cdf_target. When both ISAs are present this aligns the
            # two CDFs on the same workload; when only one is present
            # it degenerates to the figure-04 logic.
            best_qp = _pick_cdf_operating_point(sdf, cdf_target)
            if best_qp is None:
                continue
            qp_suffix = "" if len(qp_names) == 1 else f"-{qp_name}"
            for isa in isas_here:
                samples = (
                    sdf[(sdf["isa_tag"] == isa) & (sdf["quality_param"] == best_qp)][
                        "latency_ms"
                    ]
                    .sort_values()
                    .values
                )
                n = len(samples)
                if n == 0:
                    continue
                if n > 500:
                    idx = np.linspace(0, n - 1, 500).astype(int)
                    latency_pts = samples[idx]
                    cdf_pts = (idx + 1) / n
                else:
                    latency_pts = samples
                    cdf_pts = np.arange(1, n + 1) / n
                pd.DataFrame({"latency_ms": latency_pts, "cdf": cdf_pts}).to_csv(
                    outdir
                    / f"latency-cdf-scalar-vs-simd-{scheme}{qp_suffix}-{isa}.tsv",
                    sep="\t",
                    index=False,
                    float_format="%.6f",
                )


def write_scalar_vs_simd_summary(
    results_dir: pathlib.Path,
    outdir: pathlib.Path,
    cdf_target: float,
    machine_id: str | None = None,
    n_passages: int | None = None,
) -> None:
    """Figure 15 summary table. Per (scheme, qpn, isa_tag)
    one row with `mean_ms`, `median_ms`, `p95_ms`, `p99_ms`,
    `recall_at_op`, and `speedup_vs_scalar` (= scalar_mean /
    isa_mean; equals 1.0 on the scalar row, the headline ratio on
    the simd row, NaN when the scalar partition is absent for that
    scheme). Mirrors `latency-cdf-summary.tsv`'s shape with an
    `isa_tag` partition added.

    Operator-decided defaults: no statistical-significance bands;
    n is large enough at the project's
    3-rep × 1000-query sweep size that the headline ratio is
    unlikely to be misleading."""
    selected = _select_runs_by_target_features(
        results_dir, machine=machine_id, n_passages=n_passages,
    )
    if not selected:
        return

    frames: list[pd.DataFrame] = []
    for csv_path, isa_tag in selected:
        df = pd.read_csv(csv_path)
        df.columns = [c.replace("-", "_") for c in df.columns]
        df = _backfill_legacy_columns(df)
        df["latency_ms"] = df["latency_us"].astype(float) / 1000.0
        df["isa_tag"] = isa_tag
        frames.append(df)
    if not frames:
        return
    full = pd.concat(frames, ignore_index=True)
    full = full[full["batch_size"] == 1]
    if full.empty:
        return

    rows: list[dict] = []
    for scheme in sorted(full["scheme"].unique()):
        sdf_full = full[full["scheme"] == scheme]
        qp_names = sorted(sdf_full["quality_param_name"].unique())
        for qp_name in qp_names:
            sdf = sdf_full[sdf_full["quality_param_name"] == qp_name]
            if sdf.empty:
                continue
            best_qp = _pick_cdf_operating_point(sdf, cdf_target)
            if best_qp is None:
                continue
            label = scheme if len(qp_names) == 1 else f"{scheme}-{qp_name}"
            isa_means: dict[str, float] = {}
            scheme_rows: list[dict] = []
            for isa in sorted(sdf["isa_tag"].unique()):
                samples = sdf[
                    (sdf["isa_tag"] == isa) & (sdf["quality_param"] == best_qp)
                ]["latency_ms"]
                if samples.empty:
                    continue
                best_recall = float(
                    sdf[sdf["isa_tag"] == isa]
                    .groupby("quality_param")["recall_at_k"]
                    .mean()
                    .loc[best_qp]
                )
                mean_ms = float(samples.mean())
                isa_means[isa] = mean_ms
                scheme_rows.append(
                    {
                        "scheme": label,
                        "isa_tag": isa,
                        "config": f"{qp_name}={best_qp:g}",
                        "n": int(samples.shape[0]),
                        "recall_at_op": best_recall,
                        "mean_ms": mean_ms,
                        "median_ms": float(samples.median()),
                        "p95_ms": float(np.percentile(samples, 95)),
                        "p99_ms": float(np.percentile(samples, 99)),
                    }
                )
            # speedup_vs_scalar: 1.0 on the scalar row by construction;
            # scalar_mean / isa_mean on simd row (>1.0 means SIMD is
            # faster). NaN when the scalar partition is absent for that
            # scheme (the comparison is undefined without a baseline).
            scalar_mean = isa_means.get("scalar")
            for r in scheme_rows:
                if scalar_mean is None:
                    r["speedup_vs_scalar"] = float("nan")
                else:
                    r["speedup_vs_scalar"] = scalar_mean / r["mean_ms"]
            rows.extend(scheme_rows)
    if not rows:
        return
    pd.DataFrame(rows).to_csv(
        outdir / "latency-cdf-scalar-vs-simd-summary.tsv",
        sep="\t",
        index=False,
        float_format="%.4f",
    )


def _select_bntm_runs(
    results_dir: pathlib.Path,
    machine_id: str | None = None,
    n_passages: int | None = None,
) -> list[pathlib.Path]:
    """Select one raw.csv per (machine, scheme, verification_enabled)
    for BN scorers. Plan-18-figure-13 needs paired on/off data, but
    `_select_runs(all_runs=False)` picks only the latest complete run
    *per machine* — losing whichever state ran first. This helper is
    BN-aware: it walks the results tree, groups BN raw.csvs by
    (machine, scheme, state), and picks the latest complete run per
    group. Also drops on-runs that pre-date the Plan-18 wiring (no
    raw.csv row has verification_overhead_us > 0): those report 0
    for the verify column and would corrupt the verify segment in
    Panel B if mixed in.

    When `machine_id` is provided, only that machine's runs are
    considered — used by the per-machine report so that figure 13
    doesn't silently pull BN data from another machine when the
    target machine has no post-Plan-18 BN runs of its own.

    When `n_passages` is provided, only runs at that corpus size are
    considered — load-bearing because verify-on data only exists at
    100k on most machines while verify-off has spread to 8.8M; without
    this filter the on/off pair gets cross-corpus on machines that
    have run both, silently confounding the verification cost story.
    """
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return []
    grouped: dict[tuple, list[tuple[str, pathlib.Path]]] = {}
    for csv_path in runs_dir.rglob("raw.csv"):
        if _read_breakdown_flag(csv_path.parent):
            continue
        scheme = _read_scheme_name(csv_path.parent)
        if scheme not in ("bntm", "bntm-ivf"):
            continue
        if _read_status(csv_path.parent) != "complete":
            continue
        verification = _read_verification_enabled(csv_path.parent)
        if verification is None:
            continue
        rel = csv_path.relative_to(runs_dir)
        run_machine = rel.parts[0]
        if machine_id is not None and run_machine != machine_id:
            continue
        if n_passages is not None and _read_n_passages(csv_path.parent) != n_passages:
            continue
        run_id = rel.parts[2]
        # Pre-Plan-18 on-runs report verification_overhead_us=0
        # everywhere even when verification=true. Drop them — Panel B
        # mixing those in would dilute the verify segment toward zero.
        if verification:
            try:
                head = pd.read_csv(csv_path, nrows=200)
                col = "verification-overhead-us"
                if col in head.columns and head[col].max() == 0:
                    continue
            except Exception:
                pass
        key = (run_machine, scheme, bool(verification))
        grouped.setdefault(key, []).append((run_id, csv_path))
    return [max(v, key=lambda x: x[0])[1] for v in grouped.values()]


def write_bntm_verification(
    df: pd.DataFrame,
    outdir: pathlib.Path,
    results_dir: pathlib.Path | None = None,
    machine_id: str | None = None,
    n_passages: int | None = None,
) -> None:
    """Figure 13 — BN with verification on vs off.

    Two output TSVs:

    - ``bntm-verification-recall-latency.tsv`` (Panel A): one row per
      ``(scheme, state, quality_param)`` with ``recall_mean``,
      ``latency_ms_mean``, ``latency_ms_rep_ci95``. State is the string
      ``"on"`` / ``"off"``. For BN flat the only quality_param is
      verification (1.0 / 0.0) and there's one row per state. For
      BN+IVF the quality_param is nprobe and there's one row per
      (state, nprobe).

    - ``bntm-verification-summary.tsv`` (Panel B): one row per
      ``(scheme, state)`` with three stack segments —
      ``compute_us``, ``verify_us``, ``side_effects_us``.
      The Plan-18 Amendment 2 reconciliation: ``compute_us`` for the
      on-bars is ``mean(latency_us) - mean(verify_us)`` — the FULL
      non-verify time, which already INCLUDES ``side_effects_us``
      (``compute_us_on == latency_us_off + side_effects_us``).
      ``side_effects_us = compute_us_on - latency_us_off`` exposes the
      gap between "on-side compute time" and "off-side total latency"; it
      captures any non-verify-attributable cost of running with
      verification on (e.g. RNG path, branch prediction).
      WARNING for figure authors: because ``compute_us`` already
      contains ``side_effects_us``, a stacked bar must plot the compute
      segment as ``compute_us - side_effects_us`` (= the off-side
      baseline), then ``verify_segment_us``, then ``side_effects_us`` —
      otherwise side_effects is double-counted and the on-bar inflates
      ~52%. ``privacy-knobs.tex`` (the paper figure) does this
      correctly; ``13b-bntm-stacked.tex`` (internal report) was fixed to
      match.
      When ``side_effects_us``
      is small relative to ``verify_us`` the segment is barely visible
      in the figure (correct visual encoding); when it's large the
      figure makes the gap explicit instead of absorbing it silently.
    """
    # When called with a results_dir, ignore `df` and load BN runs
    # directly via `_select_bntm_runs` so we get paired on/off data
    # regardless of whether the global `--all-runs` flag was set. The
    # default `_select_runs` returns only one run per machine, which
    # collapses the on/off comparison.
    if results_dir is not None:
        csv_paths = _select_bntm_runs(
            results_dir, machine_id=machine_id, n_passages=n_passages,
        )
        if not csv_paths:
            print(
                "warning: no BN runs with verification metadata found "
                "— skipping bntm-verification-*.tsv",
                file=sys.stderr,
            )
            return
        dfs = []
        for csv_path in csv_paths:
            sub = pd.read_csv(csv_path)
            sub["verification_enabled"] = _read_verification_enabled(csv_path.parent)
            dfs.append(sub)
        bn = _backfill_legacy_columns(pd.concat(dfs, ignore_index=True))
    else:
        bn = df[
            df["scheme"].isin(["bntm", "bntm-ivf"])
            & df["verification_enabled"].notna()
        ].copy()
    if bn.empty:
        print(
            "warning: no BN rows with [scheme-config].verification-enabled "
            "— skipping bntm-verification-*.tsv",
            file=sys.stderr,
        )
        return
    bn["state"] = bn["verification_enabled"].map({True: "on", False: "off"})

    # ---- Panel A: recall vs latency ---------------------------------
    # Emit one TSV per (scheme, state) so the figure can `\addplot
    # table` directly without pgfplots filtering tricks (which interact
    # badly with string-column comparisons in the standalone style.tex
    # `discard if not` definition).
    for (scheme, state), group in bn.groupby(["scheme", "state"]):
        stats = _rep_latency_stats(group)[
            [
                "quality_param",
                "recall_mean",
                "latency_ms_mean",
                "latency_ms_rep_ci95",
            ]
        ]
        # Single-point series breaks pgfplots' log-axis bounding
        # (`! Dimension too large`); emit header-only so `\addplot
        # table` finds the file but draws no points.
        if len(stats) < 2:
            stats = stats.iloc[0:0]
        stats.to_csv(
            outdir / f"bntm-verification-recall-latency-{scheme}-{state}.tsv",
            sep="\t",
            index=False,
            float_format="%.6f",
        )

    # ---- Panel B: stacked bar (compute / verify / side-effects) -----
    # Per (scheme, state) summary: mean latency, mean verify_us. For
    # bntm-ivf the verify-on sweep may cover only a subset of the
    # verify-off sweep's nprobe range (e.g. on sacs006, on=nprobe=32
    # only while off=nprobe in {1,4,16,32}). Restrict the off-side
    # aggregation to the nprobe values present on the on-side so the
    # `side_effects = compute_on - latency_off` math compares the same
    # operating points. Without this, an off-side averaged across
    # cheap-nprobe rows under-states the baseline, inflating
    # side-effects into the compute segment of the on-bar.
    on_qp_by_scheme: dict[str, set] = {
        scheme: set(group["quality_param"].unique())
        for scheme, group in bn[bn["state"] == "on"].groupby("scheme")
    }

    def _matches_on_qp_set(row: pd.Series) -> bool:
        if row["state"] == "on":
            return True
        on_qps = on_qp_by_scheme.get(row["scheme"])
        if not on_qps:
            return True
        return row["quality_param"] in on_qps

    bn_panel_b = bn[bn.apply(_matches_on_qp_set, axis=1)]
    summary = bn_panel_b.groupby(["scheme", "state"]).agg(
        n=("latency_us", "count"),
        latency_us_mean=("latency_us", "mean"),
        verify_us_mean=("verification_overhead_us", "mean"),
    ).reset_index()
    # Pivot to wide so we can compute the off-side latency per scheme
    # and use it to define side_effects for the on-side bar.
    off_latency = summary[summary["state"] == "off"].set_index("scheme")[
        "latency_us_mean"
    ]
    panel_b_rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        scheme = row["scheme"]
        state = row["state"]
        latency = float(row["latency_us_mean"])
        verify = float(row["verify_us_mean"])
        if state == "off":
            compute = latency
            verify_seg = 0.0
            side_effects = 0.0
        else:
            compute = latency - verify
            verify_seg = verify
            # Off-side baseline may be missing (paired sweep not run
            # yet); fall back to "all compute on the on side" — the
            # side-effects segment is then 0 by construction.
            off_lat = float(off_latency.get(scheme, compute))
            side_effects = compute - off_lat
            # IMPORTANT — `compute_us` for the on-bar is the FULL non-verify
            # time (latency - verify), which by construction already INCLUDES
            # `side_effects_us` (compute_us_on == latency_off + side_effects).
            # So a stacked bar must plot the compute segment as
            # `compute_us - side_effects_us` (= off-side baseline), NOT
            # `compute_us` directly — otherwise side_effects is counted twice
            # and the on-bar inflates by ~52%. See 13b-bntm-stacked.tex /
            # privacy-knobs.tex, which do the subtraction. (This data shape is
            # deliberately kept so the off/on bars share one `compute_us`
            # column semantics; the decomposition lives in the figure.)
            # Side-effects can go negative (on-compute lower than
            # off-latency) under pure noise; clamp to >= 0 so the
            # stacked bar renders. The signed gap is in the TSV for
            # the caption to quote if it matters.
        panel_b_rows.append(
            {
                "scheme": scheme,
                "state": state,
                "n": int(row["n"]),
                "latency_us_mean": latency,
                "verify_us_mean": verify,
                "compute_us": max(compute, 0.0),
                "verify_segment_us": verify_seg,
                "side_effects_us": max(side_effects, 0.0),
                "side_effects_signed_us": side_effects,
            }
        )
    panel_b_df = pd.DataFrame(panel_b_rows)
    # Sort deterministically. Order: scheme ASC (bntm < bntm-ivf), then
    # state ASC (off < on). The figure used to rely on this order with
    # hardcoded xticklabels = {bntm/off, bntm/on, bntm-ivf/off,
    # bntm-ivf/on}, but partial machines (only on-side data, only one
    # scheme, etc.) misaligned: a single (bntm-ivf, on) row landed at
    # coordindex 0 and got labeled `bntm/off`. Now we emit a composed
    # `label` column read directly by `xticklabels from table` so each
    # bar carries its own caption regardless of how many rows survive.
    panel_b_df = panel_b_df.sort_values(
        by=["scheme", "state"], ascending=[True, True]
    ).reset_index(drop=True)
    if not panel_b_df.empty:
        panel_b_df.insert(
            0, "label",
            panel_b_df["scheme"].astype(str) + "/" + panel_b_df["state"].astype(str),
        )
    panel_b_df.to_csv(
        outdir / "bntm-verification-summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    # Per-scheme slices so fig 13b's 2×1 groupplot cells can each
    # read a dedicated TSV. Avoids `discard if not` filter scoping
    # issues inside `every axis/.append style` contexts.
    for scheme, group in panel_b_df.groupby("scheme"):
        group.to_csv(
            outdir / f"bntm-verification-summary-{scheme}.tsv",
            sep="\t",
            index=False,
            float_format="%.6f",
        )


def write_communication_summary(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    # Older CSVs lack newer columns; fill missing with 0 so the summary is
    # always complete.
    #   - `cluster_response_bytes`, `setup_bytes`: added by a later
    #     CSV-schema extension.
    #   - `pre_query_offline_bytes`: in older CSVs Tiptoe rows had it
    #     bundled into `setup_bytes`.
    #   - `pre_query_offline_up_bytes` / `pre_query_offline_down_bytes`:
    #     in the oldest CSVs Tiptoe rows had the upload term bundled
    #     into `pre_query_offline_bytes` and the download term
    #     mislabelled in `response_bytes`. We preserve the legacy view
    #     by falling the old column through to `_up_bytes` so the
    #     old bundle still lands in the offline-upload bar.
    df = df.copy()
    if "pre_query_offline_up_bytes" not in df.columns:
        df["pre_query_offline_up_bytes"] = df.get("pre_query_offline_bytes", 0)
    if "pre_query_offline_down_bytes" not in df.columns:
        df["pre_query_offline_down_bytes"] = 0
    for col in ("cluster_response_bytes", "setup_bytes"):
        if col not in df.columns:
            df[col] = 0
    comm = (
        df.groupby("scheme")
        .agg(
            query_bytes=("query_bytes", "first"),
            response_bytes=("response_bytes", "first"),
            cluster_response_bytes=("cluster_response_bytes", "first"),
            setup_bytes=("setup_bytes", "first"),
            pre_query_offline_up_bytes=("pre_query_offline_up_bytes", "first"),
            pre_query_offline_down_bytes=("pre_query_offline_down_bytes", "first"),
        )
        .reset_index()
    )
    for col in (
        "cluster_response_bytes",
        "setup_bytes",
        "pre_query_offline_up_bytes",
        "pre_query_offline_down_bytes",
    ):
        comm[col] = comm[col].fillna(0).astype(int)
    comm["total_bytes"] = comm["query_bytes"] + comm["response_bytes"]
    comm.to_csv(outdir / "communication-summary.tsv", sep="\t", index=False)

    # Plan-15-figure-03 redesign: split by comm-cost class so figure 03
    # can render a 3-panel groupplot, each panel showing only the bars
    # that have non-zero data for that class. Avoids the visual noise
    # of bars at log-zero (invisible by `log origin=infty` but still
    # spaced) and lets each panel use a tighter category axis.
    #
    # Class 1: online-only — plaintext / SAP (no setup, no offline).
    # Class 2: with-setup — BN / EMVP (matrix-encryption schemes; setup
    #     is the per-cluster encrypted matrix upload).
    # Class 3: with-offline — Tiptoe (PIR-class; both offline-up and
    #     offline-down are non-zero).
    online_only = {"plaintext", "sap", "sap-ivf"}
    with_setup = {"bntm", "bntm-ivf", "emvp", "emvp-ivf"}
    with_offline = {"tiptoe", "tiptoe-go"}
    for slug, schemes in (
        ("online-only", online_only),
        ("with-setup", with_setup),
        ("with-offline", with_offline),
    ):
        # Always emit the file with at least one row so figure-03's
        # `\pgfplotstableread \commWith…` registers the macro and
        # `xticklabels from table=\macro` (evaluated at axis-option
        # time, before the in-axis `\IfFileExists` guard) doesn't
        # error on an undefined macro / empty table. When no schemes
        # match, write a placeholder row whose bars all render at 0
        # (invisible by `log origin=infty`) — the panel shows up as
        # an empty plot frame on machines that lack that scheme
        # class' data.
        sub = comm[comm["scheme"].isin(schemes)]
        if sub.empty:
            placeholder = {col: 0 for col in comm.columns}
            placeholder["scheme"] = "(none)"
            sub = pd.DataFrame([placeholder])
        sub.to_csv(
            outdir / f"communication-{slug}.tsv",
            sep="\t",
            index=False,
        )


def write_beta_recall(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    if "sap" not in df["scheme"].values:
        print("warning: no SAP rows — skipping beta-recall.tsv", file=sys.stderr)
        return
    sap = df[df["scheme"] == "sap"]
    result = sap.groupby("quality_param").agg(
        n=("recall_at_k", "count"),
        recall_mean=("recall_at_k", "mean"),
        recall_std=("recall_at_k", "std"),
    ).reset_index()
    result["recall_se"] = result["recall_std"] / np.sqrt(result["n"])
    result["recall_ci95"] = 1.96 * result["recall_se"]
    result = result.rename(columns={"quality_param": "beta"})

    if "plaintext" in df["scheme"].values:
        plaintext_ref = float(df[df["scheme"] == "plaintext"]["recall_at_k"].max())
    else:
        plaintext_ref = float("nan")
    result["plaintext_ref"] = plaintext_ref

    result[["beta", "recall_mean", "recall_se", "recall_ci95", "plaintext_ref"]].to_csv(
        outdir / "beta-recall.tsv", sep="\t", index=False, float_format="%.6f"
    )


def write_beta_recall_sap_ivf(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    sap_ivf_beta = df[
        (df["scheme"] == "sap-ivf") & (df["quality_param_name"] == "beta")
    ]
    if sap_ivf_beta.empty:
        print("warning: no SAP-IVF beta rows — skipping beta-recall-sap-ivf.tsv", file=sys.stderr)
        return
    result = sap_ivf_beta.groupby("quality_param").agg(
        n=("recall_at_k", "count"),
        recall_mean=("recall_at_k", "mean"),
        recall_std=("recall_at_k", "std"),
    ).reset_index()
    result["recall_se"] = result["recall_std"] / np.sqrt(result["n"])
    result["recall_ci95"] = 1.96 * result["recall_se"]
    result = result.rename(columns={"quality_param": "beta"})

    if "plaintext" in df["scheme"].values:
        plaintext_ref = float(df[df["scheme"] == "plaintext"]["recall_at_k"].max())
    else:
        plaintext_ref = float("nan")
    result["plaintext_ref"] = plaintext_ref

    result[["beta", "recall_mean", "recall_se", "recall_ci95", "plaintext_ref"]].to_csv(
        outdir / "beta-recall-sap-ivf.tsv", sep="\t", index=False, float_format="%.6f"
    )


def write_build_time_summary(
    results_dir: pathlib.Path,
    outdir: pathlib.Path,
    all_runs: bool = False,
    machine_id: str | None = None,
) -> None:
    """One row per scheme: cold-build duration, cluster count, m_total.
    Filtered to runs where `[index].cache-hit = false` so warm-cache
    runs don't mislead the figure (they'd report sub-millisecond
    "build" times).

    When `machine_id` is provided, only that machine's runs are
    considered — used by the per-machine report so that figure 08
    doesn't render bars for cold builds on OTHER machines (which
    would show the same scheme name twice). Otherwise every machine's
    runs are included.

    When multiple cold-build runs exist for a scheme on the same
    machine, the latest run-id wins. With `--all-runs`, every cold
    build is emitted (figure caller can aggregate).
    """
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return

    rows: list[dict] = []
    for meta_path in sorted(runs_dir.rglob("run-metadata.toml")):
        run_dir = meta_path.parent
        idx = _read_index_block(run_dir)
        if idx is None:
            continue
        if idx.get("cache-hit", True):
            continue
        scheme = _read_scheme_name(run_dir)
        if scheme is None:
            continue
        rel = run_dir.relative_to(runs_dir)
        run_machine = rel.parts[0]
        if machine_id is not None and run_machine != machine_id:
            continue
        rows.append(
            {
                "scheme": scheme,
                "build_duration_secs": float(idx.get("build-duration-secs", 0.0)),
                "cluster_count": int(idx.get("cluster-count", 0)),
                "m_total": int(idx.get("m-total", 0)),
                "machine_id": run_machine,
                "run_id": rel.parts[2],
            }
        )

    if not rows:
        # Don't emit an empty TSV; figure 08 \IfFileExists guard skips
        # cleanly.
        return

    df = pd.DataFrame(rows)
    if not all_runs:
        # Latest cold-build run per (scheme, machine_id).
        df = df.sort_values("run_id").groupby(
            ["scheme", "machine_id"], as_index=False
        ).tail(1)
    df = df.sort_values("scheme").reset_index(drop=True)
    df.to_csv(
        outdir / "build-time-summary.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )


def _read_parallel_meta(run_dir: pathlib.Path) -> tuple[int | None, str]:
    """Parse `parallel-threads` and `numactl-binding` from
    run-metadata.toml. Returns `(None, "none")` when the run predates
    those fields — the caller skips those rather than guessing."""
    meta_path = run_dir / "run-metadata.toml"
    if not (meta_path.exists() and tomllib is not None):
        return None, "none"
    with open(meta_path, "rb") as f:
        meta = tomllib.load(f)
    threads = meta.get("parallel-threads")
    binding = meta.get("numactl-binding") or "none"
    if threads is None:
        return None, binding
    return int(threads), str(binding)


def write_parallel_scaling(
    results_dir: pathlib.Path,
    outdir: pathlib.Path,
    machine_id: str | None = None,
) -> None:
    """Figure 07: latency / efficiency vs thread count.

    Walks every raw.csv under results/runs/, joins the run's
    parallel-threads + numactl-binding from the sibling
    run-metadata.toml, and emits one TSV per scheme:

        parallel-scaling-<scheme>.tsv
        threads, binding, latency_ms_mean, qps, parallel_efficiency

    Within a scheme, the pair `(parallel_threads, numactl_binding)`
    identifies a unique data point. N=16 maps to two rows when both
    pinned and unpinned anchors are present.

    Efficiency baseline `T(1)` is the single-threaded *pinned* run
    (`numactl_binding == "physcpubind=0-15,membind=0"`) for the same
    scheme. If absent, `parallel_efficiency` is left blank (NaN) for
    that scheme — the latency curve still plots; only Panel B suffers.

    IVF parity: when candidate runs on this machine span more than one
    `[ivf]` key (n-centroids, train-seed, max-iter) — e.g. a 100k
    MS MARCO sweep coexisting with an 8.8M sweep — only runs matching
    the *active* IVF key are kept. The active key is the one belonging
    to the run with the highest run-id among candidates. Without this
    gate, the per-cell `groupby.mean` silently averaged latencies
    measured at different corpus sizes, producing 64-thread cells
    that were the arithmetic mean of incompatible measurements.
    """
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return

    candidates: list[tuple[int, pathlib.Path, tuple[int, int, int], int, str]] = []
    for csv_path in sorted(runs_dir.rglob("raw.csv")):
        run_dir = csv_path.parent
        # Per-machine filter: when machine_id is set, restrict to that
        # machine's runs so a per-machine report doesn't show another
        # machine's parallel-scaling sweep.
        if machine_id is not None:
            try:
                rel = run_dir.relative_to(runs_dir)
            except ValueError:
                continue
            if not rel.parts or rel.parts[0] != machine_id:
                continue
        # Breakdown runs emit header-only raw.csv; skip them.
        if _read_breakdown_flag(run_dir):
            continue
        if csv_path.stat().st_size == 0:
            continue
        threads, binding = _read_parallel_meta(run_dir)
        if threads is None:
            continue
        ivf = _read_ivf_block(run_dir)
        if ivf is None:
            continue
        ivf_key = (
            int(ivf.get("n-centroids", -1)),
            int(ivf.get("train-seed", -1)),
            int(ivf.get("max-iter", -1)),
        )
        try:
            run_id = int(run_dir.name)
        except ValueError:
            continue
        candidates.append((run_id, csv_path, ivf_key, threads, binding))

    if not candidates:
        return

    active_key = max(candidates, key=lambda c: c[0])[2]
    other_keys = sorted({c[2] for c in candidates if c[2] != active_key})
    if other_keys:
        dropped = sum(1 for c in candidates if c[2] != active_key)
        print(
            f"write_parallel_scaling: active IVF key {active_key}; "
            f"dropping {dropped} run(s) with non-matching keys "
            f"{other_keys} to avoid mixed-corpus averaging",
            file=sys.stderr,
        )

    rows: list[dict] = []
    for _run_id, csv_path, ivf_key, threads, binding in candidates:
        if ivf_key != active_key:
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df.columns = [c.replace("-", "_") for c in df.columns]
        scheme = str(df["scheme"].iloc[0])
        rows.append(
            {
                "scheme": scheme,
                "threads": threads,
                "binding": binding,
                "latency_ms_mean": float(df["latency_us"].mean()) / 1000.0,
            }
        )

    if not rows:
        return

    df = pd.DataFrame(rows)
    # If the same (scheme, threads, binding) appears in multiple runs
    # (e.g. re-running the sweep), keep the latest by averaging — this
    # is rare in practice; the Makefile loop produces one run per cell.
    df = (
        df.groupby(["scheme", "threads", "binding"], as_index=False)
        .agg(latency_ms_mean=("latency_ms_mean", "mean"))
    )
    df["qps"] = 1000.0 / df["latency_ms_mean"]

    # Per-scheme T(1) baseline = single-threaded pinned latency.
    pinned_label = "physcpubind=0-15,membind=0"
    baseline = (
        df[(df["threads"] == 1) & (df["binding"] == pinned_label)]
        .set_index("scheme")["latency_ms_mean"]
    )

    def efficiency(row: pd.Series) -> float:
        t1 = baseline.get(row["scheme"])
        if t1 is None:
            return float("nan")
        return float(t1 / (row["threads"] * row["latency_ms_mean"]))

    df["parallel_efficiency"] = df.apply(efficiency, axis=1)

    # Sort: pinned anchor first at each thread count so the figure-07
    # solid line precedes the dashed one in the TSV order.
    df["_binding_rank"] = (df["binding"] != pinned_label).astype(int)
    df = df.sort_values(
        ["scheme", "threads", "_binding_rank"], kind="mergesort"
    ).drop(columns=["_binding_rank"])

    for scheme, sdf in df.groupby("scheme"):
        # Skip schemes with fewer than two distinct thread points —
        # parallel-scaling lines need a sweep to be meaningful, and
        # single-point data triggers fig 07 to crash on the empty
        # `parallel_efficiency` column when no T(1) baseline exists.
        if sdf["threads"].nunique() < 2:
            continue
        sdf[
            ["threads", "binding", "latency_ms_mean", "qps", "parallel_efficiency"]
        ].to_csv(
            outdir / f"parallel-scaling-{scheme}.tsv",
            sep="\t",
            index=False,
            float_format="%.6f",
        )


def _select_breakdown_runs(
    results_dir: pathlib.Path, all_runs: bool,
    machine_id: str | None = None,
) -> list[pathlib.Path]:
    """Return substep-breakdown.csv paths to load. Latest non-empty
    breakdown run per (machine, scheme), or all when --all-runs.

    Earlier this picked one run per machine, which silently dropped
    seven of every eight scheme breakdowns on a machine that ran the
    full sweep — figures 09a/09b ended up showing only the latest
    scheme. Per-(machine, scheme) keeps each scheme's bar visible.

    When ``machine_id`` is set, only that machine's runs are
    considered — used by the per-machine report so that figures
    09a/09b don't render bars from other machines' breakdown sweeps.
    """
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return []
    grouped: dict[tuple[str, str], list[tuple[str, pathlib.Path]]] = {}
    for csv_path in sorted(runs_dir.rglob("substep-breakdown.csv")):
        if csv_path.stat().st_size == 0:
            continue
        scheme = _read_scheme_name(csv_path.parent) or "unknown"
        rel = csv_path.relative_to(runs_dir)
        run_machine = rel.parts[0]
        if machine_id is not None and run_machine != machine_id:
            continue
        run_id = rel.parts[2]
        grouped.setdefault((run_machine, scheme), []).append((run_id, csv_path))
    if not grouped:
        return []
    if all_runs:
        return [csv for runs in grouped.values() for _, csv in runs]
    return [max(runs, key=lambda x: x[0])[1] for runs in grouped.values()]


def load_breakdown(
    results_dir: pathlib.Path, all_runs: bool = False,
    machine_id: str | None = None,
) -> pd.DataFrame:
    """Load substep-breakdown.csv from selected runs into a single
    DataFrame. Empty if no breakdown runs are present.

    When ``machine_id`` is set, only that machine's runs are loaded —
    used by the per-machine report so figures 09a/09b stay
    machine-scoped."""
    paths = _select_breakdown_runs(results_dir, all_runs, machine_id=machine_id)
    if not paths:
        return pd.DataFrame()
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [c.replace("-", "_") for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Canonical substep names in figure-row order, matching
# `eval_harness::CANONICAL_SUBSTEPS`. Underscores replace dashes in
# `server-compute` so pgfplots column references stay parseable
# (`y=server_compute`, not `y=server-compute` which parses as
# subtraction).
_CANONICAL_SUBSTEPS = (
    "route",
    "encode",
    "server_compute",
    "verify",
    "decompress",
    "decode",
    "merge",
)


def write_substep_breakdown(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    """Figures 09a / 09b. One row per scheme; columns are the
    seven canonical substep names. Absolute TSV in microseconds;
    normalised TSV with each substep as a fraction of the row's sum.

    Aggregation: mean `us` across all `(scheme, substep)` rows of every
    breakdown run loaded — i.e. averaged over all queries and all
    configs in the run. If you ran multiple configs in `--breakdown`
    mode and want a single representative config, filter the
    underlying runs first or restrict the sweep.
    """
    if df.empty:
        # Emit header-only TSVs so the figure files (which reference
        # the loaded tables without checking `\subsIvf` etc. are
        # defined) find a parseable empty table and render an empty
        # chart instead of failing with `column_not_found` /
        # `Undefined control sequence`. Per-machine reports for boxes
        # that never ran `--breakdown` hit this path.
        cols = ["scheme", *_CANONICAL_SUBSTEPS]
        empty = pd.DataFrame(columns=cols)
        for name in (
            "substep-breakdown-absolute.tsv",
            "substep-breakdown-absolute-ivf.tsv",
            "substep-breakdown-absolute-ivf-no-bntm.tsv",
            "substep-breakdown-normalised.tsv",
        ):
            empty.to_csv(outdir / name, sep="\t", index=False)
        return

    df = df.copy()
    # Map kebab-cased substep `server-compute` to `server_compute` so
    # the pivoted column name is pgfplots-safe (no dash → no
    # subtraction parse).
    df["substep"] = df["substep"].replace("server-compute", "server_compute")

    grouped = df.groupby(["scheme", "substep"], as_index=False)["us"].mean()
    wide = grouped.pivot(index="scheme", columns="substep", values="us").reset_index()
    wide.columns.name = None

    # Ensure every canonical substep is a column even if the loaded
    # data has none of that substep recorded (zero-fill missing).
    for col in _CANONICAL_SUBSTEPS:
        if col not in wide.columns:
            wide[col] = 0.0

    cols = ["scheme", *_CANONICAL_SUBSTEPS]
    wide = wide[cols].copy()
    # Sort descending by total per-query time so figure 09a's xbar
    # chart reads smallest-at-top, biggest-at-bottom (top-to-bottom
    # = ascending). xbar puts y=0 at the axis bottom; with the
    # biggest at row 0 the visual order top->bottom matches "sorted
    # smallest to biggest".
    wide["_total"] = wide[list(_CANONICAL_SUBSTEPS)].sum(axis=1)
    wide = (
        wide.sort_values("_total", ascending=False)
        .drop(columns=["_total"])
        .reset_index(drop=True)
    )
    wide.to_csv(
        outdir / "substep-breakdown-absolute.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    # IVF-only subset (plaintext + *-ivf schemes) for figure 09a's
    # second panel: tiptoe's encode segment dwarfs the others on the
    # main chart, so a panel scoped to the IVF schemes lets the
    # reader read sub-ms differences cleanly.
    ivf_subset = wide[
        wide["scheme"].isin(
            ["plaintext", "sap-ivf", "emvp-ivf", "bntm-ivf"]
        )
    ].reset_index(drop=True)
    ivf_subset.to_csv(
        outdir / "substep-breakdown-absolute-ivf.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )
    # Third panel of figure 09a: same as the IVF subset minus
    # bntm-ivf, which dwarfs the rest in this slice (verify segment
    # at ~300 ms on Sec128). With it gone the remaining schemes
    # (plaintext / sap-ivf / emvp-ivf) all sit under ~15 ms.
    ivf_no_bntm = wide[
        wide["scheme"].isin(["plaintext", "sap-ivf", "emvp-ivf"])
    ].reset_index(drop=True)
    ivf_no_bntm.to_csv(
        outdir / "substep-breakdown-absolute-ivf-no-bntm.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    norm = wide.copy()
    totals = norm[list(_CANONICAL_SUBSTEPS)].sum(axis=1).replace(0, pd.NA)
    norm[list(_CANONICAL_SUBSTEPS)] = (
        norm[list(_CANONICAL_SUBSTEPS)].div(totals, axis=0).fillna(0.0)
    )
    norm.to_csv(
        outdir / "substep-breakdown-normalised.tsv",
        sep="\t",
        index=False,
        float_format="%.6f",
    )

    # Table-friendly variants for the figure-11 rationale: one row per
    # scheme, all values pre-converted to the units the table shows so
    # pgfplotstable doesn't have to do arithmetic at typeset-time. The
    # `total` column makes the sort order explicit in the printed
    # table. Row order is REVERSED relative to the chart TSVs (ascending
    # by total) so the printed table reads top-to-bottom in the same
    # order as the chart's visual y-axis (smallest at top, biggest at
    # bottom).
    abs_ms = wide.iloc[::-1].copy().reset_index(drop=True)
    for col in _CANONICAL_SUBSTEPS:
        abs_ms[col] = abs_ms[col] / 1000.0
    abs_ms["total"] = abs_ms[list(_CANONICAL_SUBSTEPS)].sum(axis=1)
    abs_ms = abs_ms[["scheme", "total", *_CANONICAL_SUBSTEPS]]
    abs_ms.to_csv(
        outdir / "substep-breakdown-absolute-ms.tsv",
        sep="\t",
        index=False,
        float_format="%.3f",
    )

    norm_pct = norm.iloc[::-1].copy().reset_index(drop=True)
    for col in _CANONICAL_SUBSTEPS:
        norm_pct[col] = norm_pct[col] * 100.0
    norm_pct.to_csv(
        outdir / "substep-breakdown-normalised-pct.tsv",
        sep="\t",
        index=False,
        float_format="%.2f",
    )


_HYDRATE_BULK_SCRIPT = (
    pathlib.Path(__file__).parent.parent / "scripts" / "hydrate_bulk.py"
)


def _auto_hydrate_from_bulk(
    results_root: pathlib.Path, machine: str | None
) -> None:
    """Idempotently fetch raw.csv / top_k.csv / substep-breakdown.csv
    from the bulk store for any run-dir under `results/runs/<machine>/`
    (or all machines, if ``machine`` is None) whose ``[bulk]`` block
    references a file not present locally.

    Runs without ``[bulk]`` blocks are silently skipped by
    ``scripts/hydrate_bulk.py``; runs whose files are already on
    disk and match the recorded sha256 are also skipped (per-file).
    Failure is non-fatal: a warning is printed and preprocess
    continues with whatever data is locally present.
    """
    if not _HYDRATE_BULK_SCRIPT.is_file():
        return
    cmd = [
        sys.executable,
        str(_HYDRATE_BULK_SCRIPT),
        "--results-root",
        str(results_root),
    ]
    if machine:
        cmd += ["--machine", machine]
    else:
        cmd += ["--all"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"warning: auto-hydrate from bulk store failed "
            f"(exit {result.returncode}); continuing with locally-present data.\n"
            f"  stderr: {result.stderr.strip()}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess eval CSVs into pgfplots TSVs.")
    parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("../results"))
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument(
        "--cdf-target",
        type=float,
        default=0.9,
        metavar="RECALL",
        help="target recall for CDF operating point (default: 0.9)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="aggregate all runs instead of latest complete run per machine",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help=(
            "Restrict input to runs from this machine-id. "
            "When set, the per-machine canonical emission path "
            "`results/aggregated/<machine-id>/` should be passed as --outdir."
        ),
    )
    parser.add_argument(
        "--n-passages",
        type=int,
        default=None,
        metavar="N",
        help=(
            "filter input to runs whose [dataset].n-passages = N. "
            "Use to disambiguate when a machine's results tree mixes "
            "corpora of different sizes (e.g. 100k flat-scheme runs "
            "alongside an 8.8M IVF sweep) — without this the cross-IVF "
            "parity guard fails the run."
        ),
    )
    args = parser.parse_args()

    if tomllib is None:
        print(
            "warning: tomllib/tomli not available — run status will be 'unknown'.\n"
            "         Install tomli (`pip install tomli`) on Python < 3.11.",
            file=sys.stderr,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    # Clear stale TSVs from a prior preprocess run. Without this, a
    # filename emitted by an earlier run set (e.g. parallel-scaling-*
    # before the --machine filter was wired through, or a scheme that
    # has since dropped out of the sweep) survives in the per-machine
    # aggregated dir and gets picked up by figure `\IfFileExists`
    # blocks, mixing stale data into the current report.
    for stale in args.outdir.glob("*.tsv"):
        stale.unlink()
    _auto_hydrate_from_bulk(args.results, args.machine)
    df = load_results(
        args.results,
        all_runs=args.all_runs,
        machine=args.machine,
        n_passages=args.n_passages,
    )
    if df.empty:
        print("no non-breakdown raw.csv runs found; skipping figures 01–07")
        schemes = []
        cpu_df = df
    else:
        # Single-device figures (01, 02, 04, 05, 06, 07, 10, 13, 14)
        # default to cpu so a scheme's gpu sweep doesn't cross-contaminate
        # an apples-to-apples cpu comparison. The cpu-vs-gpu emitter
        # (figure 11) and effective-bytes emitter (figure 12) still see
        # the full df. `_select_runs` partitions on device, so cpu and
        # gpu sweeps for the same (scheme, qpn) both survive selection
        # and surface here under their own device tag.
        cpu_df = df[df["device"] == "cpu"]
        schemes = sorted(cpu_df["scheme"].unique())
        gpu_only = sorted(set(df["scheme"].unique()) - set(schemes))
        msg = f"Loaded {len(df):,} rows (cpu {len(cpu_df):,}) across schemes: {', '.join(schemes)}"
        if gpu_only:
            msg += f"  [gpu-only, excluded from single-device figures: {', '.join(gpu_only)}]"
        print(msg)

    # Schemes that always emit per-qp-name TSVs so filenames match
    # the figure templates (which reference, e.g., emvp-ivf-nprobe).
    _always_suffix = {"sap-ivf", "emvp-ivf", "bntm-ivf"}

    for scheme in schemes:
        scheme_df = cpu_df[cpu_df["scheme"] == scheme]
        qp_names = sorted(scheme_df["quality_param_name"].unique())
        if len(qp_names) > 1 or scheme in _always_suffix:
            # Scheme has multiple sweep axes (e.g. sap-ivf has nprobe and beta sweeps).
            # Generate separate TSVs per quality_param_name to avoid mixing axes.
            for qpn in qp_names:
                slug = f"{scheme}-{qpn}"
                print(f"  [{slug}] recall-latency, recall-throughput, latency-cdf")
                write_recall_latency(cpu_df, scheme, args.outdir, qp_name=qpn, suffix=slug)
                write_recall_throughput(cpu_df, scheme, args.outdir, qp_name=qpn, suffix=slug)
                write_recall_throughput_cpu_vs_gpu(
                    df, scheme, args.outdir, qp_name=qpn, suffix=slug
                )
                write_recall_effective_bytes(
                    df, scheme, args.outdir, qp_name=qpn, suffix=slug
                )
                write_latency_cdf(cpu_df, scheme, args.outdir, args.cdf_target, qp_name=qpn, suffix=slug)
        else:
            print(f"  [{scheme}] recall-latency, recall-throughput, latency-cdf")
            write_recall_latency(cpu_df, scheme, args.outdir)
            write_recall_throughput(cpu_df, scheme, args.outdir)
            write_recall_throughput_cpu_vs_gpu(df, scheme, args.outdir)
            write_recall_effective_bytes(df, scheme, args.outdir)
            write_latency_cdf(cpu_df, scheme, args.outdir, args.cdf_target)
        # Recall-vs-nprobe applies whenever the scheme has any nprobe
        # rows (plaintext / sap-ivf / emvp-ivf / bntm-ivf). Writer
        # filters internally and skips schemes without nprobe data.
        write_recall_nprobe(cpu_df, scheme, args.outdir)

    if not df.empty:
        print(
            "  [all] communication-summary, beta-recall, beta-recall-sap-ivf, "
            "recall-nprobe, bntm-verification, throughput-vs-latency-batch"
        )
        write_communication_summary(cpu_df, args.outdir)
        write_beta_recall(cpu_df, args.outdir)
        write_beta_recall_sap_ivf(cpu_df, args.outdir)
        write_bntm_verification(
            cpu_df, args.outdir, results_dir=args.results,
            machine_id=args.machine, n_passages=args.n_passages,
        )
        write_latency_cdf_summary(cpu_df, args.outdir, args.cdf_target)
        # Figure 15 (scalar-vs-SIMD CDF). Both emitters use
        # the new `_select_runs_by_target_features` selector (sibling
        # of `_select_runs`, target-features included in the partition
        # key) so scalar + SIMD runs of the same sweep both feed the
        # figure. No-op when no runs carry the field or only one ISA
        # tag is present per (scheme, qpn) — the per-curve
        # `\IfFileExists{}` in 15-scalar-vs-simd-cdf.tex skips absent
        # series cleanly.
        write_scalar_vs_simd_cdf(
            args.results, args.outdir, args.cdf_target,
            machine_id=args.machine, n_passages=args.n_passages,
        )
        write_scalar_vs_simd_summary(
            args.results, args.outdir, args.cdf_target,
            machine_id=args.machine, n_passages=args.n_passages,
        )
        # Figure 14. No-op when no batch_size > 1 rows are
        # present (every existing run is B=1 only); safe to call
        # unconditionally. SAP-IVF: pick the nprobe sweep variant for
        # qp* selection (beta sweep would mix in a second axis the
        # figure doesn't separate).
        write_throughput_vs_latency_batch(
            cpu_df,
            args.outdir,
            qp_name_override={"sap-ivf": "nprobe"},
        )

    # Figures 08 / 09a / 09b. Both emitters are no-ops when the
    # corresponding source data isn't present (no cold-build runs in
    # the results tree, or no `--breakdown` runs respectively), so
    # they're safe to call unconditionally. `--machine` is threaded
    # through so the per-machine `results/aggregated/<id>/` only
    # contains that machine's data — otherwise a `--machine A` run
    # leaks B's parallel-scaling / build-time / breakdown into A's
    # aggregated dir and the figure templates render the wrong points.
    print("  [all] build-time-summary, substep-breakdown (absolute + normalised)")
    write_build_time_summary(
        args.results, args.outdir, all_runs=args.all_runs, machine_id=args.machine,
    )
    breakdown_df = load_breakdown(
        args.results, all_runs=args.all_runs, machine_id=args.machine,
    )
    write_substep_breakdown(breakdown_df, args.outdir)

    # Figure 07. No-op when no run carries `parallel-threads`
    # in its run-metadata.toml (older runs without the field); safe to call
    # unconditionally.
    print("  [all] parallel-scaling-<scheme>")
    write_parallel_scaling(args.results, args.outdir, machine_id=args.machine)

    # Fig. 1 (paper) groupplot panels — independent of args.n_passages
    # (each panel loads its own slice internally so the IVF parity guard
    # sees a single corpus size at a time).
    print("  [all] recall-throughput-{100k-cpu,8m-cpu,8m-gpu}-<scheme>")
    write_recall_throughput_panels(args.results, args.machine, args.outdir)

    print(f"TSVs written to {args.outdir}/")


if __name__ == "__main__":
    main()
