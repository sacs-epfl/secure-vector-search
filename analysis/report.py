#!/usr/bin/env python3
"""Generate a per-machine PDF report aggregating all schemes from the latest runs."""
import argparse
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli for Python < 3.11
    except ImportError:
        tomllib = None  # type: ignore[assignment]

import csv

FIGURES_DIR = pathlib.Path(__file__).parent / "figures"
STYLE_TEX   = pathlib.Path(__file__).parent / "style.tex"

QUALITY_FIGURES = [
    "05-beta-recall.tex",
    "06-beta-recall-sap-ivf.tex",
    "10-recall-nprobe.tex",
    "03-communication.tex",
]
PERFORMANCE_FIGURES = [
    "01-recall-latency.tex",
    "02-recall-throughput.tex",
    "04-latency-cdf.tex",
    "07-parallel-scaling.tex",
    "08-build-time.tex",
    "09a-substeps-absolute.tex",
    "09b-substeps-normalised.tex",
    "13a-bntm-recall-latency.tex",
    "13b-bntm-stacked.tex",
    "14-throughput-vs-latency-batch.tex",
    "15-scalar-vs-simd-cdf.tex",
]
ALL_FIGURES = QUALITY_FIGURES + PERFORMANCE_FIGURES

CAPTIONS: dict[str, str] = {
    "05-beta-recall.tex":         r"SAP recall@$k$ vs.\ perturbation level $\beta$.",
    "06-beta-recall-sap-ivf.tex": r"SAP+IVF recall@$k$ vs.\ perturbation level $\beta$.",
    "10-recall-nprobe.tex":       r"Recall@$k$ vs.\ \texttt{nprobe} for IVF schemes (log $x$). Reads off ``how aggressive does the probe need to be to reach $X\%$ recall'' directly, instead of inferring it from figures 01--02.",
    "03-communication.tex":       r"Communication volume: query bytes, per-cluster response, and one-time setup cost (log scale).",
    "01-recall-latency.tex":      r"Recall@$k$ vs.\ mean latency (log scale).",
    "02-recall-throughput.tex":   r"Recall@$k$ vs.\ throughput (log scale).",
    "04-latency-cdf.tex":         r"Latency CDF at the $\approx$90\% recall operating point.",
    "07-parallel-scaling.tex":    r"Parallel scaling on \texttt{sacs006}. Top: latency vs thread count (log--log). Bottom: parallel efficiency $T(1)/(N\,T(N))$. Vertical guides at $N{=}16$ (socket boundary; NUMA crossing on this dual-socket Xeon Gold 6426Y) and $N{=}32$ (SMT boundary; logical cores share execution units past this). Within-socket portion solid (1--16 pinned to \texttt{physcpubind=0-15,membind=0}); cross-socket portion dashed (16-unpinned, 32, 64).",
    "08-build-time.tex":          r"Cold-build time per scheme (log $y$). Filtered to runs with \texttt{[index].cache-hit~=~false}; warm-cache reloads are sub-millisecond and are excluded.",
    "09a-substeps-absolute.tex":  r"Per-query substep breakdown, absolute units. One stacked bar per scheme; segments share a colour palette across schemes.",
    "09b-substeps-normalised.tex": r"Per-query substep breakdown, normalised to 100\% per scheme. Reveals composition independent of absolute scale; cross-reference figure 09a for magnitude.",
    "13a-bntm-recall-latency.tex": r"BN with verification on vs.\ off — recall@$k$ vs.\ mean latency. Round markers are BN flat, square markers are BN+IVF; solid points are verify-off, dashed are verify-on. The gap between an off point and its on counterpart is the end-to-end verification overhead at wall-clock scope.",
    "13b-bntm-stacked.tex":       r"Stacked per-query breakdown of compute / verify / side-effects time, split BN flat (left) vs.\ BN+IVF (right) because the two schemes' y-scales differ by $\sim$10$\times$. The verify segment is Protocol~2 (Freivalds) at $\lambda'$~=~3 trials.",
    "14-throughput-vs-latency-batch.tex": r"Throughput vs.\ amortised per-query latency with batch size $B \in \{1, 8, 64, 256\}$ as the operating-point parameter (log--log). Lines that retain a positive throughput slope across $B$ are amortising fixed overhead; lines that flatten are compute-bound at the chosen $m$. Per-scheme operating point chosen for comparable recall $\approx 0.9$ on the $B{=}1$ baseline (Plan~23 \S~Decision~3); the recall achieved per scheme is in each TSV's \texttt{recall\_mean} column. Tiptoe omitted (ADR~010 Decision~3 cost-floor exclusion).",
    "15-scalar-vs-simd-cdf.tex": r"""Scalar-baseline vs SIMD-enabled latency CDFs per scheme at the $\approx$90\% recall operating point (Plan~26). Solid = \texttt{make eval} (baseline x86-64); dashed = \texttt{make eval-native} (\texttt{cfg(target\_feature = "avx512f")} active, Plan~25's BN matvec + L2 dispatchers compiled in). The horizontal gap between a paired solid/dashed curve is the per-query speedup attributable to compile-time SIMD gating. Schemes with only one ISA recorded on this machine render as a single line.""",
}

# Per-figure TSV dependencies, used by `_figure_has_data()` to skip
# figures with no source data on this machine. Each entry is a tuple
# of glob patterns relative to the report's data/ directory; a figure
# renders iff at least one matching TSV exists with content beyond
# the header row. Diagnostic figures (07 parallel-scaling, 08 cold
# build-time, 09a/b substeps, 13 BN verification) drop out cleanly
# when the run mode that produces their data wasn't part of the eval
# campaign on this machine.
FIGURE_DATA_FILES: dict[str, tuple[str, ...]] = {
    "01-recall-latency.tex": ("recall-latency-*.tsv",),
    "02-recall-throughput.tex": ("recall-throughput-*.tsv",),
    "03-communication.tex": (
        "communication-online-only.tsv",
        "communication-with-setup.tsv",
        "communication-with-offline.tsv",
    ),
    "04-latency-cdf.tex": ("latency-cdf-summary.tsv",),
    "05-beta-recall.tex": ("beta-recall.tsv",),
    "06-beta-recall-sap-ivf.tex": ("beta-recall-sap-ivf.tsv",),
    "07-parallel-scaling.tex": ("parallel-scaling-*.tsv",),
    "08-build-time.tex": ("build-time-summary.tsv",),
    "09a-substeps-absolute.tex": ("substep-breakdown-absolute.tsv",),
    "09b-substeps-normalised.tex": ("substep-breakdown-normalised.tsv",),
    "10-recall-nprobe.tex": ("recall-nprobe-*.tsv",),
    "13a-bntm-recall-latency.tex": ("bntm-verification-recall-latency-*.tsv",),
    "13b-bntm-stacked.tex":        ("bntm-verification-summary.tsv",),
    "14-throughput-vs-latency-batch.tex": ("throughput-vs-latency-batch-*.tsv",),
    "15-scalar-vs-simd-cdf.tex": ("latency-cdf-scalar-vs-simd-*.tsv",),
}


def _comm_class_has_data(data_dir: pathlib.Path, slug: str) -> bool:
    """Return True iff `communication-<slug>.tsv` carries at least one
    real (non-`(none)`-placeholder) data row. preprocess.py emits a
    `(none)` placeholder when no scheme on this machine matches the
    class so the figure's `\\pgfplotstableread` always succeeds; this
    helper filters that case out."""
    tsv = data_dir / f"communication-{slug}.tsv"
    if not tsv.is_file():
        return False
    with tsv.open() as f:
        next(f, None)  # header
        for line in f:
            scheme = line.split("\t", 1)[0].strip()
            if scheme and scheme != "(none)":
                return True
    return False


def _figure_has_data(name: str, data_dir: pathlib.Path) -> bool:
    """Return True iff the figure has a source TSV with rows beyond
    the header. Used to skip figure sections cleanly when the
    machine's run set doesn't carry the relevant diagnostic data."""
    patterns = FIGURE_DATA_FILES.get(name)
    if patterns is None:
        return True  # unknown figure → render by default
    for pat in patterns:
        for tsv in data_dir.glob(pat):
            if not tsv.is_file():
                continue
            with tsv.open() as f:
                # Cheap check: does any line beyond the first exist?
                next(f, None)  # header
                if next(f, None) is not None:
                    return True
    return False


# Schemes that "could appear" in each figure. `None` means "every
# scheme that has data". Filtered against the actually-loaded
# scheme_metas at sidebar-build time.
SCHEMES_BY_FIGURE: dict[str, set[str] | None] = {
    "01-recall-latency.tex": None,
    "02-recall-throughput.tex": None,
    "03-communication.tex": None,
    "04-latency-cdf.tex": None,
    "05-beta-recall.tex": {"sap"},
    "06-beta-recall-sap-ivf.tex": {"sap-ivf"},
    "10-recall-nprobe.tex": {"plaintext", "sap-ivf", "emvp-ivf", "bntm-ivf"},
    "07-parallel-scaling.tex": None,
    "08-build-time.tex": None,
    "09a-substeps-absolute.tex": None,
    "09b-substeps-normalised.tex": None,
    "13a-bntm-recall-latency.tex": {"bntm", "bntm-ivf"},
    "13b-bntm-stacked.tex":        {"bntm", "bntm-ivf"},
    "14-throughput-vs-latency-batch.tex": {
        "plaintext",
        "sap",
        "sap-ivf",
        "emvp",
        "emvp-ivf",
        "bntm",
        "bntm-ivf",
    },
    "15-scalar-vs-simd-cdf.tex": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tex_escape(s: str) -> str:
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def _read_meta(run_dir: pathlib.Path) -> dict:
    meta_path = run_dir / "run-metadata.toml"
    if meta_path.exists() and tomllib is not None:
        with open(meta_path, "rb") as f:
            return tomllib.load(f)
    return {}


def _read_status(run_dir: pathlib.Path) -> str:
    return _read_meta(run_dir).get("status", "unknown")


def _machine_info(results_dir: pathlib.Path, machine_id: str) -> dict:
    machines_csv = results_dir / "machines.csv"
    if not machines_csv.exists():
        return {}
    with open(machines_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("machine-id") == machine_id:
                return row
    return {}


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

class SkipMachine(Exception):
    """Raised when a machine has no per-scheme data to plot (e.g. a
    breakdown-only machine, or a machine whose runs directory is
    missing). Caught at the report-loop level so the machine is
    skipped without being counted as a build failure."""


def find_runs_for_machine(
    results_dir: pathlib.Path,
    machine_id: str,
) -> dict[str, pathlib.Path]:
    """Return {scheme: latest_complete_run_dir} for the given machine.

    Walks ``run-metadata.toml`` only — figure data comes from
    ``results/aggregated/<machine-id>/*.tsv`` (emitted by preprocess.py),
    so this discovery only needs the meta for scheme/status/sidebars.

    Returns an empty dict (with a warning) when the machine has no
    non-breakdown runs — caller is expected to treat this as a skip,
    not a failure: breakdown-only machines still legitimately appear
    in the results tree (their figure-09 data ships via aggregated/),
    and the per-scheme sidebar / runs-table simply can't be populated.
    """
    runs_dir = results_dir / "runs" / machine_id
    if not runs_dir.exists():
        print(
            f"warning: no runs directory for machine {machine_id} under {results_dir}",
            file=sys.stderr,
        )
        return {}

    # Layout: runs/<machine-id>/<git-sha>/<run-id>/run-metadata.toml.
    # Breakdown runs feed figures 09a/09b via
    # `substep-breakdown.csv` → preprocess.py → aggregated/; the
    # sidebar / figure 01-07 picker filters them out here.
    candidates: list[tuple[str, str, pathlib.Path, str]] = []
    for sha_dir in sorted(runs_dir.iterdir()):
        if not sha_dir.is_dir():
            continue
        for run_dir in sorted(sha_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "run-metadata.toml").exists():
                continue
            meta = _read_meta(run_dir)
            if meta.get("breakdown"):
                continue
            scheme = meta.get("scheme-config", {}).get("scheme", "unknown")
            run_id = run_dir.name
            status = meta.get("status", "unknown")
            candidates.append((scheme, run_id, run_dir, status))

    if not candidates:
        print(
            f"warning: no non-breakdown run-metadata.toml found under {runs_dir}",
            file=sys.stderr,
        )
        return {}

    # For each scheme, pick latest complete run (fall back to latest overall).
    by_scheme: dict[str, list[tuple[str, pathlib.Path, str]]] = {}
    for scheme, run_id, run_dir, status in candidates:
        by_scheme.setdefault(scheme, []).append((run_id, run_dir, status))

    result: dict[str, pathlib.Path] = {}
    for scheme, runs in by_scheme.items():
        complete = [(rid, p) for rid, p, st in runs if st == "complete"]
        if not complete:
            print(
                f"warning: no complete run for scheme {scheme} on machine {machine_id}, "
                "using latest available",
                file=sys.stderr,
            )
            pool = [(rid, p) for rid, p, _ in runs]
        else:
            pool = complete
        result[scheme] = max(pool, key=lambda x: x[0])[1]

    return result


def list_machines(results_dir: pathlib.Path) -> list[str]:
    """Return every machine-id under ``results/runs/`` that has at least
    one complete run on disk. Used by ``--all-machines`` to iterate the
    full inventory."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return []
    out = []
    for machine_dir in sorted(runs_dir.iterdir()):
        if not machine_dir.is_dir():
            continue
        for sha_dir in machine_dir.iterdir():
            if not sha_dir.is_dir():
                continue
            for run_dir in sha_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                if not (run_dir / "run-metadata.toml").exists():
                    continue
                if _read_status(run_dir) == "complete":
                    out.append(machine_dir.name)
                    break
            else:
                continue
            break
    return out


def discover_machine(results_dir: pathlib.Path) -> str:
    """Return the machine-id that has the most recent complete run."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        sys.exit(f"error: no runs/ directory under {results_dir}")

    best: tuple[str, str] | None = None  # (run_id, machine_id)
    for machine_dir in sorted(runs_dir.iterdir()):
        if not machine_dir.is_dir():
            continue
        machine_id = machine_dir.name
        for sha_dir in sorted(machine_dir.iterdir()):
            if not sha_dir.is_dir():
                continue
            for run_dir in sorted(sha_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if not (run_dir / "run-metadata.toml").exists():
                    continue
                if _read_status(run_dir) != "complete":
                    continue
                run_id = run_dir.name
                if best is None or run_id > best[0]:
                    best = (run_id, machine_id)

    if best is None:
        # No complete runs anywhere — fall back to any run.
        for machine_dir in sorted(runs_dir.iterdir()):
            if machine_dir.is_dir():
                return machine_dir.name
        sys.exit(f"error: no run directories found under {runs_dir}")

    return best[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metas(run_dirs: dict[str, pathlib.Path]) -> dict[str, dict]:
    """Return {scheme: meta_dict} from each scheme's run-metadata.toml."""
    metas: dict[str, dict] = {}
    for scheme, run_dir in run_dirs.items():
        metas[scheme] = _read_meta(run_dir)
    return metas


def copy_aggregated_tsvs(
    results_dir: pathlib.Path,
    machine_id: str,
    data_dir: pathlib.Path,
) -> int:
    """Copy ``results/aggregated/<machine-id>/*.tsv`` into the report's
    data dir. preprocess.py is the single source of truth for figure
    data; this snapshots it
    next to the report.tex so the report dir remains self-contained.

    Findings (``<basename>-finding.tex``) are NOT copied from the
    aggregated dir — they live canonically in the snapshot's data
    dir, keyed by the snapshot's source-run-set fingerprint. The
    per-machine aggregated root would silently re-attribute a
    finding to every snapshot rendered for that machine, even when
    the source data changed (e.g. a 100k-corpus sweep vs an 8.8M-corpus
    sweep); the snapshot-local convention scopes each
    finding to exactly the (scheme, run-id) tuples it describes.

    Stale ``*.tsv`` files in the destination are removed before
    copying so anything no longer in the source doesn't carry over.
    ``*-finding.tex`` files in the destination are preserved — they
    belong to this snapshot, not to the aggregated/TSV pipeline.

    Returns the number of TSV files copied.
    """
    src = results_dir / "aggregated" / machine_id
    if not src.is_dir():
        sys.exit(
            f"error: no aggregated data at {src} — "
            f"run `make preprocess MACHINE={machine_id}` first"
        )
    tsvs = sorted(p for p in src.iterdir() if p.is_file() and p.suffix == ".tsv")
    if not tsvs:
        sys.exit(
            f"error: {src} has no .tsv files — "
            f"run `make preprocess MACHINE={machine_id}` first"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale in data_dir.iterdir():
        if stale.is_file() and stale.suffix == ".tsv":
            stale.unlink()
    for f in tsvs:
        shutil.copy(f, data_dir / f.name)
    return len(tsvs)


# ---------------------------------------------------------------------------
# Figure compilation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Report LaTeX generation
# ---------------------------------------------------------------------------

def _rationale_input(name: str) -> str:
    """Build a rationale block inserted inside the `figure` environment
    so it floats with the figure (LaTeX figure floats can otherwise
    leave the rationale prose stranded on a different page from its
    chart). Convention: `analysis/figures/<basename>-rationale.tex`
    holds a tight LaTeX prose snippet (3–5 sentences) explaining
    non-obvious things about the experiment / chart.

    Wrapped in a `minipage` because the surrounding figure environment
    has `\\centering` applied, which would centre each line of the
    rationale prose (each line trimmed and centred within the column
    rather than flush-left + justified). The minipage establishes its
    own paragraph context with default LaTeX full justification.
    Renders in `\\small` to distinguish from main text. No-op when the
    file is missing — figures without rationale just render as before.
    """
    base = name.removesuffix(".tex")
    return (
        rf"\IfFileExists{{figures/{base}-rationale.tex}}{{%"
        "\n  \\par\\medskip\\begin{minipage}{\\linewidth}\\small"
        rf"\input{{figures/{base}-rationale.tex}}\end{{minipage}}"
        "\n}{}"
    )


def _finding_input(name: str) -> str:
    """Per-run finding: italicised prose specific to this snapshot's
    (scheme, run-id) source set, authored directly into the
    snapshot's `data/<basename>-finding.tex`. Not copied from
    `results/aggregated/<machine-id>/` — the aggregated root has
    machine granularity, not source-set granularity.

    Convention: a finding records non-obvious observations about THIS
    run's data — substrate effects, anomalies that need contextual
    framing, decisions a reader would otherwise question. Generic
    chart explanation belongs in the source-tree rationale; per-run
    observations belong here so they travel with the data and survive
    every re-render. Lead with `\\textbf{Finding:}` so a reader can
    tell at a glance it's a recorded observation, not chart prose.
    """
    base = name.removesuffix(".tex")
    return (
        rf"\IfFileExists{{data/{base}-finding.tex}}{{%"
        "\n  \\par\\medskip\\begin{minipage}{\\linewidth}\\small\\itshape"
        rf"\textbf{{Finding:}}~\input{{data/{base}-finding.tex}}\end{{minipage}}"
        "\n}{}"
    )


def _fig_block(name: str, sidebar: str = "") -> str:
    """Each figure becomes a `minipage` pair — figure
    on the left ~75% width, parameter sidebar on the right ~25%. When
    `sidebar` is empty (no metadata available) the block falls back to
    the original full-width figure layout. The optional rationale (a
    co-located `figures/<basename>-rationale.tex`) flows inside the
    figure environment after the caption so it floats with the chart
    rather than getting stranded on a different page."""
    caption = CAPTIONS.get(name, name)
    if not (FIGURES_DIR / name).exists():
        return f"\\textit{{[{tex_escape(name)}: source not found]}}\n\n\\bigskip\n"
    rationale = _rationale_input(name)
    finding = _finding_input(name)
    if sidebar.strip():
        return textwrap.dedent(rf"""
            \begin{{figure}}[htbp]
              \centering
              \begin{{minipage}}[c]{{0.74\linewidth}}
                \centering
                \Description{{Evaluation figure: {tex_escape(caption)}}}
                \inputplot{{figures/{name}}}
              \end{{minipage}}\hfill
              \begin{{minipage}}[c]{{0.24\linewidth}}
                \footnotesize
                {sidebar}
              \end{{minipage}}
              \caption{{{caption}}}
              {rationale}
              {finding}
            \end{{figure}}
            """).lstrip()
    return textwrap.dedent(f"""\
        \\begin{{figure}}[htbp]
          \\centering
          \\Description{{Evaluation figure: {tex_escape(caption)}}}
          \\inputplot{{figures/{name}}}
          \\caption{{{caption}}}
          {rationale}
          {finding}
        \\end{{figure}}
        """)


def _short_path(path: str) -> str:
    """Last segment of a filesystem path; for compact sidebar display."""
    if not path or path == "—":
        return path
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return parts[-1] if parts else path


def _fmt_compact_list(values: list, max_inline: int = 4) -> str:
    """`[1, 2, 4, 8, 16, 32, 64, 128]` → `1\\ldots128 (8)`; short lists
    are joined inline."""
    if not values:
        return "—"
    if len(values) <= max_inline:
        return ",".join(_fmt_value(v) for v in values)
    return f"{_fmt_value(values[0])}\\ldots{_fmt_value(values[-1])} ({len(values)})"


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _fmt_duration(secs) -> str:
    """Human-readable run duration as `HH:MM:SS`, LaTeX-safe.

    Matches the run-metadata.toml `duration-secs` scale (seconds,
    integer or float) — non-numeric falls through unchanged. Wrapped
    in \\texttt{} at call sites that want monospace alignment.
    """
    try:
        s = int(float(secs))
    except (TypeError, ValueError):
        return str(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _format_scheme_knobs(scheme: str, sc: dict) -> str:
    """Compact one-line summary of scheme-specific knobs from
    `[scheme-config]`. Unknown scheme → join whatever sweep arrays are
    present."""
    parts: list[str] = []
    if "nprobe-values" in sc:
        parts.append(f"nprobe={_fmt_compact_list(sc['nprobe-values'])}")
    if "beta-values" in sc:
        parts.append(rf"$\beta$={_fmt_compact_list(sc['beta-values'])}")
    if "quantisation-bits-values" in sc:
        parts.append(f"q={_fmt_compact_list(sc['quantisation-bits-values'])}")
    if "verification-enabled" in sc:
        parts.append(f"verify={'on' if sc['verification-enabled'] else 'off'}")
    if sc.get("params") == "Sec128" and scheme in ("emvp", "emvp-ivf", "bntm", "bntm-ivf"):
        parts.append("Sec128")
    return " | ".join(parts) if parts else "—"


def _param_sidebar(
    name: str,
    scheme_metas: dict[str, dict],
    machine_id: str,
) -> str:
    """Per-figure parameter sidebar pulled from the
    same `run-metadata.toml` files that produced the figure's TSVs.
    Single source of truth, no manual entry."""
    if not scheme_metas:
        return ""

    rows: list[tuple[str, str]] = []

    common = next(iter(scheme_metas.values()))
    ds = common.get("dataset", {})
    ivf = common.get("ivf", {})
    rows.append(("dataset", _short_path(str(ds.get("path", "—")))))
    model = str(ds.get("embedding-model", "—"))
    rows.append(("model", model.split("/")[-1]))
    rows.append(
        (
            "IVF",
            f"{ivf.get('n-centroids', '—')}/{ivf.get('train-seed', '—')}/{ivf.get('max-iter', '—')}",
        )
    )

    fig_schemes = SCHEMES_BY_FIGURE.get(name)
    schemes_in_fig = sorted(
        s for s in scheme_metas if fig_schemes is None or s in fig_schemes
    )
    if schemes_in_fig:
        rows.append(("__rule__", ""))
    for scheme in schemes_in_fig:
        sc = scheme_metas[scheme].get("scheme-config", {})
        rows.append((scheme, _format_scheme_knobs(scheme, sc)))

    if name == "08-build-time.tex":
        rows.append(("__rule__", ""))
        any_idx = False
        for scheme in schemes_in_fig:
            idx = scheme_metas[scheme].get("index")
            if idx is None:
                continue
            cold = not idx.get("cache-hit", True)
            dur = idx.get("build-duration-secs")
            if dur is None:
                continue
            rows.append(
                (
                    f"{scheme} build",
                    f"{float(dur):.1f}s ({'cold' if cold else 'warm'})",
                )
            )
            any_idx = True
        if not any_idx:
            rows.append(("[index]", "—"))

    if name in ("09a-substeps-absolute.tex", "09b-substeps-normalised.tex"):
        rows.append(("__rule__", ""))
        any_breakdown = any(m.get("breakdown") for m in scheme_metas.values())
        rows.append(("breakdown", "true" if any_breakdown else "—"))

    rows.append(("__rule__", ""))
    rows.append(("machine", str(machine_id)))
    latest = max(
        scheme_metas.values(),
        key=lambda m: str(m.get("run-id", "")),
    )
    sha = str(latest.get("git-sha", "—"))
    sha_short = sha[:8] if len(sha) >= 8 else sha
    dirty = "*" if latest.get("git-dirty") else ""
    rows.append(("git", sha_short + dirty))

    return _render_sidebar(rows)


def _render_sidebar(rows: list[tuple[str, str]]) -> str:
    body_lines: list[str] = []
    for k, v in rows:
        if k == "__rule__":
            body_lines.append("\\midrule")
            continue
        body_lines.append(f"\\texttt{{{tex_escape(k)}}} & {v} \\\\")
    body = "\n          ".join(body_lines)
    return textwrap.dedent(rf"""
        \begin{{tabular}}{{@{{}}lp{{0.55\linewidth}}@{{}}}}
          \toprule
          {body}
          \bottomrule
        \end{{tabular}}
        """).strip()


def _to_local(iso_utc: str) -> str:
    """Convert an ISO-8601 UTC timestamp from run-metadata.toml
    (`2026-05-08T06:07:07Z`) to the local timezone of the machine
    rendering the report. Falls back to the input string if parsing
    fails so an unexpected format doesn't blow up the table.
    """
    try:
        s = iso_utc.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone()
        return local.strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, AttributeError):
        return iso_utc


def _runs_table(scheme_metas: dict[str, dict], run_dirs: dict[str, pathlib.Path]) -> str:
    """Return a LaTeX tabular with one row per scheme."""
    rows = []
    # Hide the Device column when every run was CPU — keeps the legacy
    # CPU-only report shape; surface it when any run is GPU-side.
    show_device = any(
        (m.get("device") or "cpu") != "cpu" for m in scheme_metas.values()
    )
    for scheme in sorted(scheme_metas):
        meta = scheme_metas[scheme]
        run_dir = run_dirs[scheme]
        run_id  = tex_escape(str(meta.get("run-id", run_dir.name)))
        status  = tex_escape(meta.get("status", "unknown"))
        sha     = tex_escape(meta.get("git-sha", "unknown")[:12])
        branch  = tex_escape(meta.get("git-branch", "unknown"))
        dirty   = r"$\star$" if meta.get("git-dirty", False) else ""
        started_raw = meta.get("started-at", "unknown")
        started = tex_escape(
            _to_local(started_raw) if started_raw != "unknown" else started_raw
        )
        dur     = meta.get("duration-secs")
        dur_str = _fmt_duration(dur) if dur is not None else "unknown"
        sc      = meta.get("scheme-config", {})
        sweep   = tex_escape(sc.get("sweep", "?"))
        n_centroids = sc.get("n-centroids", "?")
        device  = (meta.get("device") or "cpu").upper()
        device_cell = f" & \\texttt{{{device}}}" if show_device else ""
        rows.append(
            f"  \\texttt{{{tex_escape(scheme)}}} & \\texttt{{{run_id}}} & {status} & "
            f"\\texttt{{{sha}}} ({branch}){dirty} & {sweep} & {n_centroids} & "
            f"{started} & {dur_str}{device_cell} \\\\"
        )
    body = "\n".join(rows)
    cols = "@{}llllllll" + ("l" if show_device else "") + "@{}"
    device_header = " & \\textbf{Device}" if show_device else ""
    return textwrap.dedent(rf"""
        \begin{{tabular}}{{{cols}}}
          \toprule
          \textbf{{Scheme}} & \textbf{{Run ID}} & \textbf{{Status}} & \textbf{{Git}} &
          \textbf{{Sweep}} & \textbf{{Centroids}} & \textbf{{Started}} & \textbf{{Duration}}{device_header} \\
          \midrule
        {body}
          \bottomrule
        \end{{tabular}}
        """).lstrip()


def _dataset_table(scheme_metas: dict[str, dict]) -> str:
    """Return a LaTeX tabular with dataset info from the first available scheme."""
    ds: dict = {}
    for meta in scheme_metas.values():
        ds = meta.get("dataset", {})
        if ds:
            break
    data_path  = tex_escape(ds.get("path", "?"))
    n_passages = ds.get("n-passages", "?")
    n_queries  = ds.get("n-queries", "?")
    dim        = ds.get("dimension", "?")
    model      = tex_escape(ds.get("embedding-model", "?"))
    return textwrap.dedent(rf"""
        \begin{{tabular}}{{@{{}}ll@{{}}}}
          \toprule
          \textbf{{Field}} & \textbf{{Value}} \\
          \midrule
          Path        & \texttt{{{data_path}}} \\
          Passages    & {n_passages} \\
          Queries     & {n_queries} \\
          Dimension   & {dim} \\
          Embed model & \texttt{{{model}}} \\
          \bottomrule
        \end{{tabular}}
        """).lstrip()


def write_report_tex(
    machine_id: str,
    run_dirs: dict[str, pathlib.Path],
    scheme_metas: dict[str, dict],
    generated_at: datetime,
    results_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    hw           = _machine_info(results_dir, machine_id)
    machine_name = hw.get("machine-name", "")
    cpu          = tex_escape(hw.get("cpu-model", "unknown"))
    cores        = hw.get("cores", "?")
    ram_gb       = int(hw.get("ram-bytes", 0)) / 1024 ** 3
    # Static GPU kind from machines.csv. Suppressed below when any run
    # was on GPU substrate — the dynamic [gpu] block carries the
    # actual SKU/memory, and machines.csv often holds "none" because
    # the field captures the host's static detection rather than the
    # runtime GPU assignment (especially for RunAI / cloud pods).
    gpu          = tex_escape(hw.get("gpu-kind", "none"))
    toolchain    = tex_escape(
        next((m.get("rust-toolchain", "unknown") for m in scheme_metas.values()), "unknown")
    )

    # Cgroup constraints across this machine's runs. Aggregates over
    # scheme_metas because cgroup-* fields are per-run (run-metadata.toml,
    # captured by meta.rs::capture_cgroup_*) — different runs on the
    # same machine could in principle have different limits. Both fields
    # are skipped from the TOML when no limit was enforced (or pre-50fe43b
    # binary), so the sets here may legitimately contain only None.
    quotas = sorted({
        m.get("cgroup-cpu-quota") for m in scheme_metas.values()
        if m.get("cgroup-cpu-quota") is not None
    })
    mem_limits = sorted({
        m.get("cgroup-memory-bytes") for m in scheme_metas.values()
        if m.get("cgroup-memory-bytes") is not None
    })
    if quotas:
        cgroup_cpu_str = (
            f"{quotas[0]:.2f}\\,vCPU" if len(quotas) == 1
            else f"{quotas[0]:.2f}\\,--\\,{quotas[-1]:.2f}\\,vCPU"
        )
    else:
        cgroup_cpu_str = ""
    if mem_limits:
        gbs = [b / 1024**3 for b in mem_limits]
        cgroup_mem_str = (
            f"{gbs[0]:.1f}\\,GB" if len(gbs) == 1
            else f"{gbs[0]:.1f}\\,--\\,{gbs[-1]:.1f}\\,GB"
        )
    else:
        cgroup_mem_str = ""

    cgroup_rows = ""
    if cgroup_cpu_str or cgroup_mem_str:
        cells = []
        if cgroup_cpu_str:
            cells.append(f"        Cgroup CPU      & {cgroup_cpu_str} \\\\")
        if cgroup_mem_str:
            cells.append(f"        Cgroup memory   & {cgroup_mem_str} \\\\")
        cgroup_rows = "\n".join(cells) + "\n"

    # GPU runtime info. Each run's [gpu] block is per-run, so
    # a machine could in principle have a mix of CPU and GPU runs;
    # surface that honestly. SKU / location / cloud fields aggregated
    # across scheme_metas; ranges shown when values differ.
    devices_seen = sorted({m.get("device", "cpu") or "cpu" for m in scheme_metas.values()})
    # When any run was GPU substrate, the dynamic [gpu] block below
    # carries the real SKU/memory — the static `GPU & {gpu}` row from
    # machines.csv would just say "none" (machines-csv field reflects
    # host detection, not runtime mount) and confuse the reader.
    static_gpu_row = (
        "" if "gpu" in devices_seen
        else f"          GPU             & {gpu} \\\\\n"
    )
    gpu_blocks = [m.get("gpu") for m in scheme_metas.values() if m.get("gpu")]
    gpu_rows = ""
    if "gpu" in devices_seen:
        device_str = (
            "GPU" if devices_seen == ["gpu"]
            else "mixed (CPU + GPU)"
        )
        skus = sorted({tex_escape(b.get("sku", "")) for b in gpu_blocks if b.get("sku")})
        sku_str = ", ".join(skus) if skus else "—"
        # GPU memory captured via nvidia-smi at run time. Aggregate
        # across runs because two runs on the same machine could in
        # principle land on different GPUs (rare but possible on
        # multi-GPU boxes); show range when values differ.
        gpu_mems = sorted({
            b.get("memory-bytes") for b in gpu_blocks
            if b.get("memory-bytes") is not None
        })
        if gpu_mems:
            mem_gbs = [m / 1024**3 for m in gpu_mems]
            mem_str = (
                f"{mem_gbs[0]:.1f}\\,GB" if len(mem_gbs) == 1
                else f"{mem_gbs[0]:.1f}\\,--\\,{mem_gbs[-1]:.1f}\\,GB"
            )
            sku_mem = f"\\texttt{{{sku_str}}},\\,{mem_str}"
        else:
            sku_mem = f"\\texttt{{{sku_str}}}"
        cells = [
            f"        Device          & \\textbf{{{device_str}}} \\\\",
            f"        GPU             & {sku_mem} \\\\",
        ]
        # Cloud sub-block, if any run carries it (location=cloud only).
        cloud_blocks = [b.get("cloud") for b in gpu_blocks if b.get("cloud")]
        if cloud_blocks:
            providers = sorted({tex_escape(c.get("provider", "")) for c in cloud_blocks if c.get("provider")})
            instances = sorted({tex_escape(c.get("instance-type", "")) for c in cloud_blocks if c.get("instance-type")})
            regions = sorted({tex_escape(c.get("region", "")) for c in cloud_blocks if c.get("region")})
            drivers = sorted({tex_escape(c.get("driver-version", "")) for c in cloud_blocks if c.get("driver-version")})
            cudas = sorted({tex_escape(c.get("cuda-version", "")) for c in cloud_blocks if c.get("cuda-version")})
            cloud_parts: list[str] = []
            if providers:
                cloud_parts.append(", ".join(providers))
            if instances:
                cloud_parts.append(f"\\texttt{{{', '.join(instances)}}}")
            if regions:
                cloud_parts.append(", ".join(regions))
            if cloud_parts:
                cells.append(f"        Cloud           & {' / '.join(cloud_parts)} \\\\")
            if drivers:
                cells.append(f"        NVIDIA driver   & \\texttt{{{', '.join(drivers)}}} \\\\")
            if cudas:
                cells.append(f"        CUDA version    & \\texttt{{{', '.join(cudas)}}} \\\\")
        gpu_rows = "\n".join(cells) + "\n"

    machine_str = tex_escape(machine_id) + (f" ({tex_escape(machine_name)})" if machine_name else "")
    gen_str     = tex_escape(generated_at.strftime("%Y-%m-%d %H:%M:%S %Z"))
    schemes_str = tex_escape(", ".join(sorted(scheme_metas)))

    runs_table    = _runs_table(scheme_metas, run_dirs)
    dataset_table = _dataset_table(scheme_metas)

    # Skip figure sections whose source TSVs are missing or
    # header-only — partial run sets (e.g. an eval campaign that
    # didn't include `--breakdown` / `eval-cold` / BN-with-and-without
    # verification / `eval-scaling`) drop their diagnostic figures
    # cleanly rather than rendering empty axes.
    data_dir = out_path.parent / "data"
    quality_present = [n for n in QUALITY_FIGURES if _figure_has_data(n, data_dir)]
    perf_present = [n for n in PERFORMANCE_FIGURES if _figure_has_data(n, data_dir)]
    skipped = [n for n in (QUALITY_FIGURES + PERFORMANCE_FIGURES)
               if not _figure_has_data(n, data_dir)]
    if skipped:
        print(f"  skipped (no source data): {', '.join(skipped)}")

    def fb(n: str) -> str:
        return _fig_block(n, _param_sidebar(n, scheme_metas, machine_id))

    quality_body  = "\n".join(fb(n) for n in quality_present)
    perf_body     = "\n".join(fb(n) for n in perf_present)

    tex = textwrap.dedent(rf"""
        \documentclass[manuscript, nonacm, dvipsnames]{{acmart}}
        \settopmatter{{printacmref=false, printfolios=true}}
        \renewcommand\footnotetextcopyrightpermission[1]{{}}
        \usepackage{{booktabs}}
        \usepackage{{standalone}}
        % `placeins`'s `\FloatBarrier` keeps figures inside their
        % section instead of floating across section headings — used
        % below so the "Quality" heading isn't orphaned at the bottom
        % of page 1 with its first figure floated to page 2.
        \usepackage{{placeins}}
        \input{{style}}
        \renewcommand{{\sansmath}}{{}}
        \pgfplotsset{{width=0.88\linewidth, height=0.44\linewidth}}
        \def\datadir{{data}}
        \newcommand{{\inputplot}}[1]{{\input{{#1}}}}
        \hypersetup{{hidelinks}}

        \title{{Evaluation Report}}
        \subtitle{{Machine:~\texttt{{{machine_str}}}\quad Schemes:~{schemes_str}\\[0.2em]
                  \normalsize Generated:~{gen_str}}}
        \author{{Secure Vector Search}}
        \affiliation{{\institution{{EPFL}}\city{{Lausanne}}\country{{Switzerland}}}}
        \email{{~}}

        \begin{{document}}
        \maketitle

        \section*{{Machine}}
        \begin{{tabular}}{{@{{}}ll@{{}}}}
          \toprule
          \textbf{{Field}} & \textbf{{Value}} \\
          \midrule
          Machine ID      & \texttt{{{machine_str}}} \\
          CPU             & {cpu} \\
          Cores / RAM     & {cores} cores / {ram_gb:.1f}\,GB \\
{static_gpu_row}          Rust toolchain  & \texttt{{{toolchain}}} \\
{gpu_rows}{cgroup_rows}          \bottomrule
        \end{{tabular}}

        \subsection*{{Runs}}
        \noindent\resizebox{{\linewidth}}{{!}}{{%
        {runs_table}}}

        \subsection*{{Dataset}}
        {dataset_table}

        % `\clearpage` before each section so that section headings
        % always have content following them on the same page (else
        % "Quality" gets orphaned at the bottom of page 1 while its
        % first figure floats to page 2).
        \clearpage
        \section{{Quality}}

        {quality_body}

        \clearpage
        \section{{Performance}}

        {perf_body}

        \end{{document}}
    """).lstrip()

    out_path.write_text(tex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _snapshot_key(run_dirs: dict[str, pathlib.Path]) -> str:
    """Fingerprint of the (scheme, run-id) tuples feeding this report.

    Same set of source runs → same fingerprint → same snapshot dir
    (rendering is idempotent over identical data). Adding or replacing
    a run shifts the fingerprint and lands the new render in a new dir,
    so a prior "good" report is never overwritten by an unrelated set.

    The dir name embeds the latest run-id across schemes so it sorts
    chronologically alongside its siblings, with a 12-char SHA-256
    suffix for uniqueness.
    """
    items = sorted((scheme, rd.name) for scheme, rd in run_dirs.items())
    body = "|".join(f"{s}={r}" for s, r in items)
    fp = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    latest_run_id = max((r for _, r in items), default="0000000000")
    return f"{latest_run_id}-{fp}"


def _write_manifest(
    snapshot_dir: pathlib.Path,
    run_dirs: dict[str, pathlib.Path],
    machine_id: str,
    generated_at: datetime,
) -> None:
    """Drop a ``manifest.toml`` listing the exact source runs the
    snapshot was built from. Lets a reader confirm at a glance
    whether two reports cover the same data."""
    lines = [
        f'machine-id = "{machine_id}"',
        f'generated-at = "{generated_at.isoformat()}"',
        "",
        "[runs]",
    ]
    for scheme in sorted(run_dirs):
        lines.append(f'{scheme} = "{run_dirs[scheme].name}"')
    (snapshot_dir / "manifest.toml").write_text("\n".join(lines) + "\n")


def _refresh_latest_symlink(
    reports_root: pathlib.Path, snapshot_name: str,
) -> None:
    """Point ``reports/latest`` at ``reports/<snapshot_name>``.

    On first migration (legacy ``reports/latest/`` is a real directory
    rather than a symlink), rename the existing one to a backup based
    on its mtime so its rendered PDF is preserved.
    """
    latest = reports_root / "latest"
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists():
        ts = datetime.fromtimestamp(latest.stat().st_mtime).strftime(
            "%Y-%m-%dT%H-%M-%S",
        )
        backup = reports_root / f"{ts}-legacy"
        suffix = 0
        while backup.exists():
            suffix += 1
            backup = reports_root / f"{ts}-legacy-{suffix}"
        latest.rename(backup)
        print(f"  archived prior reports/latest → reports/{backup.name}")
    # Relative target so the symlink stays valid if the reports tree
    # gets moved (e.g. between machines via rsync).
    latest.symlink_to(snapshot_name)


def build_report_for_machine(
    machine_id: str, results_dir: pathlib.Path,
) -> pathlib.Path | None:
    """Render the per-machine PDF report. Returns the generated
    ``report.pdf`` path on success, ``None`` on failure."""
    print(f"Machine: {machine_id}")

    run_dirs = find_runs_for_machine(results_dir, machine_id)
    if not run_dirs:
        raise SkipMachine(
            f"no non-breakdown runs under results/runs/{machine_id} — nothing to render"
        )
    schemes_found = sorted(run_dirs)
    print(f"  Schemes: {', '.join(schemes_found)}")
    for scheme, rd in sorted(run_dirs.items()):
        print(f"    {scheme}: {rd}")

    scheme_metas = load_metas(run_dirs)

    # Output: results/runs/<machine-id>/reports/<latest-run-id>-<fp>/
    # The dir name is a fingerprint of the source (scheme, run-id) set,
    # so rendering the *same* data twice lands in the *same* dir
    # (idempotent overwrite); a different data set = a different dir,
    # so prior reports are never silently shifted by new evals on the
    # same machine. reports/latest symlinks to the freshest snapshot.
    generated_at = datetime.now().astimezone()
    snapshot_name = _snapshot_key(run_dirs)
    reports_root = results_dir / "runs" / machine_id / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    report_dir = reports_root / snapshot_name
    report_dir.mkdir(parents=True, exist_ok=True)

    data_dir = report_dir / "data"
    n_copied = copy_aggregated_tsvs(results_dir, machine_id, data_dir)
    print(f"  TSVs → {data_dir} ({n_copied} files)")

    figs_dir = report_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Copy style.tex and figure sources so the report dir is self-contained.
    if STYLE_TEX.exists():
        shutil.copy(STYLE_TEX, report_dir / "style.tex")
    # Figure 03 has a 2-panel variant for machines without any
    # with-offline-class data (no Tiptoe runs). Inline conditionals
    # didn't survive pgfkeys' option-parser, so we pick the right
    # template here at copy time and rename it to the canonical name
    # the report.tex expects.
    has_offline = _comm_class_has_data(data_dir, "with-offline")
    for fig_name in ALL_FIGURES:
        if fig_name == "03-communication.tex" and not has_offline:
            src = FIGURES_DIR / "03-communication-2panel.tex"
            if src.exists():
                shutil.copy(src, figs_dir / fig_name)
                rationale_name = fig_name.removesuffix(".tex") + "-rationale.tex"
                rationale_tex = FIGURES_DIR / rationale_name
                if rationale_tex.exists():
                    shutil.copy(rationale_tex, figs_dir / rationale_name)
                continue
            # Falls through to the default copy if the variant is missing.
        fig_tex = FIGURES_DIR / fig_name
        if fig_tex.exists():
            shutil.copy(fig_tex, figs_dir / fig_name)
        else:
            print(f"  warning: {fig_name} not found in {FIGURES_DIR}", file=sys.stderr)
        rationale_name = fig_name.removesuffix(".tex") + "-rationale.tex"
        rationale_tex = FIGURES_DIR / rationale_name
        if rationale_tex.exists():
            shutil.copy(rationale_tex, figs_dir / rationale_name)

    report_tex = report_dir / "report.tex"
    write_report_tex(
        machine_id, run_dirs, scheme_metas,
        generated_at, results_dir, report_tex,
    )

    cmd = [
        "latexmk", "-pdf", "-interaction=nonstopmode",
        f"-outdir={report_dir}",
        str(report_tex),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=report_dir)
    if result.returncode != 0:
        print(f"error: failed to compile report.tex for {machine_id}", file=sys.stderr)
        print(result.stderr[-1000:], file=sys.stderr)
        return None

    report_pdf = report_dir / "report.pdf"
    if not report_pdf.exists():
        print(f"error: report.pdf missing after compilation for {machine_id}", file=sys.stderr)
        return None
    _write_manifest(report_dir, run_dirs, machine_id, generated_at)
    _refresh_latest_symlink(reports_root, snapshot_name)
    print(f"Report: {report_pdf}")
    print(f"Latest: {reports_root / 'latest'}")
    return report_pdf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a per-machine PDF report aggregating all schemes."
    )
    parser.add_argument(
        "--run", type=pathlib.Path, metavar="DIR",
        help="any run directory for the target machine",
    )
    parser.add_argument(
        "--machine", metavar="ID",
        help="machine-id to report on (overrides --run for machine selection)",
    )
    parser.add_argument(
        "--all-machines", action="store_true",
        help="render a report for every machine under results/runs/ "
             "(default when neither --run nor --machine is set)",
    )
    parser.add_argument(
        "--results", type=pathlib.Path, default=pathlib.Path("../results"),
        help="results root (default: ../results)",
    )
    args = parser.parse_args()

    if shutil.which("latexmk") is None:
        sys.exit("error: latexmk not found — install texlive or mactex")

    results_dir = args.results.resolve()

    # Resolve which machine(s) to render.
    if args.machine:
        machine_ids = [args.machine]
    elif args.run:
        run_dir = args.run.resolve()
        try:
            machine_ids = [run_dir.relative_to(results_dir / "runs").parts[0]]
        except ValueError:
            meta = _read_meta(run_dir)
            machine_ids = [meta.get("machine-id") or discover_machine(results_dir)]
    elif args.all_machines:
        machine_ids = list_machines(results_dir)
        if not machine_ids:
            sys.exit(f"error: no machines with complete runs under {results_dir}")
        print(f"All-machines mode: {len(machine_ids)} machine(s) — "
              f"{', '.join(machine_ids)}")
    else:
        # Default: render every machine that has data, mirroring the
        # cross-machine dossier's "show me everything" ergonomics. Falls
        # back gracefully via list_machines() (returns []  if results/
        # is empty).
        machine_ids = list_machines(results_dir)
        if not machine_ids:
            sys.exit(f"error: no machines with complete runs under {results_dir}")

    failures: list[str] = []
    skipped: list[str] = []
    generated: list[tuple[str, pathlib.Path]] = []
    for mid in machine_ids:
        try:
            pdf = build_report_for_machine(mid, results_dir)
        except SkipMachine as e:
            # build_report_for_machine already printed `Machine: {mid}`
            # before raising — just add the skip reason.
            print(f"  skip: {e}")
            skipped.append(mid)
            if len(machine_ids) > 1:
                print()
            continue
        if pdf is None:
            failures.append(mid)
        else:
            generated.append((mid, pdf))
        if len(machine_ids) > 1:
            print()

    if generated:
        print(f"Generated {len(generated)} report(s):")
        # Look up machine-name once per machine for the summary line.
        for mid, pdf in generated:
            info = _machine_info(results_dir, mid)
            name = info.get("machine-name", "").strip()
            label = f"{mid} ({name})" if name else mid
            print(f"  {label:<30}  {os.path.relpath(pdf.resolve())}")

    if skipped:
        print(f"Skipped {len(skipped)} machine(s) with no non-breakdown data: "
              f"{', '.join(skipped)}")

    if failures:
        sys.exit(f"error: report build failed for {len(failures)} machine(s): "
                 f"{', '.join(failures)}")


if __name__ == "__main__":
    main()
