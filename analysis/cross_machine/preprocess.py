#!/usr/bin/env python3
"""Cross-machine insights preprocess: TSVs + tikz fragments + tables.

Walks ``results/runs/**/raw.csv`` across every machine, joins with
each run's ``run-metadata.toml`` and the global ``machines.csv``, and
emits into ``data/``:

- ``machines-table.tex`` — tabular block listing every machine that
  produced runs, with CPU / cores / RAM / GPU / observed
  rayon-realised thread range / cgroup-quota status.
- ``coverage-table.tex`` — scheme × machine matrix of run counts so
  the dossier shows what data is actually behind each figure.
- ``header.tex`` — defines ``\\insightsSubtitle`` for the title page.
- Figure 01 — latency vs nprobe per IVF scheme. Multi-series line
  plot; one series per (machine, device, threads, cgroup) tuple.
- Figure 02 — per-machine bars at one canonical config per scheme.
  Horizontal bar chart; one bar per series, easier to read for
  direct hardware ranking than a log-log line plot.
- Figure 03 — parallel scaling. For each IVF scheme at the canonical
  nprobe, plot latency vs realised thread count where any single
  (machine, device, cgroup) tuple has at least two thread points
  (a thread-count sweep is the natural source).

The (machine, device, threads, cgroup_cpu_quota) tuple is the
canonical series key — distinct cgroup quotas land as distinct
series so the cross-machine plot doesn't fold them together. Series
labels surface ``GPU`` (the device the run measured against), ``t=N``
(rayon-pool disagrees with host cores), and ``cg=Q`` (cgroup CPU cap).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tomllib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


IVF_SCHEMES = ("plaintext", "sap-ivf", "emvp-ivf", "bntm-ivf")
ALL_SCHEMES = (
    "plaintext", "sap", "sap-ivf", "emvp", "emvp-ivf",
    "tiptoe", "tiptoe-go", "bntm", "bntm-ivf",
)

# Canonical (scheme, config-label) operating points for the bar charts.
# The IVF schemes converge at nprobe=32 (~recall 0.93); flat scorers
# pick their natural single config.
BAR_CONFIGS = (
    ("plaintext", "nprobe=32"),
    ("sap-ivf", "beta=0.0000|nprobe=32"),
    ("emvp-ivf", "nprobe=32"),
    ("bntm-ivf", "nprobe=32"),
    ("sap", "beta=0.0000"),
    ("emvp", "emvp"),
    ("bntm", "bntm"),
)

# Canonical config for the parallel-scaling figures — same nprobe as the
# bar charts so an operator reading both reads the same operating point.
SCALING_CONFIGS = {
    "plaintext": "nprobe=32",
    "sap-ivf": "beta=0.0000|nprobe=32",
    "emvp-ivf": "nprobe=32",
    "bntm-ivf": "nprobe=32",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=Path("../../results"), type=Path)
    p.add_argument("--outdir", default=Path("data"), type=Path)
    return p.parse_args()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def tex_escape(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
         .replace("&", r"\&")
         .replace("%", r"\%")
         .replace("$", r"\$")
         .replace("#", r"\#")
         .replace("_", r"\_")
         .replace("{", r"\{")
         .replace("}", r"\}")
         .replace("~", r"\textasciitilde{}")
         .replace("^", r"\textasciicircum{}")
    )


def load_machines(results: Path) -> dict[str, dict[str, str]]:
    p = results / "machines.csv"
    if not p.exists():
        return {}
    with p.open(newline="") as f:
        return {row["machine-id"]: row for row in csv.DictReader(f)}


def load_meta(run_dir: Path) -> dict:
    p = run_dir / "run-metadata.toml"
    if not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError:
        return {}


def fmt_threads_range(values: set) -> str:
    vs = sorted(v for v in values if isinstance(v, int))
    if not vs:
        return "—"
    if vs[0] == vs[-1]:
        return str(vs[0])
    return f"{vs[0]}–{vs[-1]}"


def fmt_quota_range(values: set) -> str:
    vs = sorted({v for v in values if v is not None})
    if not vs:
        return "—"
    if len(vs) == 1:
        return f"{vs[0]:.2f}"
    return f"{vs[0]:.2f}–{vs[-1]:.2f}"


def collect_all(results: Path):
    """Walk every raw.csv. Returns three structures.

    - ``samples_by_key``: ``{(scheme, config_label, mid, dev, threads,
      cpu_quota): [(latency_us, recall)]}``. Source of truth for every
      figure emitter.
    - ``machine_summary``: per-machine summary for the Machines table.
    - ``run_counts``: ``{(scheme, mid): n_complete_runs}`` for the
      Coverage table.
    """
    samples: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    machine_summary: dict[str, dict] = defaultdict(lambda: {
        "threads": set(),
        "cpu_quotas": set(),
        "memory_bytes": set(),
        "schemes": set(),
        "devices": set(),
        "n_runs": 0,
    })
    # `(scheme, machine_id) -> {device -> count}` so the coverage table
    # can annotate which runs were CPU vs GPU. CPU is the default and
    # left implicit; GPU is the surprising case the reader needs flagged.
    run_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int),
    )

    runs_root = results / "runs"
    if not runs_root.exists():
        return samples, machine_summary, run_counts

    for raw in runs_root.rglob("raw.csv"):
        meta = load_meta(raw.parent)
        if not meta or meta.get("breakdown"):
            continue
        with raw.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        scheme = rows[0].get("scheme", "")
        if scheme not in ALL_SCHEMES:
            continue
        threads = meta.get("parallel-threads")
        cpu_quota = meta.get("cgroup-cpu-quota")
        mem_bytes = meta.get("cgroup-memory-bytes")
        machine_id = meta.get("machine-id") or rows[0].get("machine-id", "")
        if not machine_id:
            continue

        # Run-level device label sourced from run-metadata.toml's `device`
        # field. CPU is the default; GPU appears when the
        # eval ran with `--device gpu`. Used to colour both the
        # machines-table "Devices used" column and the coverage cells.
        run_device = (meta.get("device") or "cpu") or "cpu"
        run_counts[(scheme, machine_id)][run_device] += 1
        s = machine_summary[machine_id]
        s["threads"].add(threads)
        s["cpu_quotas"].add(cpu_quota)
        s["memory_bytes"].add(mem_bytes)
        s["schemes"].add(scheme)
        s["devices"].add(run_device)
        s["n_runs"] += 1

        for row in rows:
            try:
                lat = float(row["latency-us"])
                rec = float(row["recall-at-k"])
            except (KeyError, ValueError, TypeError):
                # TypeError covers truncated final rows: a process
                # killed mid-write (e.g. RunAI Suspended) leaves a row
                # with fewer columns than the header, and DictReader
                # returns None for the missing trailing cells.
                continue
            cfg = row.get("config-label", "")
            mid = row.get("machine-id", "") or machine_id
            if not mid:
                continue
            dev = row.get("device", "cpu") or "cpu"
            key = (scheme, cfg, mid, dev, threads, cpu_quota)
            samples[key].append((lat, rec))

    return samples, machine_summary, run_counts


def series_label(
    mid: str, device: str, threads, cpu_quota, machines: dict,
) -> str:
    row = machines.get(mid, {})
    name = (row.get("machine-name") or "").strip() or mid or "?"
    parts = [name]
    if device == "gpu":
        parts.append("GPU")
    cores = row.get("cores", "")
    if threads is not None and cores and str(threads) != str(cores):
        parts.append(f"t={threads}")
    if cpu_quota is not None:
        parts.append(f"cg={float(cpu_quota):.1f}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Figure 01 — latency vs nprobe (per IVF scheme)
#
# Tikz lives in figures/01-latency-vs-nprobe-<scheme>.tex (hand-written
# templates that iterate over \canonicalSeries via \foreach + \IfFileExists).
# This emitter only writes the per-series TSVs they consume.
# ---------------------------------------------------------------------------

def emit_tsvs_latency_vs_nprobe(
    scheme: str, samples: dict, machines: dict, outdir: Path,
) -> int:
    series: dict[tuple, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for (sch, cfg, mid, dev, threads, cpu_quota), pts in samples.items():
        if sch != scheme:
            continue
        m = re.search(r"nprobe=(\d+)", cfg)
        if not m:
            continue
        if "beta=" in cfg and "beta=0.0000" not in cfg:
            continue
        nprobe = int(m.group(1))
        series[(mid, dev, threads, cpu_quota)][nprobe].extend(pts)

    fig = f"01-latency-vs-nprobe-{scheme}"
    n = 0
    for key, by_np in series.items():
        slug = slugify(series_label(*key, machines))
        tsv = outdir / f"{fig}-{slug}.tsv"
        with tsv.open("w") as f:
            f.write("nprobe\tlatency_ms\trecall\n")
            for nprobe in sorted(by_np.keys()):
                pts = by_np[nprobe]
                lat_ms = mean(p[0] for p in pts) / 1000.0
                rec = mean(p[1] for p in pts)
                f.write(f"{nprobe}\t{lat_ms:.4f}\t{rec:.4f}\n")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Figure 02 — per-machine bars at fixed config
# ---------------------------------------------------------------------------

def emit_figure_machine_bars(
    scheme: str, config: str, samples: dict, machines: dict, outdir: Path,
) -> bool:
    """Returns True if a figure was emitted (data was present)."""
    bars: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for (sch, cfg, mid, dev, threads, cpu_quota), pts in samples.items():
        if sch == scheme and cfg == config:
            bars[(mid, dev, threads, cpu_quota)].extend(pts)
    if not bars:
        return False

    rows = sorted(
        bars.items(),
        key=lambda kv: mean(p[0] for p in kv[1]),
    )

    cfg_slug = slugify(config)
    fig = f"02-bar-{scheme}-{cfg_slug}"
    tsv = outdir / f"{fig}.tsv"

    # Use numeric y indices + a yticklabels list rather than `symbolic y
    # coords`. The series labels carry `=` and spaces (from `t=N` and
    # `cg=Q` suffixes) which pgfplots' symbolic-coord parser treats as
    # special; numeric indices sidestep that entirely.
    labels: list[str] = []
    with tsv.open("w") as f:
        f.write("idx\tlatency_ms\n")
        for i, ((mid, dev, threads, cpu_quota), pts) in enumerate(rows):
            labels.append(series_label(mid, dev, threads, cpu_quota, machines))
            lat_ms = mean(p[0] for p in pts) / 1000.0
            f.write(f"{i}\t{lat_ms:.4f}\n")

    title = (
        f"\\texttt{{{tex_escape(scheme)}}} @ "
        f"\\texttt{{{tex_escape(config)}}}: latency per machine"
    )
    n = len(labels)
    ytick = ",".join(str(i) for i in range(n))
    yticklabels = ",".join("{" + tex_escape(lab) + "}" for lab in labels)

    tikz = outdir / f"{fig}.tikz"
    with tikz.open("w") as f:
        f.write("% Auto-generated by analysis/cross_machine/preprocess.py\n")
        f.write("\\begin{tikzpicture}\n")
        f.write("  \\begin{axis}[\n")
        f.write("    xbar,\n")
        # Single \addplot below picks the first colour from this
        # cycle list — gives a uniform fill across all bars from the
        # colorbrewer Paired-12 palette (mirrors the project's
        # bar-chart styling in analysis/figures/03-communication.tex).
        f.write("    cycle list/Paired-12,\n")
        f.write("    every axis plot/.append style={fill, fill opacity=0.85},\n")
        f.write("    bar width=5pt,\n")
        # style.tex turns y-major grids on globally; for an xbar chart
        # they fight the categorical y-axis (one gridline per bar slot
        # is just clutter). Swap to x-major grids so the eye can read
        # latencies against vertical reference lines instead.
        f.write("    ymajorgrids=false,\n")
        f.write("    xmajorgrids=true,\n")
        f.write("    xlabel={mean latency (ms)},\n")
        f.write(f"    title={{{title}}},\n")
        # Per-slot height grows linearly with n at 0.18in/bar plus a
        # 0.5in axis/title margin. The floor at 0.13×n + 0.5 was
        # crowding small-n charts (n=3 came out at the floor of 0.9in,
        # only 0.13in per bar — packed tight against the title and
        # xlabel). Linear growth keeps n=10 from being absurdly tall
        # while giving n=2-3 enough breathing room.
        f.write(f"    height={0.18 * n + 0.5:.2f}in,\n")
        f.write(f"    ytick={{{ytick}}},\n")
        f.write(f"    yticklabels={{{yticklabels}}},\n")
        # Explicit ymin/ymax with half-a-unit padding above/below the
        # outer bars; `enlarge y limits` doesn't reliably propagate
        # through xbar so set the bounds directly.
        f.write("    ymin=-0.6,\n")
        f.write(f"    ymax={n - 0.4:.1f},\n")
        f.write("    nodes near coords,\n")
        f.write("    nodes near coords align={horizontal},\n")
        # Force the value labels to black so they stay readable
        # regardless of the bar's fill colour (cycle list draws bars
        # in the Paired-12 first slot — a saturated mid-blue that
        # would otherwise tint the labels too).
        f.write(
            "    nodes near coords style={"
            "font=\\sffamily\\footnotesize, text=black},\n"
        )
        f.write("    xmin=0,\n")
        f.write("    y tick label style={font=\\sffamily\\footnotesize},\n")
        f.write("  ]\n")
        f.write(
            f"    \\addplot+[xbar] table[x=latency_ms, y=idx, col sep=tab] "
            f"{{\\datadir/{fig}.tsv}};\n",
        )
        f.write("  \\end{axis}\n")
        f.write("\\end{tikzpicture}\n")
    return True


# ---------------------------------------------------------------------------
# Figure 03 — parallel scaling
#
# Tikz lives in figures/03-parallel-scaling-<scheme>.tex; this emitter
# writes the per-series TSVs only. A series surfaces in the figure only
# when it has ≥2 distinct thread points — single-point series aren't
# scaling lines.
# ---------------------------------------------------------------------------

def emit_tsvs_parallel_scaling(
    scheme: str, config: str, samples: dict, machines: dict, outdir: Path,
) -> int:
    by_series: dict[tuple, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for (sch, cfg, mid, dev, threads, cpu_quota), pts in samples.items():
        if sch != scheme or cfg != config:
            continue
        if not isinstance(threads, int):
            continue
        by_series[(mid, dev, cpu_quota)][threads].extend(pts)

    by_series = {k: v for k, v in by_series.items() if len(v) >= 2}
    cfg_slug = slugify(config)
    fig = f"03-parallel-scaling-{scheme}-{cfg_slug}"
    n = 0
    for (mid, dev, cpu_quota), by_t in by_series.items():
        # threads is the x-axis here, so the slug omits any t=N suffix.
        slug = slugify(series_label(mid, dev, None, cpu_quota, machines))
        tsv = outdir / f"{fig}-{slug}.tsv"
        with tsv.open("w") as f:
            f.write("threads\tlatency_ms\n")
            for t in sorted(by_t.keys()):
                pts = by_t[t]
                lat_ms = mean(p[0] for p in pts) / 1000.0
                f.write(f"{t}\t{lat_ms:.4f}\n")
        n += 1
    return n


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def emit_machines_table(
    machine_summary: dict, machines: dict, outdir: Path,
) -> None:
    cols = "@{}lllrrlll@{}"
    body = []
    body.append(f"\\begin{{tabular}}{{{cols}}}")
    body.append("  \\toprule")
    body.append(
        "  \\textbf{ID} & \\textbf{Name} & \\textbf{CPU} & "
        "\\textbf{Cores} & \\textbf{RAM} & \\textbf{Devices used} & "
        "\\textbf{Threads} & \\textbf{Cgroup CPU} \\\\",
    )
    body.append("  \\midrule")

    for mid in sorted(
        machine_summary,
        key=lambda m: ((machines.get(m, {}).get("machine-name") or ""), m),
    ):
        row = machines.get(mid, {})
        name = row.get("machine-name", "") or "—"
        cpu = row.get("cpu-model", "?") or "?"
        cores = row.get("cores", "?") or "?"
        ram_bytes = row.get("ram-bytes", "0") or "0"
        try:
            ram_gb = f"{int(ram_bytes) / 1024**3:.0f}\\,GB"
        except ValueError:
            ram_gb = "?"
        s = machine_summary[mid]
        # "Devices used" reflects what each run's [run-metadata.toml]
        # `device` axis recorded — NOT what the host has available.
        # sacs006 has CUDA hardware but ran every eval at --device cpu;
        # surfacing that as "GPU: cuda" was misleading on the title
        # page. This column says "cpu" / "gpu" / "cpu, gpu" based on
        # actual run usage.
        devs = sorted(s["devices"]) if s["devices"] else ["—"]
        devices_str = ", ".join(devs)
        threads = fmt_threads_range(s["threads"])
        cgroup = fmt_quota_range(s["cpu_quotas"])
        body.append(
            "  \\texttt{" + tex_escape(mid) + "} & "
            f"{tex_escape(name)} & {tex_escape(cpu)} & "
            f"{tex_escape(cores)} & {ram_gb} & "
            f"\\texttt{{{tex_escape(devices_str)}}} & "
            f"{tex_escape(threads)} & {tex_escape(cgroup)} \\\\",
        )

    body.append("  \\bottomrule")
    body.append("\\end{tabular}")
    (outdir / "machines-table.tex").write_text("\n".join(body) + "\n")


def emit_coverage_table(
    machine_summary: dict, run_counts: dict, machines: dict, outdir: Path,
) -> None:
    machine_ids = sorted(
        machine_summary,
        key=lambda m: ((machines.get(m, {}).get("machine-name") or ""), m),
    )
    schemes = sorted({s for (s, _) in run_counts})

    if not machine_ids or not schemes:
        (outdir / "coverage-table.tex").write_text(
            "\\noindent\\emph{(no runs found)}\n",
        )
        return

    cols = "@{}l" + "r" * len(machine_ids) + "@{}"
    body = []
    body.append(f"\\begin{{tabular}}{{{cols}}}")
    body.append("  \\toprule")
    head = ["  \\textbf{Scheme}"]
    for mid in machine_ids:
        name = (machines.get(mid, {}).get("machine-name") or mid)[:12]
        head.append(f"\\textbf{{{tex_escape(name)}}}")
    body.append(" & ".join(head) + " \\\\")
    body.append("  \\midrule")

    for scheme in schemes:
        cells = [f"  \\texttt{{{tex_escape(scheme)}}}"]
        for mid in machine_ids:
            by_dev = run_counts.get((scheme, mid), {})
            cpu_n = by_dev.get("cpu", 0)
            gpu_n = by_dev.get("gpu", 0)
            if not cpu_n and not gpu_n:
                cells.append("·")
            elif gpu_n and not cpu_n:
                # Pure-GPU cell: tag it explicitly because GPU is the
                # surprising case the reader needs flagged.
                cells.append(f"{gpu_n} (gpu)")
            elif cpu_n and not gpu_n:
                # CPU-only: bare count, CPU is the implicit default.
                cells.append(str(cpu_n))
            else:
                # Both: split.
                cells.append(f"{cpu_n} + {gpu_n} (gpu)")
        body.append(" & ".join(cells) + " \\\\")

    body.append("  \\bottomrule")
    body.append("\\end{tabular}")
    (outdir / "coverage-table.tex").write_text("\n".join(body) + "\n")


def emit_canonical_series(
    samples: dict, machines: dict, outdir: Path,
) -> int:
    """Auto-generate the `\\canonicalSeries` list from the (machine,
    device, threads, cgroup) tuples we actually have data for. The
    line-plot templates iterate this list via ``\\foreach`` +
    ``\\IfFileExists`` — historically it was a hand-curated file
    under ``figures/series.tex``, but adding a new machine meant a
    one-line edit no one remembered to do, so the new V100/GPU runs
    silently dropped off the cross-machine plots. Generating from
    actual data here removes that maintenance burden.

    Series order is (machine name asc, then CPU first / GPU second,
    threads asc within each pair) for stable colours across builds
    when the set of machines doesn't change.
    """
    series_keys: set[tuple] = set()
    for (_sch, _cfg, mid, dev, threads, cpu_quota) in samples.keys():
        if not mid:
            continue
        series_keys.add((mid, dev, threads, cpu_quota))

    def sort_key(key):
        mid, dev, threads, _q = key
        name = (machines.get(mid, {}).get("machine-name") or mid).lower()
        # CPU before GPU for the same machine (paired-line readability).
        device_rank = 0 if dev == "cpu" else 1
        # `None` sorts before integers; pgfplots cycle order picks up
        # `<machine>` then `<machine> t=N` consistently.
        thread_rank = -1 if threads is None else threads
        return (name, device_rank, thread_rank)

    sorted_keys = sorted(series_keys, key=sort_key)
    entries: list[str] = []
    seen_slugs: set[str] = set()
    for key in sorted_keys:
        mid, dev, threads, cpu_quota = key
        label = series_label(mid, dev, threads, cpu_quota, machines)
        slug = slugify(label)
        # Two distinct (threads, cpu_quota) tuples can collapse to the
        # same display label when the suffix-emit logic decides neither
        # is worth annotating (e.g. cpu_quota present but
        # threads-matches-cores). Dedupe so the figure's \foreach
        # doesn't draw the same series twice.
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        entries.append(f"  {slug}/{{{label}}}")

    body = "% Auto-generated by analysis/cross_machine/preprocess.py.\n"
    body += "% Format per entry: <slug>/{<display label>}.\n"
    body += "% Sort: machine-name ASC, then CPU before GPU, then threads ASC.\n"
    if entries:
        body += "\\def\\canonicalSeries{%\n"
        body += ",%\n".join(entries) + "%\n"
        body += "}\n"
    else:
        body += "\\def\\canonicalSeries{}\n"
    (outdir / "series.tex").write_text(body)
    return len(entries)


def emit_header(machine_summary: dict, run_counts: dict, outdir: Path) -> None:
    n_machines = len(machine_summary)
    n_runs = sum(s["n_runs"] for s in machine_summary.values())
    schemes = sorted({s for (s, _) in run_counts})
    schemes_str = ", ".join(schemes)
    gen = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    body = (
        "% Auto-generated by analysis/cross_machine/preprocess.py\n"
        f"\\def\\insightsSubtitle{{Machines:~{n_machines}\\quad "
        f"Runs:~{n_runs}\\quad "
        f"Schemes:~{tex_escape(schemes_str)}\\\\[0.2em]"
        f"\\normalsize Generated:~{tex_escape(gen)}}}\n"
    )
    (outdir / "header.tex").write_text(body)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Wipe stale per-figure outputs so a removed series doesn't linger.
    for prefix in ("01-latency-vs-nprobe-", "02-bar-", "03-parallel-scaling-"):
        for old in args.outdir.glob(f"{prefix}*"):
            old.unlink()

    machines = load_machines(args.results)
    samples, machine_summary, run_counts = collect_all(args.results)

    if not machine_summary:
        sys.exit("no runs found under results/runs/**/raw.csv")

    n01_tsvs = sum(
        emit_tsvs_latency_vs_nprobe(sch, samples, machines, args.outdir)
        for sch in IVF_SCHEMES
    )
    n02 = sum(
        emit_figure_machine_bars(sch, cfg, samples, machines, args.outdir)
        for sch, cfg in BAR_CONFIGS
    )
    n03_tsvs = sum(
        emit_tsvs_parallel_scaling(sch, cfg, samples, machines, args.outdir)
        for sch, cfg in SCALING_CONFIGS.items()
    )

    n_series = emit_canonical_series(samples, machines, args.outdir)
    emit_machines_table(machine_summary, machines, args.outdir)
    emit_coverage_table(machine_summary, run_counts, machines, args.outdir)
    emit_header(machine_summary, run_counts, args.outdir)

    (args.outdir / ".preprocessed").touch()
    print(
        f"wrote tables + TSVs/tikz: 01×{n01_tsvs} latency-nprobe TSVs, "
        f"02×{n02} machine-bar tikz, 03×{n03_tsvs} parallel-scaling TSVs "
        f"(line plots fed via figures/*.tex templates) — "
        f"{sum(s['n_runs'] for s in machine_summary.values())} runs across "
        f"{len(machine_summary)} machines",
    )


if __name__ == "__main__":
    main()
