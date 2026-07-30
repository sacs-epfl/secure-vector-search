#!/usr/bin/env python3
"""
Embed passages and queries with intfloat/e5-base-v2 and write .fvecs files.

IMPORTANT — E5 prefix requirement:
  E5 models require task-specific prefixes for correct retrieval quality.
  This script prepends "passage: " to every corpus passage and "query: " to
  every query before encoding. Omitting these prefixes silently degrades recall.

Checkpointing:
  Progress is saved to data/*.fvecs.ckpt.npz every CHECKPOINT_INTERVAL batches.
  If the script is interrupted and re-run, it resumes from the last checkpoint.
  Checkpoint files are deleted automatically on successful completion.

Inputs:
  data/passages_<N>.jsonl    (from subsample_corpus.py; pick with --input-jsonl;
                              defaults to passages_100k.jsonl for back-compat)
  MS MARCO dev queries       (hard-coded path; adjust if needed)

Outputs:
  data/passages.fvecs        (N × 768 float32)
  data/queries.fvecs         (1k × 768 float32, uniformly subsampled with seed 42)

Run from repo root:
  python scripts/embed_corpus.py [--device cpu|cuda|mps]
                                 [--input-jsonl passages_<N>.jsonl]
"""
import argparse
import csv
import datetime
import json
import math
import os
import platform
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "intfloat/e5-base-v2"
SEED = 42
N_QUERIES = 1_000
BATCH_SIZE = 256
CHECKPOINT_INTERVAL = 10  # save checkpoint every N batches (~2560 texts)


def write_fvecs(path: Path, vecs: np.ndarray) -> None:
    """Write a numpy array (N, D) as .fvecs (FAISS convention, little-endian)."""
    n, d = vecs.shape
    with open(path, "wb") as f:
        for row in vecs:
            f.write(struct.pack("<I", d))
            f.write(row.astype(np.float32).tobytes())
    print(f"Wrote {n} × {d} vectors to {path}")


def embed_with_checkpoint(
    model: SentenceTransformer,
    texts: list[str],
    output_path: Path,
    desc: str,
    device: str,
) -> dict | None:
    """
    Encode `texts` in batches, checkpointing every CHECKPOINT_INTERVAL batches.
    Resumes from an existing checkpoint if one is found next to `output_path`.
    Deletes the checkpoint file on successful completion.

    Returns timing metadata for the just-finished encode pass, or None when the
    output already existed and no work was done.
    """
    checkpoint_path = output_path.with_suffix(".fvecs.ckpt.npz")

    # Skip entirely if final output already exists.
    if output_path.exists():
        print(f"  {output_path.name} already exists — skipping {desc}.")
        return None

    n = len(texts)
    n_batches = math.ceil(n / BATCH_SIZE)

    # --- resume from checkpoint if available ---
    start_batch = 0
    collected: list[np.ndarray] = []
    is_resumed = checkpoint_path.exists()
    if is_resumed:
        ckpt = np.load(checkpoint_path)
        collected = [ckpt["embeddings"]]
        start_batch = int(ckpt["next_batch"])
        n_done = collected[0].shape[0]
        print(
            f"  Resuming {desc} from batch {start_batch}/{n_batches} "
            f"({n_done:,}/{n:,} texts already encoded)."
        )
    else:
        print(f"Encoding {n:,} texts ({desc})…", flush=True)

    started_at_unix = time.time()
    started_monotonic = time.monotonic()

    # --- encode remaining batches ---
    for i in tqdm(range(start_batch, n_batches), initial=start_batch, total=n_batches):
        batch = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        collected.append(np.array(vecs, dtype=np.float32))

        # Periodically consolidate and save checkpoint.
        if (i + 1) % CHECKPOINT_INTERVAL == 0 or i == n_batches - 1:
            combined = np.concatenate(collected, axis=0)
            np.savez(checkpoint_path, embeddings=combined, next_batch=np.array(i + 1))
            collected = [combined]

    elapsed_secs = time.monotonic() - started_monotonic

    result = collected[0] if len(collected) == 1 else np.concatenate(collected, axis=0)

    # Write final output and clean up checkpoint.
    write_fvecs(output_path, result)
    checkpoint_path.unlink(missing_ok=True)

    return {
        "n": n,
        "started_at_unix": started_at_unix,
        "elapsed_secs": elapsed_secs,
        "is_resumed": is_resumed,
    }


def _collect_machine_info() -> tuple[str, int, int]:
    """Return (cpu_model, logical_cores, ram_bytes) — same fields the Rust
    eval harness FNV-1a-hashes into machine-id, so the two CSVs join."""
    cpu_model = ""
    cpu_cores = os.cpu_count() or 0
    ram_bytes = 0

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_bytes = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    elif platform.system() == "Darwin":
        import subprocess
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
        try:
            ram_bytes = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
        except Exception:
            pass

    return cpu_model, cpu_cores, ram_bytes


def _machine_id(cpu_model: str, cores: int, ram_bytes: int) -> str:
    """64-bit FNV-1a over '{cpu_model}\\n{cores}\\n{ram_bytes}', first 8 hex chars.
    Mirrors eval-harness/src/meta.rs so this CSV joins on results/machines.csv."""
    key = f"{cpu_model}\n{cores}\n{ram_bytes}".encode()
    h = 14695981039346656037
    mask = 0xFFFFFFFFFFFFFFFF
    for b in key:
        h ^= b
        h = (h * 1099511628211) & mask
    return f"{h:016x}"[:8]


def _gpu_name(device: str) -> str:
    if device == "cuda":
        try:
            import torch
            return torch.cuda.get_device_name(0)
        except Exception:
            return ""
    if device == "mps":
        return f"Apple {platform.machine()}"
    return ""


def log_embed_timing(timings_path: Path, corpus: str, info: dict, device: str) -> None:
    cpu_model, cores, ram_bytes = _collect_machine_info()
    row = {
        "started-at": datetime.datetime.fromtimestamp(
            info["started_at_unix"], tz=datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine-id": _machine_id(cpu_model, cores, ram_bytes),
        "cpu-model": cpu_model,
        "cores": cores,
        "ram-bytes": ram_bytes,
        "gpu-name": _gpu_name(device),
        "device": device,
        "model": MODEL_NAME,
        "corpus": corpus,
        "n": info["n"],
        "batch-size": BATCH_SIZE,
        "elapsed-secs": f"{info['elapsed_secs']:.3f}",
        "is-resumed": "true" if info["is_resumed"] else "false",
    }
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not timings_path.exists()
    with open(timings_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(
        f"  Logged {corpus} timing to {timings_path}: "
        f"{info['elapsed_secs']:.1f}s on {device}"
    )


def load_queries(data_dir: Path) -> list[str]:
    """
    Load MS MARCO dev queries. Tries the standard v1 and v2 locations.
    Returns a uniformly subsampled list of N_QUERIES queries with seed SEED.
    """
    candidates = [
        data_dir / "queries.dev.small.tsv",
        data_dir / "msmarco-test2019-queries.tsv",
        data_dir / "queries.dev.tsv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        sys.exit(
            "Dev queries not found. Expected one of:\n"
            + "\n".join(f"  {p}" for p in candidates)
        )
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                queries.append(parts[1])
    rng = random.Random(SEED)
    if len(queries) > N_QUERIES:
        queries = rng.sample(queries, N_QUERIES)
    print(f"Loaded {len(queries)} dev queries from {path}")
    return queries


def _detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "msmarco",
    )
    parser.add_argument(
        "--timings-csv",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "embed-timings.csv",
        help="CSV to append per-corpus embedding wall-clock timings to.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="passages_100k.jsonl",
        help="Filename (under --data-dir) of the passages JSONL to embed.",
    )
    args = parser.parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    device = _detect_device() if args.device == "auto" else args.device
    if args.device == "auto":
        print(f"Device: auto-detected → {device}")

    print(f"Loading model {MODEL_NAME} on {device}…")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # --- Passages ---
    passages_file = data_dir / args.input_jsonl
    if not passages_file.exists():
        sys.exit(f"{passages_file} not found. Run subsample_corpus.py first.")

    print("Loading passages…")
    raw_passages = []
    with open(passages_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            raw_passages.append(obj["passage"])

    # E5 requires "passage: " prefix for corpus documents.
    prefixed_passages = [f"passage: {p}" for p in raw_passages]
    info = embed_with_checkpoint(
        model, prefixed_passages, data_dir / "passages.fvecs", "passages", device
    )
    if info is not None:
        log_embed_timing(args.timings_csv, "passages", info, device)

    # --- Queries ---
    raw_queries = load_queries(data_dir)
    # E5 requires "query: " prefix for queries.
    prefixed_queries = [f"query: {q}" for q in raw_queries]
    info = embed_with_checkpoint(
        model, prefixed_queries, data_dir / "queries.fvecs", "queries", device
    )
    if info is not None:
        log_embed_timing(args.timings_csv, "queries", info, device)

    print("Done.")


if __name__ == "__main__":
    main()
