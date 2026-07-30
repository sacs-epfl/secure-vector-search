#!/usr/bin/env python3
"""
Subsample N passages from the MS MARCO passage corpus with seed 42.

Input:  <data-dir>/collection.tsv  (pid TAB passage, one per line)
Output: <data-dir>/passages_<N>.jsonl  ({"pid": ..., "passage": ...} per line)

Run from repo root:
  python scripts/subsample_corpus.py [--n N] [--data-dir data/msmarco]

Default N is 100_000. When N is greater than or equal to the observed
corpus size the reservoir is bypassed and passages are streamed directly
to the output (avoids a multi-GB in-memory reservoir at full-corpus
sizes).
"""
import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42
DEFAULT_N = 100_000


def iter_passages(path: Path):
    """Yield (pid, passage_text) from the MS MARCO TSV collection."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                yield parts[0], parts[1]


def count_passages(path: Path) -> int:
    n = 0
    for _ in iter_passages(path):
        n += 1
    return n


def stream_all(input_path: Path, output_path: Path) -> int:
    written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for pid, text in iter_passages(input_path):
            out.write(json.dumps({"pid": pid, "passage": text}, ensure_ascii=False) + "\n")
            written += 1
            if written % 1_000_000 == 0:
                print(f"  streamed {written:,} passages…", flush=True)
    return written


def reservoir_sample(input_path: Path, output_path: Path, n: int) -> int:
    rng = random.Random(SEED)
    reservoir = []
    i = -1
    for i, (pid, text) in enumerate(iter_passages(input_path)):
        if i < n:
            reservoir.append({"pid": pid, "passage": text})
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = {"pid": pid, "passage": text}
        if (i + 1) % 1_000_000 == 0:
            print(f"  processed {i + 1:,} passages…", flush=True)
    if i < 0:
        sys.exit("No passages found — is collection.tsv empty?")
    print(f"Sampled {len(reservoir):,} passages from {i + 1:,} total.")
    with open(output_path, "w", encoding="utf-8") as f:
        for obj in reservoir:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(reservoir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "msmarco",
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()

    input_path = args.data_dir / "collection.tsv"
    output_path = args.data_dir / f"passages_{args.n}.jsonl"

    if not input_path.exists():
        sys.exit(f"{input_path} not found. Run download_corpus.sh first.")
    if args.n <= 0:
        sys.exit(f"--n must be positive (got {args.n}).")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Counting passages in {input_path}…", flush=True)
    total = count_passages(input_path)
    if total == 0:
        sys.exit("No passages found — is collection.tsv empty?")
    print(f"Observed {total:,} passages.")

    if args.n >= total:
        print(f"N={args.n:,} ≥ corpus size; streaming all passages directly.", flush=True)
        written = stream_all(input_path, output_path)
    else:
        print(f"Reservoir-sampling {args.n:,} passages from {total:,}…", flush=True)
        written = reservoir_sample(input_path, output_path, args.n)

    print(f"Written {written:,} passages to {output_path}")


if __name__ == "__main__":
    main()
