#!/usr/bin/env bash
# Download MS MARCO passage corpus and dev queries.
# Usage: bash scripts/download_corpus.sh [--data-dir data/msmarco]
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data/msmarco"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$DATA_DIR"

# --- Passages (~2.9 GB compressed) ---
PASSAGES_ARCHIVE="$DATA_DIR/collection.tar.gz"
if [[ -f "$DATA_DIR/collection.tsv" ]]; then
    echo "Passages already extracted — skipping."
else
    if [[ ! -f "$PASSAGES_ARCHIVE" ]]; then
        echo "Downloading MS MARCO passage corpus (~2.9 GB compressed)…"
        curl -L --progress-bar -o "$PASSAGES_ARCHIVE" \
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
    fi
    echo "Extracting passages…"
    tar -xzf "$PASSAGES_ARCHIVE" -C "$DATA_DIR"
    echo "Extracted to $DATA_DIR/collection.tsv"
fi

# --- Dev queries (~1 MB) ---
QUERIES_FILE="$DATA_DIR/queries.dev.small.tsv"
if [[ -f "$QUERIES_FILE" ]]; then
    echo "Dev queries already present — skipping."
else
    QUERIES_ARCHIVE="$DATA_DIR/queries.tar.gz"
    if [[ ! -f "$QUERIES_ARCHIVE" ]]; then
        echo "Downloading MS MARCO dev queries (~1 MB)…"
        curl -L --progress-bar -o "$QUERIES_ARCHIVE" \
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz"
    fi
    echo "Extracting queries…"
    tar -xzf "$QUERIES_ARCHIVE" -C "$DATA_DIR"
    echo "Extracted to $QUERIES_FILE"
fi

echo "Done."
