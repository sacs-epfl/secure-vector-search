pub mod meta;
pub mod progress;
pub mod tiptoe_go_build;

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// .fvecs / .ivecs I/O  (FAISS convention, little-endian)
// ---------------------------------------------------------------------------

/// Load a `.fvecs` file. Each vector is stored as `[dim: u32][f32 × dim]`.
pub fn load_fvecs(path: &Path) -> io::Result<Vec<Vec<f32>>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut vecs = Vec::new();
    loop {
        let mut dim_buf = [0u8; 4];
        match r.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let mut data = vec![0u8; dim * 4];
        r.read_exact(&mut data)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        vecs.push(floats);
    }
    Ok(vecs)
}

/// Load a `.ivecs` file. Each vector is stored as `[dim: u32][i32 × dim]`.
pub fn load_ivecs(path: &Path) -> io::Result<Vec<Vec<u32>>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut vecs = Vec::new();
    loop {
        let mut dim_buf = [0u8; 4];
        match r.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let mut data = vec![0u8; dim * 4];
        r.read_exact(&mut data)?;
        let ids: Vec<u32> = data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32)
            .collect();
        vecs.push(ids);
    }
    Ok(vecs)
}

/// Write a `.ivecs` file from a list of id lists.
pub fn save_ivecs(path: &Path, vecs: &[Vec<u32>]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for ids in vecs {
        let dim = ids.len() as u32;
        w.write_all(&dim.to_le_bytes())?;
        for &id in ids {
            w.write_all(&(id as i32).to_le_bytes())?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Recall
// ---------------------------------------------------------------------------

/// Fraction of `returned` IDs that appear in `ground_truth[..k]`.
/// Both slices may be shorter than `k`; the shorter one sets the bound.
pub fn recall_at_k(ground_truth: &[u32], returned: &[u32], k: usize) -> f64 {
    let gt: std::collections::HashSet<u32> = ground_truth[..k.min(ground_truth.len())]
        .iter()
        .copied()
        .collect();
    let hits = returned[..k.min(returned.len())]
        .iter()
        .filter(|id| gt.contains(id))
        .count();
    hits as f64 / k.min(ground_truth.len()) as f64
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/// One row of evaluation output.
pub struct CsvRow {
    pub run_id: String,
    pub scheme: String,
    pub quality_param_name: String,
    pub quality_param: f64,
    pub config_label: String,
    pub k: u32,
    pub query_id: u32,
    pub recall_at_k: f64,
    /// Real per-query latency in microseconds.
    /// `Some(_)` only when `batch_size = 1`; `None` for batched rows
    /// (the per-query latency inside a batch isn't separately
    /// measurable, so we emit an empty cell). preprocess.py backfills
    /// CSVs lacking this column by filling `latency_us` on load.
    /// Figure 02 / figure 04 (recall-vs-throughput, latency CDF)
    /// filter on `batch_size = 1` so the column keeps a single meaning.
    pub latency_us: Option<u64>,
    pub query_bytes: u64,
    pub response_bytes: u64,
    pub cluster_response_bytes: u64,
    pub setup_bytes: u64,
    /// Per-query offline upload — what the client sends to the server
    /// in the offline phase. 0 for every scheme except Tiptoe (whose
    /// BFV-encrypted query token is per-query).
    pub pre_query_offline_up_bytes: u64,
    /// Per-query offline download — what the server returns in the
    /// offline phase. 0 for every scheme except Tiptoe (whose
    /// apply-hint result is BFV-encrypted h_s, ~0.5 MB/query).
    pub pre_query_offline_down_bytes: u64,
    /// Per-query Protocol 2 (Freivalds) verification cost, BN-only.
    /// Set to 0 for scorers without a verification step (Plaintext,
    /// SAP/SAP+IVF, EMVP/EMVP+IVF, Tiptoe).
    pub verification_overhead_us: u64,
    pub machine_id: String,
    /// Substrate this run measured against. `"cpu"` or `"gpu"`.
    /// Empty / unknown rows in legacy CSVs default to `"cpu"` in
    /// `analysis/preprocess.py`.
    pub device: String,
    /// Analytical proxy for GPU bandwidth-bound throughput (matrix
    /// elements touched × element size in bytes per query). Computable
    /// without a GPU run; the per-scheme formula differs by scheme.
    /// Mirrors `CommunicationCost::effective_bytes_per_query`.
    pub effective_bytes_per_query: u64,
    /// Batch size `B` requested for this row's chunk. Uniformly
    /// emitted across all schemes; rows from default-impl scorers also
    /// carry the column (set to whatever `B` the harness invoked — the
    /// value reflects intent, not whether the scorer actually fused
    /// the batched work).
    pub batch_size: u32,
    /// Wall-clock of the `score_batch` call covering this row's chunk.
    /// Populated for every row; at `batch_size = 1` it equals
    /// `latency_us`. Canonical column for raw-throughput accounting at
    /// the chunk granularity.
    pub wallclock_us: u64,
    /// `wallclock_us / batch_size`, per-query latency amortised across
    /// the batched chunk. Canonical column for figure 14's x-axis.
    pub amortised_latency_us: u64,
}

pub fn write_csv_header(w: &mut impl Write) -> io::Result<()> {
    writeln!(
        w,
        "run-id,scheme,quality-param-name,quality-param,config-label,k,query-id,\
         recall-at-k,latency-us,query-bytes,response-bytes,cluster-response-bytes,setup-bytes,\
         pre-query-offline-up-bytes,pre-query-offline-down-bytes,verification-overhead-us,\
         machine-id,device,effective-bytes-per-query,batch-size,wallclock-us,amortised-latency-us"
    )
}

/// One row of the per-query top-k sidecar `top_k.csv`. Written
/// alongside `raw.csv` so the validation gate's `tiptoe_diff.py` can
/// compute per-query top-10 ID overlap.
pub struct TopKRow<'a> {
    pub config_label: &'a str,
    pub query_id: u32,
    pub rank: u32,
    pub doc_id: u32,
    pub score: f32,
}

pub fn write_top_k_header(w: &mut impl Write) -> io::Result<()> {
    writeln!(w, "config-label,query-id,rank,doc-id,score")
}

pub fn write_top_k_row(w: &mut impl Write, row: &TopKRow<'_>) -> io::Result<()> {
    writeln!(
        w,
        "{},{},{},{},{:.6}",
        row.config_label, row.query_id, row.rank, row.doc_id, row.score,
    )
}

/// One row of `substep-breakdown.csv`.
///
/// Long-format: one row per `(scheme, config-label, query-id, substep)`;
/// the seven canonical substep names — `route`, `encode`,
/// `server-compute`, `verify`, `decompress`, `decode`, `merge` — are
/// emitted for every query so the schema is uniform across schemes.
/// Substeps that don't apply to a given scheme write `us = 0`.
pub struct SubstepRow<'a> {
    pub run_id: &'a str,
    pub scheme: &'a str,
    pub quality_param_name: &'a str,
    pub quality_param: f64,
    pub config_label: &'a str,
    pub query_id: u32,
    pub substep: &'a str,
    pub us: u64,
    pub machine_id: &'a str,
}

/// Canonical substep names in figure-row order. Stable across schemes
/// so figure 09 stacks the same colour bands by index.
pub const CANONICAL_SUBSTEPS: [&str; 7] = [
    "route",
    "encode",
    "server-compute",
    "verify",
    "decompress",
    "decode",
    "merge",
];

pub fn write_substep_breakdown_header(w: &mut impl Write) -> io::Result<()> {
    writeln!(
        w,
        "run-id,scheme,quality-param-name,quality-param,config-label,query-id,substep,us,machine-id"
    )
}

pub fn write_substep_breakdown_row(w: &mut impl Write, row: &SubstepRow<'_>) -> io::Result<()> {
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{}",
        row.run_id,
        row.scheme,
        row.quality_param_name,
        row.quality_param,
        row.config_label,
        row.query_id,
        row.substep,
        row.us,
        row.machine_id,
    )
}

pub fn write_csv_row(w: &mut impl Write, row: &CsvRow) -> io::Result<()> {
    // `latency-us` is empty for batched rows. Empty cell is a natural
    // pandas/csv convention for "this measurement is not separately
    // observable at this granularity"; preprocess.py reads it as NaN
    // and figure-02/figure-04 filter on batch-size = 1.
    let latency_us_cell: String = match row.latency_us {
        Some(v) => v.to_string(),
        None => String::new(),
    };
    writeln!(
        w,
        "{},{},{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        row.run_id,
        row.scheme,
        row.quality_param_name,
        row.quality_param,
        row.config_label,
        row.k,
        row.query_id,
        row.recall_at_k,
        latency_us_cell,
        row.query_bytes,
        row.response_bytes,
        row.cluster_response_bytes,
        row.setup_bytes,
        row.pre_query_offline_up_bytes,
        row.pre_query_offline_down_bytes,
        row.verification_overhead_us,
        row.machine_id,
        row.device,
        row.effective_bytes_per_query,
        row.batch_size,
        row.wallclock_us,
        row.amortised_latency_us,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect() {
        assert_eq!(recall_at_k(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], 5), 1.0);
    }

    #[test]
    fn recall_zero() {
        assert_eq!(recall_at_k(&[0, 1, 2], &[3, 4, 5], 3), 0.0);
    }

    #[test]
    fn recall_partial() {
        // 2 of 4 ground-truth found
        assert_eq!(recall_at_k(&[0, 1, 2, 3], &[0, 1, 5, 6], 4), 0.5);
    }

    #[test]
    fn recall_k_truncates() {
        // Only first 3 of ground-truth count
        assert_eq!(recall_at_k(&[0, 1, 2, 3, 4], &[0, 1, 2], 3), 1.0);
    }

    /// Header carries the three batch columns in the documented
    /// position (appended after `effective-bytes-per-query`) so
    /// preprocess.py's column-by-name access path is stable.
    #[test]
    fn raw_csv_header_carries_batch_columns() {
        let mut buf = Vec::new();
        write_csv_header(&mut buf).unwrap();
        let header = String::from_utf8(buf).unwrap();
        assert!(
            header
                .trim_end()
                .ends_with(",batch-size,wallclock-us,amortised-latency-us"),
            "header missing the W3 columns: {header}"
        );
    }

    /// Row emission shape under both `batch_size = 1` (latency-us
    /// populated, equal to wallclock) and `batch_size > 1` (latency-us
    /// empty cell). The empty-cell convention is what preprocess.py
    /// reads as NaN.
    fn make_row(latency_us: Option<u64>, batch_size: u32, wallclock_us: u64) -> CsvRow {
        CsvRow {
            run_id: "r".into(),
            scheme: "s".into(),
            quality_param_name: "q".into(),
            quality_param: 0.0,
            config_label: "c".into(),
            k: 10,
            query_id: 0,
            recall_at_k: 1.0,
            latency_us,
            query_bytes: 0,
            response_bytes: 0,
            cluster_response_bytes: 0,
            setup_bytes: 0,
            pre_query_offline_up_bytes: 0,
            pre_query_offline_down_bytes: 0,
            verification_overhead_us: 0,
            machine_id: "m".into(),
            device: "cpu".into(),
            effective_bytes_per_query: 0,
            batch_size,
            wallclock_us,
            amortised_latency_us: wallclock_us / batch_size as u64,
        }
    }

    #[test]
    fn raw_csv_row_b1_populates_latency() {
        let mut buf = Vec::new();
        write_csv_row(&mut buf, &make_row(Some(123), 1, 123)).unwrap();
        let line = String::from_utf8(buf).unwrap();
        assert!(
            line.trim_end().ends_with(",1,123,123"),
            "B=1 tail wrong: {line}"
        );
        // 22 columns → 21 commas; spot-check the latency-us cell is "123", not empty.
        let cells: Vec<&str> = line.trim_end().split(',').collect();
        assert_eq!(cells.len(), 22, "expected 22 cells");
        assert_eq!(cells[8], "123", "latency-us cell wrong");
    }

    #[test]
    fn raw_csv_row_b_gt_1_leaves_latency_empty() {
        let mut buf = Vec::new();
        write_csv_row(&mut buf, &make_row(None, 8, 4000)).unwrap();
        let line = String::from_utf8(buf).unwrap();
        let cells: Vec<&str> = line.trim_end().split(',').collect();
        assert_eq!(cells.len(), 22, "expected 22 cells");
        assert_eq!(cells[8], "", "batched row must leave latency-us empty");
        assert_eq!(cells[19], "8", "batch-size cell wrong");
        assert_eq!(cells[20], "4000", "wallclock-us cell wrong");
        assert_eq!(cells[21], "500", "amortised-latency-us cell wrong (4000/8)");
    }
}
