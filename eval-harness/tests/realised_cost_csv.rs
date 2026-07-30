//! End-to-end check that an IVF scorer's realised per-query communication
//! cost survives serialisation into the CSV. When two queries route to
//! differently-sized clusters on a corpus with uneven cluster sizes, the
//! `response-bytes` column written by `write_csv_row` must differ between
//! their rows. The per-crate tests in `scorer-emvp` already cover the
//! `score_with_realised_cost` override itself; this test covers the CSV
//! plumbing around it.

use eval_harness::{CsvRow, write_csv_header, write_csv_row};
use scorer_core::{Device, Scorer, Vector};
use scorer_emvp::{EmvpIvfConfig, EmvpIvfScorer};

#[tokio::test]
async fn eval_csv_realised_bytes_vary_across_queries() {
    // Corpus rigged for k-means cluster sizes [60, 20, 15, 5].
    let mut vectors = Vec::new();
    for _ in 0..60 {
        vectors.push(Vector(vec![0.0]));
    }
    for _ in 0..20 {
        vectors.push(Vector(vec![10.0]));
    }
    for _ in 0..15 {
        vectors.push(Vector(vec![20.0]));
    }
    for _ in 0..5 {
        vectors.push(Vector(vec![30.0]));
    }

    let scorer = EmvpIvfScorer::new();
    let config = EmvpIvfConfig {
        key_seed: [42u8; 32],
        n_centroids: 4,
        nprobe: 2,
        train_seed: 42,
        max_iter: 25,
        upload_seed: 7,
        progress: None,
        device: Device::Cpu,
        vram_budget_bytes: None,
    };
    let (handle, _) = scorer.upload_cluster(&config, &vectors).await.unwrap();

    let queries = [Vector(vec![0.0]), Vector(vec![30.0])];
    let mut buf: Vec<u8> = Vec::new();
    write_csv_header(&mut buf).unwrap();
    for (qi, query) in queries.iter().enumerate() {
        let (_, cost) = scorer
            .score_with_realised_cost(&handle, query, 10)
            .await
            .unwrap();
        write_csv_row(
            &mut buf,
            &CsvRow {
                run_id: "test".to_string(),
                scheme: "emvp-ivf".to_string(),
                quality_param_name: "nprobe".to_string(),
                quality_param: 2.0,
                config_label: "nprobe=2".to_string(),
                k: 10,
                query_id: qi as u32,
                recall_at_k: 0.0,
                latency_us: Some(0),
                query_bytes: cost.query_bytes,
                response_bytes: cost.response_bytes,
                cluster_response_bytes: cost.cluster_response_bytes,
                setup_bytes: cost.setup_bytes,
                pre_query_offline_up_bytes: cost.pre_query_offline_up_bytes,
                pre_query_offline_down_bytes: cost.pre_query_offline_down_bytes,
                verification_overhead_us: 0,
                machine_id: "test".to_string(),
                device: "cpu".to_string(),
                effective_bytes_per_query: cost.effective_bytes_per_query,
                batch_size: 1,
                wallclock_us: 0,
                amortised_latency_us: 0,
            },
        )
        .unwrap();
    }

    let csv = String::from_utf8(buf).unwrap();
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "expected exactly 2 data rows");

    // response-bytes is column index 10 (0-based); see
    // eval_harness::write_csv_header.
    let resp_bytes = |row: &str| -> u64 { row.split(',').nth(10).unwrap().parse().unwrap() };
    let r0 = resp_bytes(rows[0]);
    let r1 = resp_bytes(rows[1]);
    assert_ne!(
        r0, r1,
        "two queries on a skewed corpus must produce different \
         response-bytes; got {r0} and {r1}"
    );
}
