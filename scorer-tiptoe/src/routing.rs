//! Client-side routing and augmented-query construction.
//!
//! Tiptoe's routing is a plaintext centroid scan: the client holds a
//! cleartext copy of the IVF centroid table and picks the cluster
//! closest to its query. This is **not** a PIR round; privacy at this
//! step comes from the augmented-query construction at the ranking step
//! (the encrypted query is zero-padded across all clusters except the
//! chosen one, so the server's encrypted view is identical to scanning
//! every cluster).
//!
//! The augmented query `q̃ ∈ Z_p^{n_centroids·d}` lives in the same
//! column basis as the encoded matrix `M`: zero in every cluster's
//! `d`-block except the chosen one, which holds the encoded query.
//! See `cluster.rs` for the matrix layout.

use ivf_index::distance::l2_f32;
use ivf_index::kmeans::Centroid;

use crate::cluster::EncodedClusters;
use crate::encoding::{EncodingError, encode_vector};

/// Pick the cluster nearest `query` by L2 distance over the centroid
/// table. Returns the cluster index in `[0, centroids.len())`. Panics
/// on an empty centroid table — callers should validate `n_centroids ≥ 1`
/// at config time.
pub fn route(query: &[f32], centroids: &[Centroid]) -> usize {
    assert!(!centroids.is_empty(), "route called with no centroids");
    let mut best_idx = 0usize;
    let mut best_d = l2_f32(query, &centroids[0]);
    for (i, c) in centroids.iter().enumerate().skip(1) {
        let d = l2_f32(query, c);
        if d < best_d {
            best_d = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Build the augmented query `q̃ ∈ Z_p^{n_centroids·d}`: zero except in
/// the `cluster_idx`-th `d`-block, which holds the encoded `query`.
///
/// This is the "augmented" form: the matrix-vector product `M · q̃`
/// collapses to inner products against the chosen cluster's vectors,
/// but the wire shape is independent of `cluster_idx` so the server
/// learns nothing about the routing outcome from a single query.
pub fn build_augmented_query(
    query: &[f32],
    cluster_idx: usize,
    encoded: &EncodedClusters,
    quantisation_bits: u8,
    p: u64,
) -> Result<Vec<u64>, EncodingError> {
    debug_assert_eq!(query.len(), encoded.d());
    debug_assert!(cluster_idx < encoded.n_centroids());
    let mut q_tilde = vec![0u64; encoded.width()];
    let col_start = encoded.cluster_col_start(cluster_idx);
    let block = &mut q_tilde[col_start..col_start + encoded.d()];
    encode_vector(query, quantisation_bits, p, block)?;
    Ok(q_tilde)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::encode_ivf;
    use crate::pir::lwe::Params;
    use ivf_index::ivf::IvfIndex;

    fn handcrafted_index(
        centroids: Vec<Centroid>,
        clusters: Vec<Vec<(u32, Vec<f32>)>>,
    ) -> IvfIndex {
        IvfIndex {
            centroids,
            clusters,
        }
    }

    #[test]
    fn route_picks_nearest_centroid() {
        // Three centroids on the x-axis at -1, 0, +1. Query 0.4 → idx 1.
        let centroids: Vec<Centroid> = vec![vec![-1.0], vec![0.0], vec![1.0]];
        assert_eq!(route(&[0.4], &centroids), 1);
        assert_eq!(route(&[0.6], &centroids), 2);
        assert_eq!(route(&[-0.9], &centroids), 0);
    }

    #[test]
    fn route_breaks_ties_to_lowest_index() {
        // Equidistant from centroids 0 and 1 → first-seen wins.
        let centroids: Vec<Centroid> = vec![vec![-1.0], vec![1.0]];
        assert_eq!(route(&[0.0], &centroids), 0);
    }

    #[test]
    fn build_augmented_query_is_zero_outside_chosen_block() {
        // Two clusters at d=2. q̃ length = 4; only one d-block populated.
        let d = 2;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; 2];
        let clusters = vec![vec![(0u32, vec![0.0, 0.0])], vec![(1u32, vec![0.0, 0.0])]];
        let index = handcrafted_index(centroids, clusters);
        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();

        let q = vec![1.0, -1.0];
        let q_tilde = build_augmented_query(&q, 1, &enc, 4, p).unwrap();
        assert_eq!(q_tilde.len(), 4);
        assert_eq!(q_tilde[0], 0);
        assert_eq!(q_tilde[1], 0);
        // Block at cluster 1: cols 2..4, holding (mag, p-mag).
        let mag = crate::encoding::max_magnitude(4) as u64;
        assert_eq!(q_tilde[2], mag);
        assert_eq!(q_tilde[3], p - mag);
    }

    #[test]
    fn build_augmented_query_for_each_cluster_only_changes_one_block() {
        let d = 3;
        let n = 4;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; n];
        let clusters = vec![vec![(0u32, vec![0.0; d])]; n];
        let index = handcrafted_index(centroids, clusters);
        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();

        let q = vec![0.5, -0.5, 0.25];
        for chosen in 0..n {
            let q_tilde = build_augmented_query(&q, chosen, &enc, 4, p).unwrap();
            for c in 0..n {
                let block = &q_tilde[c * d..(c + 1) * d];
                if c == chosen {
                    assert!(
                        block.iter().any(|&v| v != 0),
                        "chosen block must be populated"
                    );
                } else {
                    assert!(
                        block.iter().all(|&v| v == 0),
                        "non-chosen block must be zero"
                    );
                }
            }
        }
    }
}
