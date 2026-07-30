//! IVF wrapping and Tiptoe matrix `M ∈ Z_p^{m_max × (n_centroids·d)}`
//! assembly.
//!
//! Uses a single-bin simplification: the shared IVF (no recursive
//! split, no double-assignment) keeps cluster sizes tight enough that
//! bin-packing offers no real benefit at our 100k corpus, so we lay
//! each cluster into a contiguous `d`-column block of `M` and zero-pad
//! rows to `m_max`.
//!
//! Layout (row-major):
//!
//! - rows = `m_max` (the largest IVF cluster size)
//! - cols = `n_centroids · d`
//! - Cluster `c`'s vector at local index `r` occupies columns
//!   `[c·d, (c+1)·d)` in row `r`. Cells outside that block are zero
//!   unless another cluster fills them at the same row.
//! - The augmented query `q̃ ∈ Z_p^{n_centroids·d}` is zero except in
//!   block `[i*·d, (i*+1)·d)`, which holds the encoded query (built
//!   in `routing.rs` once the centroid scan picks `i*`).

use ivf_index::ivf::IvfIndex;
use scorer_core::VectorId;

use crate::encoding::{EncodingError, encode_vector};

/// Server-side encoded state for a Tiptoe cluster index. Holds the
/// `Z_p` matrix `M` plus the per-cluster local-index → global
/// `VectorId` map needed to translate top-`k` results back to corpus
/// IDs after decoding.
pub struct EncodedClusters {
    matrix: Vec<u64>,
    m_max: usize,
    d: usize,
    n_centroids: usize,
    cluster_membership: Vec<Vec<VectorId>>,
}

impl EncodedClusters {
    /// Read-only access to the row-major `m_max × (n_centroids·d)`
    /// matrix. Server consumes this as the SimplePIR database.
    #[inline]
    pub fn matrix(&self) -> &[u64] {
        &self.matrix
    }

    /// Number of rows in `M` (= largest IVF cluster size).
    #[inline]
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// Embedding dimension.
    #[inline]
    pub fn d(&self) -> usize {
        self.d
    }

    /// Number of IVF clusters.
    #[inline]
    pub fn n_centroids(&self) -> usize {
        self.n_centroids
    }

    /// Width of `M` (= `n_centroids · d`); matches the augmented-query
    /// length `q̃ ∈ Z_p^{n_centroids·d}`.
    #[inline]
    pub fn width(&self) -> usize {
        self.n_centroids * self.d
    }

    /// Column offset where cluster `c`'s `d`-block starts in `M` and in
    /// the augmented query. `c < n_centroids`.
    #[inline]
    pub fn cluster_col_start(&self, c: usize) -> usize {
        debug_assert!(c < self.n_centroids);
        c * self.d
    }

    /// Local-index → global `VectorId` map for cluster `c`. The first
    /// `cluster_membership(c).len()` entries of the recovered
    /// `(M·q̃)[c-block]` row give the inner-product scores for these
    /// IDs in order; remaining rows in `[len, m_max)` are zero-padding.
    #[inline]
    pub fn cluster_membership(&self, c: usize) -> &[VectorId] {
        &self.cluster_membership[c]
    }
}

/// Encode an `IvfIndex` into a packed Tiptoe `Z_p` matrix.
///
/// Returns the assembled matrix plus per-cluster membership tables.
/// `quantisation_bits` and `p` come from `TiptoeConfig`; caller must
/// have validated them via [`crate::encoding::validate_params`].
///
/// Empty clusters are permitted (they contribute zero rows but reserve
/// their `d`-column block); an empty index (`n_centroids == 0`)
/// produces an empty matrix.
pub fn encode_ivf(
    index: &IvfIndex,
    quantisation_bits: u8,
    p: u64,
) -> Result<EncodedClusters, EncodingError> {
    let n_centroids = index.clusters.len();
    let d = index.centroids.first().map_or(0, |c| c.len());
    let m_max = index.clusters.iter().map(Vec::len).max().unwrap_or(0);
    let width = n_centroids * d;
    let mut matrix = vec![0u64; m_max * width];
    let mut cluster_membership: Vec<Vec<VectorId>> = vec![Vec::new(); n_centroids];

    for (c, cluster) in index.clusters.iter().enumerate() {
        let col_start = c * d;
        let membership = &mut cluster_membership[c];
        membership.reserve_exact(cluster.len());
        for (r, (id, vec)) in cluster.iter().enumerate() {
            membership.push(*id);
            let row_offset = r * width + col_start;
            encode_vector(
                vec,
                quantisation_bits,
                p,
                &mut matrix[row_offset..row_offset + d],
            )?;
        }
    }

    Ok(EncodedClusters {
        matrix,
        m_max,
        d,
        n_centroids,
        cluster_membership,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{dequantise_component, max_magnitude};
    use crate::pir::lwe::Params;
    use ivf_index::kmeans::Centroid;

    /// Build a tiny IvfIndex by hand. Each cluster is a Vec of (id, vec).
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
    fn shape_matches_layout() {
        // 3 clusters with sizes (2, 1, 3) at d=4 → m_max=3, width=12.
        let d = 4;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; 3];
        let clusters = vec![
            vec![(0u32, vec![0.1; d]), (1, vec![0.2; d])],
            vec![(2, vec![-0.3; d])],
            vec![(3, vec![0.4; d]), (4, vec![-0.5; d]), (5, vec![0.6; d])],
        ];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();

        assert_eq!(enc.m_max(), 3);
        assert_eq!(enc.d(), d);
        assert_eq!(enc.n_centroids(), 3);
        assert_eq!(enc.width(), 12);
        assert_eq!(enc.matrix().len(), 3 * 12);
        assert_eq!(enc.cluster_col_start(0), 0);
        assert_eq!(enc.cluster_col_start(1), 4);
        assert_eq!(enc.cluster_col_start(2), 8);
    }

    #[test]
    fn vector_lands_in_its_cluster_block() {
        // d=2, two clusters each with one vector at distinguishable values.
        let d = 2;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; 2];
        let clusters = vec![vec![(7u32, vec![1.0, 0.0])], vec![(13u32, vec![0.0, -1.0])]];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let q = 4;
        let mag = max_magnitude(q) as u64;
        let enc = encode_ivf(&index, q, p).unwrap();

        // Row 0 expected: [mag, 0, 0, p-mag]
        let row0 = &enc.matrix()[0..enc.width()];
        assert_eq!(row0[0], mag);
        assert_eq!(row0[1], 0);
        assert_eq!(row0[2], 0);
        assert_eq!(row0[3], p - mag);
    }

    #[test]
    fn padding_rows_are_zero() {
        // Cluster 0 has 2 vectors, cluster 1 has 1 → m_max=2. Row 1 of
        // cluster 1's d-block must be all zeros (padding).
        let d = 3;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; 2];
        let clusters = vec![
            vec![(0u32, vec![0.5, 0.5, 0.5]), (1u32, vec![-0.5, -0.5, -0.5])],
            vec![(2u32, vec![1.0, 1.0, 1.0])],
        ];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();

        let width = enc.width();
        let row1 = &enc.matrix()[width..2 * width];
        let cluster1_block = &row1[enc.cluster_col_start(1)..enc.cluster_col_start(1) + d];
        assert!(
            cluster1_block.iter().all(|&z| z == 0),
            "padding row must be zero, got {cluster1_block:?}"
        );
    }

    #[test]
    fn cells_outside_a_clusters_block_are_zero() {
        // Cluster 0 vector should NOT bleed into cluster 1's columns.
        let d = 2;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; 2];
        let clusters = vec![vec![(0u32, vec![1.0, 1.0])], vec![]];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();

        let width = enc.width();
        let row0 = &enc.matrix()[0..width];
        // Cluster 1's block (cols 2..4) must be zero.
        let block_start = enc.cluster_col_start(1);
        for (offset, &z) in row0[block_start..block_start + d].iter().enumerate() {
            assert_eq!(z, 0, "col {}", block_start + offset);
        }
    }

    #[test]
    fn cluster_membership_round_trips_local_to_global_ids() {
        let centroids: Vec<Centroid> = vec![vec![0.0]; 2];
        let clusters = vec![
            vec![(42u32, vec![0.1]), (17u32, vec![0.2])],
            vec![(99u32, vec![0.3])],
        ];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();
        assert_eq!(enc.cluster_membership(0), &[42, 17]);
        assert_eq!(enc.cluster_membership(1), &[99]);
    }

    #[test]
    fn empty_cluster_reserves_block_with_no_rows() {
        // Cluster 0 empty, cluster 1 has 1 vector → m_max=1; cluster 0's
        // d-block is reserved at cols [0, d) and stays zero.
        let d = 2;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]; 2];
        let clusters = vec![vec![], vec![(0u32, vec![1.0, -1.0])]];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let enc = encode_ivf(&index, 4, p).unwrap();
        assert_eq!(enc.m_max(), 1);
        assert_eq!(enc.width(), 4);
        let row0 = &enc.matrix()[0..enc.width()];
        // Cluster 0's block all zero; cluster 1 holds (mag, p-mag).
        assert_eq!(row0[0], 0);
        assert_eq!(row0[1], 0);
        let mag = max_magnitude(4) as u64;
        assert_eq!(row0[2], mag);
        assert_eq!(row0[3], p - mag);
        assert!(enc.cluster_membership(0).is_empty());
    }

    #[test]
    fn encoded_block_dequantises_back_to_original() {
        let d = 4;
        let centroids: Vec<Centroid> = vec![vec![0.0; d]];
        let v = vec![0.2, -0.5, 0.7, -1.0];
        let clusters = vec![vec![(0u32, v.clone())]];
        let index = handcrafted_index(centroids, clusters);

        let p = Params::tiptoe_text().p();
        let q = 4;
        let mag = max_magnitude(q) as f32;
        let enc = encode_ivf(&index, q, p).unwrap();
        let row0 = &enc.matrix()[0..enc.width()];
        let block = &row0[enc.cluster_col_start(0)..enc.cluster_col_start(0) + d];
        for (i, &z) in block.iter().enumerate() {
            let back = dequantise_component(z, q, p);
            // Bound is one quantisation step (round-half-away-from-zero
            // can push half-integer scaled values up to 1/mag away).
            assert!((back - v[i]).abs() < 1.0 / mag, "i={i}");
        }
    }
}
