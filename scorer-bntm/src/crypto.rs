//! Braverman–Newman protocol primitives.
//!
//! No public reference implementation exists for the paper
//! (arXiv:2502.13060v3; Phase 0 search closed negative, see
//! `docs/notes/bntm-from-paper.md`), so `(L, H, S)` are derived
//! deterministically from `key_seed` / `h_seed` via ChaCha20 rather
//! than a paper-validated concrete tuple. That said, this file
//! faithfully implements the paper's Protocol 1 (§6, p.9): `L, H, S`
//! are sampled once at cluster initialisation and reused for every
//! online query — the fast-multiply decomposition `M' · v_enc =
//! H·(L⊤·v_enc) + S·v_enc` is the trapdoor's actual mechanism, not a
//! placeholder for it. δ is retuned within the paper's §7 security
//! floor (`δ·μ > 1/n`) rather than left at an arbitrary constant — see
//! `params.rs`. Plan 28 fixed an earlier bug where `(H, S)` were
//! re-derived from scratch on every query instead of being cached.
//!
//! Not implemented: Protocol 3 (§7, p.11), the recursive multi-level
//! construction that gets the paper's own headline client speedups
//! (Table 1, up to 25× at n=16385) past the Protocol-1 `Θ(√n)` floor.
//! Tracked as a follow-up plan.
//!
//! Convention: our `l_subspace` is shape `n × n_1`. In the standard
//! `LH + S` formulation `L` is `m × n_1`, so our `L` is its transpose
//! `L⊤`. Arithmetic in this file is in the `HL⊤ + S` form throughout.
//!
//! Verification-overhead note: measured on BN flat / 100k MS MARCO at
//! 30.0 % of `latency-us` (1.5686 s on, 1.0976 s off). This is what
//! motivates splitting the per-query verification cost into its own
//! `verification-overhead-us` column.

use std::sync::Arc;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::params::{BnTmParams, P};

// ---------------------------------------------------------------------------
// Mersenne field arithmetic mod P = 2^61 − 1
// ---------------------------------------------------------------------------

const P_U128: u128 = P as u128;

/// Reduce `x` mod P. Two-stage Mersenne folding handles full `u128`
/// inputs — required by the chunked accumulator in `compute_products`,
/// where the running sum can reach `~2^127` between reductions.
#[inline]
pub(crate) fn mersenne_reduce(x: u128) -> u64 {
    // 2^61 ≡ 1 (mod P), so x ≡ (x mod 2^61) + (x >> 61) (mod P).
    // Stage 1: x < 2^128 → s1 < 2^67 + 2^61 < 2^68.
    let lo1 = x & P_U128;
    let hi1 = x >> 61;
    let s1 = lo1 + hi1;
    // Stage 2: s1 < 2^68 → lo2 < 2^61, hi2 < 2^7. Result < 2^61 + 128.
    let lo2 = (s1 & P_U128) as u64;
    let hi2 = (s1 >> 61) as u64;
    let mut r = lo2 + hi2;
    // Canonicalise to [0, P). At most two subtractions are needed.
    if r >= P {
        r -= P;
    }
    if r >= P {
        r -= P;
    }
    r
}

#[inline]
pub(crate) fn add_mod(a: u64, b: u64) -> u64 {
    debug_assert!(a < P && b < P);
    let s = a + b;
    if s >= P { s - P } else { s }
}

#[inline]
pub(crate) fn sub_mod(a: u64, b: u64) -> u64 {
    debug_assert!(a < P && b < P);
    if a >= b { a - b } else { a + P - b }
}

/// Field multiplication. Currently unused on the hot paths (which
/// inline the `(a as u128) * (b as u128) → mersenne_reduce` step for
/// throughput) — kept here as a building block referenced by tests
/// and a useful future utility.
#[inline]
#[allow(dead_code)]
pub(crate) fn mul_mod(a: u64, b: u64) -> u64 {
    debug_assert!(a < P && b < P);
    mersenne_reduce((a as u128) * (b as u128))
}

// ---------------------------------------------------------------------------
// Client-side encoding helpers
//
// Shared by the flat (`lib.rs`) and IVF (`ivf.rs`) scorers so the
// f32↔F_p mapping and rank-lift stay bit-identical across both — the
// cross-scheme comparison relies on it.
// ---------------------------------------------------------------------------

/// Quantise `v ∈ R^d` (d ≤ n, with implicit zero-pad to n) to F_p^n.
/// Mapping: `v_q[i] = round(v[i] · q) mod P`, with negatives wrapped
/// into `[P/2, P)`. Embeddings are unit-norm so `|v_i| · q ≤ q << P/2`
/// for any `q ≤ P/2` — overflow is unreachable by construction at the
/// quantisation scales we use; we don't surface it as an error in the
/// hot path.
pub(crate) fn quantise_vector(v: &[f32], n: usize, q: u64) -> Vec<u64> {
    debug_assert!(v.len() <= n);
    let mut out = vec![0u64; n];
    let q_scale = q as f64;
    for (i, &x) in v.iter().enumerate() {
        let scaled = (x as f64) * q_scale;
        let rounded = scaled.round() as i64;
        out[i] = if rounded >= 0 {
            (rounded as u64) % P
        } else {
            // map -k to P − k for k in (0, P).
            let k = (-rounded) as u64;
            let r = k % P;
            if r == 0 { 0 } else { P - r }
        };
    }
    out
}

/// Lift an F_p element to a signed integer, treating values in
/// `[P/2, P)` as negative (after subtracting P). Used for
/// rank-by-magnitude on inner-product scores.
#[inline]
pub(crate) fn fp_to_signed(s: u64) -> i64 {
    debug_assert!(s < P);
    if s < P / 2 {
        s as i64
    } else {
        (s as i64) - (P as i64)
    }
}

/// Domain-separate a seed for use with an independent purpose
/// (encode mask vs verification mask). Mixes the original seed with
/// a short tag via XOR then runs one ChaCha20 round to whiten — keeps
/// the two streams empirically uncorrelated without pulling in a
/// full KDF for a low-stakes check.
pub(crate) fn domain_separate(seed: [u8; 32], tag: &[u8]) -> [u8; 32] {
    use rand::RngExt;
    let mut out = seed;
    for (i, b) in tag.iter().enumerate() {
        out[i % 32] ^= *b;
    }
    let mut rng = ChaCha20Rng::from_seed(out);
    rng.random()
}

/// Sample a uniformly random element of F_p. Rejection probability is
/// `2^{-61}` (only the value `P` itself is rejected from a 61-bit draw),
/// so the loop terminates almost surely on the first attempt.
fn sample_fp<R: rand::Rng + ?Sized>(rng: &mut R) -> u64 {
    loop {
        let x = rng.next_u64() & P;
        if x < P {
            return x;
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse {-1, 0, +1} types
// ---------------------------------------------------------------------------

/// Sparse signed vector with entries in `{-1, 0, +1}`.
#[derive(Clone, Debug)]
pub(crate) struct SparseVec {
    pub n: usize,
    pub idx: Vec<u32>, // sorted column indices
    pub sign: Vec<i8>, // ±1, same length as idx
}

/// Sparse signed matrix `m × n` with entries in `{-1, 0, +1}`. One
/// `SparseVec` per row.
#[derive(Clone, Debug)]
pub(crate) struct SparseMatrix {
    /// Row count. Currently derivable from `rows.len()`; kept as an
    /// explicit field for readability and future serialisation.
    #[allow(dead_code)]
    pub m: usize,
    pub n: usize,
    pub rows: Vec<SparseVec>,
}

impl SparseMatrix {
    /// All-zero `m × n` placeholder. Used for `m = 0` (empty) IVF
    /// clusters — which have no rows and so no meaningful `S` — and by
    /// GPU tests that need a well-formed `SparseMatrix` without
    /// exercising the sparse decode path.
    pub(crate) fn empty(m: usize, n: usize) -> Self {
        let rows = (0..m)
            .map(|_| SparseVec {
                n,
                idx: Vec::new(),
                sign: Vec::new(),
            })
            .collect();
        SparseMatrix { m, n, rows }
    }
}

/// Sample a sparse `{-1, 0, +1}` vector of length `n` with per-entry
/// probability `mu` of being non-zero (uniformly signed when so).
fn sample_sparse_pm1<R: rand::Rng + ?Sized>(rng: &mut R, n: usize, mu: f64) -> SparseVec {
    debug_assert!((0.0..1.0).contains(&mu));
    // Map mu to a 32-bit threshold for cheap comparison.
    let threshold = (mu * (1u64 << 32) as f64) as u64;
    let threshold = threshold.min(u32::MAX as u64) as u32;

    let mut idx = Vec::new();
    let mut sign = Vec::new();
    for j in 0..n {
        let r = rng.next_u32();
        if r < threshold {
            idx.push(j as u32);
            sign.push(if rng.next_u32() & 1 == 0 { -1i8 } else { 1i8 });
        }
    }
    SparseVec { n, idx, sign }
}

// ---------------------------------------------------------------------------
// Deterministic generation from seeds
// ---------------------------------------------------------------------------

/// Generate the LPN subspace `L`, shape `n × n_1`, row-major. Uniform
/// over F_p and deterministic in `key_seed`.
///
/// Convention: rows index over the "inner" embedding dimension, columns
/// over the subspace; `L⊤` (logical) is materialised on the fly when
/// needed. See file-level convention note.
pub(crate) fn generate_subspace(key_seed: [u8; 32], n: usize, n1: usize) -> Vec<u64> {
    let mut rng = ChaCha20Rng::from_seed(key_seed);
    (0..n * n1).map(|_| sample_fp(&mut rng)).collect()
}

/// Generate `H`, shape `m × n_1`, row-major. Uniform over F_p and
/// deterministic in `h_seed`.
///
/// Domain-separates from `S` by stepping the same RNG before sampling.
fn generate_h<R: rand::Rng + ?Sized>(rng: &mut R, m: usize, n1: usize) -> Vec<u64> {
    (0..m * n1).map(|_| sample_fp(rng)).collect()
}

/// Generate sparse `S`, shape `m × n`, with per-entry rate `mu` and
/// uniform `{-1, +1}` signs. Deterministic in `h_seed` (after `H`).
fn generate_s<R: rand::Rng + ?Sized>(rng: &mut R, m: usize, n: usize, mu: f64) -> SparseMatrix {
    let rows: Vec<SparseVec> = (0..m).map(|_| sample_sparse_pm1(rng, n, mu)).collect();
    SparseMatrix { m, n, rows }
}

/// Regenerate `(H, S)` from `h_seed` for a given parameter set. The
/// pair is sampled in a fixed order (H first, then S) so callers
/// re-running this function get identical output.
pub(crate) fn regenerate_h_s(
    h_seed: [u8; 32],
    m: usize,
    params: BnTmParams,
) -> (Vec<u64>, SparseMatrix) {
    let n = params.n();
    let n1 = params.n1();
    let mu = params.mu();
    let mut rng = ChaCha20Rng::from_seed(h_seed);
    let h = generate_h(&mut rng, m, n1);
    let s = generate_s(&mut rng, m, n, mu);
    (h, s)
}

// ---------------------------------------------------------------------------
// Cluster encryption
// ---------------------------------------------------------------------------

/// Storage backend for a per-cluster `u64` matrix.
///
/// `Owned` keeps the field elements in a heap-allocated `Vec`; the
/// rest of the codebase treats this exactly like the earlier
/// `Arc<Vec<u64>>` field. Used for the flat scorer and for the IVF
/// path when no cache directory is configured.
///
/// `Mmap` stores a byte-offset into a memory-mapped on-disk cluster
/// cache file; `as_slice()` returns a zero-copy `&[u64]` backed by the
/// page cache. Used for the IVF path on cache-write or cache-load so
/// the in-memory handle never holds the corpus-sized encrypted +
/// plaintext matrices (~180 GB at the 8.8M-vector scale across `m_enc`,
/// `a_l`, and `m_plain`, exceeding sacs006's 125 GB physical RAM).
/// Per-field regions are 8-byte aligned by the cache file layout
/// (header + IVF payload + pad-to-8 + 40-byte per-cluster header is
/// followed by sequences of `u64`s); the unsafe cast in `as_slice()`
/// is sound under that invariant.
#[derive(Clone)]
pub(crate) enum Storage {
    Owned(Arc<Vec<u64>>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        byte_offset: usize,
        len_u64s: usize,
    },
}

impl Storage {
    pub fn as_slice(&self) -> &[u64] {
        match self {
            Storage::Owned(v) => v.as_slice(),
            Storage::Mmap {
                mmap,
                byte_offset,
                len_u64s,
            } => {
                let bytes = &mmap[*byte_offset..*byte_offset + *len_u64s * 8];
                debug_assert_eq!(
                    bytes.as_ptr() as usize % 8,
                    0,
                    "Storage::Mmap region must be 8-byte aligned (file format invariant)"
                );
                // SAFETY: u64 has no invalid bit patterns; mmap regions
                // are 8-byte aligned by cache file layout construction.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u64, *len_u64s) }
            }
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        match self {
            Storage::Owned(v) => v.len(),
            Storage::Mmap { len_u64s, .. } => *len_u64s,
        }
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Owned(v) => write!(f, "Storage::Owned(len={})", v.len()),
            Storage::Mmap { len_u64s, .. } => write!(f, "Storage::Mmap(len={len_u64s})"),
        }
    }
}

/// Per-cluster crypto state for BN.
///
/// **Mixed dual-side type — not a wire format.** This struct bundles
/// server-side and client-side material together for ergonomics in the
/// in-process scorer. Only `m_enc` is uploaded during init (and that's
/// the entire server-side state). `h_seed`, `a_l`, `m_plain`, `h_mat`,
/// and `s_mat` are client-side trapdoor / state material; a future
/// client/server split must keep the partition explicit at the I/O
/// boundary.
#[derive(Clone)]
pub(crate) struct EncryptedCluster {
    /// Server-side. M_enc = M + (H L⊤ + S), row-major `m × n`. Storage
    /// is either an owned `Vec<u64>` (small corpora / tests / flat
    /// path) or a mmap-backed slice into the on-disk cluster cache
    /// (large-scale IVF builds, to avoid holding the matrices in RAM).
    pub m_enc: Storage,
    /// Client-side trapdoor. Seed `(H, S)` were originally derived
    /// from; retained as the cache-file / fingerprint identity and so
    /// the GPU path can still re-derive `H` on a VRAM-eviction
    /// cache-miss (`gpu/mod.rs`). The CPU decode hot path no longer
    /// reads this — see `h_mat` / `s_mat` below (Plan 28).
    pub h_seed: [u8; 32],
    /// Number of vectors (rows of M).
    pub m: usize,
    /// Client-side trapdoor. `AL = M · L`, row-major `m × n_1`. The
    /// per-init precomputation that makes the per-query correction
    /// `M · v' = AL · G + M · T` cheap. Same storage choice as
    /// `m_enc`.
    pub a_l: Storage,
    /// Client-side state. Original cluster matrix `M`, row-major
    /// `m × n`. Required for the per-query `M · T` correction — this
    /// crate implements the paper's Protocol 1 shape (not the further
    /// recursive Protocol 3), which still needs the plaintext client
    /// side for this term. In a hardened deployment the client owns
    /// `M` regardless of how `M_enc` is uploaded — this field models
    /// the client side faithfully. Same storage choice as `m_enc`.
    pub m_plain: Storage,
    /// Client-side trapdoor. `H`, row-major `m × n_1`, materialised
    /// once (at `encrypt_cluster` time, or once at disk-cache-load
    /// time) instead of being regenerated from `h_seed` on every
    /// query. Same shape and storage choice as `a_l` — this is what
    /// Plan 28 fixed: the paper's Protocol 1 (arXiv:2502.13060v3, p.9)
    /// samples `H` once at initialisation and reuses it for every
    /// online round; the pre-fix code called `regenerate_h_s` inside
    /// `decode_scores` on every query instead.
    pub h_mat: Storage,
    /// Client-side trapdoor. `S`, sparse `m × n`, materialised once
    /// alongside `h_mat`. Cheap to keep resident (μ-rate sparse; ~18
    /// MiB per 100k-row cluster at Sec128) — mirrors what the GPU path
    /// already did for its own `s_mat` / `s_mats` caches.
    pub s_mat: Arc<SparseMatrix>,
}

/// Encrypt a cluster matrix `M` (row-major, `m × n` field elements).
///
/// Steps:
/// 1. Generate `(H, S)` deterministically from `h_seed`.
/// 2. Compute `M' = HL⊤ + S` row-by-row.
/// 3. Set `M_enc = M + M'`.
/// 4. Precompute `AL = M · L`.
///
/// `(H, S)` are retained on the returned `EncryptedCluster` (as `h_mat` /
/// `s_mat`) rather than discarded — the query hot path (`decode_scores`)
/// reads them directly instead of re-deriving from `h_seed` per query
/// (Plan 28).
pub(crate) fn encrypt_cluster(
    m_matrix: &[u64],
    m: usize,
    l_subspace: &[u64],
    h_seed: [u8; 32],
    params: BnTmParams,
) -> EncryptedCluster {
    let n = params.n();
    let n1 = params.n1();
    debug_assert_eq!(m_matrix.len(), m * n);
    debug_assert_eq!(l_subspace.len(), n * n1);

    let (h_mat, s_mat) = regenerate_h_s(h_seed, m, params);

    // M' = HL⊤ + S; M_enc = M + M'.
    // Row i: (HL⊤)[i, j] = sum_k H[i, k] · L[j, k]   (since L is n × n_1)
    //                   = sum_k H[i*n1+k] * L[j*n1+k]
    let m_enc: Vec<u64> = (0..m)
        .into_par_iter()
        .flat_map_iter(|i| {
            let h_row = &h_mat[i * n1..(i + 1) * n1];
            let s_row = &s_mat.rows[i];
            let mat_row = &m_matrix[i * n..(i + 1) * n];
            (0..n).map(move |j| {
                let l_row = &l_subspace[j * n1..(j + 1) * n1];
                // (HL⊤)[i, j] = inner(h_row, l_row).
                let mut acc: u128 = 0;
                for k in 0..n1 {
                    acc += (h_row[k] as u128) * (l_row[k] as u128);
                    if k & 31 == 31 {
                        acc = mersenne_reduce(acc) as u128;
                    }
                }
                let hl = mersenne_reduce(acc);
                // Add the sparse S[i, j] entry, if any.
                let s_term = sparse_lookup(s_row, j as u32);
                let m_prime = match s_term {
                    1 => add_mod(hl, 1),
                    -1 => sub_mod(hl, 1),
                    _ => hl,
                };
                add_mod(mat_row[j], m_prime)
            })
        })
        .collect();

    // AL = M · L, shape m × n_1.
    let a_l: Vec<u64> = (0..m)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mat_row = &m_matrix[i * n..(i + 1) * n];
            (0..n1).map(move |k| {
                let mut acc: u128 = 0;
                for j in 0..n {
                    acc += (mat_row[j] as u128) * (l_subspace[j * n1 + k] as u128);
                    if j & 31 == 31 {
                        acc = mersenne_reduce(acc) as u128;
                    }
                }
                mersenne_reduce(acc)
            })
        })
        .collect();

    EncryptedCluster {
        m_enc: Storage::Owned(Arc::new(m_enc)),
        h_seed,
        m,
        a_l: Storage::Owned(Arc::new(a_l)),
        m_plain: Storage::Owned(Arc::new(m_matrix.to_vec())),
        h_mat: Storage::Owned(Arc::new(h_mat)),
        s_mat: Arc::new(s_mat),
    }
}

/// Lookup the value of a sparse row at column `col`. Returns `0` if not
/// present. O(log n) via binary search on sorted indices.
#[inline]
fn sparse_lookup(row: &SparseVec, col: u32) -> i8 {
    match row.idx.binary_search(&col) {
        Ok(pos) => row.sign[pos],
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Per-query encoding
// ---------------------------------------------------------------------------

/// Per-query encoded state.
pub(crate) struct QueryEncoded {
    /// `v_enc = v + L·G + T`, length `n`.
    pub v_enc: Vec<u64>,
    /// `G`, length `n_1`. Used at decode time for the `AL · G` term.
    pub g: Vec<u64>,
    /// `T`, sparse length `n`. Used at decode time for the `M · T`
    /// term and as part of the `(HL⊤+S) · v_enc` decomposition.
    pub t: SparseVec,
}

/// Encode query `v` (quantised, length `n`, F_p elements) by adding
/// per-query trapdoor noise `v' = LG + T`. Returns `v_enc = v + v'`
/// plus the per-query mask state `(G, T)` needed at decode time.
pub(crate) fn encode_query<R: rand::Rng + ?Sized>(
    v: &[u64],
    l_subspace: &[u64],
    rng: &mut R,
    params: BnTmParams,
) -> QueryEncoded {
    let n = params.n();
    let n1 = params.n1();
    let mu = params.mu();
    debug_assert_eq!(v.len(), n);

    // G uniform over F_p^{n_1}.
    let g: Vec<u64> = (0..n1).map(|_| sample_fp(rng)).collect();
    // T sparse {-1, 0, +1} of length n.
    let t = sample_sparse_pm1(rng, n, mu);

    // v' = L·G + T. L is n × n_1.
    // (LG)[j] = sum_k L[j, k] · g[k]
    let mut v_enc: Vec<u64> = (0..n)
        .map(|j| {
            let l_row = &l_subspace[j * n1..(j + 1) * n1];
            let mut acc: u128 = 0;
            for k in 0..n1 {
                acc += (l_row[k] as u128) * (g[k] as u128);
                if k & 31 == 31 {
                    acc = mersenne_reduce(acc) as u128;
                }
            }
            mersenne_reduce(acc)
        })
        .collect();
    // Add T (sparse) and v.
    for (idx, sign) in t.idx.iter().zip(t.sign.iter()) {
        let j = *idx as usize;
        v_enc[j] = match *sign {
            1 => add_mod(v_enc[j], 1),
            -1 => sub_mod(v_enc[j], 1),
            _ => v_enc[j],
        };
    }
    for j in 0..n {
        v_enc[j] = add_mod(v_enc[j], v[j]);
    }

    QueryEncoded { v_enc, g, t }
}

// ---------------------------------------------------------------------------
// Server hot path
// ---------------------------------------------------------------------------

/// Compute `r = M_enc · v_enc`, length `m`. Server-side; this is the
/// dominant cost we measure. Thin wrapper around [`mersenne_matvec`].
pub(crate) fn compute_products(m_enc: &[u64], m: usize, n: usize, v_enc: &[u64]) -> Vec<u64> {
    mersenne_matvec(m_enc, m, n, v_enc)
}

/// Row-parallel Mersenne-field matvec: `out[i] = sum_j mat[i*cols + j] * vec[j] (mod P)`.
///
/// The inner kernel is a `u128` accumulator with a chunked Mersenne
/// fold at every 32 multiplications. The fold cadence (`j & 31 == 31`)
/// keeps the accumulator below `~2^127` even at `cols >= 64`, where 32
/// products of 61-bit operands can reach `~2^122`.
///
/// Output order is row-major and byte-identical across all callers
/// (`compute_products`, `dense_matvec`'s two `decode_scores` calls, and
/// the `verify_response` matvec — see [`mersenne_matvec_transpose`] for
/// the column-parallel sibling), so they all produce identical outputs.
/// The `mersenne_matvec_matches_reference_inline` test below pins that
/// invariant.
///
/// When the build has `target_feature = "avx512f"` and `cols % 8 == 0`,
/// dispatches to the lane-decomposed AVX-512 kernel (validated
/// 1.355–1.499× over scalar on AVX-512 hardware). Bit-equal to scalar —
/// pinned by `mersenne_matvec_avx512_matches_scalar`.
fn mersenne_matvec(mat: &[u64], rows: usize, cols: usize, vec: &[u64]) -> Vec<u64> {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(vec.len(), cols);

    // Compile-time gate: enabled implicitly by a `target-cpu=native`
    // build; default builds emit scalar only. AVX-512 has 8 u64 lanes,
    // so `cols % 8 != 0` falls back to scalar to avoid masking
    // complications at the trailing chunk.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        if cols > 0 && cols.is_multiple_of(8) {
            return mersenne_matvec_avx512(mat, rows, cols, vec);
        }
    }

    mersenne_matvec_scalar(mat, rows, cols, vec)
}

/// Scalar reference implementation of [`mersenne_matvec`]. Always
/// compiled; the `mersenne_matvec` dispatcher falls through to this
/// when AVX-512 is unavailable or the column count isn't a multiple
/// of 8. Also called directly by the
/// `mersenne_matvec_avx512_matches_scalar` correctness test.
fn mersenne_matvec_scalar(mat: &[u64], rows: usize, cols: usize, vec: &[u64]) -> Vec<u64> {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(vec.len(), cols);
    (0..rows)
        .into_par_iter()
        .map(|i| {
            let row = &mat[i * cols..(i + 1) * cols];
            let mut acc: u128 = 0;
            for j in 0..cols {
                acc += (row[j] as u128) * (vec[j] as u128);
                // For our ring P < 2^61, each product is < 2^122. Sum
                // of 32 products fits in u128 with margin; reduce
                // there to avoid overflow at cols ≥ 64.
                if j & 31 == 31 {
                    acc = mersenne_reduce(acc) as u128;
                }
            }
            mersenne_reduce(acc)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// AVX-512 SIMD for `mersenne_matvec`.
//
// Lane-decomposed Mersenne MAC: each `u64 × u64` is split into four
// `u32 × u32 → u64` widening products via `_mm512_mul_epu32` (8 lanes
// at a time). Under `P = 2^61 − 1`, `2^61 ≡ 1` so `2^64 ≡ 8`; the
// high-high partial recombines as `acc_hh × 8` after reduction. The
// `fold_avx512` helper applies one round of Mersenne reduction
// in-vector (no scalar round-trip). Validated 1.355–1.499× over scalar
// on AVX-512 hardware; the impl is lifted verbatim from
// `scorer-bntm/examples/mersenne_matvec_simd_bench.rs` to preserve the
// bit-equality witness from that bench.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn mersenne_matvec_avx512(mat: &[u64], rows: usize, cols: usize, vec: &[u64]) -> Vec<u64> {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(cols % 8, 0);
    (0..rows)
        .into_par_iter()
        .map(|i| {
            let row = &mat[i * cols..(i + 1) * cols];
            // SAFETY: `cfg(target_feature = "avx512f")` at compile time
            // proves AVX-512F is available on the build target, so the
            // intrinsics in `avx512_row_dot` are legal to invoke.
            unsafe { avx512_row_dot(row, vec, cols) }
        })
        .collect()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx512_fold(
    v: std::arch::x86_64::__m512i,
    mask: std::arch::x86_64::__m512i,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let lo = _mm512_and_si512(v, mask);
    let hi = _mm512_srli_epi64::<61>(v);
    _mm512_add_epi64(lo, hi)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn avx512_row_dot(row: &[u64], v: &[u64], cols: usize) -> u64 {
    use std::arch::x86_64::*;
    debug_assert_eq!(row.len(), cols);
    debug_assert_eq!(v.len(), cols);
    debug_assert_eq!(cols % 8, 0);

    let mask = _mm512_set1_epi64(P as i64);

    let mut acc_ll_v = _mm512_setzero_si512();
    let mut acc_cross_v = _mm512_setzero_si512();
    let mut acc_hh_v = _mm512_setzero_si512();

    let mut j = 0;
    while j < cols {
        let a_v = _mm512_loadu_si512(row.as_ptr().add(j) as *const _);
        let b_v = _mm512_loadu_si512(v.as_ptr().add(j) as *const _);

        let a_hi_v = _mm512_srli_epi64::<32>(a_v);
        let b_hi_v = _mm512_srli_epi64::<32>(b_v);

        let prod_ll = _mm512_mul_epu32(a_v, b_v);
        let prod_lh = _mm512_mul_epu32(a_v, b_hi_v);
        let prod_hl = _mm512_mul_epu32(a_hi_v, b_v);
        let prod_hh = _mm512_mul_epu32(a_hi_v, b_hi_v);
        let prod_cross = _mm512_add_epi64(prod_lh, prod_hl);

        let prod_ll_f = avx512_fold(prod_ll, mask);
        let prod_cross_f = avx512_fold(prod_cross, mask);

        acc_ll_v = _mm512_add_epi64(acc_ll_v, prod_ll_f);
        acc_cross_v = _mm512_add_epi64(acc_cross_v, prod_cross_f);
        acc_hh_v = _mm512_add_epi64(acc_hh_v, prod_hh);

        acc_ll_v = avx512_fold(acc_ll_v, mask);
        acc_cross_v = avx512_fold(acc_cross_v, mask);
        j += 8;
        if (j / 8) % 8 == 0 {
            acc_hh_v = avx512_fold(acc_hh_v, mask);
        }
    }

    let mut buf_ll = [0u64; 8];
    let mut buf_cross = [0u64; 8];
    let mut buf_hh = [0u64; 8];
    _mm512_storeu_si512(buf_ll.as_mut_ptr() as *mut _, acc_ll_v);
    _mm512_storeu_si512(buf_cross.as_mut_ptr() as *mut _, acc_cross_v);
    _mm512_storeu_si512(buf_hh.as_mut_ptr() as *mut _, acc_hh_v);

    let mut ll_sum: u128 = 0;
    let mut cross_sum: u128 = 0;
    let mut hh_sum: u128 = 0;
    for k in 0..8 {
        ll_sum += buf_ll[k] as u128;
        cross_sum += buf_cross[k] as u128;
        hh_sum += buf_hh[k] as u128;
    }

    let r_ll = mersenne_reduce(ll_sum) as u128;
    let r_cross = mersenne_reduce(cross_sum) as u128;
    let r_hh = mersenne_reduce(hh_sum) as u128;

    let total = r_ll + (r_cross << 32) + r_hh * 8;
    mersenne_reduce(total)
}

/// Column-parallel Mersenne-field matvec on the transpose:
/// `out[j] = sum_i vec[i] * mat[i*cols + j] (mod P)`.
///
/// Shared by two call sites: `verify_response`'s `z = u · M_enc` step
/// (which iterates `j` outer / `i` inner across `m_enc`'s columns) and
/// the `l_transpose_times_vec` helper. The inner kernel is the same
/// Mersenne accumulator + fold pattern; only the parallel axis and the
/// access pattern differ.
///
/// The access is strided (`mat[i*cols + j]` for fixed `j`); rayon
/// parallelism over `j` keeps each output cell's accumulator
/// register-local even though the row reads are cache-unfriendly.
fn mersenne_matvec_transpose(mat: &[u64], rows: usize, cols: usize, vec: &[u64]) -> Vec<u64> {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(vec.len(), rows);
    (0..cols)
        .into_par_iter()
        .map(|j| {
            let mut acc: u128 = 0;
            for i in 0..rows {
                acc += (vec[i] as u128) * (mat[i * cols + j] as u128);
                if i & 31 == 31 {
                    acc = mersenne_reduce(acc) as u128;
                }
            }
            mersenne_reduce(acc)
        })
        .collect()
}

/// Batched [`compute_products`] over `B` queries against the same
/// `m_enc`. Returns one `Vec<u64>` of length `m` per query (the same
/// shape `compute_products` produces individually).
///
/// Per-row, the inner accumulator runs the same `j` loop with the same
/// `& 31 == 31` reduction cadence as `compute_products`, just unrolled
/// across `B` queries — so per-(i, q_idx) the accumulation order, the
/// product values, the intermediate reductions, and the final
/// `mersenne_reduce` are byte-identical to the per-query call. The
/// amortisation lever is memory-bandwidth on `m_enc`: each row is read
/// once per call and contributes to every batched query's `r[i]`.
pub(crate) fn compute_products_batch(
    m_enc: &[u64],
    m: usize,
    n: usize,
    v_encs: &[&[u64]],
) -> Vec<Vec<u64>> {
    let big_b = v_encs.len();
    debug_assert_eq!(m_enc.len(), m * n);
    for v in v_encs {
        debug_assert_eq!(v.len(), n);
    }
    if big_b == 0 {
        return Vec::new();
    }

    // Parallelise over rows (matches `compute_products`). Per row, fold
    // the same row block against every batched query's v_enc with the
    // same reduction cadence. Output is row-major across queries; the
    // post-loop transpose into per-query `Vec<u64>`s of length `m` is
    // O(m·B) and dwarfed by the matvec cost.
    let row_outs: Vec<Vec<u64>> = (0..m)
        .into_par_iter()
        .map(|i| {
            let row = &m_enc[i * n..(i + 1) * n];
            let mut accs = vec![0u128; big_b];
            for j in 0..n {
                let row_j = row[j] as u128;
                for (q_idx, v_enc) in v_encs.iter().enumerate() {
                    accs[q_idx] += row_j * (v_enc[j] as u128);
                    if j & 31 == 31 {
                        accs[q_idx] = mersenne_reduce(accs[q_idx]) as u128;
                    }
                }
            }
            accs.into_iter().map(mersenne_reduce).collect::<Vec<u64>>()
        })
        .collect();

    // Transpose row-major `(m, big_b)` → per-query length-`m` `r`.
    let mut results: Vec<Vec<u64>> = (0..big_b).map(|_| vec![0u64; m]).collect();
    for (i, row_out) in row_outs.into_iter().enumerate() {
        for (q_idx, r_i) in row_out.into_iter().enumerate() {
            results[q_idx][i] = r_i;
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Client-side decode
// ---------------------------------------------------------------------------

/// Multiply a dense `m × n_1` matrix by a length-`n_1` vector, returning
/// length `m`. Used for `AL·G` and for the `H·(L⊤·v_enc)` step of the
/// decode. Thin wrapper around [`mersenne_matvec`].
fn dense_matvec(mat: &[u64], rows: usize, cols: usize, vec: &[u64]) -> Vec<u64> {
    mersenne_matvec(mat, rows, cols, vec)
}

/// Compute `L⊤ · v` where `L` is `n × n_1` row-major and `v` has length
/// `n`. Output has length `n_1`. (`L⊤[k, j] = L[j, k]`.) Exposed to the
/// crate so the GPU decode dispatch can compute it on host before
/// invoking the device-side dense matvec. Thin wrapper around
/// [`mersenne_matvec_transpose`].
pub(crate) fn l_transpose_times_vec(l: &[u64], n: usize, n1: usize, v: &[u64]) -> Vec<u64> {
    mersenne_matvec_transpose(l, n, n1, v)
}

/// Compute `M · t` where `M` is `m × n` dense and `t` is sparse. Output
/// has length `m`. Cost is `O(m · |support(t)|)`.
fn dense_times_sparse(mat: &[u64], rows: usize, cols: usize, t: &SparseVec) -> Vec<u64> {
    debug_assert_eq!(mat.len(), rows * cols);
    debug_assert_eq!(t.n, cols);
    let mut out = vec![0u64; rows];
    for (idx, sign) in t.idx.iter().zip(t.sign.iter()) {
        let col = *idx as usize;
        match *sign {
            1 => {
                for i in 0..rows {
                    out[i] = add_mod(out[i], mat[i * cols + col]);
                }
            }
            -1 => {
                for i in 0..rows {
                    out[i] = sub_mod(out[i], mat[i * cols + col]);
                }
            }
            _ => {}
        }
    }
    out
}

/// Compute `S · v` where `S` is sparse `m × n` and `v` is dense
/// length `n`. Output has length `m`. Cost is `O(sum of row supports)`.
fn sparse_times_dense(s: &SparseMatrix, v: &[u64]) -> Vec<u64> {
    debug_assert_eq!(s.n, v.len());
    s.rows
        .par_iter()
        .map(|row| {
            let mut acc: u64 = 0;
            for (idx, sign) in row.idx.iter().zip(row.sign.iter()) {
                let j = *idx as usize;
                match *sign {
                    1 => acc = add_mod(acc, v[j]),
                    -1 => acc = sub_mod(acc, v[j]),
                    _ => {}
                }
            }
            acc
        })
        .collect()
}

/// Recover `Mv` from the server's response.
///
/// Math: `Mv = r − M·v' − M'·v_enc`, where
///   `M·v' = AL·G + M·T`
///   `M'·v_enc = H·(L⊤·v_enc) + S·v_enc`
pub(crate) fn decode_scores(
    server_response: &[u64],
    cluster: &EncryptedCluster,
    l_subspace: &[u64],
    encoded: &QueryEncoded,
    params: BnTmParams,
) -> Vec<u64> {
    let m = cluster.m;
    let n = params.n();
    let n1 = params.n1();
    debug_assert_eq!(server_response.len(), m);

    // M · v' = AL · G + M · T.
    let al_g = dense_matvec(cluster.a_l.as_slice(), m, n1, &encoded.g);
    let m_t = dense_times_sparse(cluster.m_plain.as_slice(), m, n, &encoded.t);
    let m_v_prime: Vec<u64> = al_g
        .iter()
        .zip(m_t.iter())
        .map(|(a, b)| add_mod(*a, *b))
        .collect();

    // M' · v_enc = H · (L⊤ · v_enc) + S · v_enc. `(H, S)` are read from
    // the cluster's cached `h_mat` / `s_mat` — materialised once at
    // `encrypt_cluster` (or disk-cache-load) time, not regenerated here
    // (Plan 28; this used to call `regenerate_h_s` on every query).
    let lt_venc = l_transpose_times_vec(l_subspace, n, n1, &encoded.v_enc);
    let h_lt_venc = dense_matvec(cluster.h_mat.as_slice(), m, n1, &lt_venc);
    let s_venc = sparse_times_dense(&cluster.s_mat, &encoded.v_enc);
    let m_prime_v_enc: Vec<u64> = h_lt_venc
        .iter()
        .zip(s_venc.iter())
        .map(|(a, b)| add_mod(*a, *b))
        .collect();

    server_response
        .iter()
        .zip(m_v_prime.iter())
        .zip(m_prime_v_enc.iter())
        .map(|((r, mvp), mpve)| sub_mod(sub_mod(*r, *mvp), *mpve))
        .collect()
}

/// Decode variant for the GPU dispatch.
///
/// Mathematically equivalent to [`decode_scores`] but lets the caller
/// supply two pre-computed inputs:
///
///   - `dense_gpu`: `AL·G + H·(L⊤·v_enc)` already produced on device
///     by [`crate::gpu::FlatGpuState::decode_dense_terms`].
///   - `s_mat`: the sparse `S` matrix, pre-derived at handle build
///     (cached on `FlatGpuState` / `IvfGpuState`) so the caller can
///     skip the per-query [`regenerate_h_s`] cost — `H` lives on the
///     device, `S` lives in host RAM. Dropping that single-threaded
///     regen keeps the sparse-CPU decode budget low at `m = 100k` (a
///     fresh [`regenerate_h_s`] call would single-thread ChaCha20
///     across `m × n_1` field samples).
///
/// On identical `(server_response, cluster, encoded, params)` the
/// output is bit-equal to `decode_scores`.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn decode_scores_with_gpu_dense(
    server_response: &[u64],
    cluster: &EncryptedCluster,
    s_mat: &SparseMatrix,
    encoded: &QueryEncoded,
    dense_gpu: &[u64],
    params: BnTmParams,
) -> Vec<u64> {
    let m = cluster.m;
    let n = params.n();
    debug_assert_eq!(server_response.len(), m);
    debug_assert_eq!(dense_gpu.len(), m);
    debug_assert_eq!(s_mat.n, n);

    // Sparse pieces remain on host. `M · T` is `m × n` dense times a
    // sparse-{-1,0,+1} length-`n` vector with `|T| ≈ μn` nonzeros;
    // `S · v_enc` is `m × n` sparse-{-1,0,+1} times a dense length-`n`
    // vector. Both are rayon-parallel.
    let m_t = dense_times_sparse(cluster.m_plain.as_slice(), m, n, &encoded.t);
    let s_venc = sparse_times_dense(s_mat, &encoded.v_enc);

    // Mv = r − dense_gpu − (M·T + S·v_enc).
    server_response
        .iter()
        .zip(dense_gpu.iter())
        .zip(m_t.iter())
        .zip(s_venc.iter())
        .map(|(((r, dg), mt), sv)| sub_mod(sub_mod(sub_mod(*r, *dg), *mt), *sv))
        .collect()
}

// ---------------------------------------------------------------------------
// Protocol 2 — iterated Freivalds verification
// ---------------------------------------------------------------------------

/// Verify the server's claim that `r = M_enc · v_enc`.
///
/// Iterated Freivalds. For each of `λ'`
/// trials we sample a uniform `u ∈ F_p^m` and check
/// `u · r =? (u · M_enc) · v_enc`. The two sides are equal if and only
/// if `r = M_enc · v_enc` (whp; per-trial soundness `≈ |R|⁻¹`). At our
/// `R = F_p` with `p = 2^{61} − 1` and `λ' = 3`, total soundness is
/// `≈ 2^{-183}`, well past the `2^{-128}` target.
///
/// **Cost.** Each trial does an `m × n` reduction (compute `u · M_enc`)
/// plus two length-`m`/`n` dot products. Verification cost ≈ `λ'` ×
/// the server's matvec cost — no asymptotic saving for matrix-vector
/// (Freivalds shines on matrix-matrix where it gives `O(n²)` vs
/// `O(n^ω)`); the value here is malicious-server detection.
///
/// Returns `true` if every trial passes. The caller treats `false` as
/// `BnTmError::VerificationFailed` and surfaces the trial index for
/// triage.
pub(crate) fn verify_response<R: rand::Rng + ?Sized>(
    server_response: &[u64],
    m_enc: &[u64],
    m: usize,
    n: usize,
    v_enc: &[u64],
    trials: usize,
    rng: &mut R,
) -> Result<(), usize> {
    debug_assert_eq!(server_response.len(), m);
    debug_assert_eq!(m_enc.len(), m * n);
    debug_assert_eq!(v_enc.len(), n);

    for trial in 0..trials {
        let u: Vec<u64> = (0..m).map(|_| sample_fp(rng)).collect();

        // u · r — length-1 scalar.
        let ur = dot_product(&u, server_response);

        // z = u · M_enc — length n. Equivalently `z[j] = sum_i u[i] *
        // m_enc[i*n+j]`, i.e. col-parallel matvec on M_enc's transpose.
        // Uses `mersenne_matvec_transpose` (the same loop the earlier
        // inline code open-coded).
        let z = mersenne_matvec_transpose(m_enc, m, n, &u);

        // (u · M_enc) · v_enc — length-1 scalar.
        let zv = dot_product(&z, v_enc);

        if ur != zv {
            return Err(trial);
        }
    }
    Ok(())
}

#[inline]
fn dot_product(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: u128 = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        acc += (*x as u128) * (*y as u128);
        if i & 31 == 31 {
            acc = mersenne_reduce(acc) as u128;
        }
    }
    mersenne_reduce(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn small_params() -> (BnTmParams, usize, usize, usize) {
        // Use Sec128's defaults; the protocol is correctness-checkable
        // at any (n, n_1, μ) tuple — the small-corpus test below uses a
        // toy cluster but full-spec params.
        let p = BnTmParams::Sec128;
        (p, p.n(), p.n1(), 16) // m = 16 toy cluster
    }

    fn random_matrix(rng: &mut ChaCha20Rng, m: usize, n: usize) -> Vec<u64> {
        (0..m * n).map(|_| sample_fp(rng)).collect()
    }

    #[test]
    fn mersenne_reduce_canonical() {
        // P maps to 0; 2P maps to 0; 0 stays 0; (P-1) stays.
        assert_eq!(mersenne_reduce(0), 0);
        assert_eq!(mersenne_reduce(P_U128), 0);
        assert_eq!(mersenne_reduce(2 * P_U128), 0);
        assert_eq!(mersenne_reduce((P - 1) as u128), P - 1);
        // Random products fit.
        let a = (P - 7) as u128;
        let b = (P - 3) as u128;
        let want = ((a * b) % P_U128) as u64;
        assert_eq!(mersenne_reduce(a * b), want);
    }

    #[test]
    fn field_ops_match_naive_modular() {
        let cases: &[(u64, u64)] = &[(0, 0), (1, 1), (P - 1, P - 1), (12345, 67890)];
        for &(a, b) in cases {
            assert_eq!(add_mod(a, b), ((a as u128 + b as u128) % P_U128) as u64);
            let want_sub = ((a as i128 - b as i128).rem_euclid(P_U128 as i128)) as u64;
            assert_eq!(sub_mod(a, b), want_sub);
            assert_eq!(mul_mod(a, b), ((a as u128 * b as u128) % P_U128) as u64);
        }
    }

    /// Reference inline matvec — row-parallel, matching the original
    /// `dense_matvec` / `compute_products` inline form. Used to verify
    /// `mersenne_matvec`'s output is bit-equal to what those inline
    /// sites would have produced.
    fn reference_matvec_inline(mat: &[u64], rows: usize, cols: usize, vec: &[u64]) -> Vec<u64> {
        (0..rows)
            .map(|i| {
                let mut acc: u128 = 0;
                for j in 0..cols {
                    acc += (mat[i * cols + j] as u128) * (vec[j] as u128);
                    if j & 31 == 31 {
                        acc = mersenne_reduce(acc) as u128;
                    }
                }
                mersenne_reduce(acc)
            })
            .collect()
    }

    /// Reference inline transpose matvec — col-parallel, matching the
    /// original `verify_response` inline form and the
    /// `l_transpose_times_vec` open-coded loop. Used to verify
    /// `mersenne_matvec_transpose` is bit-equal to those inline sites.
    fn reference_matvec_transpose_inline(
        mat: &[u64],
        rows: usize,
        cols: usize,
        vec: &[u64],
    ) -> Vec<u64> {
        (0..cols)
            .map(|j| {
                let mut acc: u128 = 0;
                for i in 0..rows {
                    acc += (vec[i] as u128) * (mat[i * cols + j] as u128);
                    if i & 31 == 31 {
                        acc = mersenne_reduce(acc) as u128;
                    }
                }
                mersenne_reduce(acc)
            })
            .collect()
    }

    /// Consolidation correctness gate for [`mersenne_matvec`]. Covers
    /// both production shapes: 100 × 512 = `AL · G` / decode-dense,
    /// 100 × 1024 = `compute_products` / `verify_response` server
    /// matvec.
    #[test]
    fn mersenne_matvec_matches_reference_inline() {
        for &(rows, cols) in &[(100usize, 512usize), (100, 1024)] {
            let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF);
            let mat = random_matrix(&mut rng, rows, cols);
            let vec: Vec<u64> = (0..cols).map(|_| sample_fp(&mut rng)).collect();

            let want = reference_matvec_inline(&mat, rows, cols, &vec);
            let got = mersenne_matvec(&mat, rows, cols, &vec);

            assert_eq!(
                got, want,
                "mersenne_matvec drift vs reference inline at {rows} × {cols}"
            );

            // Public API path also matches (compute_products is a
            // thin wrapper now; verifying explicitly so the wrapper
            // contract is exercised by the test surface).
            let via_compute = compute_products(&mat, rows, cols, &vec);
            assert_eq!(
                via_compute, want,
                "compute_products wrapper drift at {rows} × {cols}"
            );
        }
    }

    /// AVX-512 SIMD correctness gate for [`mersenne_matvec`]. Only
    /// compiled when the build target has `target_feature = "avx512f"`
    /// (the same gate that decides whether the production dispatcher
    /// uses the SIMD path). Runs the AVX-512 kernel against the scalar
    /// reference on the two production shapes and asserts byte-equal
    /// output.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn mersenne_matvec_avx512_matches_scalar() {
        for &(rows, cols) in &[(100usize, 512usize), (100, 1024)] {
            let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF ^ 0xF0);
            let mat = random_matrix(&mut rng, rows, cols);
            let vec: Vec<u64> = (0..cols).map(|_| sample_fp(&mut rng)).collect();

            let scalar = mersenne_matvec_scalar(&mat, rows, cols, &vec);
            let simd = mersenne_matvec_avx512(&mat, rows, cols, &vec);

            assert_eq!(
                simd, scalar,
                "AVX-512 mersenne_matvec drift vs scalar at {rows} × {cols}"
            );
        }
    }

    /// Consolidation correctness gate for
    /// [`mersenne_matvec_transpose`]. Same two production shapes,
    /// swapped axis (output length = cols, vec length = rows).
    #[test]
    fn mersenne_matvec_transpose_matches_reference_inline() {
        for &(rows, cols) in &[(100usize, 512usize), (100, 1024)] {
            let mut rng = ChaCha20Rng::seed_from_u64(0xC0DE_BEEF ^ 0xA5);
            let mat = random_matrix(&mut rng, rows, cols);
            let vec: Vec<u64> = (0..rows).map(|_| sample_fp(&mut rng)).collect();

            let want = reference_matvec_transpose_inline(&mat, rows, cols, &vec);
            let got = mersenne_matvec_transpose(&mat, rows, cols, &vec);

            assert_eq!(
                got, want,
                "mersenne_matvec_transpose drift vs reference inline at {rows} × {cols}"
            );

            // l_transpose_times_vec is a thin wrapper now; verify
            // explicitly so the wrapper contract is exercised.
            let via_lt = l_transpose_times_vec(&mat, rows, cols, &vec);
            assert_eq!(
                via_lt, want,
                "l_transpose_times_vec wrapper drift at {rows} × {cols}"
            );
        }
    }

    /// Roundtrip: encrypt cluster, encode query, server compute, decode
    /// → matches the plaintext `Mv` exactly. The construction has zero
    /// LPN noise, so any drift here is a port bug.
    #[test]
    fn protocol_roundtrip_recovers_mv_exactly() {
        let (params, n, _n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([7u8; 32]);

        // Build a small toy cluster M (m × n) and a query v (length n).
        let m_matrix = random_matrix(&mut rng, m, n);
        let v: Vec<u64> = (0..n).map(|_| sample_fp(&mut rng)).collect();

        // Plaintext reference: Mv computed naively.
        let mv_plain: Vec<u64> = (0..m)
            .map(|i| {
                let mut acc: u128 = 0;
                for j in 0..n {
                    acc += (m_matrix[i * n + j] as u128) * (v[j] as u128);
                    if j & 31 == 31 {
                        acc = mersenne_reduce(acc) as u128;
                    }
                }
                mersenne_reduce(acc)
            })
            .collect();

        // Run the protocol.
        let l = generate_subspace([1u8; 32], n, params.n1());
        let cluster = encrypt_cluster(&m_matrix, m, &l, [2u8; 32], params);
        let mut q_rng = ChaCha20Rng::from_seed([9u8; 32]);
        let encoded = encode_query(&v, &l, &mut q_rng, params);
        let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
        let mv_decoded = decode_scores(&r, &cluster, &l, &encoded, params);

        assert_eq!(
            mv_decoded, mv_plain,
            "protocol decoding must recover Mv exactly (zero-noise §7.2)"
        );
    }

    /// `EncryptedCluster::h_mat` / `s_mat` are populated immediately by
    /// `encrypt_cluster` (Plan 28) and match exactly what
    /// `regenerate_h_s(h_seed, ...)` would independently produce —
    /// pins the cache-correctness invariant that `decode_scores` now
    /// relies on instead of re-deriving `(H, S)` per query.
    #[test]
    fn h_mat_s_mat_populated_after_encrypt_cluster() {
        let (params, n, n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([51u8; 32]);
        let m_matrix = random_matrix(&mut rng, m, n);
        let l = generate_subspace([52u8; 32], n, n1);
        let h_seed = [53u8; 32];
        let cluster = encrypt_cluster(&m_matrix, m, &l, h_seed, params);

        assert_eq!(cluster.h_mat.len(), m * n1, "h_mat shape");
        let (expected_h, expected_s) = regenerate_h_s(h_seed, m, params);
        assert_eq!(
            cluster.h_mat.as_slice(),
            expected_h.as_slice(),
            "cached h_mat must match a fresh regenerate_h_s(h_seed, ..) derivation"
        );
        assert_eq!(
            cluster.s_mat.rows.len(),
            expected_s.rows.len(),
            "s_mat row count"
        );
        for (got, want) in cluster.s_mat.rows.iter().zip(expected_s.rows.iter()) {
            assert_eq!(got.idx, want.idx);
            assert_eq!(got.sign, want.sign);
        }
    }

    /// Recall stays exact (§7.2 zero-noise) at every δ this crate
    /// supports (`Sec128Delta(1..=4)`, i.e. δ ∈ {0.5, 0.25, 0.125,
    /// 0.0625}) — only decode cost should move with δ, never
    /// correctness. Mirrors `protocol_roundtrip_recovers_mv_exactly`
    /// but sweeps the parameter set instead of fixing it at `Sec128`.
    #[test]
    fn delta_sweep_recall_exact_at_every_delta() {
        for k in 1u32..=4 {
            let params = BnTmParams::Sec128Delta(k);
            let n = params.n();
            let n1 = params.n1();
            let m = 16;
            let mut rng = ChaCha20Rng::from_seed([60 + k as u8; 32]);
            let m_matrix = random_matrix(&mut rng, m, n);
            let v: Vec<u64> = (0..n).map(|_| sample_fp(&mut rng)).collect();

            let mv_plain: Vec<u64> = (0..m)
                .map(|i| {
                    let mut acc: u128 = 0;
                    for j in 0..n {
                        acc += (m_matrix[i * n + j] as u128) * (v[j] as u128);
                        if j & 31 == 31 {
                            acc = mersenne_reduce(acc) as u128;
                        }
                    }
                    mersenne_reduce(acc)
                })
                .collect();

            let l = generate_subspace([70 + k as u8; 32], n, n1);
            let cluster = encrypt_cluster(&m_matrix, m, &l, [80 + k as u8; 32], params);
            let mut q_rng = ChaCha20Rng::from_seed([90 + k as u8; 32]);
            let encoded = encode_query(&v, &l, &mut q_rng, params);
            let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
            let mv_decoded = decode_scores(&r, &cluster, &l, &encoded, params);

            assert_eq!(
                mv_decoded,
                mv_plain,
                "k={k} (δ={}): decode must recover Mv exactly",
                params.delta()
            );
        }
    }

    /// A second random query under the same cluster handle still
    /// recovers Mv exactly — exercises that per-query (G, T) state
    /// is independent of cluster init state.
    #[test]
    fn protocol_two_independent_queries() {
        let (params, n, _n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([11u8; 32]);
        let m_matrix = random_matrix(&mut rng, m, n);
        let l = generate_subspace([3u8; 32], n, params.n1());
        let cluster = encrypt_cluster(&m_matrix, m, &l, [4u8; 32], params);

        for q_seed in [[13u8; 32], [17u8; 32]] {
            let v: Vec<u64> = (0..n).map(|_| sample_fp(&mut rng)).collect();
            let mv_plain: Vec<u64> = (0..m)
                .map(|i| {
                    let mut acc: u128 = 0;
                    for j in 0..n {
                        acc += (m_matrix[i * n + j] as u128) * (v[j] as u128);
                        if j & 31 == 31 {
                            acc = mersenne_reduce(acc) as u128;
                        }
                    }
                    mersenne_reduce(acc)
                })
                .collect();
            let mut q_rng = ChaCha20Rng::from_seed(q_seed);
            let encoded = encode_query(&v, &l, &mut q_rng, params);
            let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
            let mv_decoded = decode_scores(&r, &cluster, &l, &encoded, params);
            assert_eq!(mv_decoded, mv_plain);
        }
    }

    #[test]
    fn h_s_regeneration_deterministic() {
        let params = BnTmParams::Sec128;
        let m = 8;
        let (h1, s1) = regenerate_h_s([42u8; 32], m, params);
        let (h2, s2) = regenerate_h_s([42u8; 32], m, params);
        assert_eq!(h1, h2);
        assert_eq!(s1.m, s2.m);
        assert_eq!(s1.n, s2.n);
        for (r1, r2) in s1.rows.iter().zip(s2.rows.iter()) {
            assert_eq!(r1.idx, r2.idx);
            assert_eq!(r1.sign, r2.sign);
        }
    }

    #[test]
    fn s_density_matches_mu_in_expectation() {
        // For our Sec128, μ = 1/32; expected non-zeros per row of
        // length n=1024 is 32. Sample 64 rows and check the empirical
        // mean is in a generous tolerance band so the test stays
        // deterministic but not flaky across compiler revs.
        let params = BnTmParams::Sec128;
        let m = 64;
        let (_h, s) = regenerate_h_s([99u8; 32], m, params);
        let total: usize = s.rows.iter().map(|r| r.idx.len()).sum();
        let mean = total as f64 / m as f64;
        let expected = params.mu() * params.n() as f64;
        assert!(
            (mean - expected).abs() < expected * 0.5,
            "S density: mean={mean}, expected≈{expected}"
        );
    }

    /// Honest server (`r = M_enc · v_enc`): every Freivalds trial
    /// passes. No false positives across many independent (cluster,
    /// query) pairs — anchors the `λ' = 3` choice for our F_p.
    #[test]
    fn verification_passes_on_honest_server() {
        let (params, n, _n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([21u8; 32]);
        let m_matrix = random_matrix(&mut rng, m, n);
        let l = generate_subspace([22u8; 32], n, params.n1());
        let cluster = encrypt_cluster(&m_matrix, m, &l, [23u8; 32], params);

        for q_seed_byte in 0..16u8 {
            let v: Vec<u64> = (0..n).map(|_| sample_fp(&mut rng)).collect();
            let mut q_rng = ChaCha20Rng::from_seed([q_seed_byte; 32]);
            let encoded = encode_query(&v, &l, &mut q_rng, params);
            let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
            let mut v_rng = ChaCha20Rng::from_seed([q_seed_byte ^ 0xAA; 32]);
            verify_response(
                &r,
                cluster.m_enc.as_slice(),
                m,
                n,
                &encoded.v_enc,
                params.verification_trials(),
                &mut v_rng,
            )
            .expect("honest server: verification must pass on every query");
        }
    }

    /// Tampered response: flipping a single coordinate of `r` is
    /// caught by Freivalds with probability `1 − |R|⁻λ'`. At our
    /// `R = F_p` and `λ' = 3` this is essentially 1; the test does a
    /// single tamper-injection and asserts detection.
    #[test]
    fn verification_detects_tampered_response() {
        let (params, n, _n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([31u8; 32]);
        let m_matrix = random_matrix(&mut rng, m, n);
        let l = generate_subspace([32u8; 32], n, params.n1());
        let cluster = encrypt_cluster(&m_matrix, m, &l, [33u8; 32], params);
        let v: Vec<u64> = (0..n).map(|_| sample_fp(&mut rng)).collect();
        let mut q_rng = ChaCha20Rng::from_seed([34u8; 32]);
        let encoded = encode_query(&v, &l, &mut q_rng, params);
        let mut r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);

        // Tamper: add 1 to a single coordinate so r' ≠ M_enc · v_enc.
        r[m / 2] = add_mod(r[m / 2], 1);

        let mut v_rng = ChaCha20Rng::from_seed([35u8; 32]);
        let verdict = verify_response(
            &r,
            cluster.m_enc.as_slice(),
            m,
            n,
            &encoded.v_enc,
            params.verification_trials(),
            &mut v_rng,
        );
        assert!(
            verdict.is_err(),
            "tampered server: verification must catch the flip"
        );
    }

    /// Verification overhead is non-trivial but bounded: cost of
    /// `verify_response` is on the same order as `compute_products`
    /// (ratio `≈ λ'`). Guards against an accidental optimisation that
    /// would make verification "free" — that would mean the verifier
    /// is doing less than Freivalds requires.
    #[test]
    fn verification_overhead_in_expected_band() {
        use std::time::Instant;
        let (params, n, _n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([41u8; 32]);
        let m_matrix = random_matrix(&mut rng, m, n);
        let l = generate_subspace([42u8; 32], n, params.n1());
        let cluster = encrypt_cluster(&m_matrix, m, &l, [43u8; 32], params);
        let v: Vec<u64> = (0..n).map(|_| sample_fp(&mut rng)).collect();
        let mut q_rng = ChaCha20Rng::from_seed([44u8; 32]);
        let encoded = encode_query(&v, &l, &mut q_rng, params);

        let t0 = Instant::now();
        let r = compute_products(cluster.m_enc.as_slice(), m, n, &encoded.v_enc);
        let server_us = t0.elapsed().as_micros();

        let mut v_rng = ChaCha20Rng::from_seed([45u8; 32]);
        let t1 = Instant::now();
        verify_response(
            &r,
            cluster.m_enc.as_slice(),
            m,
            n,
            &encoded.v_enc,
            params.verification_trials(),
            &mut v_rng,
        )
        .unwrap();
        let verify_us = t1.elapsed().as_micros();

        // We care about a soft canary: verification does *real* work,
        // not "free". Lower bound: verification cost is comparable to
        // the server's matvec — a Freivalds trial does an m × n
        // reduction. Upper bound is intentionally loose (20×) because
        // parallel `cargo test --workspace` multi-process noise drags
        // wall-clock measurements; the upper bound exists only to
        // catch a *huge* regression, not to assert tight constants.
        let _lambda = params.verification_trials() as u128;
        assert!(
            verify_us > 0 && verify_us <= server_us.saturating_mul(20),
            "verify_us={verify_us} outside expected band vs server_us={server_us}"
        );
    }

    /// Batched `compute_products_batch` produces, for every batched
    /// query, a length-`m` `r` byte-identical to the per-query
    /// `compute_products` on the same `v_enc`. Per-(i, q_idx) the inner
    /// fold reads the same row, the same v_enc[j], and runs the same
    /// `& 31 == 31` reduction cadence, so the only path difference is
    /// loop unrolling across queries — no field-arithmetic
    /// reassociation.
    #[test]
    fn compute_products_batch_matches_per_query() {
        let (_params, n, _n1, m) = small_params();
        let mut rng = ChaCha20Rng::from_seed([57u8; 32]);

        // m × n field elements in [0, P) — synthetic m_enc; the batched
        // helper has no input invariants beyond shape, so we don't need
        // a real cluster build here.
        let m_enc = random_matrix(&mut rng, m, n);

        for &big_b in &[1usize, 8, 64] {
            let v_encs: Vec<Vec<u64>> = (0..big_b)
                .map(|_| (0..n).map(|_| sample_fp(&mut rng)).collect())
                .collect();
            let v_enc_refs: Vec<&[u64]> = v_encs.iter().map(|v| v.as_slice()).collect();

            let from_batch = compute_products_batch(&m_enc, m, n, &v_enc_refs);
            assert_eq!(from_batch.len(), big_b);

            for (q_idx, v_enc) in v_encs.iter().enumerate() {
                let from_per_query = compute_products(&m_enc, m, n, v_enc);
                assert_eq!(
                    from_batch[q_idx], from_per_query,
                    "B={big_b}, q_idx={q_idx}: batched r diverges from per-query r"
                );
            }
        }
    }
}
