//! Tiptoe scorer (private nearest-neighbour search via LWE + BFV).
//!
//! In-house Rust port of SimplePIR's linearly-homomorphic matrix-vector
//! primitive and Tiptoe's BFV second-layer compression, which shrinks
//! the per-query online download by re-encrypting the SimplePIR hint
//! product under BFV. Cryptographic parameters mirror the Go reference
//! implementation pinned in `tools/tiptoe-go-rev`.
//!
//! Architectural deviations from the original design, paired with a
//! Go-reference validation gate so the comparison stays well-defined:
//!
//! - shared IVF (no recursive split, no double-assignment)
//! - no PCA (embeddings stay 768-dim)
//! - no URL service (`Scorer::score` returns IDs only)
//!
//! IVF cache: single-entry in-memory + disk persistence, keyed by
//! `(n_centroids, train_seed, max_iter, N, dim, quantisation_bits,
//! lwe_param_fingerprint, bfv_param_fingerprint)`. On a hit, k-means +
//! matrix assembly + the multi-hour SimplePIR `preprocess`
//! (`O(r·c·n)`, ~1.6×10¹¹ u64 mults at our 100k corpus) are skipped;
//! the BFV `decompose_hint` step is fast enough to redo on load, which
//! keeps the on-disk format free of `fhe`-specific types.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use ivf_index::{
    ivf::{self, IvfIndex},
    kmeans,
};
use rand::RngExt;
use scorer_core::{BuildOutcome, CommunicationCost, Hit, ProgressReporter, Scorer, Vector, cache};

use crate::bfv::compress::{CompressError, CompressedHint, NUM_LIMBS_64, decompose_hint};
use crate::bfv::{BfvParameters, tiptoe_text_params};
use crate::cluster::{EncodedClusters, encode_ivf};
use crate::encoding::{EncodingError, validate_params};
use crate::pir::lwe::Params as LweParams;
use crate::pir::simplepir::{PirSetup, preprocess_with_progress};

pub mod bfv;
pub mod cluster;
pub mod encoding;
pub mod pir;
pub mod ranking;
pub mod routing;

/// Per-cluster configuration. LWE/BFV parameters are not user-
/// configurable — they mirror the Go reference implementation.
pub struct TiptoeConfig {
    /// Number of IVF centroids. Per the architectural invariant, all
    /// IVF-using scorers share the same default `⌈√N⌉`; the eval
    /// harness sets it.
    pub n_centroids: usize,
    /// Seed for k-means (k-means++ init + Lloyd iterations).
    pub train_seed: u64,
    /// Lloyd iterations; default 25 per the shared IVF invariant.
    pub max_iter: usize,
    /// Magnitude bits per embedding component. Default 4 (matches the
    /// deployed "signed 4-bit" range). Candidate for sweeping.
    pub quantisation_bits: u8,
    pub progress: Option<Arc<dyn ProgressReporter>>,
}

impl Default for TiptoeConfig {
    fn default() -> Self {
        Self {
            n_centroids: 1,
            train_seed: 42,
            max_iter: 25,
            quantisation_bits: 4,
            progress: None,
        }
    }
}

/// Opaque per-corpus state. Holds the plaintext-side IVF (centroids
/// drive client routing; per-cluster vector ids translate top-k results
/// back to corpus ids), the encoded `Z_p` matrix, the SimplePIR
/// preprocessing state, and the precomputed BFV limb-decomposition of
/// the hint.
///
/// **No key material.** Fresh LWE and BFV secrets are generated per
/// query (single-use token semantics).
pub struct TiptoeHandle {
    pub(crate) index: Arc<IvfIndex>,
    pub(crate) encoded: Arc<EncodedClusters>,
    pub(crate) pir_setup: Arc<PirSetup>,
    pub(crate) compressed_hint: Arc<CompressedHint>,
    pub(crate) lwe_params: LweParams,
    pub(crate) bfv_params: Arc<BfvParameters>,
    pub(crate) quantisation_bits: u8,
    pub(crate) dim: usize,
}

impl TiptoeHandle {
    /// IVF centroids. Exposed so out-of-process consumers (e.g., the
    /// Go-reference paired-run driver) can reproduce client-side
    /// routing against the same `IvfIndex` without going through
    /// `Scorer::score`.
    pub fn centroids(&self) -> &[ivf_index::kmeans::Centroid] {
        &self.index.centroids
    }

    /// Server-side encoded cluster matrix and membership tables. Holds
    /// no key material — only the `Z_p` matrix and the local→global
    /// `VectorId` map per cluster.
    pub fn encoded(&self) -> &EncodedClusters {
        &self.encoded
    }

    /// Magnitude bits used at encoding time. Needed to map the encoded
    /// `Z_p` matrix back to int8 slot values for the Go-ref CSV format.
    pub fn quantisation_bits(&self) -> u8 {
        self.quantisation_bits
    }
}

impl std::fmt::Debug for TiptoeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiptoeHandle")
            .field("dim", &self.dim)
            .field("n_centroids", &self.encoded.n_centroids())
            .field("m_max", &self.encoded.m_max())
            .field("quantisation_bits", &self.quantisation_bits)
            .field("lwe_params", &self.lwe_params)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TiptoeError {
    #[error("corpus must contain at least one vector")]
    EmptyCorpus,
    #[error("k-means produced an empty cluster (corpus too small for n_centroids?)")]
    EmptyCluster,
    #[error("n_centroids {0} exceeds corpus size")]
    TooManyCentroids(usize),
    #[error("query dimension {query_dim} does not match index dimension {index_dim}")]
    DimensionMismatch { query_dim: usize, index_dim: usize },
    #[error("quantisation: {0}")]
    QuantisationOverflow(EncodingError),
    #[error("BFV §6.2 composition failed: {0}")]
    BfvCompositionFailed(#[from] CompressError),
    #[error("spawn_blocking panicked")]
    SpawnPanic,
}

/// In-memory entry shared between sequential `upload_cluster` calls
/// against the same corpus + parameters (e.g. a `--quantisation-bits`
/// sweep across `q=3,4` builds the same IVF; only the encoded matrix
/// and downstream artifacts differ — but each `q` is its own entry
/// because the fingerprint includes `q`).
#[derive(Clone)]
struct CacheEntry {
    index: Arc<IvfIndex>,
    encoded: Arc<EncodedClusters>,
    pir_setup: Arc<PirSetup>,
    compressed_hint: Arc<CompressedHint>,
}

/// Tiptoe scorer. Stateless except for the optional cache. All per-
/// corpus state lives in [`TiptoeHandle`]; per-query state is generated
/// inside [`Scorer::score`]. Construct with [`TiptoeScorer::new`] for a
/// no-cache scorer (tests) or [`TiptoeScorer::with_cache_dir`] for
/// production runs that should hit `.tiptoe-cache-*.bin` on re-run.
pub struct TiptoeScorer {
    cache: Mutex<HashMap<String, CacheEntry>>,
    cache_dir: Option<PathBuf>,
}

impl TiptoeScorer {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            cache_dir: None,
        }
    }

    pub fn with_cache_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            cache_dir: Some(dir.into()),
        }
    }
}

impl Default for TiptoeScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cache I/O
// ---------------------------------------------------------------------------
//
// File layout:
//   [20-byte cache header]                  via scorer_core::cache
//   [u32: SCHEMA_VERSION]
//   [IvfIndex payload]                      via ivf::write_index_to
//   [u64: m_max] [u64: n_lwe] [32-byte: pir seed]
//   [m_max × n_lwe × u64 LE: pir hint]
//
// Encoded matrix and BFV `compressed_hint` are not stored — both are
// regenerated on load (encode_ivf is cheap quantise; decompose_hint is
// BFV plaintext encoding only). Storing the BFV plaintexts would
// require fhe's serialisation surface and inflate the file by ~16×.

const CACHE_SCHEMA_VERSION: u32 = 1;
const CACHE_TAG: &[u8] = b"tiptoe-v1";

fn lwe_fingerprint_bytes(p: &LweParams) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 * 3);
    v.extend_from_slice(&(p.n as u64).to_le_bytes());
    v.extend_from_slice(&(p.log2_p as u64).to_le_bytes());
    v.extend_from_slice(&p.sigma.to_bits().to_le_bytes());
    v
}

fn bfv_fingerprint_bytes(p: &BfvParameters) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&(p.degree() as u64).to_le_bytes());
    v.extend_from_slice(&(p.plaintext()).to_le_bytes());
    for &m in p.moduli_sizes() {
        v.extend_from_slice(&(m as u64).to_le_bytes());
    }
    v
}

#[allow(clippy::too_many_arguments)]
fn cache_parts(
    n_centroids: usize,
    train_seed: u64,
    max_iter: usize,
    n: usize,
    dim: usize,
    q_bits: u8,
    lwe_fp: &[u8],
    bfv_fp: &[u8],
) -> [Vec<u8>; 9] {
    [
        CACHE_TAG.to_vec(),
        (n_centroids as u64).to_le_bytes().to_vec(),
        train_seed.to_le_bytes().to_vec(),
        (max_iter as u64).to_le_bytes().to_vec(),
        (n as u64).to_le_bytes().to_vec(),
        (dim as u64).to_le_bytes().to_vec(),
        (q_bits as u64).to_le_bytes().to_vec(),
        lwe_fp.to_vec(),
        bfv_fp.to_vec(),
    ]
}

fn save_tiptoe_cache(
    path: &Path,
    parts: &[&[u8]],
    index: &IvfIndex,
    pir_setup: &PirSetup,
) -> io::Result<()> {
    let tmp = path.with_extension("bin.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut w = io::BufWriter::new(file);
    cache::write_header(&mut w, parts)?;
    w.write_all(&CACHE_SCHEMA_VERSION.to_le_bytes())?;
    ivf::write_index_to(&mut w, index)?;
    w.write_all(&(pir_setup.r as u64).to_le_bytes())?;
    w.write_all(&(pir_setup.n as u64).to_le_bytes())?;
    w.write_all(&pir_setup.seed)?;
    for &v in &pir_setup.hint {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    std::fs::rename(&tmp, path)
}

fn load_tiptoe_cache(path: &Path, parts: &[&[u8]]) -> Option<(IvfIndex, PirSetup)> {
    let mut file = std::fs::File::open(path).ok()?;
    if !cache::verify_header(&mut file, parts).ok()? {
        return None;
    }
    let mut r = io::BufReader::new(file);
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];
    let mut buf32 = [0u8; 32];

    r.read_exact(&mut buf4).ok()?;
    if u32::from_le_bytes(buf4) != CACHE_SCHEMA_VERSION {
        return None;
    }
    let index = ivf::read_index_from(&mut r).ok()?;
    r.read_exact(&mut buf8).ok()?;
    let m_max = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf8).ok()?;
    let n_lwe = u64::from_le_bytes(buf8) as usize;
    r.read_exact(&mut buf32).ok()?;
    let seed = buf32;
    let mut hint = vec![0u64; m_max * n_lwe];
    for v in hint.iter_mut() {
        r.read_exact(&mut buf8).ok()?;
        *v = u64::from_le_bytes(buf8);
    }
    let dim = index.centroids.first().map_or(0, |c| c.len());
    let n_centroids = index.clusters.len();
    let pir_setup = PirSetup {
        seed,
        hint,
        r: m_max,
        c: n_centroids * dim,
        n: n_lwe,
    };
    Some((index, pir_setup))
}

/// Derive a 32-byte SimplePIR seed from `train_seed` deterministically.
/// The seed only needs to be stable for a given corpus build; we reuse
/// `train_seed` so both sides of the IVF cache key cover this state.
fn derive_pir_seed(train_seed: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[..8].copy_from_slice(&train_seed.to_le_bytes());
    s
}

#[async_trait]
impl Scorer for TiptoeScorer {
    type Config = TiptoeConfig;
    type ClusterHandle = TiptoeHandle;
    type Error = TiptoeError;

    async fn upload_cluster(
        &self,
        config: &TiptoeConfig,
        vectors: &[Vector],
    ) -> Result<(TiptoeHandle, BuildOutcome), TiptoeError> {
        let build_start = Instant::now();
        let lwe_params = LweParams::tiptoe_text();
        let bfv_params = tiptoe_text_params();

        validate_params(config.quantisation_bits, lwe_params.log2_p)
            .map_err(TiptoeError::QuantisationOverflow)?;

        if vectors.is_empty() {
            return Err(TiptoeError::EmptyCorpus);
        }
        if config.n_centroids == 0 || config.n_centroids > vectors.len() {
            return Err(TiptoeError::TooManyCentroids(config.n_centroids));
        }

        let dim = vectors[0].0.len();
        let n_centroids = config.n_centroids;
        let train_seed = config.train_seed;
        let max_iter = config.max_iter;
        let quantisation_bits = config.quantisation_bits;
        let p = lwe_params.p();
        let progress = config.progress.clone();
        let n = vectors.len();

        let lwe_fp = lwe_fingerprint_bytes(&lwe_params);
        let bfv_fp = bfv_fingerprint_bytes(&bfv_params);
        let parts_owned = cache_parts(
            n_centroids,
            train_seed,
            max_iter,
            n,
            dim,
            quantisation_bits,
            &lwe_fp,
            &bfv_fp,
        );
        let parts_refs: Vec<&[u8]> = parts_owned.iter().map(|v| v.as_slice()).collect();
        let fp = cache::fingerprint(&parts_refs);
        let disk_path = self
            .cache_dir
            .as_deref()
            .map(|dir| dir.join(format!(".tiptoe-cache-{fp}.bin")));

        // In-memory cache hit: same scorer instance was already built.
        if let Some(entry) = self.cache.lock().unwrap().get(&fp).cloned() {
            if let Some(ref p) = progress {
                p.on_build_complete();
            }
            let outcome = BuildOutcome {
                cache_hit: true,
                build_duration: build_start.elapsed(),
            };
            return Ok((
                TiptoeHandle {
                    index: entry.index,
                    encoded: entry.encoded,
                    pir_setup: entry.pir_setup,
                    compressed_hint: entry.compressed_hint,
                    lwe_params,
                    bfv_params,
                    quantisation_bits,
                    dim,
                },
                outcome,
            ));
        }

        let data: Vec<Vec<f32>> = vectors.iter().map(|v| v.0.clone()).collect();
        let bfv_params_for_build = Arc::clone(&bfv_params);
        let disk_path_for_build = disk_path.clone();
        let parts_for_build = parts_owned.clone();

        let (index, encoded, pir_setup, compressed_hint, disk_hit) =
            tokio::task::spawn_blocking(move || {
                // Disk cache hit: skip kmeans + matrix assembly + the
                // multi-hour SimplePIR preprocess. encode_ivf and
                // decompose_hint still run on load (both are fast).
                let from_disk = disk_path_for_build
                    .as_deref()
                    .and_then(|path| if path.exists() { Some(path) } else { None })
                    .and_then(|path| {
                        let refs: Vec<&[u8]> =
                            parts_for_build.iter().map(|v| v.as_slice()).collect();
                        load_tiptoe_cache(path, &refs)
                    });

                if let Some((index, pir_setup)) = from_disk {
                    let encoded = encode_ivf(&index, quantisation_bits, p)
                        .map_err(TiptoeError::QuantisationOverflow)?;
                    debug_assert_eq!(encoded.m_max(), pir_setup.r);
                    debug_assert_eq!(encoded.width(), pir_setup.c);
                    let compressed_hint = decompose_hint(
                        &pir_setup.hint,
                        encoded.m_max(),
                        lwe_params.n,
                        &bfv_params_for_build,
                    )?;
                    if let Some(ref p) = progress {
                        p.on_build_complete();
                    }
                    return Ok::<_, TiptoeError>((
                        index,
                        encoded,
                        pir_setup,
                        compressed_hint,
                        true,
                    ));
                }

                // Cache miss: full build.
                let p_for_init = progress.clone();
                let on_init = move |i: usize| {
                    if let Some(ref pp) = p_for_init {
                        pp.on_init_centroid(i);
                    }
                };
                let p_for_iter = progress.clone();
                let on_iter = move |i: usize| {
                    if let Some(ref pp) = p_for_iter {
                        pp.on_kmeans_iter(i);
                    }
                };

                let centroids =
                    kmeans::train(&data, n_centroids, train_seed, max_iter, on_init, on_iter);
                let index = ivf::build_index(&data, centroids, dim);

                if index.clusters.iter().any(|c| c.is_empty()) {
                    return Err(TiptoeError::EmptyCluster);
                }

                let encoded = encode_ivf(&index, quantisation_bits, p)
                    .map_err(TiptoeError::QuantisationOverflow)?;

                // SimplePIR preprocess is the multi-hour step. Tell the
                // progress reporter how many extra ticks to expect (one per
                // row of the hint) before running it.
                if let Some(ref p) = progress {
                    p.on_phase_extend(encoded.m_max());
                }
                let p_for_simplepir = progress.clone();
                let on_pir_row = move |i: usize| {
                    if let Some(ref pp) = p_for_simplepir {
                        pp.on_encrypt(i);
                    }
                };

                let seed_bytes = derive_pir_seed(train_seed);
                let pir_setup = preprocess_with_progress(
                    encoded.matrix(),
                    encoded.m_max(),
                    encoded.width(),
                    seed_bytes,
                    &lwe_params,
                    on_pir_row,
                );

                if let Some(ref path) = disk_path_for_build {
                    let refs: Vec<&[u8]> = parts_for_build.iter().map(|v| v.as_slice()).collect();
                    if let Err(e) = save_tiptoe_cache(path, &refs, &index, &pir_setup) {
                        eprintln!("Warning: could not save Tiptoe cache to {path:?}: {e}");
                    }
                }

                let compressed_hint = decompose_hint(
                    &pir_setup.hint,
                    encoded.m_max(),
                    lwe_params.n,
                    &bfv_params_for_build,
                )?;

                if let Some(ref p) = progress {
                    p.on_build_complete();
                }

                Ok::<_, TiptoeError>((index, encoded, pir_setup, compressed_hint, false))
            })
            .await
            .map_err(|_| TiptoeError::SpawnPanic)??;

        let index = Arc::new(index);
        let encoded = Arc::new(encoded);
        let pir_setup = Arc::new(pir_setup);
        let compressed_hint = Arc::new(compressed_hint);

        self.cache.lock().unwrap().insert(
            fp,
            CacheEntry {
                index: Arc::clone(&index),
                encoded: Arc::clone(&encoded),
                pir_setup: Arc::clone(&pir_setup),
                compressed_hint: Arc::clone(&compressed_hint),
            },
        );

        let outcome = BuildOutcome {
            cache_hit: disk_hit,
            build_duration: build_start.elapsed(),
        };
        Ok((
            TiptoeHandle {
                index,
                encoded,
                pir_setup,
                compressed_hint,
                lwe_params,
                bfv_params,
                quantisation_bits,
                dim,
            },
            outcome,
        ))
    }

    async fn score(
        &self,
        handle: &TiptoeHandle,
        query: &Vector,
        k: usize,
    ) -> Result<Vec<Hit>, TiptoeError> {
        // Sample seeds on the async task — the OS RNG is `!Send` and
        // can't cross into spawn_blocking.
        let lwe_seed: u64 = rand::rng().random();
        let bfv_seed: u64 = rand::rng().random();

        let query_vec = query.0.clone();
        let h = TiptoeHandle {
            index: Arc::clone(&handle.index),
            encoded: Arc::clone(&handle.encoded),
            pir_setup: Arc::clone(&handle.pir_setup),
            compressed_hint: Arc::clone(&handle.compressed_hint),
            lwe_params: handle.lwe_params,
            bfv_params: Arc::clone(&handle.bfv_params),
            quantisation_bits: handle.quantisation_bits,
            dim: handle.dim,
        };

        tokio::task::spawn_blocking(move || {
            ranking::score_query(&h, &query_vec, k, lwe_seed, bfv_seed)
        })
        .await
        .map_err(|_| TiptoeError::SpawnPanic)?
    }

    fn communication_cost(&self, handle: &TiptoeHandle, _k: usize) -> CommunicationCost {
        // Per-query traffic splits into four directional fields. The
        // BFV-encrypted query token upload and the apply-hint download
        // are the offline round-trip; the LWE query/answer is the
        // online round-trip. `setup_bytes` is the per-client one-time
        // SimplePIR hint (paid once at corpus-fixation, reused across
        // every query).
        //
        // The online response is the LWE answer `M·ct ∈ Z_q^{m_max}`
        // (`m_max × 8` bytes); the BFV apply-hint result (`chunks ·
        // NUM_LIMBS_64 · bfv_ct_bytes`) is offline download, charged to
        // `pre_query_offline_down_bytes`.
        let lwe = &handle.lwe_params;
        let bfv = &handle.bfv_params;
        let degree = bfv.degree();
        let coeff_bytes = bfv.moduli_sizes().iter().sum::<usize>().div_ceil(8);
        // BFV ciphertext = 2 polynomials × `degree` coeffs × coeff_bytes.
        let bfv_ct_bytes = (2 * degree * coeff_bytes) as u64;

        let lwe_ct_bytes = (handle.encoded.width() * 8) as u64; // u64 per slot
        let lwe_answer_bytes = (handle.encoded.m_max() * 8) as u64; // online down
        let chunks = handle.encoded.m_max().div_ceil(degree) as u64;
        let apply_hint_bytes = chunks * (NUM_LIMBS_64 as u64) * bfv_ct_bytes;
        let token_bytes = (lwe.n as u64) * bfv_ct_bytes; // BFV-encrypted query token
        let hint_bytes = (handle.encoded.m_max() * lwe.n * 8) as u64;

        // GPU bandwidth proxy = SimplePIR matrix touched per query
        // (m_max × n_lwe × 8). Captures Tiptoe's ~4× penalty vs the fp32
        // matrices the candidate schemes stream — Tiptoe's 64-bit Z_q
        // elements vs 32-bit fp32 / signed-i32 EMVP scalars.
        let effective_bytes_per_query = (handle.encoded.m_max() * lwe.n * 8) as u64;

        CommunicationCost {
            query_bytes: lwe_ct_bytes,
            cluster_response_bytes: lwe_answer_bytes,
            response_bytes: lwe_answer_bytes,
            setup_bytes: hint_bytes,
            pre_query_offline_up_bytes: token_bytes,
            pre_query_offline_down_bytes: apply_hint_bytes,
            effective_bytes_per_query,
        }
    }
}

pub use ranking::TiptoeTiming;

impl TiptoeScorer {
    /// Off-hot-path equivalent of [`Scorer::score`] with per-substep
    /// timings. `bfv_decompress_us` covers the BFV limb-decomposition
    /// layer, which is the single-thread bottleneck.
    pub async fn score_with_breakdown(
        &self,
        handle: &TiptoeHandle,
        query: &Vector,
        k: usize,
    ) -> Result<(Vec<Hit>, TiptoeTiming), TiptoeError> {
        let lwe_seed: u64 = rand::rng().random();
        let bfv_seed: u64 = rand::rng().random();

        let query_vec = query.0.clone();
        let h = TiptoeHandle {
            index: Arc::clone(&handle.index),
            encoded: Arc::clone(&handle.encoded),
            pir_setup: Arc::clone(&handle.pir_setup),
            compressed_hint: Arc::clone(&handle.compressed_hint),
            lwe_params: handle.lwe_params,
            bfv_params: Arc::clone(&handle.bfv_params),
            quantisation_bits: handle.quantisation_bits,
            dim: handle.dim,
        };

        tokio::task::spawn_blocking(move || {
            ranking::score_query_with_breakdown(&h, &query_vec, k, lwe_seed, bfv_seed)
        })
        .await
        .map_err(|_| TiptoeError::SpawnPanic)?
    }
}
