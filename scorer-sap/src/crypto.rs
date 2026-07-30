use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

pub struct SapKey {
    pub s: f64,
    pub prf_key: [u8; 32],
}

pub struct Ciphertext {
    pub c: Vec<f64>,
}

/// Generate a SAP key from `[s_min, s_max]` using a seeded RNG for reproducibility.
pub fn keygen(s_min: f64, s_max: f64, seed: u64) -> SapKey {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    SapKey {
        s: rng.random_range(s_min..=s_max),
        prf_key: std::array::from_fn(|_| rng.random()),
    }
}

// XOR the nonce into both halves of the key so every (key, nonce) pair
// maps to a distinct ChaCha seed without a proper KDF.
fn prf_rng(prf_key: &[u8; 32], nonce: &[u8; 16]) -> ChaCha20Rng {
    let mut seed = *prf_key;
    for i in 0..16 {
        seed[i] ^= nonce[i];
        seed[i + 16] ^= nonce[i];
    }
    ChaCha20Rng::from_seed(seed)
}

// Uniform random point inside a d-ball of the given radius: a Gaussian
// random direction scaled by an x^(1/d) radial correction so the sample
// is uniform over the ball's volume rather than biased toward the centre.
fn perturbation(rng: &mut ChaCha20Rng, d: usize, radius: f64) -> Vec<f64> {
    let normal = Normal::new(0.0_f64, 1.0_f64).unwrap();
    let u: Vec<f64> = (0..d).map(|_| normal.sample(rng)).collect();
    let norm = u
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(f64::EPSILON);
    let r = radius * rng.random::<f64>().powf(1.0 / d as f64);
    u.iter().map(|v| r * v / norm).collect()
}

/// Encrypt `m` using a pre-generated `nonce`. Deterministic given (key, m, beta, nonce).
/// The caller is responsible for generating a fresh random nonce per encryption.
pub(crate) fn encrypt_with_nonce(
    key_s: f64,
    prf_key: &[u8; 32],
    m: &[f64],
    beta: f64,
    nonce: [u8; 16],
) -> Ciphertext {
    let mut prng = prf_rng(prf_key, &nonce);
    let lambda = perturbation(&mut prng, m.len(), key_s * beta / 4.0);
    Ciphertext {
        c: m.iter()
            .zip(&lambda)
            .map(|(mi, li)| key_s * mi + li)
            .collect(),
    }
}
