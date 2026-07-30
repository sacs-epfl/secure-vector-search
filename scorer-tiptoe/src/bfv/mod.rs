//! BFV second-layer encryption for Tiptoe's ciphertext compression.
//!
//! We adopt [`fhe`](https://docs.rs/fhe) (a.k.a. fhe.rs by Tristan
//! Lepoint) as the BFV implementation, contingent on parameter parity
//! with the Go reference's SEAL 4.1.1 defaults. Parity is verified by
//! [`seal_compat::verify`]; the limb-decomposition glue that consumes
//! these parameters lives in `compress.rs`.
//!
//! Re-exports are deliberately narrow — only the types needed by
//! `compress.rs` and downstream Tiptoe code. We do not blanket-re-
//! export `fhe::bfv::*` because doing so would let callers reach for
//! BFV configurations that have not been parity-checked.
//!
//! ### Communication-cost accounting
//!
//! The per-query BFV-encrypted query token is charged to
//! `CommunicationCost.pre_query_offline_up_bytes`, not `setup_bytes`,
//! because at our 100k MS-MARCO corpus (`n_centroids = 317`,
//! `dim = 768`, `m_max = 977`) the token dominates the one-time hint:
//!
//! - `token_bytes  = lwe.n × bfv_ct_bytes = 2048 × 20480 = 41,943,040`
//! - `hint_bytes   = m_max × lwe.n × 8   =  977 × 2048 × 8 = 16,007,168`
//!
//! `setup_bytes` therefore carries `hint_bytes` only, the
//! BFV-encrypted apply-hint result is charged to
//! `pre_query_offline_down_bytes`, and neither is folded into the
//! online `response_bytes`. Old `raw.csv` files that pre-date the split
//! keep the bundled value; `analysis/preprocess.py` defaults the new
//! columns to 0 for them so the legacy view is preserved.

use std::sync::Arc;

pub use fhe::bfv::{BfvParameters, Ciphertext, Plaintext, PublicKey, SecretKey};
pub use seal_compat::{SealCompatError, verify as verify_seal_compat};

pub mod compress;
pub mod seal_compat;

/// Build the canonical Tiptoe-text BFV parameters that mirror the Go
/// reference's SEAL 4.1.1 defaults, and verify parity through
/// [`seal_compat::verify`]. Panics with a clear `SealCompatError`
/// message if parity fails — this is the **fail-closed** path that
/// guarantees we never silently use a BFV noise budget that diverges
/// from the Go reference.
///
/// First call constructs and validates; subsequent calls return the
/// same `Arc` (see source for the cache mechanism). Either way the
/// parity check has already passed by the time the caller sees the
/// `Arc`.
pub fn tiptoe_text_params() -> Arc<BfvParameters> {
    use fhe::bfv::BfvParametersBuilder;
    use std::sync::OnceLock;

    static CACHED: OnceLock<Arc<BfvParameters>> = OnceLock::new();

    CACHED
        .get_or_init(|| {
            let params = BfvParametersBuilder::new()
                .set_degree(seal_compat::EXPECTED_POLY_DEGREE)
                .set_plaintext_modulus(seal_compat::EXPECTED_PLAINTEXT_MODULUS)
                .set_moduli_sizes(seal_compat::EXPECTED_COEFF_MODULUS_BITS)
                .build()
                .expect("canonical SEAL-parity BFV params must build under fhe");
            seal_compat::verify(&params).expect("seal_compat parity check failed at module init");
            Arc::new(params)
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fhe::bfv::{Encoding, PublicKey};
    use fhe_traits::{FheDecoder, FheDecrypter, FheEncoder, FheEncrypter};
    // fhe / fhe-traits 0.1.1 pin rand 0.9; the renamed deps in
    // Cargo.toml `[dev-dependencies]` give us a compatible RNG.
    use rand_chacha_fhe::ChaCha20Rng;
    use rand_fhe::SeedableRng;

    #[test]
    fn tiptoe_text_params_pass_parity_gate() {
        let params = tiptoe_text_params();
        // If we got here, the OnceLock init succeeded — i.e., the
        // parity check did not panic. Re-run it explicitly for
        // belt-and-braces.
        verify_seal_compat(&params).expect("parity must hold");
        assert_eq!(params.degree(), 2048);
        assert_eq!(params.plaintext(), 65537);
        assert_eq!(params.moduli_sizes(), &[38][..]);
    }

    #[test]
    fn fhe_bfv_encrypt_decrypt_roundtrip_at_tiptoe_params() {
        // Smoke test: encrypt a small plaintext and decrypt back. This
        // confirms the chosen parameters yield a working BFV instance
        // (NTT-compatible primes, valid keygen, etc.) — i.e., the
        // parity gate is not just structurally correct but
        // operationally usable.
        let params = tiptoe_text_params();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let sk = SecretKey::random(&params, &mut rng);
        let pk = PublicKey::new(&sk, &mut rng);

        // Encode a small Vec<u64> via PolyEncoding so the values land
        // in the polynomial coefficients (this is the encoding path
        // the limb decomposition uses).
        let msg: Vec<u64> = vec![1, 2, 3, 4, 5];
        let pt = Plaintext::try_encode(&msg, Encoding::poly(), &params)
            .expect("encode should succeed for short u64 slice");
        let ct: Ciphertext = pk
            .try_encrypt(&pt, &mut rng)
            .expect("encrypt should succeed");

        let pt_back = sk.try_decrypt(&ct).expect("decrypt should succeed");
        let recovered =
            Vec::<u64>::try_decode(&pt_back, Encoding::poly()).expect("decode should succeed");

        // Decoded plaintext is the polynomial-coefficient vector of
        // length `params.degree()`; the first `msg.len()` slots match
        // our message, the rest are zero-padded.
        assert_eq!(&recovered[..msg.len()], &msg[..]);
        assert!(recovered[msg.len()..].iter().all(|&v| v == 0));
    }
}
