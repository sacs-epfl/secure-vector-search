/// Mersenne prime p = 2^61 - 1.
pub const P: u64 = (1u64 << 61) - 1;

/// Bytes per field element (u64, 8 bytes).
pub const FIELD_BYTES: usize = 8;

/// Quantisation scale: f32 → F_p via round(v * Q).
/// Q = 2^20 keeps inner-product accumulations well below p for unit vectors
/// (ell0 * Q^2 ≈ 2^50 << 2^61 - 1).
pub const Q: u64 = 1u64 << 20;

/// Structural parameters for one EMVP parameter set.
pub struct EmvpParams {
    /// Codeword length of the underlying dual code.
    pub n: usize,
    /// Code dimension (n - ell0).
    pub k: usize,
    /// Number of blocks in the random block partition.
    pub s: usize,
    /// Block size (n / s).
    pub b: usize,
    /// Plaintext dimension (n - k).
    pub ell0: usize,
}

/// Sec128 EMVP parameters targeting 128-bit security.
/// Random block partition variant: λ=128, f=1.25, ℓ_0=1024.
pub const SEC128: EmvpParams = EmvpParams {
    n: 1292,
    k: 268,
    s: 76,
    b: 17,
    ell0: 1024,
};
