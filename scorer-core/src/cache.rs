//! Hashed cache-filename helper and self-verifying header used by all
//! disk-backed scorer caches.
//!
//! Filename convention: `.<scorer>-cache-<16hex>.bin` where
//! `<16hex>` is the first 16 hex chars of [`fingerprint64`] over the
//! canonical-encoded parameter tuple. The first byte slice in `parts`
//! must be a scorer-discriminating string (e.g., `b"plaintext-ivf"`,
//! `b"sap-ivf"`, `b"emvp-flat"`, `b"emvp-ivf"`) so two scorers cannot
//! produce the same fingerprint for accidentally-identical numeric
//! tuples.
//!
//! The on-disk header (32 bytes magic+version+hash) lets a load path
//! detect corruption or version drift cheaply: the header is regenerated
//! from the expected parts and compared bit-for-bit. A mismatch is
//! treated as a cache miss; the caller rebuilds from scratch.

use std::io::{self, Read, Write};

pub const MAGIC: [u8; 8] = *b"\x89SVScach";
pub const VERSION: u32 = 1;
const HEADER_BYTES: usize = MAGIC.len() + 4 + 8;

/// FNV-1a 64-bit over the length-prefixed concatenation of `parts`.
pub fn fingerprint64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for part in parts {
        let len = part.len() as u64;
        for b in len.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// First 16 hex chars of [`fingerprint64`]; used in cache filenames.
pub fn fingerprint(parts: &[&[u8]]) -> String {
    format!("{:016x}", fingerprint64(parts))
}

/// Write `MAGIC | VERSION | fingerprint64(parts)` to `file`.
pub fn write_header(file: &mut impl Write, parts: &[&[u8]]) -> io::Result<()> {
    file.write_all(&MAGIC)?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&fingerprint64(parts).to_le_bytes())?;
    Ok(())
}

/// Read the header from `file` and return whether it matches `expected_parts`.
/// I/O failures, magic/version mismatches, and hash mismatches all return
/// `Ok(false)` — every "this cache is unusable" case folds into one branch
/// at the call site.
pub fn verify_header(file: &mut impl Read, expected_parts: &[&[u8]]) -> io::Result<bool> {
    let mut buf = [0u8; HEADER_BYTES];
    if file.read_exact(&mut buf).is_err() {
        return Ok(false);
    }
    if buf[..MAGIC.len()] != MAGIC {
        return Ok(false);
    }
    let v_off = MAGIC.len();
    let h_off = v_off + 4;
    let version = u32::from_le_bytes(buf[v_off..h_off].try_into().unwrap());
    if version != VERSION {
        return Ok(false);
    }
    let hash = u64::from_le_bytes(buf[h_off..].try_into().unwrap());
    Ok(hash == fingerprint64(expected_parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn fingerprint_is_deterministic_and_distinguishes_scorers() {
        let f1 = fingerprint(&[b"plaintext-ivf", &[1, 0, 0, 0]]);
        let f2 = fingerprint(&[b"sap-ivf", &[1, 0, 0, 0]]);
        let f3 = fingerprint(&[b"plaintext-ivf", &[1, 0, 0, 0]]);
        assert_ne!(
            f1, f2,
            "different scorer tags must produce different fingerprints"
        );
        assert_eq!(f1, f3, "deterministic for identical inputs");
        assert_eq!(f1.len(), 16);
        assert!(f1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn fingerprint_distinguishes_part_boundaries() {
        // Length-prefixing means concatenation differences are caught:
        // ["ab", "c"] != ["a", "bc"].
        let f1 = fingerprint(&[b"ab", b"c"]);
        let f2 = fingerprint(&[b"a", b"bc"]);
        assert_ne!(f1, f2);
    }

    #[test]
    fn header_roundtrip_verifies() {
        let mut buf = Vec::new();
        write_header(&mut buf, &[b"emvp-ivf", &[42u8; 4]]).unwrap();
        assert_eq!(buf.len(), HEADER_BYTES);

        let mut cursor = Cursor::new(&buf);
        assert!(verify_header(&mut cursor, &[b"emvp-ivf", &[42u8; 4]]).unwrap());
    }

    #[test]
    fn header_rejects_mismatched_parts() {
        let mut buf = Vec::new();
        write_header(&mut buf, &[b"emvp-ivf", &[42u8; 4]]).unwrap();

        let mut cursor = Cursor::new(&buf);
        assert!(!verify_header(&mut cursor, &[b"emvp-ivf", &[43u8; 4]]).unwrap());
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut buf = vec![0u8; HEADER_BYTES];
        let mut cursor = Cursor::new(&buf[..]);
        assert!(!verify_header(&mut cursor, &[b"emvp-ivf"]).unwrap());

        // Truncated header → cache miss, not error.
        buf.truncate(4);
        let mut cursor = Cursor::new(&buf[..]);
        assert!(!verify_header(&mut cursor, &[b"emvp-ivf"]).unwrap());
    }
}
