use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::distance::l2_f32;

pub type Centroid = Vec<f32>;

/// Train k centroids from `data` using Lloyd's algorithm with k-means++ init.
/// Deterministic given `seed`. Panics if `k == 0` or `k > data.len()` — callers
/// validate before calling.
/// `on_init_centroid` is called after each centroid is chosen during k-means++ init
/// (1-based, so k calls total). `on_iter` is called after each Lloyd's iteration.
pub fn train(
    data: &[Vec<f32>],
    k: usize,
    seed: u64,
    max_iter: usize,
    on_init_centroid: impl Fn(usize),
    on_iter: impl Fn(usize),
) -> Vec<Centroid> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut centroids = kmeans_plus_plus_init(data, k, &mut rng, &on_init_centroid);

    for i in 0..max_iter {
        let assignments = assign(data, &centroids);
        let new_centroids = recompute(data, &assignments, k, data[0].len());
        on_iter(i + 1);
        if new_centroids == centroids {
            break;
        }
        centroids = new_centroids;
    }
    centroids
}

/// Assign each point to its nearest centroid. Returns a Vec of centroid indices,
/// one per data point.
pub fn assign(data: &[Vec<f32>], centroids: &[Centroid]) -> Vec<usize> {
    data.par_iter()
        .map(|v| {
            centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, l2_f32(v, c)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0
        })
        .collect()
}

fn recompute(data: &[Vec<f32>], assignments: &[usize], k: usize, dim: usize) -> Vec<Centroid> {
    let mut sums = vec![vec![0.0_f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (v, &c) in data.iter().zip(assignments) {
        for (s, x) in sums[c].iter_mut().zip(v) {
            *s += x;
        }
        counts[c] += 1;
    }
    sums.into_iter()
        .zip(&counts)
        .map(|(s, &n)| {
            if n == 0 {
                vec![0.0; dim]
            } else {
                s.into_iter().map(|x| x / n as f32).collect()
            }
        })
        .collect()
}

// k-means++ initialisation: first centroid chosen uniformly at random, subsequent
// centroids chosen with probability proportional to squared distance to nearest
// existing centroid. Incremental: maintain `min_dist_sq[i]` = min over already-
// chosen centroids of `l2(data[i], c)²`; each pick recomputes distance to only
// the *new* centroid and elementwise-mins into the running array. Total work is
// O(N·k) rather than the O(N·k²) of the recompute-from-scratch shape.
// Output is bit-identical to the recompute-from-scratch variant given the same
// seed: RNG draws happen in the same order (one `random_range` then `k-1`
// `random::<f32>()` calls), and `total = Σ min_dist_sq[i]` is mathematically
// equal to `Σ min over centroids of l2(v_i, c)²` by construction.
fn kmeans_plus_plus_init(
    data: &[Vec<f32>],
    k: usize,
    rng: &mut ChaCha20Rng,
    on_centroid: &impl Fn(usize),
) -> Vec<Centroid> {
    let mut centroids: Vec<Centroid> = Vec::with_capacity(k);
    let first = rng.random_range(0..data.len());
    centroids.push(data[first].clone());
    on_centroid(1);

    let mut min_dist_sq: Vec<f32> = data
        .par_iter()
        .map(|v| {
            let d = l2_f32(v, &centroids[0]);
            d * d
        })
        .collect();

    for ci in 1..k {
        let total: f32 = min_dist_sq.par_iter().sum();
        let mut threshold = rng.random::<f32>() * total;
        let mut chosen = data.len() - 1;
        for (i, &d) in min_dist_sq.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());

        let new_c = &centroids[ci];
        min_dist_sq
            .par_iter_mut()
            .zip(data.par_iter())
            .for_each(|(m, v)| {
                let d = l2_f32(v, new_c);
                let d_sq = d * d;
                if d_sq < *m {
                    *m = d_sq;
                }
            });

        on_centroid(ci + 1);
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect()
    }

    #[test]
    fn kmeans_plus_plus_init_deterministic_under_seed() {
        let data = synthetic_data(200, 4, 7);
        let mut r1 = ChaCha20Rng::seed_from_u64(42);
        let mut r2 = ChaCha20Rng::seed_from_u64(42);
        let a = kmeans_plus_plus_init(&data, 16, &mut r1, &|_| {});
        let b = kmeans_plus_plus_init(&data, 16, &mut r2, &|_| {});
        assert_eq!(a.len(), 16);
        assert_eq!(a, b, "init must be deterministic given the same seed");
    }

    #[test]
    fn train_is_deterministic_and_assigns_every_point() {
        let data = synthetic_data(400, 8, 11);
        let c1 = train(&data, 12, 99, 20, |_| {}, |_| {});
        let c2 = train(&data, 12, 99, 20, |_| {}, |_| {});
        assert_eq!(
            c1, c2,
            "train output must be deterministic given the same seed"
        );

        let assignments = assign(&data, &c1);
        assert_eq!(assignments.len(), data.len());
        assert!(assignments.iter().all(|&a| a < 12));
    }

    #[test]
    fn init_picks_distinct_centroids_for_well_separated_clusters() {
        // Three obvious clusters along axis 0; k-means++ should pick one
        // candidate from each region rather than collapsing onto a single
        // dense neighbourhood.
        let mut data = Vec::new();
        for &base in &[0.0_f32, 100.0, 200.0] {
            for i in 0..30 {
                data.push(vec![base + (i as f32) * 0.01, (i as f32) * 0.01]);
            }
        }
        let centroids = train(&data, 3, 1, 25, |_| {}, |_| {});
        let mut bases: Vec<i32> = centroids
            .iter()
            .map(|c| (c[0] / 100.0).round() as i32)
            .collect();
        bases.sort();
        assert_eq!(bases, vec![0, 1, 2]);
    }
}
