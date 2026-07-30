use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;
use scorer_core::{Hit, cache};

use crate::distance::l2_f32;
use crate::kmeans::{Centroid, assign};

pub struct IvfIndex {
    pub centroids: Vec<Centroid>,
    /// clusters[c] = list of (vector_id, vector) assigned to centroid c
    pub clusters: Vec<Vec<(u32, Vec<f32>)>>,
}

pub fn build_index(data: &[Vec<f32>], centroids: Vec<Centroid>, _dim: usize) -> IvfIndex {
    let assignments = assign(data, &centroids);
    let k = centroids.len();
    let mut clusters: Vec<Vec<(u32, Vec<f32>)>> = vec![Vec::new(); k];
    for (i, (v, &c)) in data.iter().zip(&assignments).enumerate() {
        clusters[c].push((i as u32, v.clone()));
    }
    IvfIndex {
        centroids,
        clusters,
    }
}

/// Binary format (little-endian), payload only (no cache header):
///   [u64 dim] [u64 n_centroids] [centroids: n_centroids × dim × f32]
///   [u64 n_clusters]
///     for each cluster: [u64 size] then size × ([u32 id] [dim × f32 vector])
pub fn save_index(index: &IvfIndex, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    write_index_to(&mut w, index)
}

/// Path-based payload-only loader (no cache header).
pub fn load_index(path: &Path) -> io::Result<IvfIndex> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    read_index_from(&mut r)
}

/// Save with a self-verifying cache header (see `scorer_core::cache`).
/// Filename convention is the caller's responsibility; this helper only
/// adds the header so a subsequent `load_index_with_header` call can
/// detect parameter drift.
pub fn save_index_with_header(index: &IvfIndex, path: &Path, parts: &[&[u8]]) -> io::Result<()> {
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    cache::write_header(&mut w, parts)?;
    write_index_to(&mut w, index)?;
    Ok(())
}

/// Load only if the cache header matches `parts`; mismatch returns `Ok(None)`
/// (treat as cache miss, rebuild from scratch).
pub fn load_index_with_header(path: &Path, parts: &[&[u8]]) -> io::Result<Option<IvfIndex>> {
    let mut r = BufReader::new(std::fs::File::open(path)?);
    if !cache::verify_header(&mut r, parts)? {
        return Ok(None);
    }
    Ok(Some(read_index_from(&mut r)?))
}

/// Write the IVF payload (no cache header) to an arbitrary `Write`.
/// Use this when embedding an IVF index inside a larger composite cache
/// file — the caller is responsible for the cache header.
pub fn write_index_to<W: Write>(w: &mut W, index: &IvfIndex) -> io::Result<()> {
    let dim = index.centroids.first().map_or(0, |c| c.len());
    w.write_all(&(dim as u64).to_le_bytes())?;
    w.write_all(&(index.centroids.len() as u64).to_le_bytes())?;
    for c in &index.centroids {
        for &v in c {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.write_all(&(index.clusters.len() as u64).to_le_bytes())?;
    for cluster in &index.clusters {
        w.write_all(&(cluster.len() as u64).to_le_bytes())?;
        for (id, vec) in cluster {
            w.write_all(&id.to_le_bytes())?;
            for &v in vec {
                w.write_all(&v.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

/// Read the IVF payload (no cache header) from an arbitrary `Read`.
pub fn read_index_from<R: Read>(r: &mut R) -> io::Result<IvfIndex> {
    let mut u64buf = [0u8; 8];
    let mut u32buf = [0u8; 4];

    let mut read_usize = |r: &mut R| -> io::Result<usize> {
        r.read_exact(&mut u64buf)?;
        Ok(u64::from_le_bytes(u64buf) as usize)
    };

    let dim = read_usize(r)?;
    let n_centroids = read_usize(r)?;
    let mut centroids = Vec::with_capacity(n_centroids);
    for _ in 0..n_centroids {
        let mut c = Vec::with_capacity(dim);
        for _ in 0..dim {
            r.read_exact(&mut u32buf)?;
            c.push(f32::from_le_bytes(u32buf));
        }
        centroids.push(c);
    }

    let n_clusters = read_usize(r)?;
    let mut clusters = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let size = read_usize(r)?;
        let mut cluster = Vec::with_capacity(size);
        for _ in 0..size {
            r.read_exact(&mut u32buf)?;
            let id = u32::from_le_bytes(u32buf);
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                r.read_exact(&mut u32buf)?;
                vec.push(f32::from_le_bytes(u32buf));
            }
            cluster.push((id, vec));
        }
        clusters.push(cluster);
    }

    Ok(IvfIndex {
        centroids,
        clusters,
    })
}

/// Substep timings for `probe_with_breakdown`: `route` ranks centroids
/// and picks `nprobe` clusters, `distance` scans those clusters in
/// parallel, `merge` is the top-k sort. `score` itself is unchanged and
/// stays free of `Instant::now()` calls (off the hot path).
#[derive(Debug, Clone, Copy, Default)]
pub struct ProbeTiming {
    pub route_us: u64,
    pub distance_us: u64,
    pub merge_us: u64,
}

/// Pick the `nprobe` nearest centroids by L2. Public so scorer crates
/// whose scoring path doesn't go through `probe` (EMVP / BN run their
/// own encrypted compute) can still share the routing step.
pub fn probe_route(index: &IvfIndex, query: &[f32], nprobe: usize) -> Vec<usize> {
    let mut centroid_dists: Vec<(usize, f32)> = index
        .centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_f32(query, c)))
        .collect();
    centroid_dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    let nprobe = nprobe.min(index.centroids.len());
    centroid_dists
        .into_iter()
        .take(nprobe)
        .map(|(c, _)| c)
        .collect()
}

/// Compute (id, l2) for every vector across the probed clusters,
/// rayon-parallel over the probe set.
pub fn probe_distance(index: &IvfIndex, query: &[f32], probe_set: &[usize]) -> Vec<(u32, f32)> {
    probe_set
        .par_iter()
        .flat_map(|c| {
            index.clusters[*c]
                .iter()
                .map(|(id, v)| (*id, l2_f32(query, v)))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Sort `dists` ascending by distance, take the top `k`, return as
/// `Hit` (score = −distance).
pub fn probe_merge(mut dists: Vec<(u32, f32)>, k: usize) -> Vec<Hit> {
    let k = k.min(dists.len());
    if k == 0 {
        return Vec::new();
    }
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists[..k]
        .iter()
        .map(|(id, dist)| Hit {
            id: *id,
            score: -dist,
        })
        .collect()
}

pub fn probe(index: &IvfIndex, query: &[f32], nprobe: usize, k: usize) -> Vec<Hit> {
    let probe_set = probe_route(index, query, nprobe);
    let dists = probe_distance(index, query, &probe_set);
    probe_merge(dists, k)
}

/// Like [`probe`] but also returns the realised probe set (cluster
/// indices the routing step picked). Single routing pass — the caller
/// gets both the merged hits and the cluster-id list to drive
/// per-query realised communication-cost accounting.
pub fn probe_with_route(
    index: &IvfIndex,
    query: &[f32],
    nprobe: usize,
    k: usize,
) -> (Vec<Hit>, Vec<usize>) {
    let probe_set = probe_route(index, query, nprobe);
    let dists = probe_distance(index, query, &probe_set);
    let hits = probe_merge(dists, k);
    (hits, probe_set)
}

/// Batched `probe` over `queries`. Per-query result is bit-equal to
/// calling [`probe`] in a loop: same `probe_route` per query (so same
/// probed-cluster set in the same order), same `l2_f32` per (query,
/// vector) pair (so same scalar distance), same `probe_merge` (so same
/// sort and top-k). The amortisation is memory-bandwidth: each cluster's
/// vectors are iterated once per call and contribute distances to every
/// query that probed that cluster, instead of once per query in the
/// batch.
pub fn probe_batch(index: &IvfIndex, queries: &[&[f32]], nprobe: usize, k: usize) -> Vec<Vec<Hit>> {
    if queries.is_empty() {
        return Vec::new();
    }

    let probe_sets: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| probe_route(index, q, nprobe))
        .collect();

    // Invert: for each cluster, list (query_idx, position_in_probe_set)
    // pairs that probed it.
    let mut per_cluster_qpos: Vec<Vec<(usize, usize)>> = vec![Vec::new(); index.clusters.len()];
    for (q_idx, probes) in probe_sets.iter().enumerate() {
        for (pos, &c_idx) in probes.iter().enumerate() {
            per_cluster_qpos[c_idx].push((q_idx, pos));
        }
    }

    // For each cluster touched by at least one query, iterate its
    // vectors once and produce a per-(query, position) bin of
    // (id, l2_f32) pairs in cluster-iteration order. Bins are scattered
    // back into per-query slots downstream; each cluster task is
    // independent so the par_iter has no cross-task writes.
    type Bin = (usize, usize, Vec<(u32, f32)>);
    let cluster_contribs: Vec<Vec<Bin>> = per_cluster_qpos
        .par_iter()
        .enumerate()
        .map(|(c_idx, qpos_list)| {
            if qpos_list.is_empty() {
                return Vec::new();
            }
            let cluster = &index.clusters[c_idx];
            let mut bins: Vec<Bin> = qpos_list
                .iter()
                .map(|&(q_idx, pos)| (q_idx, pos, Vec::with_capacity(cluster.len())))
                .collect();
            for (id, v) in cluster {
                for entry in bins.iter_mut() {
                    let q_idx = entry.0;
                    entry.2.push((*id, l2_f32(queries[q_idx], v)));
                }
            }
            bins
        })
        .collect();

    // Scatter contribs into per-query slots indexed by probe position,
    // preserving the per-query dist ordering `probe_distance` would
    // have produced (cluster[probe_set[0]] then cluster[probe_set[1]],
    // …, with cluster-iteration order inside each).
    let mut bins_per_query: Vec<Vec<Vec<(u32, f32)>>> = probe_sets
        .iter()
        .map(|ps| vec![Vec::new(); ps.len()])
        .collect();
    for cluster_list in cluster_contribs {
        for (q_idx, pos, bin) in cluster_list {
            bins_per_query[q_idx][pos] = bin;
        }
    }

    bins_per_query
        .into_iter()
        .map(|bins| {
            let dists: Vec<(u32, f32)> = bins.into_iter().flatten().collect();
            probe_merge(dists, k)
        })
        .collect()
}

/// Like [`probe_batch`] but with separate routing and scoring queries.
/// Used by encrypted-IVF scorers (e.g. SAP-IVF) that route on a
/// plaintext query against plaintext centroids but score with an
/// encrypted query against encrypted per-cluster contents. Per-query
/// result is bit-equal to a `probe_route(routing_queries[i]) +
/// probe_distance(scoring_queries[i], probe_set) + probe_merge` loop;
/// the amortisation is the same memory-bandwidth win `probe_batch`
/// gets — each cluster's vectors are iterated once per call instead of
/// once per query.
pub fn probe_batch_split(
    index: &IvfIndex,
    routing_queries: &[&[f32]],
    scoring_queries: &[&[f32]],
    nprobe: usize,
    k: usize,
) -> Vec<Vec<Hit>> {
    assert_eq!(
        routing_queries.len(),
        scoring_queries.len(),
        "probe_batch_split: routing and scoring query slices must align 1:1"
    );
    if routing_queries.is_empty() {
        return Vec::new();
    }

    let probe_sets: Vec<Vec<usize>> = routing_queries
        .iter()
        .map(|q| probe_route(index, q, nprobe))
        .collect();

    let mut per_cluster_qpos: Vec<Vec<(usize, usize)>> = vec![Vec::new(); index.clusters.len()];
    for (q_idx, probes) in probe_sets.iter().enumerate() {
        for (pos, &c_idx) in probes.iter().enumerate() {
            per_cluster_qpos[c_idx].push((q_idx, pos));
        }
    }

    type Bin = (usize, usize, Vec<(u32, f32)>);
    let cluster_contribs: Vec<Vec<Bin>> = per_cluster_qpos
        .par_iter()
        .enumerate()
        .map(|(c_idx, qpos_list)| {
            if qpos_list.is_empty() {
                return Vec::new();
            }
            let cluster = &index.clusters[c_idx];
            let mut bins: Vec<Bin> = qpos_list
                .iter()
                .map(|&(q_idx, pos)| (q_idx, pos, Vec::with_capacity(cluster.len())))
                .collect();
            for (id, v) in cluster {
                for entry in bins.iter_mut() {
                    let q_idx = entry.0;
                    entry.2.push((*id, l2_f32(scoring_queries[q_idx], v)));
                }
            }
            bins
        })
        .collect();

    let mut bins_per_query: Vec<Vec<Vec<(u32, f32)>>> = probe_sets
        .iter()
        .map(|ps| vec![Vec::new(); ps.len()])
        .collect();
    for cluster_list in cluster_contribs {
        for (q_idx, pos, bin) in cluster_list {
            bins_per_query[q_idx][pos] = bin;
        }
    }

    bins_per_query
        .into_iter()
        .map(|bins| {
            let dists: Vec<(u32, f32)> = bins.into_iter().flatten().collect();
            probe_merge(dists, k)
        })
        .collect()
}

/// `probe` with per-substep timings. Off the hot path: the
/// `Instant::now()` pairs add nanoseconds per substep. Use only from
/// `score_with_breakdown`.
pub fn probe_with_breakdown(
    index: &IvfIndex,
    query: &[f32],
    nprobe: usize,
    k: usize,
) -> (Vec<Hit>, ProbeTiming) {
    let t = std::time::Instant::now();
    let probe_set = probe_route(index, query, nprobe);
    let route_us = t.elapsed().as_micros() as u64;

    let t = std::time::Instant::now();
    let dists = probe_distance(index, query, &probe_set);
    let distance_us = t.elapsed().as_micros() as u64;

    let t = std::time::Instant::now();
    let hits = probe_merge(dists, k);
    let merge_us = t.elapsed().as_micros() as u64;

    (
        hits,
        ProbeTiming {
            route_us,
            distance_us,
            merge_us,
        },
    )
}
