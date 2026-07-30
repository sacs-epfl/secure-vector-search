//! Streaming cluster store with LRU eviction.
//!
//! See the crate-level docs for the design rationale. This module
//! holds the trait + the eviction-bookkeeping machinery; the actual
//! CUDA primitives (context, stream, NVML) live in the caller.
//!
//! Internally the bookkeeping is factored into [`LruInner`] which
//! doesn't reference any CUDA types — that's the surface the unit
//! tests exercise without requiring a working CUDA driver on CI.
//! [`ClusterStore`] wraps `LruInner` with the streams + context the
//! upload closure needs.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaStream};

use crate::error::ClusterStoreError;

// ---------------------------------------------------------------------------
// Streaming-store telemetry
// ---------------------------------------------------------------------------
//
// Process-wide aggregate counters across every `ClusterStore` in this
// process. The eval-harness runs one scheme per binary invocation, so a
// global is the simplest uniform read surface (no per-scorer accessor
// plumbing). The eval-harness snapshots `global_stats()` near
// `started_at` and again at end-of-run; the delta lands in
// `run-metadata.toml [memory]` alongside CPU page-fault counters.
//
// `Ordering::Relaxed` is sound here: the counters are monotonic, the
// eval-harness reads them sequentially after the scoring loop has
// quiesced, and we don't synchronise other state through them.

static GLOBAL_GETS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_UPLOADS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_EVICTIONS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_BYTES_UPLOADED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the process-wide `ClusterStore` counters. Read once at
/// the start of the scoring loop and once at the end; subtract for the
/// per-run aggregate.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterStoreStats {
    /// Total `get_or_upload` calls (hits + misses).
    pub gets: u64,
    /// Misses that ran the upload closure successfully (insert
    /// completed). Excludes the race short-circuit case where another
    /// thread won the insert.
    pub uploads: u64,
    /// LRU tail pops driven by `pre_evict_for`. A re-uploaded cluster
    /// is at least one eviction + one upload.
    pub evictions: u64,
    /// Sum of `DeviceCluster::device_bytes()` across every successful
    /// upload. Captures the H2D bandwidth the streaming pattern paid;
    /// `bytes_uploaded > resident_high_water` is the thrashing
    /// signature.
    pub bytes_uploaded: u64,
}

impl ClusterStoreStats {
    pub fn delta(self, base: Self) -> Self {
        ClusterStoreStats {
            gets: self.gets.saturating_sub(base.gets),
            uploads: self.uploads.saturating_sub(base.uploads),
            evictions: self.evictions.saturating_sub(base.evictions),
            bytes_uploaded: self.bytes_uploaded.saturating_sub(base.bytes_uploaded),
        }
    }
}

/// Snapshot the process-wide `ClusterStore` counters.
pub fn global_stats() -> ClusterStoreStats {
    ClusterStoreStats {
        gets: GLOBAL_GETS.load(Ordering::Relaxed),
        uploads: GLOBAL_UPLOADS.load(Ordering::Relaxed),
        evictions: GLOBAL_EVICTIONS.load(Ordering::Relaxed),
        bytes_uploaded: GLOBAL_BYTES_UPLOADED.load(Ordering::Relaxed),
    }
}

/// Per-cluster device-resident payload.
///
/// `device_bytes()` is consulted on miss-path insert to track the
/// store's bookkeeping `resident_bytes` for budget enforcement.
/// Implementors must report bytes that *they themselves* allocated on
/// device (the `CudaSlice<u64>` size, the cuVS `BfIndex`'s device
/// payload size, etc.). Internal cuVS / cudarc / RMM allocations the
/// implementor didn't make are tracked by the [`crate::PeakVramSampler`]
/// instead — it samples NVML directly and is the source of truth for
/// the realised peak that lands in `run-metadata.toml`.
///
/// Bound: `Send + 'static`. The payload moves into the store's
/// `Arc<Mutex<C>>`, which is in turn accessed by the
/// `tokio::task::spawn_blocking` worker pool that the scorers route
/// score calls through. `Sync` is **not** required — the store wraps
/// each cluster in a `Mutex` so single-threaded access is the
/// public invariant.
pub trait DeviceCluster: Send + 'static {
    fn device_bytes(&self) -> u64;
}

/// Handle returned by [`ClusterStore::get_or_upload`]. Locks the
/// underlying cluster on demand via [`LruCell::lock`]; dropping the
/// handle does not evict, only releases the caller's strong count.
pub struct LruCell<C: DeviceCluster> {
    inner: Arc<Mutex<C>>,
}

impl<C: DeviceCluster> Clone for LruCell<C> {
    fn clone(&self) -> Self {
        LruCell {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<C: DeviceCluster> LruCell<C> {
    pub fn lock(&self) -> std::sync::MutexGuard<'_, C> {
        self.inner.lock().expect("LruCell mutex poisoned")
    }
}

struct ClusterEntry<C: DeviceCluster> {
    payload: Arc<Mutex<C>>,
    bytes: u64,
}

/// LRU + budget bookkeeping. CUDA-free so unit tests can exercise it
/// on any host; [`ClusterStore`] wraps this together with the CUDA
/// context + streams the caller needs to actually do uploads.
struct LruInner<C: DeviceCluster> {
    entries: HashMap<usize, ClusterEntry<C>>,
    /// Recency queue. Front = LRU (next evicted), back = MRU.
    /// We update via O(n) `remove` + `push_back` on hit — fine for
    /// nprobe ≤ a few thousand, which dominates the v1 measurement
    /// scope. If the per-query nprobe grows past low-thousands the
    /// bump can switch to a doubly-linked-list LRU; deferred.
    order: VecDeque<usize>,
    /// Bookkeeping sum of `device_bytes()` across `entries`. Used
    /// for budget enforcement; the actual NVML peak is sampled
    /// separately (see [`crate::PeakVramSampler`]).
    resident_bytes: u64,
    budget_bytes: u64,
}

impl<C: DeviceCluster> LruInner<C> {
    fn new(budget_bytes: u64) -> Self {
        LruInner {
            entries: HashMap::new(),
            order: VecDeque::new(),
            resident_bytes: 0,
            budget_bytes,
        }
    }

    /// Look up `cluster_id` and bump its recency on hit. `None` on
    /// miss. The caller runs its upload closure outside the LRU
    /// lock, then calls [`Self::insert_or_keep`].
    fn touch(&mut self, cluster_id: usize) -> Option<Arc<Mutex<C>>> {
        let entry = self.entries.get(&cluster_id)?;
        let payload = Arc::clone(&entry.payload);
        if let Some(pos) = self.order.iter().position(|&id| id == cluster_id) {
            self.order.remove(pos);
        }
        self.order.push_back(cluster_id);
        Some(payload)
    }

    /// Pre-evict LRU tail until `requested_bytes` would fit on top
    /// of `resident_bytes`. Called by the miss path *before* the
    /// upload closure runs, so the closure executes with the budget
    /// margin already cleared.
    ///
    /// `requested_bytes` is a caller-supplied a-priori hint —
    /// usually the deterministic device-side size of the payload
    /// (`m × dim × 4` for plaintext fp32, `m × n × 8` for emvp,
    /// etc.). The post-upload `payload.device_bytes()` is the
    /// authoritative value recorded in `resident_bytes`; the
    /// caller should set the hint ≥ the actual to keep the budget
    /// honest.
    fn pre_evict_for<E>(&mut self, requested_bytes: u64) -> Result<(), ClusterStoreError<E>> {
        if requested_bytes > self.budget_bytes {
            return Err(ClusterStoreError::BudgetTooSmall {
                budget_bytes: self.budget_bytes,
                requested_bytes,
            });
        }
        while self.resident_bytes + requested_bytes > self.budget_bytes {
            let Some(evicted_id) = self.order.pop_front() else {
                return Err(ClusterStoreError::BudgetTooSmall {
                    budget_bytes: self.budget_bytes,
                    requested_bytes,
                });
            };
            if let Some(evicted) = self.entries.remove(&evicted_id) {
                self.resident_bytes -= evicted.bytes;
                GLOBAL_EVICTIONS.fetch_add(1, Ordering::Relaxed);
                // C: Drop runs here; for cuVS that's BfIndex's
                // destructor returning device memory to RMM, for
                // cudarc it's CudaSlice's free. If a concurrent
                // scorer still holds an Arc the Drop is deferred;
                // bookkeeping `resident_bytes` is logical, not a
                // real-time VRAM-fenced count. The NVML sampler is
                // the source of truth for realised peak VRAM.
                drop(evicted);
            }
        }
        Ok(())
    }

    /// Insert `payload` under `cluster_id`. Caller has already
    /// pre-evicted via [`Self::pre_evict_for`]; this is purely the
    /// bookkeeping update. Returns the just-inserted Arc.
    ///
    /// Benign-race handling: if another thread inserted the same
    /// `cluster_id` between the caller's miss check and this call,
    /// the existing entry is returned and the caller's payload
    /// drops on the spot (returning its device memory via the
    /// `C: Drop` impl).
    fn insert(&mut self, cluster_id: usize, payload: C) -> Arc<Mutex<C>> {
        if let Some(entry) = self.entries.get(&cluster_id) {
            return Arc::clone(&entry.payload);
        }
        let bytes = payload.device_bytes();
        let payload_arc = Arc::new(Mutex::new(payload));
        self.entries.insert(
            cluster_id,
            ClusterEntry {
                payload: Arc::clone(&payload_arc),
                bytes,
            },
        );
        self.order.push_back(cluster_id);
        self.resident_bytes += bytes;
        payload_arc
    }

    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// LRU cache of device-resident cluster handles, bounded by a hard
/// byte budget. See the crate docs for the streaming flow.
pub struct ClusterStore<C: DeviceCluster> {
    ctx: Arc<CudaContext>,
    upload_stream: Arc<CudaStream>,
    compute_stream: Arc<CudaStream>,
    budget_bytes: u64,
    inner: Mutex<LruInner<C>>,
}

impl<C: DeviceCluster> ClusterStore<C> {
    pub fn new(
        ctx: Arc<CudaContext>,
        budget_bytes: u64,
        upload_stream: Arc<CudaStream>,
        compute_stream: Arc<CudaStream>,
    ) -> Self {
        ClusterStore {
            ctx,
            upload_stream,
            compute_stream,
            budget_bytes,
            inner: Mutex::new(LruInner::new(budget_bytes)),
        }
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Upload stream — H2D copies go here. Caller schedules its work
    /// on this stream from inside the `upload` closure of
    /// [`Self::get_or_upload`].
    pub fn upload_stream(&self) -> &Arc<CudaStream> {
        &self.upload_stream
    }

    /// Compute stream — kernel launches and D2H readbacks go here.
    /// Caller serialises against upload via a stream event when
    /// pipelining cluster N+1's upload against cluster N's compute.
    pub fn compute_stream(&self) -> &Arc<CudaStream> {
        &self.compute_stream
    }

    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Snapshot of the bookkeeping `resident_bytes`. Reflects what
    /// the LRU thinks is resident, not what NVML sees — see the
    /// [`DeviceCluster::device_bytes`] doc for the distinction.
    pub fn resident_bytes(&self) -> u64 {
        let guard = self.inner.lock().expect("ClusterStore mutex poisoned");
        guard.resident_bytes()
    }

    /// Resolve `cluster_id` to a device-resident handle.
    ///
    /// Fast path (cache hit): bump recency, return the cached cell.
    ///
    /// Miss path:
    ///
    /// 1. Pre-evict LRU tail until `payload_bytes_hint` would fit
    ///    on top of `resident_bytes`. `payload_bytes_hint` is the
    ///    caller's a-priori estimate of how much device memory
    ///    the upload will consume — for plaintext fp32 that's
    ///    `m × dim × 4`, for emvp `m × n × 8`, etc. Setting the
    ///    hint ≥ the actual `device_bytes()` keeps the budget
    ///    honest.
    /// 2. Invoke `upload(upload_stream)` outside the LRU lock so
    ///    cache-hit queries on other clusters aren't serialised
    ///    behind a long upload.
    /// 3. Insert the result. If another thread inserted the same
    ///    `cluster_id` while our closure ran, the existing entry
    ///    is returned and our payload drops on the spot.
    ///
    /// If pre-eviction can't free enough budget even after
    /// popping every resident entry, returns
    /// [`ClusterStoreError::BudgetTooSmall`]; the closure does
    /// not run in that case (so no wasted device allocation
    /// when the operator-pinned budget is fundamentally too low
    /// for the per-cluster shape).
    pub fn get_or_upload<F, E>(
        &self,
        cluster_id: usize,
        payload_bytes_hint: u64,
        upload: F,
    ) -> Result<LruCell<C>, ClusterStoreError<E>>
    where
        F: FnOnce(&Arc<CudaStream>) -> Result<C, E>,
    {
        GLOBAL_GETS.fetch_add(1, Ordering::Relaxed);

        // Hit path + pre-eviction. Record whether any resident cluster
        // was actually evicted (resident_bytes drops iff it was).
        let evicted = {
            let mut guard = self.inner.lock().expect("ClusterStore mutex poisoned");
            if let Some(payload) = guard.touch(cluster_id) {
                return Ok(LruCell { inner: payload });
            }
            let before = guard.resident_bytes();
            guard.pre_evict_for::<E>(payload_bytes_hint)?;
            guard.resident_bytes() < before
        };

        // Evicted cluster buffers are dropped (cuMemFreeAsync) on the
        // upload stream, while per-query scratch frees on the compute
        // stream — same device mempool, different streams. cudarc's
        // stream-ordered async pool only reuses a stream's own frees
        // without a sync, so cross-stream freed blocks accumulate and
        // reserved VRAM climbs per query until CUDA_ERROR_OUT_OF_MEMORY
        // (a deferred pool-trim gap; hits emvp/bntm at 8.8M, never the
        // RMM-backed cuVS path). A context synchronize after
        // eviction lets the pool reclaim before the next upload
        // allocates. Gated on `evicted` so non-evicting hits and the
        // budget-fits case keep full upload/compute async overlap.
        if evicted {
            self.ctx.synchronize().map_err(ClusterStoreError::Sync)?;
        }

        // Closure runs outside the lock — cache-hit queries on
        // other clusters can complete concurrently.
        let payload = upload(&self.upload_stream).map_err(ClusterStoreError::Upload)?;
        let uploaded_bytes = payload.device_bytes();
        debug_assert!(
            uploaded_bytes <= payload_bytes_hint,
            "DeviceCluster reported {uploaded_bytes} bytes after upload but caller's hint was {payload_bytes_hint}; \
             this under-estimate makes the LRU budget bookkeeping leaky",
        );

        let mut guard = self.inner.lock().expect("ClusterStore mutex poisoned");
        let arc = guard.insert(cluster_id, payload);
        GLOBAL_UPLOADS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_BYTES_UPLOADED.fetch_add(uploaded_bytes, Ordering::Relaxed);
        Ok(LruCell { inner: arc })
    }
}

#[cfg(test)]
mod tests {
    //! CUDA-free unit tests for the LRU bookkeeping. The streaming
    //! flow's CUDA-touching parts (`ClusterStore::get_or_upload`'s
    //! stream pass-through, the cuVS / cudarc `DeviceCluster`
    //! `Drop` impls in caller crates) need a working driver and are
    //! covered by the per-scorer `gpu`-feature integration tests on
    //! the sacs006 self-hosted runner.

    use super::*;

    #[derive(Debug)]
    struct FakeCluster {
        bytes: u64,
        id: usize,
        drops: Arc<Mutex<Vec<usize>>>,
    }

    impl DeviceCluster for FakeCluster {
        fn device_bytes(&self) -> u64 {
            self.bytes
        }
    }

    impl Drop for FakeCluster {
        fn drop(&mut self) {
            self.drops
                .lock()
                .expect("drop log mutex poisoned")
                .push(self.id);
        }
    }

    fn fake(bytes: u64, id: usize, drops: &Arc<Mutex<Vec<usize>>>) -> FakeCluster {
        FakeCluster {
            bytes,
            id,
            drops: Arc::clone(drops),
        }
    }

    fn insert_one(
        inner: &mut LruInner<FakeCluster>,
        id: usize,
        bytes: u64,
        drops: &Arc<Mutex<Vec<usize>>>,
    ) {
        inner
            .pre_evict_for::<std::io::Error>(bytes)
            .expect("pre-evict");
        let _arc = inner.insert(id, fake(bytes, id, drops));
    }

    #[test]
    fn budget_too_small_surfaces_when_first_payload_overflows() {
        let mut inner = LruInner::<FakeCluster>::new(100);
        let res: Result<(), ClusterStoreError<std::io::Error>> = inner.pre_evict_for(200);
        match res {
            Err(ClusterStoreError::BudgetTooSmall {
                budget_bytes,
                requested_bytes,
            }) => {
                assert_eq!(budget_bytes, 100);
                assert_eq!(requested_bytes, 200);
            }
            other => panic!("expected BudgetTooSmall, got {other:?}"),
        }
    }

    #[test]
    fn lru_evicts_oldest_then_inserts_new() {
        let mut inner = LruInner::<FakeCluster>::new(250);
        let drops = Arc::new(Mutex::new(Vec::new()));
        for id in 0..3 {
            insert_one(&mut inner, id, 100, &drops);
        }
        let dropped = drops.lock().unwrap().clone();
        assert_eq!(
            dropped,
            vec![0],
            "cluster 0 should evict first when 2's insert hits the budget"
        );
        assert_eq!(inner.resident_bytes(), 200);
    }

    #[test]
    fn touch_returns_existing_payload_and_bumps_recency() {
        let mut inner = LruInner::<FakeCluster>::new(1_000);
        let drops = Arc::new(Mutex::new(Vec::new()));
        insert_one(&mut inner, 7, 100, &drops);
        let touched = inner.touch(7).expect("hit");
        let guard = touched.lock().expect("lock");
        assert_eq!(guard.id, 7);
    }

    #[test]
    fn recency_bump_protects_recently_used_cluster() {
        let mut inner = LruInner::<FakeCluster>::new(250);
        let drops = Arc::new(Mutex::new(Vec::new()));
        insert_one(&mut inner, 0, 100, &drops);
        insert_one(&mut inner, 1, 100, &drops);
        // Bump 0 → it's MRU now.
        inner.touch(0).expect("hit on 0");
        // Insert 2 → must evict 1, not 0.
        insert_one(&mut inner, 2, 100, &drops);
        let dropped = drops.lock().unwrap().clone();
        assert_eq!(
            dropped,
            vec![1],
            "cluster 1 should evict (cluster 0 was bumped)"
        );
    }

    #[test]
    fn insert_returns_existing_on_race() {
        // Benign race: another thread inserted `cluster_id` between
        // our miss check and our own insert call. The second insert
        // returns the existing entry and drops the just-built payload.
        let mut inner = LruInner::<FakeCluster>::new(500);
        let drops = Arc::new(Mutex::new(Vec::new()));

        insert_one(&mut inner, 3, 100, &drops);
        let bytes_before = inner.resident_bytes();

        // Pretend another thread "already inserted" id=3 — pre_evict
        // would have passed, but the insert short-circuits.
        let _arc = inner.insert(3, fake(100, 99, &drops));

        assert_eq!(inner.resident_bytes(), bytes_before);
        let dropped = drops.lock().unwrap().clone();
        assert_eq!(
            dropped,
            vec![99],
            "second-payload (id=99) drops on race; resident_bytes unchanged"
        );
    }

    #[test]
    fn pre_evict_only_evicts_enough_for_request() {
        // Budget=250, resident=200 (two 100-byte clusters). A
        // 100-byte request needs 50 more bytes of headroom →
        // evict exactly one cluster (the LRU one).
        let mut inner = LruInner::<FakeCluster>::new(250);
        let drops = Arc::new(Mutex::new(Vec::new()));
        insert_one(&mut inner, 0, 100, &drops);
        insert_one(&mut inner, 1, 100, &drops);

        inner
            .pre_evict_for::<std::io::Error>(100)
            .expect("pre-evict");
        // 0 evicted, 1 still resident.
        assert_eq!(inner.resident_bytes(), 100);
        let dropped = drops.lock().unwrap().clone();
        assert_eq!(dropped, vec![0]);
    }
}
