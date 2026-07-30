//! Indicatif-backed `ProgressReporter` impl shared across all scorers.
//!
//! Every scorer Config carries the same
//! `progress: Option<Arc<dyn ProgressReporter>>` field. The eval binary
//! constructs an `IndicatifProgress` per scheme; tests pass `None`.
//!
//! `indicatif::ProgressBar`'s default draw target silently hides when stderr
//! is not a TTY (e.g. when the harness is invoked under `tee` for log
//! capture), which hides build progress entirely in that case. To keep
//! long-running builds observable in log files, every `on_*` event also
//! emits a throttled `[progress]` line on stderr via `eprintln!` —
//! pretty bar on TTY, plain heartbeat in logs, same source.

use indicatif::ProgressBar;
use scorer_core::ProgressReporter;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Minimum wall-clock between two `[progress]` eprintlns from the same
/// `IndicatifProgress`. Picked so a 25-iter Lloyd's pass at ~30 s/iter
/// emits roughly every iter, and a 2967-pick k-means++ init emits every
/// few percent rather than once per pick.
const LOG_INTERVAL: Duration = Duration::from_secs(5);

/// Forwards every progress milestone to a single `ProgressBar`, and also
/// emits a throttled plaintext heartbeat on stderr so non-TTY sessions
/// (under `tee`, in unattended job runners) still see liveness.
pub struct IndicatifProgress {
    bar: ProgressBar,
    start: Instant,
    last_log: Mutex<(Instant, u64)>,
}

impl IndicatifProgress {
    pub fn new(bar: ProgressBar) -> Self {
        let now = Instant::now();
        Self {
            bar,
            start: now,
            last_log: Mutex::new((now, 0)),
        }
    }

    fn maybe_log(&self) {
        let pos = self.bar.position();
        let total = self.bar.length().unwrap_or(0);
        if let Ok(mut state) = self.last_log.try_lock() {
            let (last_t, last_pos) = *state;
            if pos > last_pos && last_t.elapsed() >= LOG_INTERVAL {
                let pct = if total > 0 {
                    pos as f64 * 100.0 / total as f64
                } else {
                    0.0
                };
                eprintln!(
                    "[progress] {}/{} ({:.1}%) elapsed {}",
                    pos,
                    total,
                    pct,
                    fmt_elapsed(self.start.elapsed()),
                );
                *state = (Instant::now(), pos);
            }
        }
    }
}

fn fmt_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{}h {:02}m {:02}s", h, m, s)
    } else if m > 0 {
        format!("{}m {:02}s", m, s)
    } else {
        format!("{}s", s)
    }
}

impl ProgressReporter for IndicatifProgress {
    fn on_init_centroid(&self, _idx: usize) {
        self.bar.inc(1);
        self.maybe_log();
    }
    fn on_kmeans_iter(&self, _iter: usize) {
        self.bar.inc(1);
        self.maybe_log();
    }
    fn on_encrypt(&self, _idx: usize) {
        self.bar.inc(1);
        self.maybe_log();
    }
    fn on_phase_extend(&self, additional: usize) {
        let cur = self.bar.length().unwrap_or(0);
        self.bar.set_length(cur + additional as u64);
    }
    fn on_build_complete(&self) {
        let pos = self.bar.position();
        let total = self.bar.length().unwrap_or(0);
        eprintln!(
            "[progress] build complete {}/{} elapsed {}",
            pos,
            total,
            fmt_elapsed(self.start.elapsed()),
        );
        self.bar.finish_and_clear();
    }
}
