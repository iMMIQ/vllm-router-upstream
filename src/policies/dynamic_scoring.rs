//! Dynamic scoring load balancing policy for heterogeneous workers
//!
//! Scores workers based on normalized queue depth relative to their safe capacity.
//! Workers with lower scores are preferred, enabling fair routing across workers
//! with different maximum concurrency limits.

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::info;

/// Configuration for the dynamic scoring policy
#[derive(Debug, Clone)]
pub struct DynamicScoringConfig {
    /// Default safe capacity for workers without explicit configuration
    pub default_safe_capacity: f64,
    /// Weight for normalized queue depth component
    pub alpha: f64,
    /// Weight for latency component (reserved for future use)
    pub beta: f64,
    /// Weight for error penalty component (reserved for future use)
    pub gamma: f64,
    /// Per-worker safe capacity overrides (URL -> capacity)
    pub worker_safe_capacities: HashMap<String, f64>,
}

impl Default for DynamicScoringConfig {
    fn default() -> Self {
        Self {
            default_safe_capacity: 100.0,
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            worker_safe_capacities: HashMap::new(),
        }
    }
}

/// Dynamic scoring policy
///
/// Computes a score for each healthy worker:
///   score = α * (inflight / safe_capacity)
///
/// The worker with the lowest score is selected. This naturally balances load
/// across heterogeneous workers with different safe operating capacities.
#[derive(Debug)]
pub struct DynamicScoringPolicy {
    /// Per-worker safe capacity (URL -> safe working point)
    safe_capacities: RwLock<HashMap<String, f64>>,
    /// Default safe capacity for unconfigured workers
    default_safe_capacity: f64,
    /// Cached load information from external monitoring
    cached_loads: RwLock<HashMap<String, isize>>,
    /// Scoring weights
    alpha: f64,
    #[allow(dead_code)]
    beta: f64,
    #[allow(dead_code)]
    gamma: f64,
}

impl DynamicScoringPolicy {
    pub fn new() -> Self {
        Self::with_config(DynamicScoringConfig::default())
    }

    pub fn with_config(config: DynamicScoringConfig) -> Self {
        Self {
            safe_capacities: RwLock::new(config.worker_safe_capacities),
            default_safe_capacity: config.default_safe_capacity,
            cached_loads: RwLock::new(HashMap::new()),
            alpha: config.alpha,
            beta: config.beta,
            gamma: config.gamma,
        }
    }

    /// Configure safe capacities for specific workers
    pub fn configure_worker_capacities(&self, caps: &HashMap<String, f64>) {
        if let Ok(mut safe_caps) = self.safe_capacities.write() {
            for (url, cap) in caps {
                safe_caps.insert(url.clone(), *cap);
            }
        }
    }

    /// Get the safe capacity for a worker URL
    fn get_safe_capacity(&self, worker_url: &str) -> f64 {
        if let Ok(caps) = self.safe_capacities.read() {
            // Try exact match first
            if let Some(&cap) = caps.get(worker_url) {
                return cap;
            }
            // For DP-aware URLs (e.g., "http://host:8000@1"), try matching by base URL
            if let Some(at_pos) = worker_url.rfind('@') {
                let base_url = &worker_url[..at_pos];
                if let Some(&cap) = caps.get(base_url) {
                    return cap;
                }
            }
        }
        self.default_safe_capacity
    }

    /// Get the current effective load for a worker.
    ///
    /// Combines the cached external load (from periodic polling) with the local
    /// in-flight count. This prevents the "thundering herd" problem where all
    /// requests go to the same worker between load polls, because the local
    /// in-flight count increases with each dispatched request.
    fn get_worker_load(&self, worker: &dyn Worker) -> f64 {
        let local_inflight = worker.load() as f64;
        if let Ok(loads) = self.cached_loads.read() {
            if let Some(&cached) = loads.get(worker.url()) {
                return cached as f64 + local_inflight;
            }
        }
        local_inflight
    }

    /// Compute the score for a worker (lower is better)
    fn compute_score(&self, worker: &dyn Worker) -> f64 {
        let load = self.get_worker_load(worker);
        let safe_cap = self.get_safe_capacity(worker.url());
        self.alpha * (load / safe_cap)
    }
}

impl LoadBalancingPolicy for DynamicScoringPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        _request_text: Option<&str>,
        _headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        if healthy_indices.len() == 1 {
            let idx = healthy_indices[0];
            workers[idx].increment_processed();
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
            return Some(idx);
        }

        // Compute scores for all healthy workers
        let scores: Vec<(usize, f64)> = healthy_indices
            .iter()
            .map(|&idx| (idx, self.compute_score(workers[idx].as_ref())))
            .collect();

        let best_score = scores.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);

        // Collect all workers tied at the best score (within epsilon)
        const EPSILON: f64 = 1e-9;
        let tied: Vec<usize> = scores
            .iter()
            .filter(|(_, s)| (*s - best_score).abs() < EPSILON)
            .map(|(idx, _)| *idx)
            .collect();

        // Random tie-breaking among tied workers
        let best_idx = if tied.len() == 1 {
            tied[0]
        } else {
            let mut rng = rand::rng();
            tied[rng.random_range(0..tied.len())]
        };

        info!(
            "Dynamic scoring selection: {} (score={:.3}, tied={}) from {} healthy workers",
            workers[best_idx].url(),
            best_score,
            tied.len(),
            healthy_indices.len()
        );

        workers[best_idx].increment_processed();
        RouterMetrics::record_processed_request(workers[best_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[best_idx].url());

        Some(best_idx)
    }

    fn name(&self) -> &'static str {
        "dynamic_scoring"
    }

    fn update_loads(&self, loads: &HashMap<String, isize>) {
        if let Ok(mut cached) = self.cached_loads.write() {
            *cached = loads.clone();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Default for DynamicScoringPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};

    #[test]
    fn test_dynamic_scoring_prefers_lower_normalized_load() {
        let mut caps = HashMap::new();
        caps.insert("http://workerA:8000".to_string(), 500.0);
        caps.insert("http://workerB:8000".to_string(), 60.0);

        let config = DynamicScoringConfig {
            default_safe_capacity: 100.0,
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            worker_safe_capacities: caps,
        };
        let policy = DynamicScoringPolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://workerA:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://workerB:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // A: 400/500 = 0.8, B: 50/60 = 0.83 → select A
        let mut loads = HashMap::new();
        loads.insert("http://workerA:8000".to_string(), 400);
        loads.insert("http://workerB:8000".to_string(), 50);
        policy.update_loads(&loads);

        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(0)); // Worker A has lower normalized load

        // A: 500/500 = 1.0, B: 30/60 = 0.5 → select B
        let mut loads2 = HashMap::new();
        loads2.insert("http://workerA:8000".to_string(), 500);
        loads2.insert("http://workerB:8000".to_string(), 30);
        policy.update_loads(&loads2);

        let selected2 = policy.select_worker(&workers, None);
        assert_eq!(selected2, Some(1)); // Worker B has lower normalized load
    }

    #[test]
    fn test_dynamic_scoring_single_worker() {
        let policy = DynamicScoringPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(BasicWorker::new(
            "http://w1:8000".to_string(),
            WorkerType::Regular,
        ))];

        assert_eq!(policy.select_worker(&workers, None), Some(0));
    }

    #[test]
    fn test_dynamic_scoring_no_workers() {
        let policy = DynamicScoringPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];

        assert_eq!(policy.select_worker(&workers, None), None);
    }

    #[test]
    fn test_dynamic_scoring_uses_default_capacity() {
        let config = DynamicScoringConfig {
            default_safe_capacity: 200.0,
            ..Default::default()
        };
        let policy = DynamicScoringPolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Both workers use default capacity of 200
        // w1: 100/200 = 0.5, w2: 150/200 = 0.75 → select w1
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 100);
        loads.insert("http://w2:8000".to_string(), 150);
        policy.update_loads(&loads);

        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn test_dynamic_scoring_falls_back_to_local_load() {
        let policy = DynamicScoringPolicy::new();
        let worker1 = BasicWorker::new("http://w1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://w2:8000".to_string(), WorkerType::Regular);

        // Set local loads
        for _ in 0..10 {
            worker1.increment_load();
        }
        for _ in 0..5 {
            worker2.increment_load();
        }

        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker1), Arc::new(worker2)];

        // No cached loads, should use local load counters
        // w1: 10/100 = 0.1, w2: 5/100 = 0.05 → select w2
        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn test_configure_worker_capacities() {
        let policy = DynamicScoringPolicy::new();

        let mut caps = HashMap::new();
        caps.insert("http://w1:8000".to_string(), 500.0);
        caps.insert("http://w2:8000".to_string(), 60.0);
        policy.configure_worker_capacities(&caps);

        assert_eq!(policy.get_safe_capacity("http://w1:8000"), 500.0);
        assert_eq!(policy.get_safe_capacity("http://w2:8000"), 60.0);
        assert_eq!(
            policy.get_safe_capacity("http://unknown:8000"),
            100.0 // default
        );
    }

    #[test]
    fn test_dynamic_scoring_random_tiebreak_at_zero_load() {
        let mut caps = HashMap::new();
        caps.insert("http://w1:8000".to_string(), 60.0);
        caps.insert("http://w2:8000".to_string(), 500.0);
        caps.insert("http://w3:8000".to_string(), 500.0);

        let config = DynamicScoringConfig {
            worker_safe_capacities: caps,
            ..Default::default()
        };
        let policy = DynamicScoringPolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w3:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // All loads are 0 → all scores are 0 → should randomly distribute
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 0);
        loads.insert("http://w2:8000".to_string(), 0);
        loads.insert("http://w3:8000".to_string(), 0);
        policy.update_loads(&loads);

        let mut counts = [0usize; 3];
        for _ in 0..300 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                counts[idx] += 1;
            }
        }

        // All three workers should be selected (not just the first one)
        assert!(counts[0] > 0, "w1 should be selected at least once");
        assert!(counts[1] > 0, "w2 should be selected at least once");
        assert!(counts[2] > 0, "w3 should be selected at least once");
    }

    #[test]
    fn test_get_safe_capacity_dp_aware_urls() {
        let mut caps = HashMap::new();
        caps.insert("http://w1:8000".to_string(), 64.0);
        caps.insert("http://w2:8000".to_string(), 256.0);

        let config = DynamicScoringConfig {
            default_safe_capacity: 100.0,
            worker_safe_capacities: caps,
            ..Default::default()
        };
        let policy = DynamicScoringPolicy::with_config(config);

        // Exact match
        assert_eq!(policy.get_safe_capacity("http://w1:8000"), 64.0);
        assert_eq!(policy.get_safe_capacity("http://w2:8000"), 256.0);

        // DP-aware URLs should resolve to base URL capacity
        assert_eq!(policy.get_safe_capacity("http://w1:8000@0"), 64.0);
        assert_eq!(policy.get_safe_capacity("http://w1:8000@3"), 64.0);
        assert_eq!(policy.get_safe_capacity("http://w2:8000@1"), 256.0);

        // Unknown worker still falls back to default
        assert_eq!(policy.get_safe_capacity("http://unknown:8000@0"), 100.0);
    }

    #[test]
    fn test_dynamic_scoring_unhealthy_workers_excluded() {
        let policy = DynamicScoringPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Mark w1 as unhealthy
        workers[0].set_healthy(false);

        // Should only select w2
        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn test_dynamic_scoring_combines_cached_and_local_load() {
        // Regression test: without combining cached + local inflight,
        // a worker with the lowest cached load gets ALL requests between polls.
        let mut caps = HashMap::new();
        caps.insert("http://w1:8000".to_string(), 100.0);
        caps.insert("http://w2:8000".to_string(), 100.0);

        let config = DynamicScoringConfig {
            worker_safe_capacities: caps,
            ..Default::default()
        };
        let policy = DynamicScoringPolicy::with_config(config);

        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(BasicWorker::new(
                "http://w1:8000".to_string(),
                WorkerType::Regular,
            )),
            Arc::new(BasicWorker::new(
                "http://w2:8000".to_string(),
                WorkerType::Regular,
            )),
        ];

        // Cached loads: w1=50, w2=48 (w2 slightly lower)
        let mut loads = HashMap::new();
        loads.insert("http://w1:8000".to_string(), 50);
        loads.insert("http://w2:8000".to_string(), 48);
        policy.update_loads(&loads);

        // Simulate w2 receiving 5 in-flight requests (local load)
        for _ in 0..5 {
            workers[1].increment_load();
        }

        // w1 effective = 50 + 0 = 50, score = 50/100 = 0.5
        // w2 effective = 48 + 5 = 53, score = 53/100 = 0.53
        // Should select w1 because its effective load is lower
        let selected = policy.select_worker(&workers, None);
        assert_eq!(selected, Some(0), "w1 should be selected: lower effective load (cached + inflight)");
    }
}
