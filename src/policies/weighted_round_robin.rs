//! Weighted round-robin load balancing policy
//!
//! Implements the Smooth Weighted Round-Robin (SWRR) algorithm, the same approach
//! used by nginx. This distributes requests proportionally to worker weights while
//! producing an interleaved schedule (e.g., weights [5,1] → A,A,A,B,A,A rather
//! than A,A,A,A,A,B).

use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use std::sync::{Arc, Mutex};

/// Per-worker state for the SWRR algorithm
#[derive(Debug, Clone)]
struct WorkerWeight {
    /// The configured effective weight (from worker labels)
    effective_weight: i64,
    /// The current weight that changes each round
    current_weight: i64,
}

/// Weighted round-robin selection policy (Smooth Weighted Round-Robin)
///
/// Distributes requests proportionally to worker weights. Workers with higher
/// weight receive proportionally more requests. The smooth algorithm ensures
/// requests are interleaved rather than batched.
///
/// Worker weights are read from the `weight` label (default: 1).
/// This policy requires initialization via `init_workers()`.
#[derive(Debug)]
pub struct WeightedRoundRobinPolicy {
    /// Per-worker weights, indexed by worker position.
    /// Protected by a Mutex since SWRR is inherently stateful.
    state: Mutex<Vec<WorkerWeight>>,
}

impl WeightedRoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(Vec::new()),
        }
    }
}

impl Default for WeightedRoundRobinPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for WeightedRoundRobinPolicy {
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
            RouterMetrics::record_processed_request(workers[idx].url());
            RouterMetrics::record_policy_decision(self.name(), workers[idx].url());
            return Some(idx);
        }

        let mut state = self.state.lock().unwrap();

        // If state is empty or mismatched, re-initialize from current workers
        if state.len() != workers.len() {
            *state = workers
                .iter()
                .map(|w| WorkerWeight {
                    effective_weight: w.weight() as i64,
                    current_weight: 0,
                })
                .collect();
        }

        // SWRR algorithm over healthy workers only:
        // 1. Add effective_weight to current_weight for each healthy worker
        // 2. Select the healthy worker with the highest current_weight
        // 3. Subtract total_weight (of healthy workers) from the selected worker

        let total_weight: i64 = healthy_indices
            .iter()
            .map(|&idx| state[idx].effective_weight)
            .sum();

        // Step 1: increment current_weight for healthy workers
        for &idx in &healthy_indices {
            state[idx].current_weight += state[idx].effective_weight;
        }

        // Step 2: find the healthy worker with the highest current_weight
        let &selected_idx = healthy_indices
            .iter()
            .max_by_key(|&&idx| state[idx].current_weight)
            .unwrap(); // safe: healthy_indices is non-empty

        // Step 3: subtract total_weight from selected worker
        state[selected_idx].current_weight -= total_weight;

        RouterMetrics::record_processed_request(workers[selected_idx].url());
        RouterMetrics::record_policy_decision(self.name(), workers[selected_idx].url());

        Some(selected_idx)
    }

    fn name(&self) -> &'static str {
        "weighted_round_robin"
    }

    fn requires_initialization(&self) -> bool {
        true
    }

    fn init_workers(&self, workers: &[Arc<dyn Worker>]) {
        let mut state = self.state.lock().unwrap();
        *state = workers
            .iter()
            .map(|w| WorkerWeight {
                effective_weight: w.weight() as i64,
                current_weight: 0,
            })
            .collect();
    }

    fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        for w in state.iter_mut() {
            w.current_weight = 0;
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::collections::HashMap;

    fn make_worker(url: &str, weight: u32) -> Arc<dyn Worker> {
        let mut labels = HashMap::new();
        labels.insert("weight".to_string(), weight.to_string());
        Arc::new(
            BasicWorker::new(url.to_string(), WorkerType::Decode).with_labels(labels),
        )
    }

    #[test]
    fn test_weighted_distribution() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", 3),
            make_worker("http://w2:8000", 1),
        ];
        policy.init_workers(&workers);

        let mut counts = [0usize; 2];
        for _ in 0..400 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                counts[idx] += 1;
            }
        }

        // With weights 3:1, worker 0 should get ~300, worker 1 ~100
        assert_eq!(counts[0], 300);
        assert_eq!(counts[1], 100);
    }

    #[test]
    fn test_smooth_interleaving() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", 5),
            make_worker("http://w2:8000", 1),
        ];
        policy.init_workers(&workers);

        // Collect first 6 selections — SWRR should interleave, not batch
        let selections: Vec<usize> = (0..6)
            .filter_map(|_| policy.select_worker(&workers, None))
            .collect();

        // With weights 5:1, worker 1 should appear at index 2 or 3 (interleaved),
        // not at the very end
        assert_eq!(selections.len(), 6);
        let w1_positions: Vec<usize> = selections
            .iter()
            .enumerate()
            .filter(|(_, &s)| s == 1)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(w1_positions.len(), 1);
        // Worker 1 should NOT be at position 5 (last) — SWRR interleaves
        assert_ne!(w1_positions[0], 5);
    }

    #[test]
    fn test_single_worker() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![make_worker("http://w1:8000", 5)];
        policy.init_workers(&workers);

        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, None), Some(0));
        }
    }

    #[test]
    fn test_unhealthy_worker_skipped() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", 3),
            make_worker("http://w2:8000", 2),
            make_worker("http://w3:8000", 1),
        ];
        policy.init_workers(&workers);

        // Mark worker 0 unhealthy
        workers[0].set_healthy(false);

        let mut counts = [0usize; 3];
        for _ in 0..300 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                counts[idx] += 1;
            }
        }

        // Worker 0 should never be selected
        assert_eq!(counts[0], 0);
        // Workers 1 and 2 should be selected with ratio 2:1
        assert_eq!(counts[1], 200);
        assert_eq!(counts[2], 100);
    }

    #[test]
    fn test_equal_weights() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", 1),
            make_worker("http://w2:8000", 1),
            make_worker("http://w3:8000", 1),
        ];
        policy.init_workers(&workers);

        let mut counts = [0usize; 3];
        for _ in 0..300 {
            if let Some(idx) = policy.select_worker(&workers, None) {
                counts[idx] += 1;
            }
        }

        // Equal weights → equal distribution
        assert_eq!(counts[0], 100);
        assert_eq!(counts[1], 100);
        assert_eq!(counts[2], 100);
    }

    #[test]
    fn test_reset() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", 3),
            make_worker("http://w2:8000", 1),
        ];
        policy.init_workers(&workers);

        // Advance a few rounds
        for _ in 0..3 {
            policy.select_worker(&workers, None);
        }

        // Reset and verify sequence restarts
        policy.reset();

        let selections: Vec<usize> = (0..4)
            .filter_map(|_| policy.select_worker(&workers, None))
            .collect();

        // After reset, should produce the same deterministic sequence as fresh init
        let policy2 = WeightedRoundRobinPolicy::new();
        policy2.init_workers(&workers);
        let fresh_selections: Vec<usize> = (0..4)
            .filter_map(|_| policy2.select_worker(&workers, None))
            .collect();

        assert_eq!(selections, fresh_selections);
    }

    #[test]
    fn test_no_workers() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        policy.init_workers(&workers);

        assert_eq!(policy.select_worker(&workers, None), None);
    }

    #[test]
    fn test_all_unhealthy() {
        let policy = WeightedRoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            make_worker("http://w1:8000", 3),
            make_worker("http://w2:8000", 1),
        ];
        policy.init_workers(&workers);

        workers[0].set_healthy(false);
        workers[1].set_healthy(false);

        assert_eq!(policy.select_worker(&workers, None), None);
    }
}
