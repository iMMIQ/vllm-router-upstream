// vLLM PD (Prefill-Decode) Router Implementation
// This module provides a standalone vLLM-specific two-stage processing router
// with service discovery integration and ZMQ-based KV transfer support
use super::dp_utils;
use super::logprobs_merge;
use super::pd_types::PDRouterError;
use super::vllm_service_discovery::{ServiceRegistry, ServiceType};
use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorker, CircuitBreakerConfig, HealthConfig, Worker, WorkerFactory, WorkerRegistry, WorkerType,
};
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::routers::header_utils;
use crate::routers::{RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// vLLM PD Router - standalone router for vLLM prefill-decode disaggregation
#[derive(Debug)]
pub struct VllmPDRouter {
    /// Worker registry for prefill/decode workers
    pub worker_registry: Arc<WorkerRegistry>,
    /// Policy registry for load balancing
    pub policy_registry: Arc<PolicyRegistry>,
    /// Worker startup timeout in seconds
    pub worker_startup_timeout_secs: u64,
    /// Worker startup check interval in seconds
    pub worker_startup_check_interval_secs: u64,
    /// Worker loads for monitoring
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    /// Handle to the load monitoring task
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    /// HTTP client for making requests
    pub client: Client,
    /// Dedicated client for prefill fire-and-forget requests
    pub prefill_client: Client,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker_config: CircuitBreakerConfig,
    /// Channel for sending prefill responses to background workers for draining
    prefill_drain_tx: mpsc::Sender<reqwest::Response>,
    /// Service discovery registry for dynamic ZMQ address resolution
    service_registry: Arc<ServiceRegistry>,
    /// HTTP client for making requests to discovered services
    http_client: reqwest::Client,
    /// Whether this router uses service discovery (true) or direct URLs (false)
    use_discovery: bool,
    /// Enable profiling calls to vLLM workers
    enable_profiling: bool,
    /// Profiling timeout in seconds
    profile_timeout_secs: u64,
    /// Active profiling timeout tasks keyed by worker URL
    profiling_tasks: Arc<Mutex<HashMap<String, tokio::task::AbortHandle>>>,
    /// Intra-node data parallel size for DP-aware routing (automatically enabled when > 1)
    intra_node_data_parallel_size: usize,
}

impl VllmPDRouter {
    /// Generate vLLM-specific request ID with prefill/decode addressing
    fn generate_vllm_request_id(prefill_addr: &str, decode_addr: &str) -> String {
        let uuid = Uuid::new_v4().to_string().replace('-', "");
        format!(
            "___prefill_addr_{}___decode_addr_{}_{}",
            prefill_addr, decode_addr, uuid
        )
    }

    /// Get ZMQ address for a worker URL using service discovery
    fn get_zmq_address(&self, http_url: &str, service_type: ServiceType) -> String {
        // Extract just the host:port from the URL
        let http_address = http_url.replace("http://", "").replace("https://", "");

        // Try to get ZMQ address from service discovery
        if let Some(zmq_addr) = self
            .service_registry
            .get_zmq_address(&http_address, service_type.clone())
        {
            debug!(
                "Using discovered ZMQ address: {} ({:?}) -> {}",
                http_address, service_type, zmq_addr
            );
            return zmq_addr;
        }

        // Fallback: use HTTP address as ZMQ address
        debug!(
            "No ZMQ discovery result for {} ({:?}), using fallback: {}",
            http_address, service_type, http_address
        );
        http_address
    }

    /// Private helper method to perform health check on a new server
    async fn wait_for_server_health(&self, url: &str) -> Result<(), PDRouterError> {
        crate::routers::http::router::Router::wait_for_healthy_workers(
            &[url.to_string()],
            self.worker_startup_timeout_secs,
            self.worker_startup_check_interval_secs,
        )
        .await
        .map_err(|_| PDRouterError::HealthCheckFailed {
            url: url.to_string(),
        })
    }

    /// Start profiling on a backend server
    async fn do_start_profiling(&self, worker_url: &str) {
        // Extract base URL if worker_url is in DP-aware format (e.g., http://127.0.0.1:8081@2)
        let (base_url, _) = dp_utils::parse_worker_url(worker_url);

        let url = format!("{}/start_profile", base_url);
        match self.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Started profiling on {}", base_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to start profiling on {}: status {}",
                    base_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error starting profiling on {}: {}", base_url, e);
            }
        }
    }

    /// Stop profiling on a backend server
    async fn do_stop_profiling(&self, worker_url: &str) {
        // Extract base URL if worker_url is in DP-aware format (e.g., http://127.0.0.1:8081@2)
        let (base_url, _) = dp_utils::parse_worker_url(worker_url);

        let url = format!("{}/stop_profile", base_url);
        match self.client.post(&url).send().await {
            Ok(res) if res.status().is_success() => {
                info!("Stopped profiling on {}", base_url);
            }
            Ok(res) => {
                warn!(
                    "Failed to stop profiling on {}: status {}",
                    base_url,
                    res.status()
                );
            }
            Err(e) => {
                warn!("Error stopping profiling on {}: {}", base_url, e);
            }
        }
    }

    /// Helper: Start profiling on a backend server with timeout
    async fn start_profiling(&self, worker_url: &str) {
        // Only profile if enabled
        if !self.enable_profiling {
            return;
        }

        // Start profiling on the worker
        self.do_start_profiling(worker_url).await;

        // Spawn a timeout task that will call stop_profiling if timeout is reached
        let timeout_secs = self.profile_timeout_secs;
        let worker_url_owned = worker_url.to_string();
        let client = self.client.clone();
        let profiling_tasks_clone = self.profiling_tasks.clone();

        let task_handle = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_secs(timeout_secs)).await;

            info!(
                "Profiling timeout reached for {}, stopping profiling",
                worker_url_owned
            );

            // Stop profiling directly
            let (base_url, _) = dp_utils::parse_worker_url(&worker_url_owned);
            let url = format!("{}/stop_profile", base_url);
            if let Err(e) = client.post(&url).send().await {
                warn!("Error stopping profiling on {}: {}", base_url, e);
            }

            // Remove ourselves from the tasks map
            let mut tasks = profiling_tasks_clone.lock().await;
            tasks.remove(&worker_url_owned);
        });

        // Store the abort handle
        let mut tasks = self.profiling_tasks.lock().await;
        if let Some(old_handle) = tasks.insert(worker_url.to_string(), task_handle.abort_handle()) {
            // Cancel any existing timeout task for this worker
            old_handle.abort();
        }
    }

    /// Helper: Stop profiling on a backend server and cancel timeout task
    async fn stop_profiling(&self, worker_url: &str) {
        // Only stop profiling if it was enabled
        if !self.enable_profiling {
            return;
        }

        // Cancel the timeout task if it exists
        let mut tasks = self.profiling_tasks.lock().await;
        if let Some(handle) = tasks.remove(worker_url) {
            handle.abort();
            info!("Cancelled profiling timeout task for {}", worker_url);
        }

        // Stop profiling on the worker
        self.do_stop_profiling(worker_url).await;
    }

    /// Generic helper for proxying to a specific worker
    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        // Extract base URL if DP-aware format (e.g., http://127.0.0.1:8081@0 → http://127.0.0.1:8081)
        let (base_url, _) = dp_utils::parse_worker_url(&worker_url);
        let url = format!("{}/{}", base_url, endpoint);
        let mut request_builder = self.client.get(&url);

        // Add headers if provided
        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(axum::body::Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (status, format!("{} server returned status: ", res.status())).into_response()
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to proxy request: {}", e),
                )
                    .into_response()
            }
        }
    }

    /// Helper for proxying requests to the first prefill worker
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available".to_string(),
            )
                .into_response()
        }
    }

    /// Helper function to select a worker using the policy
    fn pick_worker_by_policy(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        // Check if we have any workers
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        // Filter available workers (healthy + circuit breaker not open)
        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        // Let policy select from available workers
        let selected_idx = policy
            .select_worker(&available_workers, request_text)
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        // Return the selected Arc worker
        Ok(available_workers[selected_idx].clone())
    }

    /// Select a pair of prefill and decode servers
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        // Get workers from registry - filter by model if provided
        let prefill_workers = if let Some(model) = model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        // Use separate policies for prefill and decode
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill = Self::pick_worker_by_policy(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            "prefill",
        )?;

        let decode = Self::pick_worker_by_policy(
            &decode_workers,
            &*decode_policy,
            request_text,
            "decode",
        )?;

        Ok((prefill, decode))
    }

    /// Modify request for prefill stage (set max_tokens=1)
    fn prepare_prefill_request(mut request: Value) -> Value {
        request["max_tokens"] = json!(1);
        if request.get("max_completion_tokens").is_some() {
            request["max_completion_tokens"] = json!(1);
        }
        // Force non-streaming for prefill to get JSON response with kv_transfer_params
        request["stream"] = json!(false);
        // Remove stream_options since we're setting stream=false
        if let Some(obj) = request.as_object_mut() {
            obj.remove("stream_options");
        }
        request
    }

    /// Convert service discovery instances to Worker objects for policy selection
    fn instances_to_workers(instances: &[(String, String)]) -> Vec<Arc<dyn Worker>> {
        instances
            .iter()
            .map(|(http_addr, _zmq_addr)| {
                let full_url =
                    if http_addr.starts_with("http://") || http_addr.starts_with("https://") {
                        http_addr.clone()
                    } else {
                        format!("http://{}", http_addr)
                    };
                Arc::new(BasicWorker::new(full_url, WorkerType::Regular)) as Arc<dyn Worker>
            })
            .collect()
    }

    /// Select worker using policy-based load balancing
    fn select_worker_with_policy(
        &self,
        instances: &[(String, String)],
        is_prefill: bool,
        request_text: Option<&str>,
    ) -> Option<usize> {
        if instances.is_empty() {
            return None;
        }

        // Convert instances to workers for policy selection
        let workers = Self::instances_to_workers(instances);

        // Get the appropriate policy
        let policy = if is_prefill {
            self.policy_registry.get_prefill_policy()
        } else {
            self.policy_registry.get_decode_policy()
        };

        // Use policy to select worker
        policy.select_worker(&workers, request_text)
    }

    /// Process vLLM request using pure service discovery
    async fn process_vllm_request(&self, request_json: Value, path: &str) -> Response {
        debug!("Processing vLLM request for path: {}", path);
        debug!(
            "Request JSON: {}",
            serde_json::to_string_pretty(&request_json).unwrap_or_default()
        );

        // Get available instances from service discovery
        let prefill_instances = self.service_registry.get_prefill_instances();
        let decode_instances = self.service_registry.get_decode_instances();

        debug!(
            "Found {} prefill instances, {} decode instances from service discovery",
            prefill_instances.len(),
            decode_instances.len()
        );

        if prefill_instances.is_empty() || decode_instances.is_empty() {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No workers available via service discovery: {} prefill, {} decode",
                    prefill_instances.len(),
                    decode_instances.len()
                ),
            )
                .into_response();
        }

        // Use policy-based load balancing to select prefill and decode workers
        let request_text = serde_json::to_string(&request_json).ok();
        let request_str = request_text.as_deref();

        let prefill_idx =
            match self.select_worker_with_policy(&prefill_instances, true, request_str) {
                Some(idx) => idx,
                None => {
                    return (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        "Prefill policy failed to select a worker".to_string(),
                    )
                        .into_response();
                }
            };

        let decode_idx = match self.select_worker_with_policy(&decode_instances, false, request_str)
        {
            Some(idx) => idx,
            None => {
                return (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "Decode policy failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let (prefill_http, prefill_zmq) = &prefill_instances[prefill_idx];
        let (decode_http, decode_zmq) = &decode_instances[decode_idx];

        let prefill_policy_name = self.policy_registry.get_prefill_policy().name();
        let decode_policy_name = self.policy_registry.get_decode_policy().name();

        debug!(
            "vLLM policy-based routing: prefill={}({}) [policy:{}], decode={}({}) [policy:{}]",
            prefill_http,
            prefill_zmq,
            prefill_policy_name,
            decode_http,
            decode_zmq,
            decode_policy_name
        );

        // Process two-stage vLLM request with discovered endpoints
        match self
            .process_vllm_two_stage_request_discovered(
                request_json,
                prefill_http,
                prefill_zmq,
                decode_http,
                decode_zmq,
                path,
            )
            .await
        {
            Ok(response) => {
                debug!("Two-stage processing completed successfully");
                response
            }
            Err(e) => {
                debug!("Two-stage processing failed: {}", e);
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request processing failed: {}", e),
                )
                    .into_response()
            }
        }
    }

    /// Process vLLM request with direct URLs (non-discovery mode)
    async fn process_direct_url_request(&self, request_json: Value, path: &str) -> Response {
        debug!("Processing direct URL vLLM request for path: {}", path);

        // Get prefill and decode workers from worker_registry
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();

        info!(
            "Found {} prefill workers, {} decode workers from worker_registry",
            prefill_workers.len(),
            decode_workers.len()
        );

        if prefill_workers.is_empty() || decode_workers.is_empty() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No workers available: {} prefill, {} decode",
                    prefill_workers.len(),
                    decode_workers.len()
                ),
            )
                .into_response();
        }

        // Select workers using policy
        let request_text = serde_json::to_string(&request_json).ok();
        let request_str = request_text.as_deref();

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill_idx = match prefill_policy.select_worker(&prefill_workers, request_str) {
            Some(idx) => idx,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Prefill policy failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let decode_idx = match decode_policy.select_worker(&decode_workers, request_str) {
            Some(idx) => idx,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Decode policy failed to select a worker".to_string(),
                )
                    .into_response();
            }
        };

        let prefill_worker = &prefill_workers[prefill_idx];
        let decode_worker = &decode_workers[decode_idx];

        info!(
            "Selected prefill={} [policy:{}], decode={} [policy:{}]",
            prefill_worker.url(),
            prefill_policy.name(),
            decode_worker.url(),
            decode_policy.name()
        );

        // Execute dual dispatch with vLLM two-stage processing
        match self
            .process_vllm_two_stage_request(
                request_json,
                prefill_worker.clone(),
                decode_worker.clone(),
                path,
            )
            .await
        {
            Ok(response) => {
                info!("Two-stage processing completed successfully");
                response
            }
            Err(e) => {
                info!("Two-stage processing failed: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request processing failed: {}", e),
                )
                    .into_response()
            }
        }
    }

    /// Two-stage request processing for vLLM disaggregated mode using discovered endpoints
    async fn process_vllm_two_stage_request_discovered(
        &self,
        request_json: Value,
        prefill_http: &str,
        prefill_zmq: &str,
        decode_http: &str,
        decode_zmq: &str,
        path: &str,
    ) -> Result<Response, String> {
        debug!("ENTERED process_vllm_two_stage_request_discovered method");
        debug!(
            "Prefill: HTTP={}, ZMQ={}, Decode: HTTP={}, ZMQ={}, Path: {}",
            prefill_http, prefill_zmq, decode_http, decode_zmq, path
        );

        let request_id = Self::generate_vllm_request_id(prefill_zmq, decode_zmq);
        debug!(
            "Generated vLLM request ID for P2P coordination: {}",
            request_id
        );

        // DO NOT add P2P metadata to internal request_id - let vLLM generate clean internal IDs
        // The P2P metadata will be sent in X-Request-Id header instead

        // Prepare prefill request (max_tokens=1 to force prefill-only mode)
        let mut prefill_request = request_json.clone();
        prefill_request["max_tokens"] = serde_json::Value::Number(serde_json::Number::from(1));
        if prefill_request.get("max_completion_tokens").is_some() {
            prefill_request["max_completion_tokens"] =
                serde_json::Value::Number(serde_json::Number::from(1));
        }
        // Force non-streaming for prefill to get JSON response with kv_transfer_params
        prefill_request["stream"] = serde_json::Value::Bool(false);
        // Remove stream_options since we're setting stream=false
        prefill_request
            .as_object_mut()
            .and_then(|obj| obj.remove("stream_options"));

        // Add kv_transfer_params for NixlConnector support at top level
        // This enables the prefill instance to prepare for remote decode
        prefill_request["kv_transfer_params"] = json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": serde_json::Value::Null,
            "remote_block_ids": serde_json::Value::Null,
            "remote_host": serde_json::Value::Null,
            "remote_port": serde_json::Value::Null
        });

        debug!("Added kv_transfer_params to prefill request for NixlConnector support");

        let prefill_request_str = serde_json::to_string(&prefill_request)
            .map_err(|e| format!("Failed to serialize prefill request: {}", e))?;

        // Stage 1: Send to prefill server with max_tokens=1 and P2P coordination header
        debug!(
            "Stage 1: Sending prefill-only request (max_tokens=1) to prefill server at http://{}",
            prefill_http
        );

        // Extract dp_rank from prefill_http if intra_node_data_parallel_size > 1
        let (prefill_base_http, prefill_dp_rank) = if self.intra_node_data_parallel_size > 1 {
            let prefill_url = format!("http://{}", prefill_http);
            let (base, rank) = dp_utils::parse_worker_url(&prefill_url);
            let base_http = base.replace("http://", "").replace("https://", "");
            (base_http, rank)
        } else {
            (prefill_http.to_string(), None)
        };

        // Start profiling on prefill server
        self.start_profiling(&format!("http://{}", prefill_base_http))
            .await;

        let mut prefill_request_builder = self
            .http_client
            .post(format!("http://{}{}", prefill_base_http, path))
            .header("Content-Type", "application/json")
            .header("X-Request-Id", &request_id); // P2P coordination metadata in header

        // Add X-data-parallel-rank header using shared utility
        prefill_request_builder =
            dp_utils::add_dp_rank_header(prefill_request_builder, prefill_dp_rank);
        if let Some(rank) = prefill_dp_rank {
            debug!(
                "Added X-data-parallel-rank={} header to prefill request",
                rank
            );
        }

        let prefill_response = prefill_request_builder
            .body(prefill_request_str)
            .send()
            .await
            .map_err(|e| format!("Prefill request failed: {}", e))?;

        let prefill_status = prefill_response.status();
        debug!("Prefill server responded with status: {}", prefill_status);

        if !prefill_status.is_success() {
            let error_body = prefill_response.text().await.unwrap_or_default();
            return Err(format!(
                "Prefill server error {}: {}",
                prefill_status, error_body
            ));
        }

        // Extract kv_transfer_params from prefill response
        let prefill_response_text = prefill_response
            .text()
            .await
            .map_err(|e| format!("Failed to read prefill response: {}", e))?;

        debug!("Prefill response body: {}", prefill_response_text);

        let prefill_response_json: Value = serde_json::from_str(&prefill_response_text)
            .map_err(|e| format!("Failed to parse prefill response as JSON: {}", e))?;

        // Extract kv_transfer_params from prefill response if present
        let kv_transfer_params = prefill_response_json.get("kv_transfer_params").cloned();

        if let Some(ref params) = kv_transfer_params {
            debug!(
                "Extracted kv_transfer_params from prefill response: {}",
                serde_json::to_string_pretty(params).unwrap_or_default()
            );
        } else {
            debug!("No kv_transfer_params found in prefill response, will proceed without them");
        }

        // Prepare decode request with kv_transfer_params from prefill response at top level
        let mut decode_request = request_json.clone();
        if let Some(params) = kv_transfer_params {
            decode_request["kv_transfer_params"] = params;
            debug!("Added kv_transfer_params to decode request");
        }

        let decode_request_str = serde_json::to_string(&decode_request)
            .map_err(|e| format!("Failed to serialize decode request: {}", e))?;

        // Stop profiling on prefill server after its work is done
        self.stop_profiling(&format!("http://{}", prefill_base_http))
            .await;

        // Stage 2: Send to decode server with original request and same P2P coordination header
        debug!(
            "Stage 2: Sending original request to decode server at http://{}",
            decode_http
        );

        // Extract dp_rank from decode_http if intra_node_data_parallel_size > 1
        let (decode_base_http, decode_dp_rank) = if self.intra_node_data_parallel_size > 1 {
            let decode_url = format!("http://{}", decode_http);
            let (base, rank) = dp_utils::parse_worker_url(&decode_url);
            let base_http = base.replace("http://", "").replace("https://", "");
            (base_http, rank)
        } else {
            (decode_http.to_string(), None)
        };

        // Start profiling on decode server
        self.start_profiling(&format!("http://{}", decode_base_http))
            .await;

        let mut decode_request_builder = self
            .http_client
            .post(format!("http://{}{}", decode_base_http, path))
            .header("Content-Type", "application/json")
            .header("X-Request-Id", &request_id); // Same P2P coordination metadata in header

        // Add X-data-parallel-rank header using shared utility
        decode_request_builder =
            dp_utils::add_dp_rank_header(decode_request_builder, decode_dp_rank);
        if let Some(rank) = decode_dp_rank {
            debug!(
                "Added X-data-parallel-rank={} header to decode request",
                rank
            );
        }

        let decode_response = decode_request_builder
            .body(decode_request_str)
            .send()
            .await
            .map_err(|e| format!("Decode request failed: {}", e))?;

        debug!(
            "Decode server responded with status: {}",
            decode_response.status()
        );

        // Stop profiling on decode server after response received
        self.stop_profiling(&format!("http://{}", decode_base_http))
            .await;

        // Check if logprobs merging is needed
        let needs_logprobs = request_json.get("logprobs").is_some()
            || request_json
                .get("echo")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        let is_streaming = request_json
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // If logprobs requested and non-streaming, merge prefill and decode logprobs
        if needs_logprobs && !is_streaming {
            debug!("Logprobs requested and non-streaming - merging prefill and decode logprobs");

            let status = decode_response.status();
            let headers = decode_response.headers().clone();
            let decode_body = decode_response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read decode response: {}", e))?;

            // Parse decode response as JSON
            let mut decode_json: Value = serde_json::from_slice(&decode_body)
                .map_err(|e| format!("Failed to parse decode response as JSON: {}", e))?;

            // Merge logprobs from prefill into decode response
            let merged =
                logprobs_merge::merge_logprobs_in_json(&prefill_response_json, &mut decode_json);
            if merged {
                debug!("Successfully merged logprobs from prefill and decode responses");
            } else {
                warn!("No logprobs were merged (might be expected if logprobs not in response)");
            }

            // Serialize merged response
            let merged_body = serde_json::to_vec(&decode_json)
                .map_err(|e| format!("Failed to serialize merged response: {}", e))?;

            let mut response_builder = axum::http::Response::builder().status(status);
            for (name, value) in headers.iter() {
                response_builder = response_builder.header(name, value);
            }

            let response = response_builder
                .body(axum::body::Body::from(merged_body))
                .map_err(|e| format!("Failed to build response: {}", e))?;

            Ok(response)
        } else {
            // No logprobs merging needed - return decode response as-is
            debug!(
                "No logprobs merging needed (streaming={}, needs_logprobs={})",
                is_streaming, needs_logprobs
            );

            let status = decode_response.status();
            let headers = decode_response.headers().clone();
            let body = decode_response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read decode response: {}", e))?;

            let mut response_builder = axum::http::Response::builder().status(status);
            for (name, value) in headers.iter() {
                response_builder = response_builder.header(name, value);
            }

            let response = response_builder
                .body(axum::body::Body::from(body))
                .map_err(|e| format!("Failed to build response: {}", e))?;

            Ok(response)
        }
    }

    /// Two-stage request processing for vLLM disaggregated mode
    async fn process_vllm_two_stage_request(
        &self,
        original_request: Value,
        prefill_worker: Arc<dyn Worker>,
        decode_worker: Arc<dyn Worker>,
        path: &str,
    ) -> Result<Response, PDRouterError> {
        debug!("ENTERED process_vllm_two_stage_request method");
        debug!(
            "Prefill worker: {}, Decode worker: {}, Path: {}",
            prefill_worker.url(),
            decode_worker.url(),
            path
        );

        let prefill_zmq_addr = self.get_zmq_address(prefill_worker.url(), ServiceType::Prefill);
        let decode_zmq_addr = self.get_zmq_address(decode_worker.url(), ServiceType::Decode);
        let request_id = Self::generate_vllm_request_id(&prefill_zmq_addr, &decode_zmq_addr);

        debug!("Generated vLLM request ID: {}", request_id);
        debug!("vLLM Proxy Comparison:");
        debug!("  vLLM Proxy Request ID format: ___prefill_addr_{{zmq_addr}}___decode_addr_{{zmq_addr}}_{{uuid}}");
        debug!("  Our Request ID format: ___prefill_addr_{{http_addr}}___decode_addr_{{http_addr}}_{{uuid}}");
        debug!("  vLLM Proxy headers: Authorization: Bearer $OPENAI_API_KEY, X-Request-Id: {{request_id}}");
        debug!(
            "  Our headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
            request_id
        );

        // Stage 1: Prepare prefill request with max_tokens=1 and kv_transfer_params
        let mut prefill_request = Self::prepare_prefill_request(original_request.clone());

        // Add kv_transfer_params for NixlConnector support at top level
        // This enables the prefill instance to prepare for remote decode
        prefill_request["kv_transfer_params"] = json!({
            "do_remote_decode": true,
            "do_remote_prefill": false,
            "remote_engine_id": serde_json::Value::Null,
            "remote_block_ids": serde_json::Value::Null,
            "remote_host": serde_json::Value::Null,
            "remote_port": serde_json::Value::Null
        });

        debug!("Added kv_transfer_params to prefill request for NixlConnector support");

        // Extract base URL and dp_rank if intra_node_data_parallel_size > 1
        let (prefill_base_url, prefill_dp_rank) = if self.intra_node_data_parallel_size > 1 {
            match dp_utils::extract_dp_rank(prefill_worker.url()) {
                Ok((base, rank)) => (base.to_string(), Some(rank)),
                Err(e) => {
                    return Err(PDRouterError::NetworkError {
                        message: format!(
                            "Failed to extract dp_rank from prefill worker URL {}: {}",
                            prefill_worker.url(),
                            e
                        ),
                    });
                }
            }
        } else {
            (prefill_worker.url().to_string(), None)
        };

        let prefill_url = format!("{}{}", prefill_base_url, path);

        debug!(
            "vLLM Stage 1 - Prefill: {} with request_id: {}",
            prefill_url, request_id
        );
        if let Some(rank) = prefill_dp_rank {
            debug!("Prefill request headers: Authorization=Bearer [REDACTED], X-Request-Id={}, X-data-parallel-rank={}", request_id, rank);
        } else {
            debug!(
                "Prefill request headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
                request_id
            );
        }
        debug!(
            "Prefill request payload: {}",
            serde_json::to_string_pretty(&prefill_request).unwrap_or_default()
        );

        // Start profiling on prefill server
        self.start_profiling(&prefill_base_url).await;

        let mut prefill_request_builder = self
            .client
            .post(&prefill_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        // Add X-data-parallel-rank header if intra_node_data_parallel_size > 1
        if let Some(rank) = prefill_dp_rank {
            prefill_request_builder =
                prefill_request_builder.header("X-data-parallel-rank", rank.to_string());
        }

        let prefill_response = prefill_request_builder
            .json(&prefill_request)
            .send()
            .await
            .map_err(|e| PDRouterError::NetworkError {
                message: format!("Prefill request failed to {}: {}", prefill_url, e),
            })?;

        debug!("Prefill response status: {}", prefill_response.status());
        debug!(
            "Prefill response headers: {:?}",
            prefill_response.headers()
        );

        // Extract prefill response body to get kv_transfer_params
        let prefill_bytes =
            prefill_response
                .bytes()
                .await
                .map_err(|e| PDRouterError::NetworkError {
                    message: format!(
                        "Failed to read prefill response from {}: {}",
                        prefill_url, e
                    ),
                })?;

        debug!(
            "Prefill response body size: {} bytes",
            prefill_bytes.len()
        );
        if prefill_bytes.len() < 1024 {
            debug!(
                "Prefill response body content: {}",
                String::from_utf8_lossy(&prefill_bytes)
            );
        }

        // Parse prefill response to extract kv_transfer_params
        let prefill_response_json: Value =
            serde_json::from_slice(&prefill_bytes).map_err(|e| PDRouterError::NetworkError {
                message: format!("Failed to parse prefill response as JSON: {}", e),
            })?;

        // Extract kv_transfer_params from prefill response if present
        let kv_transfer_params = prefill_response_json.get("kv_transfer_params").cloned();

        if let Some(ref params) = kv_transfer_params {
            debug!(
                "Extracted kv_transfer_params from prefill response: {}",
                serde_json::to_string_pretty(params).unwrap_or_default()
            );
        } else {
            debug!("No kv_transfer_params found in prefill response, will proceed without them");
        }

        // Stop profiling on prefill server after its work is done
        self.stop_profiling(&prefill_base_url).await;

        debug!("vLLM Stage 1 completed, starting Stage 2 - Decode");

        // Stage 2: Prepare decode request with kv_transfer_params from prefill response at top level
        let mut decode_request = original_request.clone();
        if let Some(params) = kv_transfer_params {
            decode_request["kv_transfer_params"] = params;
            debug!("Added kv_transfer_params to decode request");
        }

        // Extract base URL and dp_rank if intra_node_data_parallel_size > 1
        let (decode_base_url, decode_dp_rank) = if self.intra_node_data_parallel_size > 1 {
            match dp_utils::extract_dp_rank(decode_worker.url()) {
                Ok((base, rank)) => (base.to_string(), Some(rank)),
                Err(e) => {
                    return Err(PDRouterError::NetworkError {
                        message: format!(
                            "Failed to extract dp_rank from decode worker URL {}: {}",
                            decode_worker.url(),
                            e
                        ),
                    });
                }
            }
        } else {
            (decode_worker.url().to_string(), None)
        };

        let decode_url = format!("{}{}", decode_base_url, path);

        debug!(
            "vLLM Stage 2 - Decode: {} with request_id: {}",
            decode_url, request_id
        );
        if let Some(rank) = decode_dp_rank {
            debug!("Decode request headers: Authorization=Bearer [REDACTED], X-Request-Id={}, X-data-parallel-rank={}", request_id, rank);
        } else {
            debug!(
                "Decode request headers: Authorization=Bearer [REDACTED], X-Request-Id={}",
                request_id
            );
        }
        debug!(
            "Decode request payload: {}",
            serde_json::to_string_pretty(&decode_request).unwrap_or_default()
        );

        // Start profiling on decode server
        self.start_profiling(&decode_base_url).await;

        let mut decode_request_builder = self
            .client
            .post(&decode_url)
            .header("Content-Type", "application/json")
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                ),
            )
            .header("X-Request-Id", &request_id);

        // Add X-data-parallel-rank header if intra_node_data_parallel_size > 1
        if let Some(rank) = decode_dp_rank {
            decode_request_builder =
                decode_request_builder.header("X-data-parallel-rank", rank.to_string());
        }

        let decode_response = decode_request_builder
            .json(&decode_request)
            .send()
            .await
            .map_err(|e| PDRouterError::NetworkError {
                message: format!("Decode request failed to {}: {}", decode_url, e),
            })?;

        // Stop profiling on decode server after response received
        self.stop_profiling(&decode_base_url).await;

        let status = decode_response.status();
        let headers = decode_response.headers().clone();

        info!("Decode response status: {}", status);
        info!("Decode response headers: {:?}", headers);

        // Check if logprobs merging is needed
        let needs_logprobs = original_request.get("logprobs").is_some()
            || original_request
                .get("echo")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        let is_streaming = original_request
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // If logprobs requested and non-streaming, merge prefill and decode logprobs
        if needs_logprobs && !is_streaming {
            debug!("Logprobs requested and non-streaming - merging prefill and decode logprobs");

            // Read decode response body
            let decode_body =
                decode_response
                    .bytes()
                    .await
                    .map_err(|e| PDRouterError::NetworkError {
                        message: format!(
                            "Failed to read decode response from {}: {}",
                            decode_url, e
                        ),
                    })?;

            // Parse decode response as JSON
            let mut decode_json: Value =
                serde_json::from_slice(&decode_body).map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to parse decode response as JSON: {}", e),
                })?;

            // Merge logprobs from prefill into decode response
            let merged =
                logprobs_merge::merge_logprobs_in_json(&prefill_response_json, &mut decode_json);
            if merged {
                debug!("Successfully merged logprobs from prefill and decode responses");
            } else {
                warn!("No logprobs were merged (might be expected if logprobs not in response)");
            }

            // Serialize merged response
            let merged_body =
                serde_json::to_vec(&decode_json).map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to serialize merged response: {}", e),
                })?;

            let mut response_builder = Response::builder().status(status);
            for (key, value) in headers.iter() {
                if key != "transfer-encoding" && key != "content-length" {
                    response_builder = response_builder.header(key, value);
                }
            }

            response_builder.body(Body::from(merged_body)).map_err(|e| {
                PDRouterError::NetworkError {
                    message: format!("Failed to build response from {}: {}", decode_url, e),
                }
            })
        } else {
            // No logprobs merging needed - return decode response as-is (streaming or no logprobs)
            debug!(
                "No logprobs merging needed (streaming={}, needs_logprobs={})",
                is_streaming, needs_logprobs
            );

            let mut response_builder = Response::builder().status(status);
            for (key, value) in headers.iter() {
                if key != "transfer-encoding" && key != "content-length" {
                    response_builder = response_builder.header(key, value);
                }
            }

            let body = Body::from_stream(decode_response.bytes_stream());
            response_builder
                .body(body)
                .map_err(|e| PDRouterError::NetworkError {
                    message: format!("Failed to build response from {}: {}", decode_url, e),
                })
        }
    }

    /// Background task to monitor worker loads
    async fn monitor_worker_loads_with_client(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        client: Client,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
    ) {
        loop {
            let mut loads = HashMap::new();

            let futures: Vec<_> = worker_urls
                .iter()
                .map(|url| {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let load = Self::get_worker_load_static(&client, &url).await.unwrap_or(0);
                        (url, load)
                    }
                })
                .collect();

            let results = futures_util::future::join_all(futures).await;

            for (url, load) in results {
                loads.insert(url, load);
            }

            debug!("Worker loads updated: {:?}", loads);

            // Update both policies with current loads
            prefill_policy.update_loads(&loads);
            decode_policy.update_loads(&loads);

            // Check if receiver is still active
            if tx.send(loads).is_err() {
                info!("Load monitor receiver dropped, shutting down monitor task");
                break;
            }

            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    }

    /// Get worker load (static version for use in async contexts)
    async fn get_worker_load_static(client: &Client, url: &str) -> Option<isize> {
        // Extract base URL if DP-aware format
        let (base_url, _) = dp_utils::parse_worker_url(url);
        let load_url = format!("{}/load", base_url);

        match client.get(&load_url).send().await {
            Ok(res) if res.status().is_success() => {
                if let Ok(text) = res.text().await {
                    text.trim().parse().ok()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Create a new vLLM PD router
    /// Supports two modes:
    /// 1. Discovery mode: discovery_address is Some, prefill_urls and decode_urls are empty
    /// 2. Direct URL mode: discovery_address is None, prefill_urls and decode_urls are provided
    pub async fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        discovery_address: Option<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Build a dedicated prefill client for fire-and-forget semantics
        let prefill_client = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .http1_only()
            .connect_timeout(Duration::from_millis(300))
            .timeout(Duration::from_secs(ctx.router_config.request_timeout_secs))
            .build()
            .map_err(|e| format!("Failed to build prefill client: {}", e))?;

        // Create bounded channel for prefill response draining
        let (prefill_drain_tx, mut prefill_drain_rx) = mpsc::channel::<reqwest::Response>(2000);

        // Spawn a coordinator with limited concurrent drain tasks
        tokio::spawn(async move {
            info!("Prefill drain coordinator started");

            let max_concurrent_drains = 100;
            let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_drains));

            while let Some(response) = prefill_drain_rx.recv().await {
                let permit = semaphore.clone().acquire_owned().await;

                match permit {
                    Ok(permit) => {
                        tokio::spawn(async move {
                            use futures_util::StreamExt;
                            let url = response.url().to_string();
                            let status = response.status();

                            if !status.is_success() {
                                error!("Prefill drain: error status={} url={}", status, url);
                            }

                            let start = std::time::Instant::now();
                            let mut stream = response.bytes_stream();
                            let mut bytes_drained = 0;

                            while let Some(chunk_result) = stream.next().await {
                                match chunk_result {
                                    Ok(chunk) => bytes_drained += chunk.len(),
                                    Err(e) => {
                                        debug!(
                                            "Prefill drain: error streaming url={} error={}",
                                            url, e
                                        );
                                        break;
                                    }
                                }
                            }

                            let elapsed = start.elapsed();
                            if elapsed > Duration::from_millis(100) {
                                debug!(
                                    "Prefill drain: slow drain {} bytes from {} in {:?}",
                                    bytes_drained, url, elapsed
                                );
                            }

                            drop(permit);
                        });
                    }
                    Err(_) => {
                        break;
                    }
                }
            }
            info!("Prefill drain coordinator shutting down");
        });

        if let Some(ref addr) = discovery_address {
            // Discovery mode
            info!(
                "VllmPDRouter::new called in discovery mode with address: {}",
                addr
            );

            // Initialize service discovery
            let mut service_registry = ServiceRegistry::new();

            info!("Starting vLLM service discovery on {}", addr);
            service_registry
                .start_listener(addr)
                .await
                .map_err(|e| format!("Failed to start service discovery: {}", e))?;

            // Set up background load monitoring (no workers to monitor in discovery mode)
            let (_tx, rx) = tokio::sync::watch::channel(HashMap::new());
            let worker_loads = Arc::new(rx);

            info!("VllmPDRouter created successfully with pure service discovery");

            Ok(Self {
                worker_registry: Arc::clone(&ctx.worker_registry),
                policy_registry: Arc::clone(&ctx.policy_registry),
                worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
                worker_startup_check_interval_secs: ctx.router_config.worker_startup_check_interval_secs,
                worker_loads,
                load_monitor_handle: None,
                client: ctx.client.clone(),
                prefill_client,
                retry_config: ctx.router_config.effective_retry_config(),
                circuit_breaker_config: core_cb_config,
                prefill_drain_tx,
                service_registry: Arc::new(service_registry),
                http_client: reqwest::Client::new(),
                use_discovery: true,
                enable_profiling: ctx.router_config.enable_profiling,
                profile_timeout_secs: ctx.router_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
            })
        } else {
            // Direct URL mode
            info!(
                "VllmPDRouter::new called in direct URL mode with {} prefill, {} decode workers",
                prefill_urls.len(),
                decode_urls.len()
            );

            // Automatically expand to DP-aware format when intra_node_data_parallel_size > 1
            let (expanded_prefill_urls, expanded_decode_urls) =
                if ctx.router_config.intra_node_data_parallel_size > 1 {
                    info!(
                        "DP-aware mode enabled (intra_node_data_parallel_size={}), expanding worker URLs",
                        ctx.router_config.intra_node_data_parallel_size
                    );

                    let prefill_base_urls: Vec<String> =
                        prefill_urls.iter().map(|(url, _)| url.clone()).collect();

                    let expanded_prefill = dp_utils::get_dp_aware_workers(
                        &prefill_base_urls,
                        &ctx.router_config.api_key,
                        ctx.router_config.intra_node_data_parallel_size,
                    )
                    .await
                    .map_err(|e| format!("Failed to expand prefill workers: {}", e))?;

                    let expanded_decode = dp_utils::get_dp_aware_workers(
                        &decode_urls,
                        &ctx.router_config.api_key,
                        ctx.router_config.intra_node_data_parallel_size,
                    )
                    .await
                    .map_err(|e| format!("Failed to expand decode workers: {}", e))?;

                    info!(
                        "Expanded {} prefill URLs to {} DP-aware URLs",
                        prefill_base_urls.len(),
                        expanded_prefill.len()
                    );
                    info!(
                        "Expanded {} decode URLs to {} DP-aware URLs",
                        decode_urls.len(),
                        expanded_decode.len()
                    );

                    let prefill_with_ports: Vec<(String, Option<u16>)> = expanded_prefill
                        .into_iter()
                        .map(|url| {
                            let port = prefill_urls.first().and_then(|(_, p)| *p);
                            (url, port)
                        })
                        .collect();

                    (prefill_with_ports, expanded_decode)
                } else {
                    info!("DP-aware mode disabled, using original worker URLs");
                    (prefill_urls, decode_urls)
                };

            // Register prefill workers in the registry
            for (url, port) in &expanded_prefill_urls {
                let worker = BasicWorker::new(
                    url.clone(),
                    WorkerType::Prefill {
                        bootstrap_port: *port,
                    },
                )
                .with_circuit_breaker_config(core_cb_config.clone())
                .with_health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                });
                ctx.worker_registry.register(Arc::new(worker));
            }

            // Register decode workers in the registry
            for url in &expanded_decode_urls {
                let worker = BasicWorker::new(url.clone(), WorkerType::Decode)
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(HealthConfig {
                        timeout_secs: ctx.router_config.health_check.timeout_secs,
                        check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                        endpoint: ctx.router_config.health_check.endpoint.clone(),
                        failure_threshold: ctx.router_config.health_check.failure_threshold,
                        success_threshold: ctx.router_config.health_check.success_threshold,
                    });
                ctx.worker_registry.register(Arc::new(worker));
            }

            // Get all workers from registry for health check
            let all_workers = ctx.worker_registry.get_all();
            let all_urls: Vec<String> = all_workers
                .iter()
                .map(|worker| worker.url().to_string())
                .collect();
            if !all_urls.is_empty() {
                crate::routers::http::router::Router::wait_for_healthy_workers(
                    &all_urls,
                    ctx.router_config.worker_startup_timeout_secs,
                    ctx.router_config.worker_startup_check_interval_secs,
                )
                .await?;
            }

            // Set up background load monitoring for power-of-two selection
            let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
            let worker_loads = Arc::new(rx);

            let prefill_policy = ctx.policy_registry.get_prefill_policy();
            let decode_policy = ctx.policy_registry.get_decode_policy();

            let load_monitor_handle =
                if prefill_policy.name() == "power_of_two" || decode_policy.name() == "power_of_two" {
                    let monitor_urls = all_urls.clone();
                    let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
                    let monitor_client = ctx.client.clone();
                    let prefill_policy_clone = Arc::clone(&prefill_policy);
                    let decode_policy_clone = Arc::clone(&decode_policy);

                    Some(Arc::new(tokio::spawn(async move {
                        Self::monitor_worker_loads_with_client(
                            monitor_urls,
                            tx,
                            monitor_interval,
                            monitor_client,
                            prefill_policy_clone,
                            decode_policy_clone,
                        )
                        .await;
                    })))
                } else {
                    None
                };

            // No service discovery in direct URL mode
            let service_registry = ServiceRegistry::new();

            info!("VllmPDRouter created successfully with direct URLs");

            Ok(Self {
                worker_registry: Arc::clone(&ctx.worker_registry),
                policy_registry: Arc::clone(&ctx.policy_registry),
                worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
                worker_startup_check_interval_secs: ctx.router_config.worker_startup_check_interval_secs,
                worker_loads,
                load_monitor_handle,
                client: ctx.client.clone(),
                prefill_client,
                retry_config: ctx.router_config.effective_retry_config(),
                circuit_breaker_config: core_cb_config,
                prefill_drain_tx,
                service_registry: Arc::new(service_registry),
                http_client: reqwest::Client::new(),
                use_discovery: false,
                enable_profiling: ctx.router_config.enable_profiling,
                profile_timeout_secs: ctx.router_config.profile_timeout_secs,
                profiling_tasks: Arc::new(Mutex::new(HashMap::new())),
                intra_node_data_parallel_size: ctx.router_config.intra_node_data_parallel_size,
            })
        }
    }

    /// Add a prefill server to the router
    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        // Check if already exists
        if self.worker_registry.get_by_url(&url).is_some() {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }

        // Create Worker for the new prefill server with circuit breaker configuration
        let worker = WorkerFactory::create_prefill_with_config(
            url.clone(),
            bootstrap_port,
            self.circuit_breaker_config.clone(),
        );

        let worker_arc: Arc<dyn Worker> = Arc::from(worker);

        // Register the worker in the registry
        self.worker_registry.register(worker_arc.clone());

        // Notify PolicyRegistry about the new worker
        let model_id = worker_arc.model_id();
        let policy = self.policy_registry.on_worker_added(model_id, None);

        // If this is a cache-aware policy, update it with all workers for this model
        if policy.name() == "cache_aware" {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                let model_workers = self.worker_registry.get_by_model_fast(model_id);
                cache_aware.init_workers(&model_workers);
            }
        }

        info!("Added prefill server: {}", url);
        Ok(format!("Successfully added prefill server: {}", url))
    }

    /// Add a decode server to the router
    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        // Check if already exists
        if self.worker_registry.get_by_url(&url).is_some() {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }

        // Create Worker for the new decode server with circuit breaker configuration
        let worker = WorkerFactory::create_decode_with_config(
            url.clone(),
            self.circuit_breaker_config.clone(),
        );

        let worker_arc: Arc<dyn Worker> = Arc::from(worker);

        // Register the worker in the registry
        self.worker_registry.register(worker_arc.clone());

        // Notify PolicyRegistry about the new worker
        let model_id = worker_arc.model_id();
        let policy = self.policy_registry.on_worker_added(model_id, None);

        // If this is a cache-aware policy, update it with all workers for this model
        if policy.name() == "cache_aware" {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                let model_workers = self.worker_registry.get_by_model_fast(model_id);
                cache_aware.init_workers(&model_workers);
            }
        }

        info!("Added decode server: {}", url);
        Ok(format!("Successfully added decode server: {}", url))
    }

    /// Remove a prefill server from the router
    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        // Check if worker exists and get model_id
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        // Remove from registry
        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            // Get the policy for this model to update cache-aware if needed
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed prefill server: {}", url);
            Ok(format!("Successfully removed prefill server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    /// Remove a decode server from the router
    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        // Check if worker exists and get model_id
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        // Remove from registry
        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            // Get the policy for this model to update cache-aware if needed
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed decode server: {}", url);
            Ok(format!("Successfully removed decode server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    /// Get a reference to the worker registry
    pub fn worker_registry(&self) -> &WorkerRegistry {
        &self.worker_registry
    }
}

// Self-contained RouterTrait implementation
#[async_trait]
impl RouterTrait for VllmPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("health", Some(headers))
            .await
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("health_generate", Some(headers))
            .await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("get_server_info", Some(headers))
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        let headers = header_utils::copy_request_headers(&req);
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // For now, route_generate is not supported in vLLM PD mode
        // This would need additional implementation for the /generate endpoint
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not implemented for vLLM PD mode",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        info!(
            "vLLM route_chat called, use_discovery={}",
            self.use_discovery
        );

        // Convert to generic request
        let request_json = match serde_json::to_value(body) {
            Ok(json) => {
                info!(
                    "Serialized chat request: {}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
                json
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing with service discovery
            info!("Using service discovery mode, processing vLLM two-stage request");
            self.process_vllm_request(request_json, "/v1/chat/completions")
                .await
        } else {
            // Direct URL mode - use worker registry
            info!("Using direct URL mode with VllmPDRouter's own routing logic");
            self.process_direct_url_request(request_json, "/v1/chat/completions")
                .await
        }
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        info!(
            "vLLM route_completion called, use_discovery={}",
            self.use_discovery
        );

        // Convert to generic request
        let request_json = match serde_json::to_value(body) {
            Ok(json) => {
                info!(
                    "Serialized completion request: {}",
                    serde_json::to_string_pretty(&json).unwrap_or_default()
                );
                json
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Serialization error: {}", e),
                )
                    .into_response()
            }
        };

        if self.use_discovery {
            // Discovery mode - use vLLM-specific two-stage processing with service discovery
            info!("Using service discovery mode, processing vLLM two-stage request");
            self.process_vllm_request(request_json, "/v1/completions")
                .await
        } else {
            // Direct URL mode - use worker registry
            info!("Using direct URL mode with VllmPDRouter's own routing logic");
            self.process_direct_url_request(request_json, "/v1/completions")
                .await
        }
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses endpoint not implemented for vLLM PD mode",
        )
            .into_response()
    }

    async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Get response endpoint not implemented for vLLM PD mode",
        )
            .into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Cancel response endpoint not implemented for vLLM PD mode",
        )
            .into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Embeddings endpoint not implemented for vLLM PD mode",
        )
            .into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Rerank endpoint not implemented for vLLM PD mode",
        )
            .into_response()
    }

    async fn flush_cache(&self) -> Response {
        // Flush cache on all workers
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();

        let mut results = Vec::new();
        let mut errors = Vec::new();

        for worker in prefill_workers.iter().chain(decode_workers.iter()) {
            let (base_url, _) = dp_utils::parse_worker_url(worker.url());
            let url = format!("{}/flush_cache", base_url);
            match self.client.post(&url).send().await {
                Ok(res) if res.status().is_success() => {
                    results.push(format!("{}: OK", worker.url()));
                }
                Ok(res) => {
                    errors.push(format!("{} returned status: {}", worker.url(), res.status()));
                }
                Err(e) => {
                    errors.push(format!("{} error: {}", worker.url(), e));
                }
            }
        }

        if errors.is_empty() {
            (StatusCode::OK, format!("Flushed cache: {:?}", results)).into_response()
        } else {
            (
                StatusCode::PARTIAL_CONTENT,
                format!("Results: {:?}, Errors: {:?}", results, errors),
            )
                .into_response()
        }
    }

    async fn get_worker_loads(&self) -> Response {
        let loads = self.worker_loads.borrow().clone();
        (StatusCode::OK, serde_json::to_string(&loads).unwrap_or_default()).into_response()
    }

    fn router_type(&self) -> &'static str {
        "vllm_pd"
    }

    fn readiness(&self) -> Response {
        // Check if we have any workers available
        let prefill_count = self.worker_registry.get_prefill_workers().len();
        let decode_count = self.worker_registry.get_decode_workers().len();

        if self.use_discovery {
            // In discovery mode, check service registry
            let prefill_instances = self.service_registry.get_prefill_instances();
            let decode_instances = self.service_registry.get_decode_instances();

            if !prefill_instances.is_empty() && !decode_instances.is_empty() {
                (StatusCode::OK, "Ready (discovery mode)").into_response()
            } else {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "Not ready: {} prefill, {} decode instances discovered",
                        prefill_instances.len(),
                        decode_instances.len()
                    ),
                )
                    .into_response()
            }
        } else {
            // In direct URL mode, check worker registry
            if prefill_count > 0 && decode_count > 0 {
                (StatusCode::OK, "Ready").into_response()
            } else {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "Not ready: {} prefill, {} decode workers",
                        prefill_count, decode_count
                    ),
                )
                    .into_response()
            }
        }
    }
}

// Self-contained WorkerManagement implementation
#[async_trait]
impl WorkerManagement for VllmPDRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        // VllmPDRouter requires separate add_prefill_server/add_decode_server calls
        Err("VllmPDRouter requires add_prefill_server or add_decode_server".to_string())
    }

    fn remove_worker(&self, worker_url: &str) {
        // Remove from registry directly
        self.worker_registry.remove_by_url(worker_url);
    }

    fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }
}
