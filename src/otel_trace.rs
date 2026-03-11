//! OpenTelemetry tracing integration.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    },
    time::Duration,
};

use anyhow::Result;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use opentelemetry::{global, trace::TracerProvider as _, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::TraceContextPropagator,
    runtime,
    trace::{BatchConfigBuilder, BatchSpanProcessor, Tracer as SdkTracer, TracerProvider},
    Resource,
};
use tokio::task::spawn_blocking;
use tracing::{Metadata, Subscriber};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{
    layer::{Context, Filter},
    Layer,
};

/// Whether OpenTelemetry tracing is enabled.
///
/// This flag guards access to TRACER and PROVIDER. We use Release/Acquire
/// ordering to ensure proper synchronization: writes to TRACER/PROVIDER
/// happen-before the Release store, and Acquire loads happen-before reads.
static ENABLED: AtomicBool = AtomicBool::new(false);
static TRACER: OnceLock<SdkTracer> = OnceLock::new();
static PROVIDER: OnceLock<TracerProvider> = OnceLock::new();

const OTEL_SPAN_TARGET: &str = "vllm_router_rs::otel-trace";

/// Filter that only allows specific module targets to be exported to OTEL.
#[derive(Clone, Copy, Default)]
pub(crate) struct CustomOtelFilter;

impl CustomOtelFilter {
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    fn is_allowed(target: &str) -> bool {
        target.starts_with(OTEL_SPAN_TARGET)
    }
}

impl<S> Filter<S> for CustomOtelFilter
where
    S: Subscriber,
{
    #[inline]
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        Self::is_allowed(meta.target())
    }

    #[inline]
    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> tracing::subscriber::Interest {
        if Self::is_allowed(meta.target()) {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
}

pub fn otel_tracing_init(enable: bool, otlp_endpoint: Option<&str>) -> Result<()> {
    if !enable {
        ENABLED.store(false, Ordering::Release);
        return Ok(());
    }

    let endpoint = otlp_endpoint.unwrap_or("localhost:4317");
    let endpoint = if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        format!("http://{endpoint}")
    } else {
        endpoint.to_string()
    };

    global::set_text_map_propagator(TraceContextPropagator::new());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .with_protocol(opentelemetry_otlp::Protocol::Grpc)
        .build()
        .map_err(|e| {
            eprintln!("[tracing] Failed to create OTLP exporter: {e}");
            anyhow::anyhow!("Failed to create OTLP exporter: {e}")
        })?;

    let batch_config = BatchConfigBuilder::default()
        .with_scheduled_delay(Duration::from_millis(500))
        .with_max_export_batch_size(64)
        .build();

    let span_processor = BatchSpanProcessor::builder(exporter, runtime::Tokio)
        .with_batch_config(batch_config)
        .build();

    let resource = Resource::default().merge(&Resource::new(vec![KeyValue::new(
        "service.name",
        "vllm-router",
    )]));

    let provider = TracerProvider::builder()
        .with_span_processor(span_processor)
        .with_resource(resource)
        .build();

    PROVIDER
        .set(provider.clone())
        .map_err(|_| anyhow::anyhow!("Provider already initialized"))?;

    let tracer = provider.tracer("vllm-router");

    TRACER
        .set(tracer)
        .map_err(|_| anyhow::anyhow!("Tracer already initialized"))?;

    let _ = global::set_tracer_provider(provider);

    ENABLED.store(true, Ordering::Release);

    eprintln!("[tracing] OpenTelemetry initialized successfully");
    Ok(())
}

/// Get the OpenTelemetry tracing layer. Must be called after `otel_tracing_init`.
pub fn get_otel_layer<S>() -> Result<Box<dyn Layer<S> + Send + Sync + 'static>>
where
    S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
{
    if !is_otel_enabled() {
        anyhow::bail!("OpenTelemetry is not enabled");
    }

    let tracer = TRACER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Tracer not initialized. Call otel_tracing_init first."))?
        .clone();

    let layer = tracing_opentelemetry::layer()
        .with_tracer(tracer)
        .with_filter(CustomOtelFilter::new());

    Ok(Box::new(layer))
}

/// Check if OpenTelemetry tracing is enabled.
#[inline]
pub fn is_otel_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

pub async fn flush_spans_async() -> Result<()> {
    if !is_otel_enabled() {
        return Ok(());
    }

    let provider = PROVIDER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Provider not initialized"))?
        .clone();

    spawn_blocking(move || provider.force_flush())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to flush spans: {e}"))?;

    Ok(())
}

pub fn shutdown_otel() {
    if ENABLED.load(Ordering::Acquire) {
        global::shutdown_tracer_provider();
        ENABLED.store(false, Ordering::Release);
        eprintln!("[tracing] OpenTelemetry shut down");
    }
}

/// Inject W3C trace context headers into an HTTP request's HeaderMap.
///
/// When OTel is enabled, this injects the current span's trace context
/// (traceparent, tracestate) into outgoing requests, making the router's
/// span the parent of backend spans.
///
/// When OTel is disabled, this is a no-op.
#[inline]
pub fn inject_trace_context_http(headers: &mut HeaderMap) {
    if !is_otel_enabled() {
        return;
    }

    let context = tracing::Span::current().context();

    struct HeaderInjector<'a>(&'a mut HeaderMap);

    impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
        #[inline]
        fn set(&mut self, key: &str, value: String) {
            if let Ok(header_name) = HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(header_value) = HeaderValue::from_str(&value) {
                    self.0.insert(header_name, header_value);
                }
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}
