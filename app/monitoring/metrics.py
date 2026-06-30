"""Prometheus metrics and OpenTelemetry tracing for the platform.

Exposes:
- API request latency histograms
- Token usage counters
- Queue depth gauges
- AI provider response time histograms
- Cache hit/miss counters
- Active workflow gauges

OpenTelemetry traces are exported to the configured backend (Jaeger, OTLP).
"""

from __future__ import annotations

from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================


def setup_prometheus() -> Any:
    """Initialize Prometheus metrics registry.

    Returns:
        CollectorRegistry: Prometheus registry with all registered metrics.
    """
    try:
        from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

        registry = CollectorRegistry()

        # HTTP request metrics
        http_request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint", "status_code"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=registry,
        )

        http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=registry,
        )

        # AI token usage
        ai_tokens_used = Counter(
            "ai_tokens_used_total",
            "Total AI tokens consumed",
            ["provider", "model", "operation"],
            registry=registry,
        )

        # AI response time
        ai_response_time = Histogram(
            "ai_response_time_seconds",
            "AI provider response time",
            ["provider", "operation"],
            registry=registry,
        )

        # RAG metrics
        rag_retrieval_duration = Histogram(
            "rag_retrieval_duration_seconds",
            "RAG retrieval latency",
            ["collection"],
            registry=registry,
        )

        rag_chunks_retrieved = Histogram(
            "rag_chunks_retrieved",
            "Number of chunks retrieved per query",
            registry=registry,
        )

        # Cache metrics
        cache_hits_total = Counter(
            "cache_hits_total",
            "Cache hit count",
            ["cache_type"],
            registry=registry,
        )

        cache_misses_total = Counter(
            "cache_misses_total",
            "Cache miss count",
            ["cache_type"],
            registry=registry,
        )

        # Celery task metrics
        celery_tasks_total = Counter(
            "celery_tasks_total",
            "Total Celery tasks",
            ["task_name", "status"],
            registry=registry,
        )

        celery_task_duration = Histogram(
            "celery_task_duration_seconds",
            "Celery task execution time",
            ["task_name"],
            registry=registry,
        )

        # Active workflows
        active_workflows = Gauge(
            "active_workflows",
            "Number of currently running workflows",
            registry=registry,
        )

        logger.info("Prometheus metrics initialized")
        return registry

    except ImportError:
        logger.warning("prometheus-client not installed; metrics disabled")
        return None


# =============================================================================
# OpenTelemetry Tracing
# =============================================================================


def setup_tracing(
    service_name: str = "ai-automation-platform",
    otlp_endpoint: str | None = None,
) -> None:
    """Configure OpenTelemetry distributed tracing.

    Args:
        service_name: Service identifier for trace metadata.
        otlp_endpoint: OTLP gRPC or HTTP endpoint (e.g., Jaeger, Tempo).
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": service_name})
        tracer_provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info("OTLP trace exporter configured", endpoint=otlp_endpoint)
            except ImportError:
                logger.warning("opentelemetry-exporter-otlp not installed")

        trace.set_tracer_provider(tracer_provider)
        logger.info("OpenTelemetry tracing initialized", service=service_name)

    except ImportError:
        logger.warning("opentelemetry not installed; tracing disabled")


def get_tracer(name: str) -> Any:
    """Get an OpenTelemetry tracer.

    Args:
        name: Tracer name (usually __name__).

    Returns:
        Tracer: OpenTelemetry tracer instance (or no-op if not configured).
    """
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return None


# =============================================================================
# FastAPI Prometheus Instrumentation Middleware
# =============================================================================


def setup_fastapi_metrics(app: Any) -> None:
    """Add Prometheus instrumentation to a FastAPI application.

    Args:
        app: FastAPI application instance.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import]

        Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/health", "/metrics"],
        ).instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])

        logger.info("FastAPI Prometheus instrumentation enabled")

    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not installed; /metrics disabled")
