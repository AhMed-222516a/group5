"""
deployment/app.py
=================
FastAPI application that exposes the Intelligent Support Ticket Classification
RAG system as a production-ready REST API.

This module is ONLY responsible for:
    * Defining the FastAPI application and its lifecycle.
    * Declaring Pydantic request / response models.
    * Routing HTTP requests to ``deployment.rag_pipeline.generate_response``.
    * Global error handling and request-logging middleware.

It does NOT implement retrieval, prompt engineering, model loading, embedding
generation, FAISS construction, dataset preprocessing, Azure SDK calls, or
authentication.  Those responsibilities belong to their respective modules.

Run locally::

    uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload

Or via the ``__main__`` block::

    python -m deployment.app
"""

from __future__ import annotations

import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from deployment.config import (
    EMBEDDING_CONFIG,
    FASTAPI_CONFIG,
    GENERATION_CONFIG,
    RUNTIME_CONFIG,
    logger,
)
from deployment.rag_pipeline import RAGResponse, generate_response, warm_up
from deployment.utils import format_latency

# ---------------------------------------------------------------------------
# Application start time (used to compute uptime in /health)
# ---------------------------------------------------------------------------

_APP_START_TIME: float = time.monotonic()


# ===========================================================================
# Pydantic models
# ===========================================================================

class PredictionRequest(BaseModel):
    """Request body for the ``POST /predict`` endpoint.

    Attributes:
        query: Free-text customer support question.  Must be non-empty.
        top_k: Optional override for the number of retrieved tickets.
        min_similarity: Optional override for the minimum similarity threshold.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Free-text customer support question.",
        examples=["I cannot login to my account."],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of nearest-neighbour tickets to retrieve (1–20).",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold for retrieved tickets (0–1).",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        """Reject queries that contain only whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be blank or whitespace-only.")
        return stripped


class RetrievalResultResponse(BaseModel):
    """Serialisable representation of a single retrieved support ticket.

    Mirrors :class:`~deployment.retrieval.RetrievalResult` but uses Pydantic
    for automatic JSON serialisation and OpenAPI schema generation.
    """

    rank: int = Field(..., description="1-based rank among retrieved results.")
    score: float = Field(..., description="Cosine similarity score (0–1).")
    ticket_id: str = Field(..., description="Unique ticket identifier.")
    category: str = Field(..., description="Ticket category label.")
    priority: str = Field(..., description="Ticket priority label.")
    subject: str = Field(..., description="Short ticket subject line.")
    resolution_note: str = Field(..., description="Resolution note from the ticket.")
    knowledge_article: str = Field(..., description="Associated knowledge-base article.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata from the ticket row.",
    )


class PredictionResponse(BaseModel):
    """Response body for the ``POST /predict`` endpoint.

    Attributes:
        request_id: Unique identifier for this prediction request (UUID4).
        query: The sanitised query string that was processed.
        answer: The model-generated answer grounded in the retrieved context.
        retrieved_context: Plain-text concatenation of the context forwarded
            to the generator.
        retrieval_results: Ranked list of retrieved support tickets.
        retrieval_time: Seconds spent in the retrieval stage.
        generation_time: Seconds spent in the generation stage.
        total_latency: End-to-end seconds for the full pipeline call.
        generation_model: Identifier of the generation model used.
        timestamp: ISO-8601 UTC timestamp of the response.
        metadata: Additional pipeline diagnostics (device, environment, etc.).
    """

    request_id: str = Field(..., description="Unique request identifier (UUID4).")
    query: str = Field(..., description="Sanitised query string that was processed.")
    answer: str = Field(..., description="Model-generated answer.")
    retrieved_context: str = Field(
        ..., description="Plain-text context forwarded to the generator."
    )
    retrieval_results: List[RetrievalResultResponse] = Field(
        ..., description="Ranked retrieved support tickets."
    )
    retrieval_time: float = Field(..., description="Retrieval stage latency in seconds.")
    generation_time: float = Field(..., description="Generation stage latency in seconds.")
    total_latency: float = Field(..., description="End-to-end pipeline latency in seconds.")
    generation_model: str = Field(..., description="HuggingFace model identifier used.")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the response.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pipeline diagnostics.",
    )


class HealthResponse(BaseModel):
    """Response body for the ``GET /health`` endpoint."""

    status: str
    device: str
    generation_model: str
    embedding_model: str
    faiss_loaded: bool
    api_version: str
    uptime_seconds: float
    timestamp: str


class RootResponse(BaseModel):
    """Response body for the ``GET /`` endpoint."""

    status: str
    service: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Standardised error response body returned on all error conditions."""

    error: str
    detail: str
    request_id: str
    timestamp: str


# ===========================================================================
# Lifespan (startup / shutdown)
# ===========================================================================

@asynccontextmanager
async def _lifespan(application: FastAPI):  # type: ignore[type-arg]
    """FastAPI lifespan context: warm up resources on startup, log on shutdown.

    Loads the FAISS index, metadata records, and all model singletons before
    the application begins serving traffic.  Uses
    :func:`~deployment.rag_pipeline.warm_up` so that no models are
    instantiated inside this module.

    Args:
        application: The FastAPI application instance (unused directly).

    Yields:
        Control to the ASGI runtime while the application is live.
    """
    logger.info("=" * 60)
    logger.info("Starting %s  v%s", FASTAPI_CONFIG.title, FASTAPI_CONFIG.version)
    logger.info("Environment : %s", RUNTIME_CONFIG.environment.value)
    logger.info("Device      : %s", RUNTIME_CONFIG.device)
    logger.info("=" * 60)

    try:
        warm_up()
        logger.info("All resources loaded — application is ready to serve traffic.")
    except Exception as exc:
        logger.exception("FATAL: resource warm-up failed: %s", exc)
        raise RuntimeError(
            f"Application failed to start: resource warm-up error — {exc}"
        ) from exc

    yield  # ── application is live ──────────────────────────────────────────

    logger.info("Shutting down %s …", FASTAPI_CONFIG.title)
    logger.info("Shutdown complete.")


# ===========================================================================
# FastAPI application
# ===========================================================================

app = FastAPI(
    title=FASTAPI_CONFIG.title,
    description=FASTAPI_CONFIG.description,
    version=FASTAPI_CONFIG.version,
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "health",
            "description": "Health and liveness probes.",
        },
        {
            "name": "prediction",
            "description": "RAG-powered support ticket classification and response generation.",
        },
    ],
)

# ---------------------------------------------------------------------------
# CORS middleware (permissive default — restrict in production via config)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ===========================================================================
# Request-logging middleware
# ===========================================================================

@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next: Any) -> Any:
    """Log every incoming request and its outcome with latency.

    Attaches a ``X-Request-ID`` header to every response for traceability.

    Args:
        request: Incoming HTTP request.
        call_next: ASGI callable for the next middleware / route handler.

    Returns:
        The HTTP response produced by the downstream handler.
    """
    request_id: str = str(uuid.uuid4())
    start = time.perf_counter()

    logger.info(
        "→ %s %s  [request_id=%s]",
        request.method,
        request.url.path,
        request_id,
    )

    response = await call_next(request)

    elapsed = time.perf_counter() - start
    logger.info(
        "← %s %s  status=%d  latency=%s  [request_id=%s]",
        request.method,
        request.url.path,
        response.status_code,
        format_latency(elapsed),
        request_id,
    )

    response.headers["X-Request-ID"] = request_id
    return response


# ===========================================================================
# Global exception handlers
# ===========================================================================

@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a structured 422 response for Pydantic validation failures.

    Args:
        request: Incoming request that triggered the error.
        exc: Pydantic validation exception.

    Returns:
        JSON response with HTTP 422 status.
    """
    request_id = str(uuid.uuid4())
    logger.warning(
        "Validation error on %s %s [request_id=%s]: %s",
        request.method,
        request.url.path,
        request_id,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(
            ErrorResponse(
                error="ValidationError",
                detail=str(exc.errors()),
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ),
    )


@app.exception_handler(HTTPException)
async def _http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Return a structured JSON response for all ``HTTPException`` instances.

    Args:
        request: Incoming request.
        exc: FastAPI HTTP exception.

    Returns:
        JSON response with the exception's status code.
    """
    request_id = str(uuid.uuid4())
    logger.warning(
        "HTTP %d on %s %s [request_id=%s]: %s",
        exc.status_code,
        request.method,
        request.url.path,
        request_id,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=jsonable_encoder(
            ErrorResponse(
                error="HTTPException",
                detail=str(exc.detail),
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler that prevents internal stack traces from leaking.

    Args:
        request: Incoming request.
        exc: Unhandled exception.

    Returns:
        JSON response with HTTP 500 status.
    """
    request_id = str(uuid.uuid4())
    logger.exception(
        "Unhandled exception on %s %s [request_id=%s]: %s",
        request.method,
        request.url.path,
        request_id,
        exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(
            ErrorResponse(
                error=type(exc).__name__,
                detail="An unexpected internal error occurred. Please try again later.",
                request_id=request_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        ),
    )


# ===========================================================================
# Helper: convert RAGResponse → PredictionResponse
# ===========================================================================

def _build_prediction_response(
    rag_response: RAGResponse,
    request_id: str,
) -> PredictionResponse:
    """Map a :class:`~deployment.rag_pipeline.RAGResponse` to a Pydantic model.

    Args:
        rag_response: Pipeline output returned by ``generate_response()``.
        request_id: UUID string assigned to this HTTP request.

    Returns:
        Fully populated :class:`PredictionResponse` ready for serialisation.
    """
    retrieval_results_out: List[RetrievalResultResponse] = [
        RetrievalResultResponse(
            rank=r.rank,
            score=r.score,
            ticket_id=r.ticket_id,
            category=r.category,
            priority=r.priority,
            subject=r.subject,
            resolution_note=r.resolution_note,
            knowledge_article=r.knowledge_article,
            metadata=r.metadata,
        )
        for r in rag_response.retrieval_results
    ]

    return PredictionResponse(
        request_id=request_id,
        query=rag_response.user_query,
        answer=rag_response.generated_response,
        retrieved_context=rag_response.retrieved_context,
        retrieval_results=retrieval_results_out,
        retrieval_time=rag_response.retrieval_time,
        generation_time=rag_response.generation_time,
        total_latency=rag_response.total_latency,
        generation_model=rag_response.generation_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        metadata=rag_response.metadata,
    )


# ===========================================================================
# Routes
# ===========================================================================

@app.get(
    "/",
    response_model=RootResponse,
    summary="Root liveness check",
    description="Returns the service name, version, and a UTC timestamp.",
    tags=["health"],
)
async def root() -> RootResponse:
    """Root liveness probe.

    Returns:
        :class:`RootResponse` with service identity and timestamp.
    """
    return RootResponse(
        status="ok",
        service=FASTAPI_CONFIG.title,
        version=FASTAPI_CONFIG.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Detailed health check",
    description=(
        "Returns the operational status of every pipeline component. "
        "Intended for use by Azure health probes and monitoring systems."
    ),
    tags=["health"],
)
async def health() -> HealthResponse:
    """Detailed health and readiness probe.

    Reports device, model identifiers, FAISS index availability, API version,
    and uptime.  Always returns HTTP 200 while the service is running;
    Azure health probes interpret a non-200 status as unhealthy.

    Returns:
        :class:`HealthResponse` populated from runtime configuration.
    """
    from deployment.rag_pipeline import _faiss_index  # local import to avoid circularity

    faiss_loaded: bool = _faiss_index is not None
    uptime: float = time.monotonic() - _APP_START_TIME

    logger.debug("Health check — faiss_loaded=%s, uptime=%.2fs", faiss_loaded, uptime)

    return HealthResponse(
        status="ok",
        device=RUNTIME_CONFIG.device,
        generation_model=GENERATION_CONFIG.primary_model,
        embedding_model=EMBEDDING_CONFIG.model_name,
        faiss_loaded=faiss_loaded,
        api_version=FASTAPI_CONFIG.version,
        uptime_seconds=round(uptime, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify and resolve a support ticket",
    description=(
        "Accepts a free-text customer query, retrieves semantically similar "
        "resolved support tickets from the FAISS index, and generates a "
        "grounded, professional response using the Flan-T5 generation model."
    ),
    tags=["prediction"],
    responses={
        200: {"description": "Successful prediction."},
        422: {"description": "Request validation error.", "model": ErrorResponse},
        500: {"description": "Internal pipeline error.", "model": ErrorResponse},
    },
)
async def predict(request_body: PredictionRequest, request: Request) -> PredictionResponse:
    """Execute the full RAG pipeline for a customer support query.

    Delegates entirely to :func:`~deployment.rag_pipeline.generate_response`.
    No retrieval, model loading, or prompt engineering is performed here.

    Args:
        request_body: Validated :class:`PredictionRequest` from the caller.
        request: Raw FastAPI request (used for the attached ``X-Request-ID``).

    Returns:
        :class:`PredictionResponse` containing the generated answer, retrieved
        context, and latency metrics.

    Raises:
        HTTPException: 400 if the query is rejected by the pipeline validator.
        HTTPException: 500 if the pipeline raises a ``RuntimeError``.
    """
    request_id: str = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    logger.info(
        "Prediction request [request_id=%s] — query: %r",
        request_id,
        request_body.query[:120],
    )

    try:
        rag_response: RAGResponse = generate_response(
            request_body.query,
            top_k=request_body.top_k,
            min_similarity=request_body.min_similarity,
        )
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Invalid prediction request [request_id=%s]: %s", request_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception(
            "Pipeline runtime error [request_id=%s]: %s", request_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The RAG pipeline encountered an internal error. Please try again.",
        ) from exc

    response = _build_prediction_response(rag_response, request_id)

    logger.info(
        "Prediction complete [request_id=%s] — retrieval: %s, generation: %s, total: %s",
        request_id,
        format_latency(response.retrieval_time),
        format_latency(response.generation_time),
        format_latency(response.total_latency),
    )

    return response


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run(
        "deployment.app:app",
        host=FASTAPI_CONFIG.host,
        port=FASTAPI_CONFIG.port,
        reload=FASTAPI_CONFIG.debug,
        log_level="info",
    )
