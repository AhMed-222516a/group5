"""
deployment/config.py
====================
Single source of truth for the Intelligent Support Ticket Classification
RAG system deployment configuration.

All future deployment modules (model_loader.py, retrieval.py,
rag_pipeline.py, app.py) should import exclusively from this file.

Usage:
    from deployment.config import (
        PATHS, EMBEDDING_CONFIG, GENERATION_CONFIG,
        RETRIEVAL_CONFIG, RUNTIME_CONFIG, FASTAPI_CONFIG,
        AZURE_CONFIG, OUTPUT_CONFIG, get_device,
        ensure_directories, validate_paths, print_configuration,
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-level logger.

    Args:
        name:  Logger name (typically ``__name__``).
        level: Initial logging level; overridden at runtime via
               ``RUNTIME_CONFIG.logging_level`` after that object is created.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger: logging.Logger = _build_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Environment(str, Enum):
    """Deployment environment identifiers."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SimilarityMetric(str, Enum):
    """Supported vector-similarity metrics for FAISS."""

    COSINE = "cosine"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"


class FAISSMetric(int, Enum):
    """FAISS internal metric constants (mirrors ``faiss.METRIC_*``)."""

    INNER_PRODUCT = 0   # faiss.METRIC_INNER_PRODUCT – use for cosine after normalisation
    L2 = 1              # faiss.METRIC_L2


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Resolve the absolute project root directory at runtime.

    The project root is defined as the *parent* of the ``deployment/``
    package directory, derived from the location of this file so that
    the project remains portable across machines.

    Returns:
        Absolute :class:`~pathlib.Path` pointing to the project root.

    Raises:
        RuntimeError: If the resolved root does not exist.
    """
    # __file__ → .../Project/deployment/config.py
    # .parent   → .../Project/deployment/
    # .parent   → .../Project/
    root: Path = Path(__file__).resolve().parent.parent
    if not root.exists():
        raise RuntimeError(
            f"Resolved project root does not exist: {root}. "
            "Ensure the project folder structure is intact."
        )
    logger.debug("Project root resolved to: %s", root)
    return root


# ---------------------------------------------------------------------------
# Path configuration dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathConfig:
    """Immutable collection of all project-wide path constants.

    Paths are resolved relative to *project_root* so the project remains
    portable – no drive letters or hard-coded absolute paths are used.

    Attributes:
        project_root:        Absolute path to the project root.
        deployment_dir:      ``<root>/src/``
        data_dir:            ``<root>/data/``
        raw_data_dir:        ``<root>/data/raw/``
        graphs_dir:          ``<root>/reports/figures/``
        model_dev_dir:       ``<root>/reports/``
        outputs_dir:         ``<root>/reports/figures/``
        cache_dir:           ``<root>/src/cache/``
        logs_dir:            ``<root>/src/logs/``
        temp_dir:            ``<root>/src/temp/``
        models_dir:          ``<root>/data/models/``
        faiss_dir:           ``<root>/data/models/faiss/``
        dataset_filename:    CSV filename for the preprocessed dataset.
        dataset_path:        Full path to the preprocessed CSV.
    """

    project_root: Path
    deployment_dir: Path
    data_dir: Path
    raw_data_dir: Path
    graphs_dir: Path
    model_dev_dir: Path
    outputs_dir: Path
    cache_dir: Path
    logs_dir: Path
    temp_dir: Path
    models_dir: Path
    faiss_dir: Path
    dataset_filename: str
    dataset_path: Path

    @classmethod
    def build(cls) -> "PathConfig":
        """Construct a :class:`PathConfig` from the runtime project root.

        Returns:
            A fully populated, immutable :class:`PathConfig`.
        """
        root = get_project_root()
        deployment = root / "src"
        data = root / "data"
        raw = data / "raw"
        graphs = root / "reports" / "figures"
        model_dev = root / "reports"
        outputs = root / "reports" / "figures"
        cache = deployment / "cache"
        logs = deployment / "logs"
        temp = deployment / "temp"
        models = root / "data" / "models"
        faiss = models / "faiss"
        filename = "support_tickets_preprocessed.csv"
        dataset = raw / filename

        return cls(
            project_root=root,
            deployment_dir=deployment,
            data_dir=data,
            raw_data_dir=raw,
            graphs_dir=graphs,
            model_dev_dir=model_dev,
            outputs_dir=outputs,
            cache_dir=cache,
            logs_dir=logs,
            temp_dir=temp,
            models_dir=models,
            faiss_dir=faiss,
            dataset_filename=filename,
            dataset_path=dataset,
        )


# ---------------------------------------------------------------------------
# Embedding model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the sentence-transformer embedding model.

    Attributes:
        model_name:        HuggingFace model identifier.
        batch_size:        Number of sentences encoded per forward pass.
        embedding_dim:     Dimensionality of the output embedding vectors.
        normalize:         Whether to L2-normalise embeddings before indexing.
        cache_dir:         Directory used to cache downloaded model weights.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    embedding_dim: int = 384
    normalize: bool = True
    cache_dir: Optional[Path] = None  # resolved to PATHS.cache_dir at runtime


# ---------------------------------------------------------------------------
# Generation model configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for the seq2seq text-generation model.

    Attributes:
        primary_model:     Primary HuggingFace model identifier.
        fallback_model:    Fallback model used when the primary fails to load.
        max_new_tokens:    Maximum number of tokens the model may generate.
        temperature:       Sampling temperature (lower → more deterministic).
        top_p:             Nucleus-sampling probability mass.
        top_k:             Top-K sampling cut-off.
        repetition_penalty: Penalty applied to repeated n-grams (>1 discourages).
        max_input_length:  Maximum token length accepted by the model's encoder.
    """

    primary_model: str = "google/flan-t5-base"
    fallback_model: str = "google/flan-t5-small"
    max_new_tokens: int = 128
    temperature: float = 0.3
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.1
    max_input_length: int = 512


# ---------------------------------------------------------------------------
# Retrieval / FAISS configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    """Configuration for FAISS-based dense retrieval.

    Attributes:
        index_filename:        File name of the persisted FAISS index.
        similarity_metric:     Human-readable similarity metric name.
        faiss_metric:          Corresponding FAISS metric constant.
        top_k:                 Number of nearest neighbours to retrieve.
        min_similarity:        Minimum similarity score to accept a result.
        max_contexts:          Hard cap on contexts passed to the generator.
    """

    index_filename: str = "support_tickets.index"
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    faiss_metric: FAISSMetric = FAISSMetric.INNER_PRODUCT
    top_k: int = 3
    min_similarity: float = 0.5
    max_contexts: int = 2


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime execution and system-level configuration.

    Attributes:
        random_seed:     Global seed for reproducibility.
        device:          Torch device string (``"cuda"`` or ``"cpu"``).
        cpu_threads:     Number of intra-op threads for CPU inference.
        environment:     Active deployment environment.
        logging_level:   Python logging level integer.
        default_encoding: Default text encoding for all I/O operations.
    """

    random_seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cpu_threads: int = os.cpu_count() or 4
    environment: Environment = Environment.DEVELOPMENT
    logging_level: int = logging.INFO
    default_encoding: str = "utf-8"


# ---------------------------------------------------------------------------
# FastAPI configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FastAPIConfig:
    """Configuration for the FastAPI application server.

    Attributes:
        title:       OpenAPI document title.
        version:     API semantic version string.
        description: Short description shown in the auto-generated docs.
        host:        Bind address for Uvicorn.
        port:        Bind port for Uvicorn.
        debug:       Enable Uvicorn/FastAPI debug mode (disable in production).
    """

    title: str = "Intelligent Support Ticket Classification API"
    version: str = "1.0.0"
    description: str = (
        "RAG-powered REST API for classifying and resolving support tickets "
        "using dense retrieval over a FAISS index and a Flan-T5 generator."
    )
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


# ---------------------------------------------------------------------------
# Azure configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AzureConfig:
    """Azure cloud resource configuration.

    All values are loaded from environment variables so that no credentials
    are ever committed to source control.

    Attributes:
        workspace_name:       Azure ML Workspace name.
        subscription_id:      Azure Subscription UUID.
        resource_group:       Azure Resource Group name.
        search_endpoint:      Azure AI Search service endpoint URL.
        search_index:         Azure AI Search index name.
        storage_account:      Azure Storage Account name.
        container_registry:   Azure Container Registry login server.
        key_vault:            Azure Key Vault name.
    """

    workspace_name: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_ML_WORKSPACE")
    )
    subscription_id: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID")
    )
    resource_group: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_RESOURCE_GROUP")
    )
    search_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_SEARCH_ENDPOINT")
    )
    search_index: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_SEARCH_INDEX", "support-tickets-index")
    )
    storage_account: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_STORAGE_ACCOUNT")
    )
    container_registry: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_CONTAINER_REGISTRY")
    )
    key_vault: Optional[str] = field(
        default_factory=lambda: os.environ.get("AZURE_KEY_VAULT")
    )


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputConfig:
    """Configuration for all deployment output directories.

    Mirrors :class:`PathConfig` directory attributes so that output modules
    can depend solely on :class:`OutputConfig` without importing
    :class:`PathConfig` directly.

    Attributes:
        outputs_dir: Artefacts and model outputs.
        logs_dir:    Application log files.
        cache_dir:   Cached embeddings and model states.
        temp_dir:    Ephemeral working files.
    """

    outputs_dir: Path
    logs_dir: Path
    cache_dir: Path
    temp_dir: Path

    @classmethod
    def from_paths(cls, paths: PathConfig) -> "OutputConfig":
        """Build an :class:`OutputConfig` from a :class:`PathConfig`.

        Args:
            paths: Populated :class:`PathConfig` instance.

        Returns:
            :class:`OutputConfig` with directories pointing into
            *paths.deployment_dir*.
        """
        return cls(
            outputs_dir=paths.outputs_dir,
            logs_dir=paths.logs_dir,
            cache_dir=paths.cache_dir,
            temp_dir=paths.temp_dir,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Detect and return the optimal Torch compute device string.

    Prefers CUDA when available; falls back to CPU transparently.

    Returns:
        ``"cuda"`` if a CUDA-capable GPU is detected, otherwise ``"cpu"``.
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA device detected: %s – using GPU acceleration.", gpu_name)
    else:
        device = "cpu"
        logger.info("No CUDA device detected – falling back to CPU.")
    return device


def ensure_directories(paths: PathConfig) -> None:
    """Create all required deployment directories if they do not yet exist.

    Uses ``pathlib.Path.mkdir(parents=True, exist_ok=True)`` so that the
    operation is idempotent and safe to call on every import.

    Args:
        paths: Populated :class:`PathConfig` instance.

    Raises:
        OSError: If a directory cannot be created due to permission issues.
    """
    directories: list[Path] = [
        paths.outputs_dir,
        paths.cache_dir,
        paths.logs_dir,
        paths.temp_dir,
        paths.models_dir,
        paths.faiss_dir,
        paths.graphs_dir,
        paths.raw_data_dir,
    ]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug("Directory ensured: %s", directory)
        except OSError as exc:
            logger.error("Failed to create directory %s: %s", directory, exc)
            raise


def validate_paths(paths: PathConfig) -> None:
    """Validate that critical paths exist and raise informative errors otherwise.

    Checks:
        * Dataset CSV file exists.
        * Project root directory exists.
        * Deployment directory exists.

    Args:
        paths: Populated :class:`PathConfig` instance.

    Raises:
        FileNotFoundError: If the dataset CSV is missing.
        NotADirectoryError: If the project root or deployment directory is
            not a valid directory.
    """
    if not paths.dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at expected path: {paths.dataset_path}\n"
            "Ensure 'support_tickets_preprocessed.csv' exists inside "
            f"'{paths.raw_data_dir}' before running the application."
        )
    logger.debug("Dataset validated: %s", paths.dataset_path)

    if not paths.project_root.is_dir():
        raise NotADirectoryError(
            f"Project root is not a directory: {paths.project_root}"
        )

    if not paths.deployment_dir.is_dir():
        raise NotADirectoryError(
            f"Deployment directory is not a directory: {paths.deployment_dir}"
        )

    logger.info("All critical paths validated successfully.")


def print_configuration(
    paths: PathConfig,
    embedding: EmbeddingConfig,
    generation: GenerationConfig,
    retrieval: RetrievalConfig,
    runtime: RuntimeConfig,
    fastapi: FastAPIConfig,
    azure: AzureConfig,
) -> None:
    """Print a structured summary of all active configuration objects.

    Intended for start-up diagnostics and CI validation runs.

    Args:
        paths:      Active :class:`PathConfig`.
        embedding:  Active :class:`EmbeddingConfig`.
        generation: Active :class:`GenerationConfig`.
        retrieval:  Active :class:`RetrievalConfig`.
        runtime:    Active :class:`RuntimeConfig`.
        fastapi:    Active :class:`FastAPIConfig`.
        azure:      Active :class:`AzureConfig`.
    """
    separator = "=" * 72

    logger.info(separator)
    logger.info("RAG DEPLOYMENT CONFIGURATION SUMMARY")
    logger.info(separator)

    logger.info("[PATHS]")
    logger.info("  project_root   : %s", paths.project_root)
    logger.info("  deployment_dir : %s", paths.deployment_dir)
    logger.info("  data_dir       : %s", paths.data_dir)
    logger.info("  raw_data_dir   : %s", paths.raw_data_dir)
    logger.info("  dataset_path   : %s", paths.dataset_path)
    logger.info("  models_dir     : %s", paths.models_dir)
    logger.info("  faiss_dir      : %s", paths.faiss_dir)
    logger.info("  cache_dir      : %s", paths.cache_dir)
    logger.info("  logs_dir       : %s", paths.logs_dir)
    logger.info("  temp_dir       : %s", paths.temp_dir)
    logger.info("  outputs_dir    : %s", paths.outputs_dir)

    logger.info("[EMBEDDING]")
    logger.info("  model          : %s", embedding.model_name)
    logger.info("  batch_size     : %d", embedding.batch_size)
    logger.info("  dim            : %d", embedding.embedding_dim)
    logger.info("  normalize      : %s", embedding.normalize)

    logger.info("[GENERATION]")
    logger.info("  primary        : %s", generation.primary_model)
    logger.info("  fallback       : %s", generation.fallback_model)
    logger.info("  max_new_tokens : %d", generation.max_new_tokens)
    logger.info("  temperature    : %.2f", generation.temperature)
    logger.info("  top_p          : %.2f", generation.top_p)
    logger.info("  top_k          : %d", generation.top_k)

    logger.info("[RETRIEVAL]")
    logger.info("  index_filename : %s", retrieval.index_filename)
    logger.info("  similarity     : %s", retrieval.similarity_metric.value)
    logger.info("  top_k          : %d", retrieval.top_k)
    logger.info("  min_similarity : %.2f", retrieval.min_similarity)
    logger.info("  max_contexts   : %d", retrieval.max_contexts)

    logger.info("[RUNTIME]")
    logger.info("  device         : %s", runtime.device)
    logger.info("  cpu_threads    : %d", runtime.cpu_threads)
    logger.info("  environment    : %s", runtime.environment.value)
    logger.info("  random_seed    : %d", runtime.random_seed)

    logger.info("[FASTAPI]")
    logger.info("  host           : %s", fastapi.host)
    logger.info("  port           : %d", fastapi.port)
    logger.info("  debug          : %s", fastapi.debug)
    logger.info("  version        : %s", fastapi.version)

    logger.info("[AZURE]")
    logger.info("  workspace      : %s", azure.workspace_name or "<not set>")
    logger.info("  subscription   : %s", azure.subscription_id or "<not set>")
    logger.info("  resource_group : %s", azure.resource_group or "<not set>")
    logger.info("  search_endpoint: %s", azure.search_endpoint or "<not set>")
    logger.info("  search_index   : %s", azure.search_index or "<not set>")
    logger.info("  storage_account: %s", azure.storage_account or "<not set>")
    logger.info("  key_vault      : %s", azure.key_vault or "<not set>")

    logger.info(separator)


# ---------------------------------------------------------------------------
# Module-level singleton initialisation
# ---------------------------------------------------------------------------

# Build paths first – every other config object may depend on them.
PATHS: PathConfig = PathConfig.build()

# Embedding / retrieval / generation / runtime singletons.
EMBEDDING_CONFIG: EmbeddingConfig = EmbeddingConfig(
    cache_dir=PATHS.cache_dir,
)

GENERATION_CONFIG: GenerationConfig = GenerationConfig()

RETRIEVAL_CONFIG: RetrievalConfig = RetrievalConfig(
    index_filename="support_tickets.index",
)

RUNTIME_CONFIG: RuntimeConfig = RuntimeConfig(
    device=get_device(),
)

# Update logger level to match RuntimeConfig after it is constructed.
logger.setLevel(RUNTIME_CONFIG.logging_level)

FASTAPI_CONFIG: FastAPIConfig = FastAPIConfig()

AZURE_CONFIG: AzureConfig = AzureConfig()

OUTPUT_CONFIG: OutputConfig = OutputConfig.from_paths(PATHS)

# Derive the FAISS index path from the resolved directories.
FAISS_INDEX_PATH: Path = PATHS.faiss_dir / RETRIEVAL_CONFIG.index_filename

# ---------------------------------------------------------------------------
# Bootstrap: directory creation and path validation on import.
# ---------------------------------------------------------------------------

ensure_directories(PATHS)
validate_paths(PATHS)

logger.info(
    "Configuration loaded successfully – environment=%s, device=%s.",
    RUNTIME_CONFIG.environment.value,
    RUNTIME_CONFIG.device,
)
