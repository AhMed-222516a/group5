"""
deployment/retrieval.py
=======================
Semantic retrieval module for the Intelligent Support Ticket Classification
RAG system.

Responsibilities
----------------
* Dataset loading and validation.
* Corpus embedding generation via the shared embedding model.
* FAISS index construction, persistence, and loading.
* Semantic search and Top-K retrieval with similarity scoring.

This module NEVER generates responses, builds prompts, loads transformer
models directly, creates APIs, or performs Azure deployment operations.

All configuration is imported exclusively from ``deployment.config``.
All models are obtained exclusively from ``deployment.model_loader``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from src.config import (
    EMBEDDING_CONFIG,
    FAISS_INDEX_PATH,
    PATHS,
    RETRIEVAL_CONFIG,
    SimilarityMetric,
    logger,
)
from src.model_loader import get_embedding_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METADATA_SUFFIX: str = ".metadata.json"

# Columns read from the preprocessed dataset and surfaced in retrieval results.
_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "ticket_id",
    "category",
    "priority",
    "subject",
    "clean_text",
    "resolution_note",
    "knowledge_article",
    "rag_text",
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieved support ticket with its similarity score.

    Attributes:
        rank:               1-based rank among retrieved results.
        score:              Cosine / inner-product similarity score (0–1).
        ticket_id:          Unique ticket identifier from the dataset.
        category:           Ticket category label (e.g. ``"billing"``).
        priority:           Ticket priority label (e.g. ``"high"``).
        subject:            Short ticket subject line.
        clean_text:         Pre-processed ticket body text.
        resolution_note:    Resolution note from the original ticket.
        knowledge_article:  Knowledge-base article associated with the ticket.
        metadata:           Additional key-value pairs from the row (may be empty).
    """

    rank: int
    score: float
    ticket_id: str
    category: str
    priority: str
    subject: str
    clean_text: str
    resolution_note: str
    knowledge_article: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load and validate the preprocessed support ticket dataset.

    Reads the CSV file whose path is provided by :data:`PATHS.dataset_path`.
    Validates that all columns required for retrieval are present.

    Returns
    -------
    pd.DataFrame
        The loaded dataset with at least the columns defined in
        ``_REQUIRED_COLUMNS``.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the configured path.
    ValueError
        If one or more required columns are absent from the loaded DataFrame.
    RuntimeError
        If the CSV cannot be parsed or any other I/O error occurs.
    """
    dataset_path: Path = PATHS.dataset_path

    logger.info("Loading dataset from: %s", dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}. "
            "Ensure 'support_tickets_preprocessed.csv' exists inside "
            f"'{PATHS.raw_data_dir}'."
        )

    try:
        df: pd.DataFrame = pd.read_csv(
            dataset_path,
            encoding=PATHS.project_root and "utf-8",  # always utf-8
            low_memory=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse dataset CSV at '{dataset_path}': {exc}"
        ) from exc

    logger.info(
        "Dataset loaded: %d rows × %d columns.", len(df), len(df.columns)
    )

    #--------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Build rag_text dynamically if it does not already exist
    # ------------------------------------------------------------------
    if "rag_text" not in df.columns:

        logger.info("Column 'rag_text' not found. Building it dynamically...")

        subject = df["subject"].fillna("").astype(str)

        clean_text = df["clean_text"].fillna("").astype(str)

        knowledge = df["knowledge_article"].fillna("").astype(str)

        df["rag_text"] = (
            subject + " " +
            clean_text + " " +
            knowledge
        ).str.strip()

        logger.info("Successfully created 'rag_text' column.")

    #--------------------------------------------------------------------------------------------

    missing: List[str] = [
        col for col in _REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Coerce text columns to str and fill NaNs to avoid downstream failures.
    text_cols = [
        "ticket_id", "category", "priority", "subject",
        "clean_text", "resolution_note", "knowledge_article", "rag_text",
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    logger.info("Dataset validated successfully.")
    return df


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate normalised sentence embeddings for a list of texts.

    Uses the singleton embedding model obtained from
    :func:`~deployment.model_loader.get_embedding_model`.  Normalisation
    is applied when :attr:`~deployment.config.EmbeddingConfig.normalize` is
    ``True`` (the default), which is required for cosine similarity via
    FAISS inner-product search.

    Parameters
    ----------
    texts:
        List of raw text strings to embed.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(len(texts), embedding_dim)``.

    Raises
    ------
    ValueError
        If ``texts`` is empty.
    RuntimeError
        If the embedding model fails to encode the input.
    """
    if not texts:
        raise ValueError("Cannot generate embeddings for an empty text list.")

    model = get_embedding_model()
    batch_size: int = EMBEDDING_CONFIG.batch_size
    normalize: bool = EMBEDDING_CONFIG.normalize

    logger.info(
        "Generating embeddings for %d texts (batch_size=%d, normalize=%s).",
        len(texts),
        batch_size,
        normalize,
    )

    t0 = time.perf_counter()
    try:
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Embedding generation failed for {len(texts)} texts: {exc}"
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        "Embeddings generated in %.2fs — shape: %s, dtype: %s, memory: %.1f MB.",
        elapsed,
        embeddings.shape,
        embeddings.dtype,
        embeddings.nbytes / 1e6,
    )
    return embeddings


# ---------------------------------------------------------------------------
# FAISS index path helpers
# ---------------------------------------------------------------------------

def _index_path() -> Path:
    """Return the configured path for the persisted FAISS index file."""
    return FAISS_INDEX_PATH


def _metadata_path() -> Path:
    """Return the path for the JSON metadata file that accompanies the index."""
    return _index_path().with_suffix("").with_suffix("") / (
        _index_path().name + _METADATA_SUFFIX
    )


def _resolve_metadata_path() -> Path:
    """Return the metadata JSON path as a sibling of the FAISS index file."""
    idx_path = _index_path()
    return idx_path.parent / (idx_path.name + _METADATA_SUFFIX)


# ---------------------------------------------------------------------------
# FAISS index existence check
# ---------------------------------------------------------------------------

def index_exists() -> bool:
    """Check whether a persisted FAISS index exists on disk.

    Returns
    -------
    bool
        ``True`` if both the index file and its companion metadata file are
        present; ``False`` otherwise.
    """
    idx_exists: bool = _index_path().is_file()
    meta_exists: bool = _resolve_metadata_path().is_file()
    return idx_exists and meta_exists


# ---------------------------------------------------------------------------
# Save index
# ---------------------------------------------------------------------------

def save_index(
    index: faiss.Index,
    df: pd.DataFrame,
) -> None:
    """Persist a FAISS index and its companion metadata to disk.

    The FAISS binary is written to :data:`FAISS_INDEX_PATH`.  A companion
    JSON file containing the dataset rows (as a list of dicts) is written
    alongside the index so that retrieval results can be reconstructed
    without keeping the full DataFrame in memory.

    Parameters
    ----------
    index:
        A trained and populated FAISS index.
    df:
        The DataFrame whose rows were used to build the index.  Row order
        must match the vector insertion order.

    Raises
    ------
    OSError
        If the index or metadata cannot be written to disk.
    RuntimeError
        If FAISS raises an internal error during serialisation.
    """
    idx_path: Path = _index_path()
    meta_path: Path = _resolve_metadata_path()

    idx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving FAISS index to: %s", idx_path)
    try:
        faiss.write_index(index, str(idx_path))
    except Exception as exc:
        raise RuntimeError(
            f"FAISS failed to write index to '{idx_path}': {exc}"
        ) from exc
    logger.info("FAISS index saved (%d vectors).", index.ntotal)

    logger.info("Saving retrieval metadata to: %s", meta_path)
    metadata_records: List[Dict[str, Any]] = df[list(_REQUIRED_COLUMNS)].to_dict(
        orient="records"
    )
    try:
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata_records, fh, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise OSError(
            f"Could not write metadata file to '{meta_path}': {exc}"
        ) from exc
    logger.info("Retrieval metadata saved (%d records).", len(metadata_records))


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------

def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load a persisted FAISS index and its companion metadata from disk.

    Returns
    -------
    Tuple[faiss.Index, List[Dict[str, Any]]]
        A ``(index, metadata_records)`` pair where ``metadata_records`` is a
        list of dicts — one per indexed document — containing all fields in
        ``_REQUIRED_COLUMNS``.

    Raises
    ------
    FileNotFoundError
        If the FAISS index file or its companion metadata file does not exist.
    RuntimeError
        If FAISS cannot deserialise the index or the metadata JSON is corrupt.
    """
    idx_path: Path = _index_path()
    meta_path: Path = _resolve_metadata_path()

    if not idx_path.is_file():
        raise FileNotFoundError(
            f"FAISS index not found at: {idx_path}. "
            "Call build_index() to create it before loading."
        )
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Retrieval metadata not found at: {meta_path}. "
            "The index may be incomplete — rebuild with build_index()."
        )

    logger.info("Loading existing FAISS index from: %s", idx_path)
    try:
        index: faiss.Index = faiss.read_index(str(idx_path))
    except Exception as exc:
        raise RuntimeError(
            f"FAISS failed to load index from '{idx_path}': {exc}. "
            "The index file may be corrupted — delete it and rebuild."
        ) from exc
    logger.info(
        "FAISS index loaded: %d vectors, dim=%d.",
        index.ntotal,
        index.d,
    )

    logger.info("Loading retrieval metadata from: %s", meta_path)
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            metadata_records: List[Dict[str, Any]] = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not load retrieval metadata from '{meta_path}': {exc}. "
            "Delete the index and metadata files and rebuild with build_index()."
        ) from exc
    logger.info("Retrieval metadata loaded (%d records).", len(metadata_records))

    return index, metadata_records


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def build_index(
    df: Optional[pd.DataFrame] = None,
    *,
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Build (or reload) the FAISS index for the support ticket corpus.

    If a valid index already exists on disk and ``force_rebuild`` is
    ``False``, the existing index is loaded and returned immediately to avoid
    unnecessary recomputation.

    Otherwise, the function:

    1. Loads the dataset (if ``df`` is not provided).
    2. Generates corpus embeddings via :func:`generate_embeddings`.
    3. Constructs a FAISS flat index (``IndexFlatIP`` for cosine similarity,
       ``IndexFlatL2`` for L2 distance).
    4. Persists the index and metadata to disk via :func:`save_index`.

    Parameters
    ----------
    df:
        Optional pre-loaded DataFrame.  When ``None``, the dataset is loaded
        automatically from :data:`PATHS.dataset_path`.
    force_rebuild:
        When ``True``, always rebuild the index even if one already exists.

    Returns
    -------
    Tuple[faiss.Index, List[Dict[str, Any]]]
        A ``(index, metadata_records)`` pair ready for retrieval.

    Raises
    ------
    FileNotFoundError
        If the dataset is not found and ``df`` was not supplied.
    RuntimeError
        If embedding generation or FAISS index construction fails.
    """
    if not force_rebuild and index_exists():
        logger.info(
            "Existing FAISS index found at '%s'. Loading instead of rebuilding.",
            _index_path(),
        )
        return load_index()

    if force_rebuild:
        logger.info("force_rebuild=True — rebuilding FAISS index from scratch.")
    else:
        logger.info("No existing FAISS index found — building a new one.")

    # 1. Dataset
    if df is None:
        df = load_dataset()

    # 2. Embeddings
    logger.info("Building embeddings for %d corpus documents.", len(df))
    corpus_texts: List[str] = df["rag_text"].tolist()
    embeddings: np.ndarray = generate_embeddings(corpus_texts)

    # 3. FAISS index construction
    embedding_dim: int = embeddings.shape[1]
    similarity_metric: SimilarityMetric = RETRIEVAL_CONFIG.similarity_metric

    logger.info(
        "Building FAISS index — metric=%s, dim=%d, vectors=%d.",
        similarity_metric.value,
        embedding_dim,
        len(embeddings),
    )

    if similarity_metric in (SimilarityMetric.COSINE, SimilarityMetric.INNER_PRODUCT):
        # Embeddings are L2-normalised → inner product equals cosine similarity.
        index: faiss.Index = faiss.IndexFlatIP(embedding_dim)
    else:
        index = faiss.IndexFlatL2(embedding_dim)

    vectors: np.ndarray = embeddings.astype(np.float32)
    index.add(vectors)

    logger.info(
        "FAISS index built: type=%s, ntotal=%d.",
        type(index).__name__,
        index.ntotal,
    )

    # 4. Persist
    save_index(index, df)

    metadata_records: List[Dict[str, Any]] = df[list(_REQUIRED_COLUMNS)].to_dict(
        orient="records"
    )
    return index, metadata_records


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

def _encode_query(query: str) -> np.ndarray:
    """Encode a single query string into a normalised float32 vector.

    Parameters
    ----------
    query:
        Raw free-text query from the caller.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, embedding_dim)`` suitable for
        ``faiss.Index.search``.

    Raises
    ------
    ValueError
        If ``query`` is an empty string after stripping.
    RuntimeError
        If the embedding model fails to encode the query.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must be a non-empty string.")

    logger.info("Encoding query: %r", query[:120])
    model = get_embedding_model()

    try:
        q_vec: np.ndarray = model.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # always normalise query for cosine search
        ).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to encode query '{query[:80]}': {exc}"
        ) from exc

    return q_vec


# ---------------------------------------------------------------------------
# Core semantic search
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    index: faiss.Index,
    metadata_records: List[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> List[RetrievalResult]:
    """Search the FAISS index and return ranked retrieval results.

    Parameters
    ----------
    query:
        Free-text user query to retrieve similar tickets for.
    index:
        A populated FAISS index (returned by :func:`build_index` or
        :func:`load_index`).
    metadata_records:
        List of row dicts aligned with the FAISS index vectors.
    top_k:
        Number of nearest neighbours to retrieve.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.top_k`.
    min_similarity:
        Minimum similarity score threshold.  Results below this value are
        discarded.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.min_similarity`.

    Returns
    -------
    List[RetrievalResult]
        Ranked list of :class:`RetrievalResult` instances, ordered by
        descending similarity score.  May be empty if no result clears the
        ``min_similarity`` threshold.

    Raises
    ------
    ValueError
        If ``query`` is empty or ``top_k`` is not a positive integer.
    RuntimeError
        If the FAISS search operation fails.
    """
    effective_top_k: int = top_k if top_k is not None else RETRIEVAL_CONFIG.top_k
    effective_min_sim: float = (
        min_similarity if min_similarity is not None
        else RETRIEVAL_CONFIG.min_similarity
    )

    if effective_top_k < 1:
        raise ValueError(f"top_k must be a positive integer; got {effective_top_k}.")

    # Guard: cannot retrieve more vectors than are indexed.
    k_capped: int = min(effective_top_k, index.ntotal)

    # 1. Encode query
    q_vec: np.ndarray = _encode_query(query)

    # 2. FAISS search
    logger.info(
        "Searching FAISS index (ntotal=%d) for top-%d results.",
        index.ntotal,
        k_capped,
    )
    t0 = time.perf_counter()
    try:
        scores: np.ndarray
        indices: np.ndarray
        scores, indices = index.search(q_vec, k_capped)
    except Exception as exc:
        raise RuntimeError(
            f"FAISS search failed for query '{query[:80]}': {exc}"
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info("FAISS search completed in %.4fs.", elapsed)

    # 3. Build result list
    results: List[RetrievalResult] = []
    rank = 0
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            # FAISS returns -1 for missing neighbours (e.g. when ntotal < k).
            continue

        sim_score = float(score)
        if sim_score < effective_min_sim:
            logger.debug(
                "Skipping result at index %d: score %.4f < min_similarity %.4f.",
                idx,
                sim_score,
                effective_min_sim,
            )
            continue

        rank += 1
        row: Dict[str, Any] = metadata_records[idx]

        # Extract standard fields; fall back to empty string for robustness.
        extra_keys = set(row.keys()) - set(_REQUIRED_COLUMNS)
        extra_metadata = {k: row[k] for k in extra_keys}

        results.append(
            RetrievalResult(
                rank=rank,
                score=sim_score,
                ticket_id=str(row.get("ticket_id", "")),
                category=str(row.get("category", "")),
                priority=str(row.get("priority", "")),
                subject=str(row.get("subject", "")),
                clean_text=str(row.get("clean_text", "")),
                resolution_note=str(row.get("resolution_note", "")),
                knowledge_article=str(row.get("knowledge_article", "")),
                metadata=extra_metadata,
            )
        )

    logger.info(
        "Retrieval completed: %d result(s) returned for query %r.",
        len(results),
        query[:80],
    )
    return results


# ---------------------------------------------------------------------------
# High-level convenience wrappers
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str,
    index: faiss.Index,
    metadata_records: List[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> List[RetrievalResult]:
    """Retrieve the most semantically similar support tickets for a query.

    Convenience wrapper around :func:`semantic_search` that exposes the same
    interface expected by ``rag_pipeline.py``.

    Parameters
    ----------
    query:
        Free-text user query.
    index:
        Populated FAISS index.
    metadata_records:
        Metadata list aligned with the FAISS index.
    top_k:
        Number of results to return.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.top_k`.
    min_similarity:
        Minimum similarity threshold.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.min_similarity`.

    Returns
    -------
    List[RetrievalResult]
        Ranked retrieval results, capped at
        :attr:`~deployment.config.RetrievalConfig.max_contexts` entries.

    Raises
    ------
    ValueError
        If ``query`` is empty.
    RuntimeError
        Propagated from :func:`semantic_search` on FAISS failure.
    """
    results: List[RetrievalResult] = semantic_search(
        query,
        index,
        metadata_records,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    # Enforce hard cap on contexts forwarded to the generator.
    max_contexts: int = RETRIEVAL_CONFIG.max_contexts
    if len(results) > max_contexts:
        logger.debug(
            "Capping results from %d to max_contexts=%d.",
            len(results),
            max_contexts,
        )
        results = results[:max_contexts]

    return results


def retrieve_documents(
    query: str,
    index: faiss.Index,
    metadata_records: List[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Retrieve similar documents and return them as plain dictionaries.

    Equivalent to :func:`retrieve_context` but returns raw ``dict`` objects
    instead of :class:`RetrievalResult` dataclasses.  Useful for JSON
    serialisation in API response payloads.

    Parameters
    ----------
    query:
        Free-text user query.
    index:
        Populated FAISS index.
    metadata_records:
        Metadata list aligned with the FAISS index.
    top_k:
        Number of results to return.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.top_k`.
    min_similarity:
        Minimum similarity threshold.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.min_similarity`.

    Returns
    -------
    List[Dict[str, Any]]
        Each dict mirrors the fields of :class:`RetrievalResult`, keyed by
        attribute name.

    Raises
    ------
    ValueError
        If ``query`` is empty.
    RuntimeError
        Propagated from :func:`semantic_search` on FAISS failure.
    """
    results: List[RetrievalResult] = retrieve_context(
        query,
        index,
        metadata_records,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    return [
        {
            "rank": r.rank,
            "score": r.score,
            "ticket_id": r.ticket_id,
            "category": r.category,
            "priority": r.priority,
            "subject": r.subject,
            "clean_text": r.clean_text,
            "resolution_note": r.resolution_note,
            "knowledge_article": r.knowledge_article,
            "metadata": r.metadata,
        }
        for r in results
    ]
