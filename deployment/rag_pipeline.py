"""
deployment/rag_pipeline.py
==========================
Orchestration layer for the Intelligent Support Ticket Classification RAG system.

This module connects every component of the RAG workflow into a single,
production-ready pipeline:

    User Query
        ↓
    Semantic Retrieval  (deployment.retrieval)
        ↓
    Retrieved Context
        ↓
    Prompt Construction
        ↓
    Response Generation (deployment.model_loader)
        ↓
    RAGResponse

This module NEVER:
    * Instantiates SentenceTransformer, AutoTokenizer, AutoModelForSeq2SeqLM,
      or any HuggingFace Pipeline directly.
    * Rebuilds FAISS embeddings or the FAISS index.
    * Loads the dataset from disk.
    * Implements FastAPI endpoints or Azure deployment logic.

All configuration is imported exclusively from ``deployment.config``.
All models are obtained exclusively from ``deployment.model_loader``.
All retrieval operations are delegated exclusively to ``deployment.retrieval``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import faiss

from deployment.config import (
    GENERATION_CONFIG,
    RETRIEVAL_CONFIG,
    RUNTIME_CONFIG,
    logger,
)
from deployment.model_loader import (
    get_generation_pipeline,
    get_tokenizer,
)
from deployment.retrieval import (
    RetrievalResult,
    build_index,
    retrieve_context,
)

# ---------------------------------------------------------------------------
# Module-level resource singletons
# ---------------------------------------------------------------------------

_faiss_index: Optional[faiss.Index] = None
_metadata_records: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    """Structured output produced by the RAG pipeline for a single query.

    Attributes
    ----------
    user_query:
        The original free-text query supplied by the caller.
    generated_response:
        The model-generated answer grounded in the retrieved context.
    retrieved_context:
        Concatenated plain-text context string forwarded to the generator.
    retrieval_results:
        Ranked list of :class:`~deployment.retrieval.RetrievalResult` objects
        returned by the retrieval layer.
    generation_model:
        Identifier of the generation model used to produce the response.
    retrieval_time:
        Wall-clock seconds spent on semantic retrieval.
    generation_time:
        Wall-clock seconds spent on response generation.
    total_latency:
        End-to-end wall-clock seconds for the full pipeline call.
    metadata:
        Arbitrary key-value pairs for downstream consumers (e.g. top_k used,
        number of contexts, device, environment).
    """

    user_query: str
    generated_response: str
    retrieved_context: str
    retrieval_results: List[RetrievalResult]
    generation_model: str
    retrieval_time: float
    generation_time: float
    total_latency: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Resource initialisation
# ---------------------------------------------------------------------------

def _load_resources() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load all pipeline resources, using module-level singletons for caching.

    On the first call, the FAISS index and its companion metadata are loaded
    (or built) via :func:`~deployment.retrieval.build_index`.  Subsequent
    calls return the already-loaded singletons without any I/O.

    Returns
    -------
    Tuple[faiss.Index, List[Dict[str, Any]]]
        A ``(index, metadata_records)`` pair ready for retrieval.

    Raises
    ------
    FileNotFoundError
        If neither an existing index nor the source dataset is available.
    RuntimeError
        If the index cannot be loaded or built.
    """
    global _faiss_index, _metadata_records

    if _faiss_index is not None and _metadata_records is not None:
        logger.debug("Reusing cached FAISS index (%d vectors).", _faiss_index.ntotal)
        return _faiss_index, _metadata_records

    logger.info("Initialising FAISS index and metadata (first call) …")
    _faiss_index, _metadata_records = build_index()
    logger.info(
        "Resources ready — FAISS index: %d vectors, metadata records: %d.",
        _faiss_index.ntotal,
        len(_metadata_records),
    )
    return _faiss_index, _metadata_records


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(query: str, retrieval_results: List[RetrievalResult]) -> str:
    """Construct a grounded instruction prompt for the generation model.

    Formats the retrieved support ticket context and the user query into a
    single string following a clear system instruction that encourages
    accurate, professional, context-aware responses.

    Parameters
    ----------
    query:
        The original user query.
    retrieval_results:
        Ranked retrieval results whose fields are used to build the context block.

    Returns
    -------
    str
        A fully-formatted prompt ready to be passed to the generation pipeline.
    """
    divider = "=" * 49
    ticket_sep = "-" * 44

    # ------------------------------------------------------------------
    # SYSTEM INSTRUCTION
    # ------------------------------------------------------------------
    system_block = (
        f"{divider}\n"
        "SYSTEM INSTRUCTION\n\n"
        "You are an expert customer support assistant for a technical help desk.\n"
        "Answer the customer's question ONLY using the retrieved support ticket context below.\n"
        "Never invent information. Never answer from general knowledge. Never hallucinate.\n"
        "If the answer cannot be derived from the retrieved context, reply exactly:\n"
        '"I could not find sufficient information in the retrieved support knowledge."\n'
        "Be concise, professional, and step-by-step when appropriate.\n"
        "Do NOT include generic greetings or endings like "
        '"Is there anything else I can help you with?"\n'
        f"{divider}"
    )

    # ------------------------------------------------------------------
    # RETRIEVED SUPPORT TICKETS
    # ------------------------------------------------------------------
    if retrieval_results:
        ticket_lines: List[str] = []
        for result in retrieval_results:
            ticket_lines.append(f"Ticket #{result.rank}  (Similarity Score: {result.score:.4f})")
            ticket_lines.append(f"Subject    : {result.subject or 'N/A'}")
            ticket_lines.append(f"Category   : {result.category or 'N/A'}")
            ticket_lines.append(f"Priority   : {result.priority or 'N/A'}")
            ticket_lines.append(f"Resolution : {result.resolution_note or 'N/A'}")
            ticket_lines.append(f"Article    : {result.knowledge_article or 'N/A'}")
            ticket_lines.append("")            
            ticket_lines.append(ticket_sep)
        context_body = "\n".join(ticket_lines)
    else:
        context_body = "No relevant support tickets were retrieved for this query."

    context_block = (
        f"{divider}\n"
        "RETRIEVED SUPPORT TICKETS\n\n"
        f"{context_body}\n"
        f"{divider}"
    )

    # ------------------------------------------------------------------
    # FEW-SHOT BEHAVIORAL EXAMPLE
    # ------------------------------------------------------------------
    example_block = (
        f"{divider}\n"
        "EXAMPLE\n\n"
        'Question:\n'
        '"My account is locked."\n\n'
        "Answer:\n"
        "Based on the retrieved support tickets, verify the customer's identity, "
        "reset the credentials if necessary, and unlock the account.\n"
        f"{divider}"
    )

    # ------------------------------------------------------------------
    # USER QUESTION
    # ------------------------------------------------------------------
    question_block = (
        f"{divider}\n"
        "USER QUESTION\n\n"
        f"{query}\n"
        f"{divider}"
    )

    # ------------------------------------------------------------------
    # ANSWER PROMPT
    # ------------------------------------------------------------------
    answer_block = (
        f"{divider}\n"
        "ANSWER\n\n"
    )

    prompt = "\n\n".join([
        system_block,
        context_block,
        example_block,
        question_block,
        answer_block,
    ])

    logger.debug(
        "Prompt constructed — length: %d characters, context blocks: %d.",
        len(prompt),
        len(retrieval_results),
    )
    return prompt


def construct_prompt(query: str, retrieval_results: List[RetrievalResult]) -> str:
    """Alias for :func:`build_prompt` for API symmetry.

    Parameters
    ----------
    query:
        The original user query.
    retrieval_results:
        Ranked retrieval results.

    Returns
    -------
    str
        Fully-formatted generation prompt.
    """
    return build_prompt(query, retrieval_results)

# ---------------------------------------------------------------------------
# Context serialisation
# ---------------------------------------------------------------------------

def _format_retrieved_context(retrieval_results: List[RetrievalResult]) -> str:
    """Serialise retrieved results into a single human-readable context string.

    Parameters
    ----------
    retrieval_results:
        Ranked list of retrieval results.

    Returns
    -------
    str
        A single string that concatenates all context fields, separated by
        newlines.  Used to populate :attr:`RAGResponse.retrieved_context`.
    """
    if not retrieval_results:
        return ""

    sections: List[str] = []
    for result in retrieval_results:
        section_lines: List[str] = [
            f"[Rank {result.rank} | Score: {result.score:.4f}]",
            f"Ticket ID : {result.ticket_id}",
            f"Subject   : {result.subject}",
            f"Category  : {result.category}",
            f"Priority  : {result.priority}",
            f"Resolution: {result.resolution_note}",
        ]
        if result.knowledge_article:
            section_lines.append(f"Article   : {result.knowledge_article}")
        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------

def _validate_query(query: str) -> str:
    """Strip and validate the incoming user query.

    Parameters
    ----------
    query:
        Raw user query string.

    Returns
    -------
    str
        Stripped, non-empty query string.

    Raises
    ------
    ValueError
        If the query is empty or contains only whitespace.
    TypeError
        If ``query`` is not a string.
    """
    if not isinstance(query, str):
        raise TypeError(
            f"query must be a str, got {type(query).__name__!r}."
        )
    cleaned = query.strip()
    if not cleaned:
        raise ValueError(
            "query must be a non-empty string; received an empty or whitespace-only value."
        )
    return cleaned


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(prompt: str) -> str:
    """Run the generation pipeline on a fully-constructed prompt.

    Delegates to the singleton Hugging Face ``text2text-generation`` pipeline
    obtained from :func:`~deployment.model_loader.get_generation_pipeline`.
    The tokenizer is used only for logging the token length of the prompt;
    all generation parameters are already baked into the pipeline at
    construction time via ``GENERATION_CONFIG``.

    Parameters
    ----------
    prompt:
        The fully-formatted prompt string produced by :func:`build_prompt`.

    Returns
    -------
    str
        The generated response text, stripped of leading/trailing whitespace.

    Raises
    ------
    RuntimeError
        If the generation pipeline raises an exception.
    """
    tokenizer = get_tokenizer()
    gen_pipeline = get_generation_pipeline()

    # Log approximate prompt length in tokens (informational only).
    try:
        token_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=GENERATION_CONFIG.max_input_length,
            return_tensors=None,
        )
        prompt_token_count: int = len(token_ids["input_ids"])
        logger.info(
            "Prompt token count: %d (max_input_length=%d).",
            prompt_token_count,
            GENERATION_CONFIG.max_input_length,
        )
    except Exception as tok_exc:
        logger.warning("Could not compute prompt token count: %s", tok_exc)

    logger.info("Starting response generation …")
    try:
        outputs = gen_pipeline(prompt, max_new_tokens=GENERATION_CONFIG.max_new_tokens)
        raw_text: str = outputs[0].get("generated_text", "")
        response_text: str = raw_text.strip()
    except Exception as exc:
        logger.exception("Generation pipeline failed: %s", exc)
        raise RuntimeError(
            f"Response generation failed for the given prompt. "
            f"Underlying error: {exc}"
        ) from exc

    logger.info(
        "Generation completed — response length: %d characters.", len(response_text)
    )
    return response_text


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_response(
    query: str,
    *,
    top_k: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> RAGResponse:
    """Execute the full RAG pipeline for a single user query.

    This is the primary public interface of the module and is intended to be
    called directly by the FastAPI application layer.

    The function performs the following steps in sequence:

    1. Validate and sanitise the input query.
    2. Load (or reuse) the FAISS index and metadata records.
    3. Retrieve the most semantically similar support tickets.
    4. Construct a grounded generation prompt.
    5. Generate a context-aware response using the cached generation pipeline.
    6. Package all artefacts into a :class:`RAGResponse` and return it.

    Parameters
    ----------
    query:
        Free-text user query.  Must be a non-empty string.
    top_k:
        Override for the number of nearest neighbours to retrieve.  Defaults
        to :attr:`~deployment.config.RetrievalConfig.top_k`.
    min_similarity:
        Override for the minimum similarity score threshold.  Defaults to
        :attr:`~deployment.config.RetrievalConfig.min_similarity`.

    Returns
    -------
    RAGResponse
        A fully-populated response object containing the generated answer,
        retrieved context, latency measurements, and metadata.

    Raises
    ------
    TypeError
        If ``query`` is not a string.
    ValueError
        If ``query`` is empty or contains only whitespace.
    FileNotFoundError
        If the FAISS index cannot be found and the source dataset is missing.
    RuntimeError
        If retrieval, prompt construction, or generation fails.
    """
    pipeline_start: float = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Validate query
    # ------------------------------------------------------------------
    logger.info("RAG pipeline invoked — incoming query: %r", query[:120])
    cleaned_query: str = _validate_query(query)

    # ------------------------------------------------------------------
    # 2. Load resources (idempotent; returns singletons after first call)
    # ------------------------------------------------------------------
    index, metadata_records = _load_resources()

    # ------------------------------------------------------------------
    # 3. Retrieval
    # ------------------------------------------------------------------
    effective_top_k: int = top_k if top_k is not None else RETRIEVAL_CONFIG.top_k
    effective_min_sim: float = (
        min_similarity if min_similarity is not None
        else RETRIEVAL_CONFIG.min_similarity
    )

    logger.info(
        "Starting retrieval — top_k=%d, min_similarity=%.2f.",
        effective_top_k,
        effective_min_sim,
    )
    retrieval_start: float = time.perf_counter()

    try:
        retrieval_results: List[RetrievalResult] = retrieve_context(
            cleaned_query,
            index,
            metadata_records,
            top_k=effective_top_k,
            min_similarity=effective_min_sim,
        )
    except Exception as exc:
        logger.exception("Retrieval failed for query %r: %s", cleaned_query[:80], exc)
        raise RuntimeError(
            f"Context retrieval failed for query '{cleaned_query[:80]}'. "
            f"Underlying error: {exc}"
        ) from exc

    retrieval_time: float = time.perf_counter() - retrieval_start
    logger.info(
        "Retrieval completed in %.4fs — %d result(s) returned.",
        retrieval_time,
        len(retrieval_results),
    )

    # ------------------------------------------------------------------
    # 4. Prompt construction
    # ------------------------------------------------------------------
    logger.info("Constructing generation prompt …")
    try:
        prompt: str = build_prompt(cleaned_query, retrieval_results)
    except Exception as exc:
        logger.exception("Prompt construction failed: %s", exc)
        raise RuntimeError(
            f"Failed to construct generation prompt: {exc}"
        ) from exc

    logger.info("Prompt construction completed.")

    # ------------------------------------------------------------------
    # 5. Response generation
    # ------------------------------------------------------------------
    generation_start: float = time.perf_counter()

    generated_response: str = _generate(prompt)

    generation_time: float = time.perf_counter() - generation_start
    total_latency: float = time.perf_counter() - pipeline_start

    logger.info(
        "Pipeline completed — retrieval: %.4fs, generation: %.4fs, total: %.4fs.",
        retrieval_time,
        generation_time,
        total_latency,
    )

    # ------------------------------------------------------------------
    # 6. Build and return RAGResponse
    # ------------------------------------------------------------------
    retrieved_context_str: str = _format_retrieved_context(retrieval_results)

    response_metadata: Dict[str, Any] = {
        "top_k_requested": effective_top_k,
        "min_similarity": effective_min_sim,
        "num_contexts_retrieved": len(retrieval_results),
        "max_contexts_cap": RETRIEVAL_CONFIG.max_contexts,
        "device": RUNTIME_CONFIG.device,
        "environment": RUNTIME_CONFIG.environment.value,
        "max_new_tokens": GENERATION_CONFIG.max_new_tokens,
        "temperature": GENERATION_CONFIG.temperature,
        "top_p": GENERATION_CONFIG.top_p,
        "top_k_generation": GENERATION_CONFIG.top_k,
        "repetition_penalty": GENERATION_CONFIG.repetition_penalty,
        "faiss_index_size": index.ntotal,
    }

    return RAGResponse(
        user_query=cleaned_query,
        generated_response=generated_response,
        retrieved_context=retrieved_context_str,
        retrieval_results=retrieval_results,
        generation_model=GENERATION_CONFIG.primary_model,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_latency=total_latency,
        metadata=response_metadata,
    )


# ---------------------------------------------------------------------------
# Warm-up utility
# ---------------------------------------------------------------------------

def warm_up() -> None:
    """Pre-load all pipeline resources to eliminate cold-start latency.

    Intended to be called once during application startup (e.g. from a
    FastAPI ``lifespan`` context manager or a ``startup`` event handler).
    Safe to call multiple times — subsequent calls are no-ops because all
    resources are cached as module-level singletons.

    Raises
    ------
    FileNotFoundError
        If the FAISS index and dataset cannot be found.
    RuntimeError
        If any resource fails to load.
    """
    logger.info("Warming up RAG pipeline — loading all resources …")

    # Load FAISS index + metadata
    _load_resources()

    # Load generation pipeline (triggers model_loader singletons)
    get_generation_pipeline()

    logger.info("RAG pipeline warm-up complete — all resources are ready.")
