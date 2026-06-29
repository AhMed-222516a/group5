"""
deployment/model_loader.py

Single source of truth for loading all ML models used in the RAG deployment pipeline.
No other module should directly instantiate SentenceTransformer, AutoTokenizer,
AutoModelForSeq2SeqLM, or HuggingFace Pipeline — use this module exclusively.
"""

from __future__ import annotations

import torch
from functools import lru_cache
from typing import Tuple

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from deployment.config import (
    PATHS,
    EMBEDDING_CONFIG,
    GENERATION_CONFIG,
    RUNTIME_CONFIG,
    logger,
    get_device,
)

# ---------------------------------------------------------------------------
# Internal module-level singletons (never access these directly from outside)
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None
_tokenizer: PreTrainedTokenizer | None = None
_generation_model: PreTrainedModel | None = None
_generation_pipeline: Pipeline | None = None


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

def load_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer embedding model.

    Loads the model specified by ``EMBEDDING_CONFIG.model_name`` and moves it
    to the device returned by :func:`get_device`.  The model is placed in
    evaluation mode and gradients are disabled for inference efficiency.

    Returns
    -------
    SentenceTransformer
        The loaded (and device-placed) embedding model.

    Raises
    ------
    RuntimeError
        If the model cannot be loaded from the local cache or Hugging Face Hub.
    """
    global _embedding_model

    if _embedding_model is not None:
        return _embedding_model

    model_name: str = EMBEDDING_CONFIG.model_name
    cache_dir: str = str(PATHS.cache_dir)
    device = get_device()

    logger.info("Loading embedding model: %s", model_name)
    logger.info("Using device: %s", device)
    logger.info("Cache directory: %s", cache_dir)

    try:
        model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_dir,
            device=str(device),
        )
        model.eval()

        # Disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False

        _embedding_model = model
        logger.info("Embedding model loaded successfully: %s", model_name)

    except Exception as exc:
        logger.exception("Failed to load embedding model '%s': %s", model_name, exc)
        raise RuntimeError(
            f"Could not load embedding model '{model_name}'. "
            "Check your network connection, model name, and cache directory."
        ) from exc

    return _embedding_model


def get_embedding_model() -> SentenceTransformer:
    """Return the cached embedding model, loading it on first call.

    This is the preferred public API for obtaining the embedding model.
    Guaranteed to load the model at most once per process.

    Returns
    -------
    SentenceTransformer
        The singleton embedding model instance.

    Raises
    ------
    RuntimeError
        Propagated from :func:`load_embedding_model` if loading fails.
    """
    return load_embedding_model()


# ---------------------------------------------------------------------------
# Generation model (tokenizer + seq2seq model)
# ---------------------------------------------------------------------------

def load_generation_model() -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """Load and cache the tokenizer and generation model.

    Attempts to load the primary model specified by
    ``GENERATION_CONFIG.primary_model``.  On any failure, automatically falls
    back to ``GENERATION_CONFIG.fallback_model``.  Both the tokenizer and the
    model are cached as module-level singletons.

    The model is moved to the configured device, set to evaluation mode, and
    gradient computation is disabled.

    Returns
    -------
    Tuple[PreTrainedTokenizer, PreTrainedModel]
        A ``(tokenizer, model)`` pair ready for inference.

    Raises
    ------
    RuntimeError
        If both the primary and fallback models fail to load.
    """
    global _tokenizer, _generation_model

    if _tokenizer is not None and _generation_model is not None:
        return _tokenizer, _generation_model

    cache_dir: str = str(PATHS.cache_dir)
    device = get_device()
    primary_model: str = GENERATION_CONFIG.primary_model
    fallback_model: str = GENERATION_CONFIG.fallback_model

    def _try_load(model_name: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        logger.info("Loading tokenizer: %s", model_name)
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info("Tokenizer loaded: %s", model_name)

        logger.info("Loading generation model: %s", model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        mdl = mdl.to(device)
        mdl.eval()

        for param in mdl.parameters():
            param.requires_grad = False

        logger.info("Generation model loaded: %s  |  device: %s", model_name, device)
        return tok, mdl

    # --- Primary attempt ---
    try:
        _tokenizer, _generation_model = _try_load(primary_model)
    except Exception as primary_exc:
        logger.warning(
            "Primary model '%s' failed to load (%s). Falling back to '%s'.",
            primary_model,
            primary_exc,
            fallback_model,
        )
        logger.info("Falling back to %s.", fallback_model)

        # --- Fallback attempt ---
        try:
            _tokenizer, _generation_model = _try_load(fallback_model)
        except Exception as fallback_exc:
            logger.exception(
                "Fallback model '%s' also failed to load: %s", fallback_model, fallback_exc
            )
            raise RuntimeError(
                f"Could not load primary model '{primary_model}' or fallback "
                f"model '{fallback_model}'. Check your network, model names, "
                "and cache directory."
            ) from fallback_exc

    return _tokenizer, _generation_model


def get_generation_model() -> PreTrainedModel:
    """Return the cached generation model, loading it on first call.

    Returns
    -------
    PreTrainedModel
        The singleton generation model instance.

    Raises
    ------
    RuntimeError
        Propagated from :func:`load_generation_model` if loading fails.
    """
    _, model = load_generation_model()
    return model


def get_tokenizer() -> PreTrainedTokenizer:
    """Return the cached tokenizer, loading it on first call.

    Returns
    -------
    PreTrainedTokenizer
        The singleton tokenizer instance.

    Raises
    ------
    RuntimeError
        Propagated from :func:`load_generation_model` if loading fails.
    """
    tokenizer, _ = load_generation_model()
    return tokenizer


# ---------------------------------------------------------------------------
# Hugging Face text2text-generation pipeline
# ---------------------------------------------------------------------------

def get_generation_pipeline() -> Pipeline:
    """Return the cached Hugging Face text2text-generation pipeline.

    Builds (once) a ``transformers.pipeline`` for ``text2text-generation``
    using the already-cached tokenizer and generation model.  All generation
    hyper-parameters (``max_new_tokens``, ``temperature``, ``top_p``,
    ``top_k``, ``repetition_penalty``) are read exclusively from
    ``GENERATION_CONFIG``.

    Returns
    -------
    Pipeline
        The singleton ``text2text-generation`` pipeline.

    Raises
    ------
    RuntimeError
        If the underlying model or tokenizer cannot be loaded, or if the
        pipeline itself fails to initialise.
    """
    global _generation_pipeline

    if _generation_pipeline is not None:
        return _generation_pipeline

    logger.info("Initialising text2text-generation pipeline …")

    tokenizer = get_tokenizer()
    model = get_generation_model()
    device = get_device()
    # Hugging Face pipeline expects:
    #   0  -> first CUDA GPU
    #  -1  -> CPU
    # Map torch.device → int index expected by HF pipeline
    # (-1 forces CPU; >=0 selects the CUDA device index)
    device_index = 0 if device == "cuda" else -1

    try:
        gen_cfg = GENERATION_CONFIG
        _generation_pipeline = pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_index,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            repetition_penalty=gen_cfg.repetition_penalty,
        )
        logger.info("text2text-generation pipeline initialised successfully.")
    except Exception as exc:
        logger.exception("Failed to create generation pipeline: %s", exc)
        raise RuntimeError(
            "Could not initialise the text2text-generation pipeline. "
            "Ensure the generation model and tokenizer loaded correctly."
        ) from exc

    return _generation_pipeline
