"""
deployment/utils.py
===================
Reusable helper utilities shared across the Intelligent Support Ticket
Classification RAG deployment package.

This module provides ONLY general-purpose utilities.  It contains NO
business logic and performs NO retrieval, generation, model loading,
dataset loading, FAISS operations, FastAPI endpoint handling, or Azure
deployment operations.  Those responsibilities belong exclusively to
their respective modules.

All logging is performed through the singleton logger defined in
``deployment.config``.  No additional logger is created here.

Usage::

    from deployment.utils import (
        load_json, save_json,
        clean_whitespace, truncate_text,
        Timer, measure_execution_time,
        dataclass_to_dict, safe_json_dumps,
        validate_query, validate_top_k, validate_similarity,
        format_retrieval_results, format_latency, format_response,
        ensure_directory, safe_read_text, safe_write_text,
    )
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union

from deployment.config import logger

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound=Callable[..., Any])
JsonValue = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


# ===========================================================================
# 1. JSON Utilities
# ===========================================================================

def load_json(path: Union[str, Path], *, encoding: str = "utf-8") -> JsonValue:
    """Load and parse a JSON file from disk.

    Args:
        path: Path to the JSON file.
        encoding: File encoding.  Defaults to ``"utf-8"``.

    Returns:
        The deserialised Python object (dict, list, str, int, …).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file content is not valid JSON.
        OSError: If the file cannot be opened.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {file_path}"
        )
    logger.debug("Loading JSON from: %s", file_path)
    try:
        with file_path.open("r", encoding=encoding) as fh:
            data: JsonValue = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON content in '{file_path}': {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Could not read JSON file '{file_path}': {exc}"
        ) from exc
    logger.debug("JSON loaded successfully from: %s", file_path)
    return data


# Alias
read_json = load_json


def save_json(
    data: JsonValue,
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    create_parents: bool = True,
) -> None:
    """Serialise ``data`` to a JSON file on disk.

    Args:
        data: Python object to serialise.
        path: Destination file path.
        encoding: File encoding.  Defaults to ``"utf-8"``.
        indent: JSON indentation level.  Defaults to ``2``.
        ensure_ascii: Escape non-ASCII characters when ``True``.
        create_parents: Automatically create missing parent directories
            when ``True`` (default).

    Raises:
        TypeError: If ``data`` contains non-JSON-serialisable objects.
        OSError: If the file cannot be written.
    """
    file_path = Path(path)
    if create_parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Saving JSON to: %s", file_path)
    try:
        with file_path.open("w", encoding=encoding) as fh:
            json.dump(data, fh, indent=indent, ensure_ascii=ensure_ascii)
    except TypeError as exc:
        raise TypeError(
            f"Data contains non-JSON-serialisable objects: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"Could not write JSON file '{file_path}': {exc}"
        ) from exc
    logger.debug("JSON saved successfully to: %s", file_path)


# Alias
write_json = save_json


# ===========================================================================
# 2. Text Utilities
# ===========================================================================

def clean_whitespace(text: str) -> str:
    """Collapse all runs of whitespace into a single space and strip.

    Args:
        text: Input string.

    Returns:
        Whitespace-normalised string.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}.")
    return re.sub(r"\s+", " ", text).strip()


def normalize_newlines(text: str, *, replacement: str = "\n") -> str:
    """Replace all CRLF and CR line endings with ``replacement``.

    Args:
        text: Input string.
        replacement: The normalised line ending to use.  Defaults to ``"\\n"``.

    Returns:
        String with unified line endings.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}.")
    return text.replace("\r\n", replacement).replace("\r", replacement)


def truncate_text(text: str, max_length: int, *, suffix: str = "…") -> str:
    """Truncate ``text`` to ``max_length`` characters, appending ``suffix``.

    If ``text`` is already within the limit, it is returned unchanged.
    The returned string (including ``suffix``) never exceeds ``max_length``
    characters.

    Args:
        text: Input string.
        max_length: Maximum number of characters in the returned string.
        suffix: Truncation indicator appended when text is shortened.
            Defaults to ``"…"``.

    Returns:
        Possibly truncated string.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If ``max_length`` is not a positive integer or is shorter
            than ``suffix``.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}.")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1; got {max_length}.")
    if len(suffix) > max_length:
        raise ValueError(
            f"suffix length ({len(suffix)}) exceeds max_length ({max_length})."
        )
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def safe_strip(value: Any) -> str:
    """Convert ``value`` to a string and strip leading/trailing whitespace.

    Args:
        value: Any object; coerced via ``str()``.

    Returns:
        Stripped string representation of ``value``.
    """
    return str(value).strip()


# ===========================================================================
# 3. Timing Utilities
# ===========================================================================

class Timer:
    """Context manager that measures elapsed wall-clock time.

    Attributes:
        elapsed: Elapsed seconds after the context exits.  ``0.0`` before.
        label: Optional human-readable label used in log messages.

    Example::

        with Timer(label="retrieval") as t:
            results = retrieve_context(query, index, metadata)
        print(f"Retrieval took {t.elapsed:.4f}s")
    """

    def __init__(self, label: str = "", *, log: bool = True) -> None:
        """Initialise the timer.

        Args:
            label: Descriptive label for log messages.
            log: Emit a DEBUG log entry on exit when ``True`` (default).
        """
        self.label: str = label
        self.elapsed: float = 0.0
        self._log: bool = log
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self._log:
            label_str = f"[{self.label}] " if self.label else ""
            logger.debug("%sElapsed: %.6fs", label_str, self.elapsed)


def measure_execution_time(func: _F) -> _F:
    """Decorator that logs the execution time of the wrapped function.

    The elapsed time is emitted at DEBUG level through the shared logger.

    Args:
        func: Callable to wrap.

    Returns:
        Wrapped callable with identical signature.

    Example::

        @measure_execution_time
        def my_function(x: int) -> int:
            return x * 2
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(
                "Function '%s' completed in %.6fs.", func.__qualname__, elapsed
            )
        return result

    return wrapper  # type: ignore[return-value]


@contextlib.contextmanager
def timed_block(label: str) -> Generator[None, None, None]:
    """Context manager that logs the duration of an arbitrary code block.

    Args:
        label: Descriptive label included in the log message.

    Yields:
        None

    Example::

        with timed_block("FAISS search"):
            scores, indices = index.search(q_vec, top_k)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.debug("[%s] completed in %.6fs.", label, elapsed)


# ===========================================================================
# 4. Serialisation Utilities
# ===========================================================================

def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Recursively convert a dataclass instance to a plain dictionary.

    Works with nested dataclasses.  Non-dataclass values are returned as-is.

    Args:
        obj: A dataclass instance (e.g. ``RAGResponse``, ``RetrievalResult``).

    Returns:
        A plain ``dict`` representation of ``obj``.

    Raises:
        TypeError: If ``obj`` is not a dataclass instance.

    Example::

        response_dict = dataclass_to_dict(rag_response)
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(
            f"Expected a dataclass instance, got {type(obj).__name__!r}."
        )
    return dataclasses.asdict(obj)


def safe_json_dumps(
    obj: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    default: Optional[Callable[[Any], Any]] = None,
) -> str:
    """Serialise ``obj`` to a JSON string, handling common non-serialisable types.

    Falls back to ``repr()`` for objects that are not JSON-serialisable when
    no custom ``default`` is provided.

    Args:
        obj: Object to serialise.  Dataclass instances are automatically
            converted via :func:`dataclass_to_dict`.
        indent: Indentation level.  Defaults to ``2``.
        ensure_ascii: Escape non-ASCII chars when ``True``.
        default: Optional callable for non-serialisable types.

    Returns:
        JSON string representation of ``obj``.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)

    def _default(value: Any) -> Any:
        if default is not None:
            return default(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.asdict(value)
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=_default)


# Alias
object_to_json = safe_json_dumps


# ===========================================================================
# 5. Validation Utilities
# ===========================================================================

def validate_query(query: Any, *, param_name: str = "query") -> str:
    """Validate and sanitise a user query string.

    Args:
        query: Value to validate.
        param_name: Name used in exception messages.  Defaults to ``"query"``.

    Returns:
        Stripped, non-empty query string.

    Raises:
        TypeError: If ``query`` is not a string.
        ValueError: If ``query`` is empty or whitespace-only after stripping.
    """
    if not isinstance(query, str):
        raise TypeError(
            f"'{param_name}' must be a str; got {type(query).__name__!r}."
        )
    cleaned = query.strip()
    if not cleaned:
        raise ValueError(
            f"'{param_name}' must be a non-empty string; "
            "received an empty or whitespace-only value."
        )
    return cleaned


def ensure_non_empty(value: str, *, param_name: str = "value") -> str:
    """Assert that a string value is non-empty after stripping.

    Args:
        value: String to check.
        param_name: Name used in exception messages.

    Returns:
        Stripped, non-empty string.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty or whitespace-only.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"'{param_name}' must be a str; got {type(value).__name__!r}."
        )
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"'{param_name}' must not be empty.")
    return stripped


def validate_top_k(top_k: Any, *, param_name: str = "top_k") -> int:
    """Validate that ``top_k`` is a positive integer.

    Args:
        top_k: Value to validate.
        param_name: Name used in exception messages.

    Returns:
        Validated positive integer.

    Raises:
        TypeError: If ``top_k`` is not an integer.
        ValueError: If ``top_k`` is less than 1.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(
            f"'{param_name}' must be an int; got {type(top_k).__name__!r}."
        )
    if top_k < 1:
        raise ValueError(
            f"'{param_name}' must be a positive integer (>= 1); got {top_k}."
        )
    return top_k


def validate_similarity(
    score: Any,
    *,
    param_name: str = "min_similarity",
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """Validate that a similarity score falls within ``[low, high]``.

    Args:
        score: Value to validate.
        param_name: Name used in exception messages.
        low: Inclusive lower bound.  Defaults to ``0.0``.
        high: Inclusive upper bound.  Defaults to ``1.0``.

    Returns:
        Validated float score.

    Raises:
        TypeError: If ``score`` is not a numeric type.
        ValueError: If ``score`` is outside ``[low, high]``.
    """
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise TypeError(
            f"'{param_name}' must be a float; got {type(score).__name__!r}."
        )
    score_f = float(score)
    if not (low <= score_f <= high):
        raise ValueError(
            f"'{param_name}' must be in [{low}, {high}]; got {score_f}."
        )
    return score_f


# ===========================================================================
# 6. Formatting Utilities
# ===========================================================================

def format_latency(seconds: float, *, precision: int = 4) -> str:
    """Format a latency value in seconds to a human-readable string.

    Args:
        seconds: Elapsed time in seconds.
        precision: Number of decimal places.  Defaults to ``4``.

    Returns:
        Formatted string such as ``"12.3456s"`` or ``"0.0023s"``.

    Raises:
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError(f"Latency cannot be negative; got {seconds}.")
    return f"{seconds:.{precision}f}s"


def format_retrieval_results(results: List[Any]) -> str:
    """Format a list of ``RetrievalResult`` objects as a human-readable string.

    Produces a numbered block per result that includes all key fields.
    Returns an empty string when ``results`` is empty.

    Args:
        results: List of :class:`~deployment.retrieval.RetrievalResult`
            instances.  Accessed via attribute names; duck-typing is used so
            that this module does not create a circular import.

    Returns:
        Multi-line formatted string, or an empty string for empty input.
    """
    if not results:
        return ""

    separator = "-" * 44
    sections: List[str] = []
    for result in results:
        lines = [
            f"[Rank {getattr(result, 'rank', '?')} | "
            f"Score: {getattr(result, 'score', 0.0):.4f}]",
            f"Ticket ID  : {getattr(result, 'ticket_id', 'N/A')}",
            f"Subject    : {getattr(result, 'subject', 'N/A')}",
            f"Category   : {getattr(result, 'category', 'N/A')}",
            f"Priority   : {getattr(result, 'priority', 'N/A')}",
            f"Resolution : {getattr(result, 'resolution_note', 'N/A')}",
            f"Article    : {getattr(result, 'knowledge_article', 'N/A')}",
            separator,
        ]
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def format_response(response: Any) -> str:
    """Format a ``RAGResponse`` object as a human-readable summary string.

    Accesses fields via ``getattr`` to avoid a circular import.

    Args:
        response: A :class:`~deployment.rag_pipeline.RAGResponse` instance.

    Returns:
        Multi-line string summarising the response.
    """
    divider = "=" * 49
    lines = [
        divider,
        "RAG PIPELINE RESPONSE",
        divider,
        f"Query         : {getattr(response, 'user_query', 'N/A')}",
        f"Model         : {getattr(response, 'generation_model', 'N/A')}",
        f"Retrieval     : {format_latency(getattr(response, 'retrieval_time', 0.0))}",
        f"Generation    : {format_latency(getattr(response, 'generation_time', 0.0))}",
        f"Total Latency : {format_latency(getattr(response, 'total_latency', 0.0))}",
        divider,
        "Generated Response:",
        "",
        str(getattr(response, "generated_response", "")),
        divider,
    ]
    return "\n".join(lines)


def format_metadata(metadata: Dict[str, Any], *, title: str = "Metadata") -> str:
    """Format a metadata dictionary as an aligned, human-readable block.

    Args:
        metadata: Key-value pairs to format.
        title: Section header.  Defaults to ``"Metadata"``.

    Returns:
        Multi-line formatted string.
    """
    if not metadata:
        return f"{title}: (empty)"

    max_key_len = max(len(str(k)) for k in metadata)
    lines = [f"{title}:"]
    for key, value in metadata.items():
        lines.append(f"  {str(key).ljust(max_key_len)} : {value}")
    return "\n".join(lines)


# ===========================================================================
# 7. File Utilities
# ===========================================================================

def ensure_directory(path: Union[str, Path], *, parents: bool = True) -> Path:
    """Create a directory (and optionally its parents) if it does not exist.

    Idempotent: safe to call when the directory already exists.

    Args:
        path: Directory path to create.
        parents: Create missing parent directories when ``True`` (default).

    Returns:
        Resolved :class:`~pathlib.Path` of the directory.

    Raises:
        OSError: If the directory cannot be created.
    """
    dir_path = Path(path).resolve()
    try:
        dir_path.mkdir(parents=parents, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Failed to create directory '{dir_path}': {exc}"
        ) from exc
    logger.debug("Directory ensured: %s", dir_path)
    return dir_path


def safe_exists(path: Union[str, Path]) -> bool:
    """Return ``True`` if ``path`` exists on the filesystem, ``False`` otherwise.

    Swallows ``PermissionError`` and returns ``False`` in that case.

    Args:
        path: Path to check.

    Returns:
        ``True`` if the path exists and is accessible; ``False`` otherwise.
    """
    try:
        return Path(path).exists()
    except (PermissionError, OSError):
        return False


def safe_delete(path: Union[str, Path], *, missing_ok: bool = True) -> bool:
    """Delete a file or directory tree safely.

    Args:
        path: Path to the file or directory to delete.
        missing_ok: If ``True`` (default), do not raise an error when the
            path does not exist.

    Returns:
        ``True`` if the path was deleted; ``False`` if it did not exist and
        ``missing_ok`` is ``True``.

    Raises:
        FileNotFoundError: If the path does not exist and ``missing_ok`` is
            ``False``.
        OSError: If deletion fails for any other reason.
    """
    target = Path(path)
    if not target.exists():
        if missing_ok:
            logger.debug("safe_delete: path does not exist (missing_ok=True): %s", target)
            return False
        raise FileNotFoundError(f"Path not found: {target}")

    try:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as exc:
        raise OSError(f"Failed to delete '{target}': {exc}") from exc

    logger.debug("Deleted: %s", target)
    return True


def safe_read_text(
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
) -> str:
    """Read the entire contents of a text file as a string.

    Args:
        path: Path to the text file.
        encoding: File encoding.  Defaults to ``"utf-8"``.

    Returns:
        File content as a string.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        OSError: If the file cannot be read.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        content = file_path.read_text(encoding=encoding)
    except OSError as exc:
        raise OSError(f"Could not read file '{file_path}': {exc}") from exc
    logger.debug("Read %d characters from: %s", len(content), file_path)
    return content


def safe_write_text(
    content: str,
    path: Union[str, Path],
    *,
    encoding: str = "utf-8",
    create_parents: bool = True,
) -> None:
    """Write a string to a text file, creating parent directories as needed.

    Args:
        content: String content to write.
        path: Destination file path.
        encoding: File encoding.  Defaults to ``"utf-8"``.
        create_parents: Automatically create missing parent directories when
            ``True`` (default).

    Raises:
        TypeError: If ``content`` is not a string.
        OSError: If the file cannot be written.
    """
    if not isinstance(content, str):
        raise TypeError(f"content must be a str; got {type(content).__name__!r}.")

    file_path = Path(path)
    if create_parents:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        file_path.write_text(content, encoding=encoding)
    except OSError as exc:
        raise OSError(f"Could not write file '{file_path}': {exc}") from exc

    logger.debug("Wrote %d characters to: %s", len(content), file_path)


# ===========================================================================
# 8. Logging Helpers
# ===========================================================================

def log_section(title: str, *, width: int = 49) -> None:
    """Emit a prominent section divider to the shared logger at INFO level.

    Args:
        title: Section heading text.
        width: Total width of the divider line.  Defaults to ``49``.
    """
    divider = "=" * width
    logger.info(divider)
    logger.info(title)
    logger.info(divider)


def log_dict(data: Dict[str, Any], *, title: str = "", level: str = "info") -> None:
    """Log every key-value pair in ``data`` through the shared logger.

    Args:
        data: Dictionary to log.
        title: Optional header emitted before the key-value pairs.
        level: Logging level name (``"debug"``, ``"info"``, ``"warning"``,
            ``"error"``).  Defaults to ``"info"``.

    Raises:
        ValueError: If ``level`` is not a recognised logging level name.
    """
    emit: Callable[[str], None] = getattr(logger, level.lower(), None)  # type: ignore[assignment]
    if emit is None:
        raise ValueError(
            f"Unrecognised logging level: {level!r}. "
            "Choose from 'debug', 'info', 'warning', 'error'."
        )
    if title:
        emit(title)
    max_key = max((len(str(k)) for k in data), default=0)
    for key, value in data.items():
        emit("  %s : %s", str(key).ljust(max_key), value)


def log_timing(
    label: str,
    retrieval_time: float,
    generation_time: float,
    total_latency: float,
) -> None:
    """Log a structured timing summary at INFO level.

    Intended as a convenience wrapper for callers that already hold the three
    latency measurements produced by ``generate_response()``.

    Args:
        label: Context label prepended to the log line.
        retrieval_time: Seconds spent in the retrieval stage.
        generation_time: Seconds spent in the generation stage.
        total_latency: End-to-end elapsed seconds.
    """
    logger.info(
        "%s — retrieval: %s, generation: %s, total: %s",
        label,
        format_latency(retrieval_time),
        format_latency(generation_time),
        format_latency(total_latency),
    )
