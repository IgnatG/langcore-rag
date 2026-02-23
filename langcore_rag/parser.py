"""RAG query parser that decomposes natural language into
semantic terms and structured filters using an LLM.

The :class:`QueryParser` inspects a Pydantic schema to discover
filterable fields and builds a system prompt that instructs the
LLM to split an incoming query into vector-search terms and
metadata filters suitable for hybrid RAG retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, get_args, get_origin

import litellm
from pydantic import BaseModel

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

logger = logging.getLogger(__name__)

# Maximum characters of LLM response we will attempt to parse.
_MAX_RESPONSE_LEN = 8_192

# Fields with these Python types are considered filterable
# (i.e. they can appear inside ``structured_filters``).
_FILTERABLE_TYPES: set[type] = {
    int,
    float,
    str,
    bool,
    date,
    datetime,
}


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """Result of parsing a natural-language query.

    Attributes:
        semantic_terms: Free-text terms for vector / similarity
            search.
        structured_filters: Metadata filters keyed by schema
            field name.  Values may use MongoDB-style operators
            such as ``$gte``, ``$lte``, ``$eq``, ``$in``.
        confidence: A 0.0 - 1.0 score indicating how confident
            the parser is in the decomposition.
        explanation: Human-readable rationale for the parse.
    """

    semantic_terms: list[str] = field(default_factory=list)
    structured_filters: dict[str, Any] = field(
        default_factory=dict,
    )
    confidence: float = 0.0
    explanation: str = ""


# ------------------------------------------------------------------
# Schema introspection helpers
# ------------------------------------------------------------------


def _python_type_label(annotation: Any) -> str:
    """Return a human-friendly label for a type annotation.

    Handles ``Optional[X]``, ``list[X]``, plain types, and
    falls back to ``str(annotation)`` for anything exotic.
    """
    origin = get_origin(annotation)

    # ``Optional[X]`` is ``Union[X, None]``.
    if origin is type(None):
        return "None"

    args = get_args(annotation)

    # Handle Union / Optional
    if origin is not None and hasattr(origin, "__name__"):
        inner = ", ".join(_python_type_label(a) for a in args)
        return f"{origin.__name__}[{inner}]"

    # Strip ``NoneType`` from Optional
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_label(non_none[0])
        labels = ", ".join(_python_type_label(a) for a in non_none)
        return f"Union[{labels}]"

    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation)


def _is_filterable(annotation: Any) -> bool:
    """Return ``True`` when *annotation* maps to a filterable type.

    Supports ``Optional[X]`` — unwraps the union and checks the
    inner type.  Generic containers like ``list[str]`` or
    ``dict[str, int]`` are **not** considered filterable.
    """
    if annotation in _FILTERABLE_TYPES:
        return True

    origin = get_origin(annotation)
    # Generic containers (list, dict, set, …) are not scalar
    # filterable types even if their type args are filterable.
    if origin is not None and origin not in (type(None),):
        # ``Optional[X]`` is ``Union[X, None]`` — origin is
        # ``types.UnionType`` or ``typing.Union``.
        import types as _types
        import typing as _typing

        _union_origins = {_typing.Union}
        # Python 3.10+ ``X | Y`` produces ``types.UnionType``.
        if hasattr(_types, "UnionType"):
            _union_origins.add(_types.UnionType)

        if origin not in _union_origins:
            return False

    args = get_args(annotation)
    if args:
        non_none = [a for a in args if a is not type(None)]
        return any(a in _FILTERABLE_TYPES for a in non_none)

    return False


def _describe_fields(schema: type[BaseModel]) -> str:
    """Build a Markdown-like description of filterable fields.

    Returns a block of text listing each field name, its type,
    and its description (if present).
    """
    lines: list[str] = []
    model_fields: dict[str, FieldInfo] = schema.model_fields  # type: ignore[type-arg]

    for name, info in model_fields.items():
        annotation = info.annotation
        if annotation is None:
            continue
        if not _is_filterable(annotation):
            continue

        type_label = _python_type_label(annotation)
        desc = info.description or ""
        line = f"- {name} ({type_label})"
        if desc:
            line += f": {desc}"
        lines.append(line)

    return "\n".join(lines) if lines else "(no filterable fields)"


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = (
    """\
You are a query decomposition engine for a RAG (Retrieval-Augmented \
Generation) system.

Given a natural-language query and a list of filterable metadata \
fields, your job is to split the query into:

1. **semantic_terms** - free-text keywords / phrases best suited \
for vector similarity search.
2. **structured_filters** - precise metadata filters using the \
fields listed below.  Use MongoDB-style operators where \
appropriate: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.

### Filterable fields

{field_descriptions}

### Output format

Return **only** a JSON object with exactly these keys:

{{
  "semantic_terms": ["term1", "term2"],
  "structured_filters": {{"field": {{"$op": value}}}},
  "confidence": 0.95,
  "explanation": "short rationale"
}}

Rules:
- ``confidence`` must be a float between 0.0 and 1.0.
- If a part of the query does not map to any field, put it in \
``semantic_terms``.
- Dates should use ISO-8601 strings (YYYY-MM-DD).
- Do **not** invent fields that are not listed above.
- If no structured filters apply, return an empty dict for \
``structured_filters``.
- Do **not** wrap the JSON in markdown code fences.
"""
    ""
)


def _build_system_prompt(schema: type[BaseModel]) -> str:
    """Build the system prompt for a given Pydantic schema.

    Parameters:
        schema: The Pydantic model whose fields define the
            filterable metadata.

    Returns:
        A fully-rendered system prompt string.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(
        field_descriptions=_describe_fields(schema),
    )


# ------------------------------------------------------------------
# Response parsing helpers
# ------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text*.

    Tries the raw text first; falls back to extracting from a
    Markdown code fence.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    text = text.strip()
    if len(text) > _MAX_RESPONSE_LEN:
        text = text[:_MAX_RESPONSE_LEN]

    # Try raw parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try inside code fences.
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract a JSON object from LLM response: {text[:200]!r}"
    )


def _to_parsed_query(raw: dict[str, Any]) -> ParsedQuery:
    """Convert a raw dict into a validated :class:`ParsedQuery`.

    Applies type coercion and sensible defaults so that a
    partially-correct LLM response still yields a usable result.
    """
    semantic = raw.get("semantic_terms", [])
    if not isinstance(semantic, list):
        semantic = [str(semantic)]
    semantic = [str(t) for t in semantic]

    filters = raw.get("structured_filters", {})
    if not isinstance(filters, dict):
        filters = {}

    confidence = raw.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    explanation = str(raw.get("explanation", ""))

    return ParsedQuery(
        semantic_terms=semantic,
        structured_filters=filters,
        confidence=confidence,
        explanation=explanation,
    )


# ------------------------------------------------------------------
# QueryParser
# ------------------------------------------------------------------


class QueryParser:
    """Parse natural-language queries into semantic terms and
    structured metadata filters using an LLM.

    The parser inspects a Pydantic *schema* to discover which
    fields are filterable, then instructs the LLM (via
    `LiteLLM <https://docs.litellm.ai/>`_) to decompose the
    user's query.

    Parameters:
        schema: A Pydantic ``BaseModel`` subclass whose fields
            define the metadata that can appear in
            ``structured_filters``.
        model_id: Any model identifier accepted by LiteLLM
            (e.g. ``"gpt-4o"``, ``"anthropic/claude-3-opus"``,
            ``"gemini/gemini-2.5-flash"``).
        temperature: Sampling temperature for the LLM call.
            Lower values produce more deterministic parses.
        max_tokens: Maximum tokens the LLM may generate.
        **litellm_kwargs: Extra keyword arguments forwarded to
            ``litellm.completion()`` /
            ``litellm.acompletion()`` (e.g. ``api_key``,
            ``api_base``, ``timeout``).

    Example::

        from langcore_rag import QueryParser

        class Invoice(BaseModel):
            amount: float
            due_date: str
            vendor: str

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        parsed = parser.parse("invoices over $5000 due in March")
        print(parsed.structured_filters)
    """

    def __init__(
        self,
        schema: type[BaseModel],
        model_id: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 2,
        **litellm_kwargs: Any,
    ) -> None:
        """Initialize the query parser.

        Parameters:
            schema: Pydantic model defining filterable fields.
            model_id: LiteLLM-compatible model identifier.
            temperature: Sampling temperature (default 0.0).
            max_tokens: Max tokens to generate (default 1024).
            max_retries: Number of retry attempts when the LLM
                returns malformed JSON (default 2). Set to 0 to
                disable retries.
            **litellm_kwargs: Extra args for litellm calls.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                f"schema must be a Pydantic BaseModel subclass, got {type(schema)!r}"
            )
        self._schema = schema
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._litellm_kwargs = litellm_kwargs
        self._system_prompt = _build_system_prompt(schema)

        logger.info(
            "QueryParser initialised for schema=%s model=%s",
            schema.__name__,
            model_id,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def schema(self) -> type[BaseModel]:
        """The Pydantic schema used for field discovery."""
        return self._schema

    @property
    def model_id(self) -> str:
        """The LiteLLM model identifier."""
        return self._model_id

    @property
    def system_prompt(self) -> str:
        """The generated system prompt (read-only, useful for
        debugging).
        """
        return self._system_prompt

    # ------------------------------------------------------------------
    # Core parsing
    # ------------------------------------------------------------------

    def _build_messages(self, query_text: str) -> list[dict[str, str]]:
        """Build the chat messages for the LLM call.

        Parameters:
            query_text: The user's natural-language query.

        Returns:
            A list of message dicts in OpenAI chat format.
        """
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query_text},
        ]

    def _call_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for litellm calls."""
        kw: dict[str, Any] = {
            "model": self._model_id,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        kw.update(self._litellm_kwargs)
        return kw

    def parse(self, query_text: str) -> ParsedQuery:
        """Parse a natural-language query synchronously.

        Retries up to ``max_retries`` times when the LLM returns
        unparseable JSON, falling back to a low-confidence
        ``ParsedQuery`` if all attempts fail.

        Parameters:
            query_text: The user's search query.

        Returns:
            A :class:`ParsedQuery` with semantic terms,
            structured filters, confidence, and explanation.
        """
        if not query_text or not query_text.strip():
            return ParsedQuery(explanation="Empty query")

        messages = self._build_messages(query_text)
        call_kw = self._call_kwargs()
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            logger.debug(
                "Calling litellm.completion model=%s (attempt %d/%d)",
                self._model_id,
                attempt + 1,
                1 + self._max_retries,
            )
            response = litellm.completion(
                messages=messages,
                **call_kw,
            )

            content: str = response.choices[0].message.content or ""
            try:
                raw = _extract_json(content)
                return _to_parsed_query(raw)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d: failed to parse LLM response — %s",
                    attempt + 1,
                    1 + self._max_retries,
                    exc,
                )

        # All retries exhausted — return best-effort result
        logger.warning(
            "All %d parse attempts failed; returning fallback ParsedQuery",
            1 + self._max_retries,
        )
        return ParsedQuery(
            semantic_terms=[query_text],
            confidence=0.0,
            explanation=f"Failed to parse LLM response: {last_error}",
        )

    async def async_parse(self, query_text: str) -> ParsedQuery:
        """Parse a natural-language query asynchronously.

        Retries up to ``max_retries`` times when the LLM returns
        unparseable JSON, falling back to a low-confidence
        ``ParsedQuery`` if all attempts fail.

        Parameters:
            query_text: The user's search query.

        Returns:
            A :class:`ParsedQuery` with semantic terms,
            structured filters, confidence, and explanation.
        """
        if not query_text or not query_text.strip():
            return ParsedQuery(explanation="Empty query")

        messages = self._build_messages(query_text)
        call_kw = self._call_kwargs()
        last_error: Exception | None = None

        for attempt in range(1 + self._max_retries):
            logger.debug(
                "Calling litellm.acompletion model=%s (attempt %d/%d)",
                self._model_id,
                attempt + 1,
                1 + self._max_retries,
            )
            response = await litellm.acompletion(
                messages=messages,
                **call_kw,
            )

            content: str = response.choices[0].message.content or ""
            try:
                raw = _extract_json(content)
                return _to_parsed_query(raw)
            except ValueError as exc:
                last_error = exc
                logger.warning(
                    "Attempt %d/%d: failed to parse LLM response (async) — %s",
                    attempt + 1,
                    1 + self._max_retries,
                    exc,
                )

        logger.warning(
            "All %d async parse attempts failed; returning fallback ParsedQuery",
            1 + self._max_retries,
        )
        return ParsedQuery(
            semantic_terms=[query_text],
            confidence=0.0,
            explanation=f"Failed to parse LLM response: {last_error}",
        )

    def parse_sync_from_async(self, query_text: str) -> ParsedQuery:
        """Convenience wrapper: run ``async_parse`` from sync code.

        Creates a new event loop if none is running.  Prefer
        :meth:`parse` in purely synchronous code and
        :meth:`async_parse` when already inside ``async def``.

        Parameters:
            query_text: The user's search query.

        Returns:
            A :class:`ParsedQuery`.
        """
        return asyncio.run(self.async_parse(query_text))
