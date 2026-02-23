"""Comprehensive tests for langcore_rag.parser module.

All LLM calls are mocked so that no real API keys or network
access is required.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from langcore_rag import ParsedQuery, QueryParser, __version__
from langcore_rag.parser import (
    _build_system_prompt,
    _describe_fields,
    _extract_json,
    _is_filterable,
    _python_type_label,
    _to_parsed_query,
)

# ------------------------------------------------------------------
# Test schemas
# ------------------------------------------------------------------


class Invoice(BaseModel):
    """Sample schema for testing."""

    amount: float = Field(description="Total invoice amount in USD")
    due_date: str = Field(description="Due date in ISO-8601 format")
    vendor: str = Field(description="Vendor / supplier name")
    paid: bool = Field(description="Whether the invoice is paid")


class EmptyModel(BaseModel):
    """Model with no filterable fields."""

    pass


class MixedModel(BaseModel):
    """Model with both filterable and non-filterable fields."""

    name: str = Field(description="Name")
    score: float = Field(description="Score value")
    tags: list[str] = Field(default_factory=list, description="Tags")


class OptionalFields(BaseModel):
    """Model with optional fields."""

    title: str | None = Field(default=None, description="Title")
    count: int | None = Field(default=None, description="Count")


# ------------------------------------------------------------------
# Helpers for building mock LiteLLM responses
# ------------------------------------------------------------------


def _make_response(content: str) -> MagicMock:
    """Build a mock LiteLLM response object."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _json_response(obj: dict) -> MagicMock:
    """Build a mock response whose content is a JSON string."""
    return _make_response(json.dumps(obj))


# ------------------------------------------------------------------
# Tests — ParsedQuery dataclass
# ------------------------------------------------------------------


class TestParsedQuery:
    """Tests for the ParsedQuery dataclass."""

    def test_defaults(self) -> None:
        """Default values are sensible."""
        pq = ParsedQuery()
        assert pq.semantic_terms == []
        assert pq.structured_filters == {}
        assert pq.confidence == 0.0
        assert pq.explanation == ""

    def test_custom_values(self) -> None:
        """Constructor accepts all fields."""
        pq = ParsedQuery(
            semantic_terms=["invoices"],
            structured_filters={"amount": {"$gte": 5000}},
            confidence=0.95,
            explanation="Parsed amount filter.",
        )
        assert pq.semantic_terms == ["invoices"]
        assert pq.structured_filters == {"amount": {"$gte": 5000}}
        assert pq.confidence == 0.95
        assert pq.explanation == "Parsed amount filter."

    def test_frozen(self) -> None:
        """ParsedQuery instances are immutable (frozen)."""
        pq = ParsedQuery()
        with pytest.raises(FrozenInstanceError):
            pq.confidence = 0.5  # type: ignore[misc]


# ------------------------------------------------------------------
# Tests — type introspection helpers
# ------------------------------------------------------------------


class TestTypeHelpers:
    """Tests for _python_type_label and _is_filterable."""

    @pytest.mark.parametrize(
        "annotation, expected",
        [
            (int, "int"),
            (float, "float"),
            (str, "str"),
            (bool, "bool"),
        ],
    )
    def test_python_type_label_simple(self, annotation: type, expected: str) -> None:
        """Simple types produce their __name__."""
        assert _python_type_label(annotation) == expected

    def test_is_filterable_basic(self) -> None:
        """Basic filterable types return True."""
        for t in (int, float, str, bool):
            assert _is_filterable(t) is True

    def test_is_filterable_list(self) -> None:
        """list[str] is NOT filterable."""
        assert _is_filterable(list[str]) is False

    def test_is_filterable_optional(self) -> None:
        """Optional[int] IS filterable."""
        assert _is_filterable(int | None) is True

    def test_is_filterable_optional_non_filterable(
        self,
    ) -> None:
        """Optional[list[str]] is NOT filterable."""
        assert _is_filterable(list[str] | None) is False


# ------------------------------------------------------------------
# Tests — _describe_fields
# ------------------------------------------------------------------


class TestDescribeFields:
    """Tests for _describe_fields."""

    def test_invoice_schema(self) -> None:
        """Invoice schema produces four filterable field lines."""
        desc = _describe_fields(Invoice)
        assert "amount (float)" in desc
        assert "due_date (str)" in desc
        assert "vendor (str)" in desc
        assert "paid (bool)" in desc

    def test_empty_schema(self) -> None:
        """Empty model produces the fallback text."""
        desc = _describe_fields(EmptyModel)
        assert desc == "(no filterable fields)"

    def test_mixed_schema_excludes_list(self) -> None:
        """list[str] fields are excluded from description."""
        desc = _describe_fields(MixedModel)
        assert "name (str)" in desc
        assert "score (float)" in desc
        assert "tags" not in desc

    def test_optional_fields(self) -> None:
        """Optional fields are listed with their inner type."""
        desc = _describe_fields(OptionalFields)
        assert "title" in desc
        assert "count" in desc


# ------------------------------------------------------------------
# Tests — _build_system_prompt
# ------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt."""

    def test_contains_field_names(self) -> None:
        """The prompt includes field descriptions."""
        prompt = _build_system_prompt(Invoice)
        assert "amount" in prompt
        assert "due_date" in prompt
        assert "semantic_terms" in prompt
        assert "structured_filters" in prompt

    def test_empty_schema_includes_fallback(self) -> None:
        """An empty schema still generates a valid prompt."""
        prompt = _build_system_prompt(EmptyModel)
        assert "(no filterable fields)" in prompt


# ------------------------------------------------------------------
# Tests — _extract_json
# ------------------------------------------------------------------


class TestExtractJson:
    """Tests for _extract_json."""

    def test_plain_json(self) -> None:
        """Plain JSON string parses correctly."""
        raw = '{"semantic_terms": ["test"], "confidence": 0.9}'
        result = _extract_json(raw)
        assert result["semantic_terms"] == ["test"]

    def test_json_in_code_fence(self) -> None:
        """JSON inside markdown fences is extracted."""
        raw = '```json\n{"a": 1}\n```'
        result = _extract_json(raw)
        assert result == {"a": 1}

    def test_code_fence_no_language(self) -> None:
        """Code fences without ``json`` tag also work."""
        raw = '```\n{"b": 2}\n```'
        result = _extract_json(raw)
        assert result == {"b": 2}

    def test_invalid_json_raises(self) -> None:
        """Non-JSON text raises ValueError."""
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_json("this is not json at all")

    def test_json_array_raises(self) -> None:
        """A JSON array (not object) raises ValueError."""
        with pytest.raises(ValueError, match="Could not extract"):
            _extract_json("[1, 2, 3]")

    def test_whitespace_padding(self) -> None:
        """Leading/trailing whitespace is handled."""
        raw = '  \n  {"x": 42}  \n  '
        result = _extract_json(raw)
        assert result == {"x": 42}

    def test_text_before_fence(self) -> None:
        """Text before a code fence is ignored."""
        raw = 'Here is the result:\n```json\n{"k": "v"}\n```'
        result = _extract_json(raw)
        assert result == {"k": "v"}


# ------------------------------------------------------------------
# Tests — _to_parsed_query
# ------------------------------------------------------------------


class TestToParsedQuery:
    """Tests for _to_parsed_query."""

    def test_full_dict(self) -> None:
        """A well-formed dict converts correctly."""
        pq = _to_parsed_query(
            {
                "semantic_terms": ["invoices"],
                "structured_filters": {"amount": {"$gte": 5000}},
                "confidence": 0.95,
                "explanation": "ok",
            }
        )
        assert pq.semantic_terms == ["invoices"]
        assert pq.structured_filters == {"amount": {"$gte": 5000}}
        assert pq.confidence == 0.95
        assert pq.explanation == "ok"

    def test_missing_keys(self) -> None:
        """Missing keys default gracefully."""
        pq = _to_parsed_query({})
        assert pq.semantic_terms == []
        assert pq.structured_filters == {}
        assert pq.confidence == 0.0
        assert pq.explanation == ""

    def test_confidence_clamped_high(self) -> None:
        """Confidence > 1.0 is clamped to 1.0."""
        pq = _to_parsed_query({"confidence": 5.0})
        assert pq.confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        """Confidence < 0.0 is clamped to 0.0."""
        pq = _to_parsed_query({"confidence": -3.0})
        assert pq.confidence == 0.0

    def test_confidence_non_numeric(self) -> None:
        """Non-numeric confidence falls back to 0.0."""
        pq = _to_parsed_query({"confidence": "high"})
        assert pq.confidence == 0.0

    def test_semantic_terms_coerced(self) -> None:
        """Non-list semantic_terms is wrapped in a list."""
        pq = _to_parsed_query({"semantic_terms": "hello"})
        assert pq.semantic_terms == ["hello"]

    def test_filters_non_dict(self) -> None:
        """Non-dict structured_filters defaults to {}."""
        pq = _to_parsed_query({"structured_filters": "invalid"})
        assert pq.structured_filters == {}


# ------------------------------------------------------------------
# Tests — QueryParser.__init__
# ------------------------------------------------------------------


class TestQueryParserInit:
    """Tests for QueryParser construction."""

    def test_valid_init(self) -> None:
        """Instantiation with a valid schema succeeds."""
        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        assert parser.schema is Invoice
        assert parser.model_id == "gpt-4o"

    def test_invalid_schema_raises(self) -> None:
        """Passing a non-BaseModel raises TypeError."""
        with pytest.raises(TypeError, match="BaseModel subclass"):
            QueryParser(schema=dict, model_id="gpt-4o")  # type: ignore[arg-type]

    def test_system_prompt_populated(self) -> None:
        """The system prompt is generated at init time."""
        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        assert "amount" in parser.system_prompt
        assert "semantic_terms" in parser.system_prompt

    def test_custom_temperature(self) -> None:
        """Temperature kwarg is stored."""
        parser = QueryParser(
            schema=Invoice,
            model_id="gpt-4o",
            temperature=0.7,
        )
        kw = parser._call_kwargs()
        assert kw["temperature"] == 0.7

    def test_extra_kwargs_forwarded(self) -> None:
        """Extra litellm kwargs are forwarded in _call_kwargs."""
        parser = QueryParser(
            schema=Invoice,
            model_id="gpt-4o",
            api_key="sk-test",
            timeout=30,
        )
        kw = parser._call_kwargs()
        assert kw["api_key"] == "sk-test"
        assert kw["timeout"] == 30


# ------------------------------------------------------------------
# Tests — QueryParser.parse (sync)
# ------------------------------------------------------------------


class TestParseSynchronous:
    """Tests for the synchronous parse() method."""

    @patch("langcore_rag.parser.litellm")
    def test_basic_parse(self, mock_litellm: MagicMock) -> None:
        """A standard LLM response is parsed correctly."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": ["invoices"],
                "structured_filters": {
                    "amount": {"$gte": 5000},
                    "due_date": {
                        "$gte": "2024-03-01",
                        "$lte": "2024-03-31",
                    },
                },
                "confidence": 0.92,
                "explanation": "Extracted amount and date filters.",
            }
        )

        parser = QueryParser(schema=Invoice, model_id="gemini-2.5-flash")
        result = parser.parse("invoices over $5000 due in March 2024")

        assert result.semantic_terms == ["invoices"]
        assert result.structured_filters["amount"] == {"$gte": 5000}
        assert result.confidence == pytest.approx(0.92)
        mock_litellm.completion.assert_called_once()

    @patch("langcore_rag.parser.litellm")
    def test_empty_query_returns_empty(self, mock_litellm: MagicMock) -> None:
        """An empty query returns an empty ParsedQuery."""
        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        result = parser.parse("")

        assert result.semantic_terms == []
        assert result.structured_filters == {}
        assert result.explanation == "Empty query"
        mock_litellm.completion.assert_not_called()

    @patch("langcore_rag.parser.litellm")
    def test_whitespace_query_returns_empty(self, mock_litellm: MagicMock) -> None:
        """A whitespace-only query returns an empty ParsedQuery."""
        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        result = parser.parse("   \n  ")

        assert result.explanation == "Empty query"
        mock_litellm.completion.assert_not_called()

    @patch("langcore_rag.parser.litellm")
    def test_parse_with_code_fence_response(self, mock_litellm: MagicMock) -> None:
        """LLM response wrapped in code fences is handled."""
        content = (
            "Here is the result:\n```json\n"
            '{"semantic_terms": ["test"], '
            '"structured_filters": {}, '
            '"confidence": 0.8, '
            '"explanation": "No filters."}\n```'
        )
        mock_litellm.completion.return_value = _make_response(content)

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        result = parser.parse("test query")

        assert result.semantic_terms == ["test"]
        assert result.confidence == pytest.approx(0.8)

    @patch("langcore_rag.parser.litellm")
    def test_parse_invalid_json_raises(self, mock_litellm: MagicMock) -> None:
        """An unparseable LLM response raises ValueError."""
        mock_litellm.completion.return_value = _make_response(
            "I don't know how to parse that."
        )

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        with pytest.raises(ValueError, match="Could not extract"):
            parser.parse("some query")

    @patch("langcore_rag.parser.litellm")
    def test_parse_no_filters(self, mock_litellm: MagicMock) -> None:
        """A query with no filterable terms has empty filters."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": ["machine", "learning"],
                "structured_filters": {},
                "confidence": 0.7,
                "explanation": "No fields match.",
            }
        )

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        result = parser.parse("machine learning papers")

        assert result.semantic_terms == [
            "machine",
            "learning",
        ]
        assert result.structured_filters == {}

    @patch("langcore_rag.parser.litellm")
    def test_messages_contain_system_and_user(self, mock_litellm: MagicMock) -> None:
        """The messages list has system + user messages."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": [],
                "structured_filters": {},
                "confidence": 0.5,
                "explanation": "",
            }
        )

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        parser.parse("hello")

        call_kw = mock_litellm.completion.call_args
        messages = call_kw.kwargs.get(
            "messages", call_kw.args[0] if call_kw.args else None
        )
        # Try keyword first, then positional
        if messages is None:
            messages = call_kw[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "hello"


# ------------------------------------------------------------------
# Tests — QueryParser.async_parse
# ------------------------------------------------------------------


class TestParseAsync:
    """Tests for the asynchronous async_parse() method."""

    @patch("langcore_rag.parser.litellm")
    async def test_async_parse_basic(self, mock_litellm: MagicMock) -> None:
        """Basic async parse works."""
        mock_litellm.acompletion = AsyncMock(
            return_value=_json_response(
                {
                    "semantic_terms": ["contracts"],
                    "structured_filters": {"vendor": {"$eq": "Acme"}},
                    "confidence": 0.88,
                    "explanation": "Filtered by vendor.",
                }
            )
        )

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        result = await parser.async_parse("contracts from Acme")

        assert result.semantic_terms == ["contracts"]
        assert result.structured_filters == {"vendor": {"$eq": "Acme"}}
        assert result.confidence == pytest.approx(0.88)
        mock_litellm.acompletion.assert_awaited_once()

    @patch("langcore_rag.parser.litellm")
    async def test_async_empty_query(self, mock_litellm: MagicMock) -> None:
        """An empty query returns early without LLM call."""
        mock_litellm.acompletion = AsyncMock()

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        result = await parser.async_parse("")

        assert result.explanation == "Empty query"
        mock_litellm.acompletion.assert_not_awaited()

    @patch("langcore_rag.parser.litellm")
    async def test_async_invalid_json_raises(self, mock_litellm: MagicMock) -> None:
        """Async parse with bad JSON raises ValueError."""
        mock_litellm.acompletion = AsyncMock(return_value=_make_response("not json"))

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        with pytest.raises(ValueError, match="Could not extract"):
            await parser.async_parse("test")


# ------------------------------------------------------------------
# Tests — edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case coverage."""

    def test_version_string(self) -> None:
        """__version__ follows semver."""
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    @patch("langcore_rag.parser.litellm")
    def test_empty_schema(self, mock_litellm: MagicMock) -> None:
        """A schema with no fields still works."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": ["everything"],
                "structured_filters": {},
                "confidence": 0.5,
                "explanation": "No filterable fields.",
            }
        )

        parser = QueryParser(schema=EmptyModel, model_id="gpt-4o")
        result = parser.parse("show me everything")
        assert result.semantic_terms == ["everything"]

    @patch("langcore_rag.parser.litellm")
    def test_optional_field_schema(self, mock_litellm: MagicMock) -> None:
        """Optional fields appear in the prompt and work."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": ["docs"],
                "structured_filters": {"count": {"$gte": 10}},
                "confidence": 0.85,
                "explanation": "Filtered by count.",
            }
        )

        parser = QueryParser(schema=OptionalFields, model_id="gpt-4o")
        assert "title" in parser.system_prompt
        assert "count" in parser.system_prompt

        result = parser.parse("docs with count >= 10")
        assert result.structured_filters == {"count": {"$gte": 10}}

    @patch("langcore_rag.parser.litellm")
    def test_llm_returns_empty_content(self, mock_litellm: MagicMock) -> None:
        """An empty LLM response raises ValueError."""
        mock_litellm.completion.return_value = _make_response("")

        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        with pytest.raises(ValueError):
            parser.parse("test")

    @patch("langcore_rag.parser.litellm")
    def test_model_id_forwarded(self, mock_litellm: MagicMock) -> None:
        """The model_id is forwarded to litellm.completion."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": [],
                "structured_filters": {},
                "confidence": 0.5,
                "explanation": "",
            }
        )

        parser = QueryParser(
            schema=Invoice,
            model_id="anthropic/claude-3-opus",
        )
        parser.parse("test")

        call_kw = mock_litellm.completion.call_args
        assert call_kw.kwargs["model"] == ("anthropic/claude-3-opus")

    @patch("langcore_rag.parser.litellm")
    def test_temperature_forwarded(self, mock_litellm: MagicMock) -> None:
        """Custom temperature is fowarded to litellm."""
        mock_litellm.completion.return_value = _json_response(
            {
                "semantic_terms": [],
                "structured_filters": {},
                "confidence": 0.5,
                "explanation": "",
            }
        )

        parser = QueryParser(
            schema=Invoice,
            model_id="gpt-4o",
            temperature=0.3,
        )
        parser.parse("test")

        call_kw = mock_litellm.completion.call_args
        assert call_kw.kwargs["temperature"] == 0.3

    def test_build_messages(self) -> None:
        """_build_messages produces system and user roles."""
        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        msgs = parser._build_messages("hello world")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello world"

    def test_call_kwargs_defaults(self) -> None:
        """_call_kwargs includes model, temperature, max_tokens."""
        parser = QueryParser(schema=Invoice, model_id="gpt-4o")
        kw = parser._call_kwargs()
        assert kw["model"] == "gpt-4o"
        assert kw["temperature"] == 0.0
        assert kw["max_tokens"] == 1024
