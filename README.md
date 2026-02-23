# LangCore RAG — Query Parsing for Hybrid Retrieval

A plugin for [LangExtract](https://github.com/google/langextract) that parses natural-language queries into **semantic terms** (for vector search) and **structured metadata filters** (for database / index filtering), enabling hybrid RAG retrieval pipelines. Inspired by [LangStruct](https://github.com/langstruct/langstruct)'s `.query()` method.

> **Note**: This is a third-party plugin for LangExtract. For the main LangExtract library, visit [google/langextract](https://github.com/google/langextract).

## Installation

Install from source:

```bash
git clone <repo-url>
cd langcore-rag
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

## Features at a Glance

| Feature | langcore-rag | LangStruct |
|---|---|---|
| **Query → semantic terms + filters** | ✅ `QueryParser.parse()` | ✅ `.query()` |
| **Async support** | ✅ `async_parse()` | ✅ |
| **Pydantic schema introspection** | ✅ Auto-discovers filterable fields | ✅ |
| **MongoDB-style operators** | ✅ `$eq`, `$gte`, `$lte`, `$in`, `$nin`, etc. | ✅ |
| **Confidence score** | ✅ 0.0 – 1.0 | ❌ |
| **Explanation / rationale** | ✅ Human-readable | ❌ |
| **Any LLM backend** | ✅ Via LiteLLM (100+ providers) | ✅ |
| **Robust JSON parsing** | ✅ Raw JSON + Markdown fences + graceful fallback | ⚠️ |

## Quick Start

### 1. Define a Schema

```python
from pydantic import BaseModel, Field

class Invoice(BaseModel):
    amount: float = Field(description="Total invoice amount in USD")
    due_date: str = Field(description="Due date in ISO-8601 format")
    vendor: str = Field(description="Vendor / supplier name")
    paid: bool = Field(description="Whether the invoice is paid")
```

### 2. Parse a Query

```python
from langcore_rag import QueryParser

parser = QueryParser(schema=Invoice, model_id="gemini/gemini-2.5-flash")
parsed = parser.parse("invoices over $5000 due in March 2024")

print(parsed.semantic_terms)
# → ["invoices"]

print(parsed.structured_filters)
# → {"amount": {"$gte": 5000}, "due_date": {"$gte": "2024-03-01", "$lte": "2024-03-31"}}

print(parsed.confidence)
# → 0.92

print(parsed.explanation)
# → "Extracted amount ≥ 5000 and date range for March 2024."
```

### 3. Async Usage

```python
import asyncio
from langcore_rag import QueryParser

async def main():
    parser = QueryParser(schema=Invoice, model_id="gpt-4o")
    parsed = await parser.async_parse("unpaid invoices from Acme Corp")
    print(parsed.structured_filters)
    # → {"paid": {"$eq": false}, "vendor": {"$eq": "Acme Corp"}}

asyncio.run(main())
```

## API Reference

### `QueryParser`

```python
QueryParser(
    schema: type[BaseModel],
    model_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **litellm_kwargs,
)
```

| Parameter | Type | Description |
|---|---|---|
| `schema` | `type[BaseModel]` | Pydantic model whose fields define filterable metadata |
| `model_id` | `str` | Any LiteLLM-compatible model ID (e.g. `"gpt-4o"`, `"gemini/gemini-2.5-flash"`, `"anthropic/claude-3-opus"`) |
| `temperature` | `float` | Sampling temperature (default `0.0` for deterministic output) |
| `max_tokens` | `int` | Maximum tokens to generate (default `1024`) |
| `**litellm_kwargs` | | Extra kwargs forwarded to `litellm.completion()` (e.g. `api_key`, `api_base`, `timeout`) |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `parse` | `(query_text: str) -> ParsedQuery` | Synchronous query parsing |
| `async_parse` | `(query_text: str) -> ParsedQuery` | Asynchronous query parsing |

#### Properties

| Property | Type | Description |
|---|---|---|
| `schema` | `type[BaseModel]` | The Pydantic schema used for field discovery |
| `model_id` | `str` | The LiteLLM model identifier |
| `system_prompt` | `str` | The generated system prompt (useful for debugging) |

### `ParsedQuery`

An immutable (frozen) dataclass returned by `parse()` / `async_parse()`.

| Field | Type | Description |
|---|---|---|
| `semantic_terms` | `list[str]` | Free-text terms for vector / similarity search |
| `structured_filters` | `dict[str, Any]` | Metadata filters with MongoDB-style operators |
| `confidence` | `float` | 0.0 – 1.0 confidence in the parse quality |
| `explanation` | `str` | Human-readable rationale for the decomposition |

## How It Works

1. **Schema introspection** — `QueryParser` inspects the Pydantic model's fields to identify which ones are scalar/filterable (`int`, `float`, `str`, `bool`, `date`, `datetime`). Complex types like `list[str]` are excluded.

2. **System prompt generation** — A system prompt is built listing the filterable fields with their types and descriptions, instructing the LLM to output a JSON object with `semantic_terms`, `structured_filters`, `confidence`, and `explanation`.

3. **LLM call** — The query text is sent as a user message alongside the system prompt via `litellm.completion()` (sync) or `litellm.acompletion()` (async).

4. **Response parsing** — The LLM's text response is parsed as JSON (handling both raw JSON and Markdown code fences). Values are type-coerced and clamped to produce a valid `ParsedQuery`.

## Supported Filter Operators

The parser instructs the LLM to use MongoDB-style operators:

| Operator | Meaning | Example |
|---|---|---|
| `$eq` | Equals | `{"vendor": {"$eq": "Acme"}}` |
| `$ne` | Not equals | `{"paid": {"$ne": true}}` |
| `$gt` | Greater than | `{"amount": {"$gt": 1000}}` |
| `$gte` | Greater than or equal | `{"amount": {"$gte": 5000}}` |
| `$lt` | Less than | `{"amount": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"due_date": {"$lte": "2024-12-31"}}` |
| `$in` | In list | `{"vendor": {"$in": ["Acme", "Globex"]}}` |
| `$nin` | Not in list | `{"vendor": {"$nin": ["Initech"]}}` |

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check langcore_rag/ tests/

# Format
uv run ruff format langcore_rag/ tests/
```

## License

[Apache 2.0](LICENSE)
