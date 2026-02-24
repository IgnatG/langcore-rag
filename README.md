# LangCore RAG

> Plugin for [LangCore](https://github.com/ignatg/langcore) — parse natural-language queries into semantic search terms and structured metadata filters for hybrid RAG pipelines.

[![PyPI version](https://img.shields.io/pypi/v/langcore-rag)](https://pypi.org/project/langcore-rag/)
[![Python](https://img.shields.io/pypi/pyversions/langcore-rag)](https://pypi.org/project/langcore-rag/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

**langcore-rag** is a plugin for [LangCore](https://github.com/ignatg/langcore) that decomposes natural-language queries into **semantic terms** (for vector/similarity search) and **structured metadata filters** (for database or index filtering). It introspects your Pydantic schema to auto-discover filterable fields, calls an LLM to parse the query, and returns MongoDB-style filter operators ready for your retrieval backend.

---

## Features

- **Query decomposition** — splits free-form queries into semantic search terms and structured filter conditions
- **Pydantic schema introspection** — automatically discovers filterable fields (`int`, `float`, `str`, `bool`, `date`, `datetime`) from your schema
- **MongoDB-style operators** — `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin` for precise filter generation
- **Confidence scoring** — 0.0–1.0 confidence score indicating parse quality
- **Human-readable explanation** — rationale for how the query was decomposed
- **Sync and async** — both `parse()` and `async_parse()` methods
- **Robust JSON parsing** — handles raw JSON, Markdown fences, and graceful fallback
- **Any LLM backend** — uses LiteLLM for access to 100+ model providers
- **Zero manual prompt engineering** — system prompt is auto-generated from your schema

---

## Installation

```bash
pip install langcore-rag
```

---

## Quick Start

### 1. Define a Schema

Define a Pydantic model whose fields represent the filterable metadata in your document store:

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

### 3. Use in a RAG Pipeline

Feed the parsed output into your vector store and metadata filter layer:

```python
from langcore_rag import QueryParser

parser = QueryParser(schema=Invoice, model_id="gpt-4o")
parsed = parser.parse("unpaid invoices from Acme Corp over $10,000")

# Semantic search with your vector store
vector_results = vector_store.similarity_search(
    query=" ".join(parsed.semantic_terms),
    k=20,
)

# Apply structured filters to narrow results
filtered = [
    doc for doc in vector_results
    if apply_filters(doc.metadata, parsed.structured_filters)
]
```

### 4. Async Usage

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

---

## Integration with LangCore

langcore-rag uses LangCore's LLM ecosystem (via LiteLLM) for query parsing. It works with any model supported by LiteLLM:

```python
from langcore_rag import QueryParser

# Use any LiteLLM-compatible model
parser = QueryParser(
    schema=Invoice,
    model_id="gpt-4o",          # or "gemini/gemini-2.5-flash", "anthropic/claude-3-opus", etc.
    temperature=0.0,             # Deterministic output
    max_tokens=1024,
    api_key="sk-...",            # Optional — override env var
)
```

When deployed via **langcore-api**, the RAG parser is available as a REST endpoint (`POST /api/v1/rag/parse`) with full configuration via environment variables.

---

## API Reference

### QueryParser

```python
QueryParser(
    schema: type[BaseModel],
    model_id: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 2,
    **litellm_kwargs,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `type[BaseModel]` | Pydantic model whose fields define filterable metadata |
| `model_id` | `str` | Any LiteLLM-compatible model ID |
| `temperature` | `float` | Sampling temperature (default `0.0` for deterministic output) |
| `max_tokens` | `int` | Maximum tokens to generate (default `1024`) |
| `max_retries` | `int` | Number of retry attempts on malformed LLM responses (default `2`, meaning 3 total attempts) |
| `**litellm_kwargs` | | Extra kwargs forwarded to `litellm.completion()` (e.g., `api_key`, `api_base`, `timeout`) |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `parse` | `(query_text: str) -> ParsedQuery` | Synchronous query parsing |
| `async_parse` | `(query_text: str) -> ParsedQuery` | Asynchronous query parsing |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `schema` | `type[BaseModel]` | The Pydantic schema used for field discovery |
| `model_id` | `str` | The LiteLLM model identifier |
| `system_prompt` | `str` | The auto-generated system prompt (useful for debugging) |

### ParsedQuery

An immutable (frozen) dataclass returned by `parse()` / `async_parse()`:

| Field | Type | Description |
|-------|------|-------------|
| `semantic_terms` | `list[str]` | Free-text terms for vector / similarity search |
| `structured_filters` | `dict[str, Any]` | Metadata filters with MongoDB-style operators |
| `confidence` | `float` | 0.0–1.0 confidence in the parse quality |
| `explanation` | `str` | Human-readable rationale for the decomposition |

---

## Supported Filter Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{"vendor": {"$eq": "Acme"}}` |
| `$ne` | Not equals | `{"paid": {"$ne": true}}` |
| `$gt` | Greater than | `{"amount": {"$gt": 1000}}` |
| `$gte` | Greater than or equal | `{"amount": {"$gte": 5000}}` |
| `$lt` | Less than | `{"amount": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"due_date": {"$lte": "2024-12-31"}}` |
| `$in` | In list | `{"vendor": {"$in": ["Acme", "Globex"]}}` |
| `$nin` | Not in list | `{"vendor": {"$nin": ["Initech"]}}` |

---

## How It Works

1. **Schema introspection** — inspects the Pydantic model's fields to identify filterable types (`int`, `float`, `str`, `bool`, `date`, `datetime`). Complex types like `list[str]` are excluded.
2. **System prompt generation** — builds a prompt listing filterable fields with types and descriptions, instructing the LLM to output structured JSON.
3. **LLM call** — sends the query as a user message with the system prompt via `litellm.completion()` or `litellm.acompletion()`.
4. **Response parsing** — parses the response as JSON (handling fences and edge cases), type-coerces values, and clamps confidence to produce a valid `ParsedQuery`.
5. **Retry on failure** — if the LLM returns malformed JSON, the parser retries up to `max_retries` times (default 2, so 3 total attempts). Each retry is logged.
6. **Graceful fallback** — if all retries are exhausted, returns a `ParsedQuery(semantic_terms=[query_text], structured_filters={}, confidence=0.0)` so callers always receive a usable result.

---

## Composing with Other Plugins

langcore-rag complements the extraction plugins. Use it to find relevant documents, then extract structured data:

```python
import langcore as lx
from langcore_rag import QueryParser

# Step 1: Parse the user's query
parser = QueryParser(schema=Invoice, model_id="gpt-4o")
parsed = parser.parse("invoices from Acme over $5000")

# Step 2: Retrieve relevant documents from your store
docs = document_store.search(
    query=parsed.semantic_terms,
    filters=parsed.structured_filters,
)

# Step 3: Extract structured entities from retrieved documents
for doc in docs:
    result = lx.extract(
        text_or_documents=doc.text,
        model_id="gemini-2.5-flash",
        prompt_description="Extract invoice details.",
        examples=[...],
    )
    print(result)
```

---

## Development

```bash
uv sync                                    # Install dependencies
uv run pytest tests/ -v                    # Run tests
uv run ruff check langcore_rag/ tests/     # Lint
uv run ruff format langcore_rag/ tests/    # Format
```

## Requirements

- Python ≥ 3.12
- `langcore`
- `litellm` ≥ 1.81.13
- `pydantic` ≥ 2.12.0

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
