# CHANGELOG

<!-- version list -->

## v1.0.1 (2026-02-23)

### Bug Fixes

- Update type hints and improve code clarity in parser and tests
  ([`305f64d`](https://github.com/IgnatG/langcore-rag/commit/305f64d15e23d5e69a9eb8364838320dac59853e))


## v1.0.0 (2026-02-23)

- Initial Release

## v1.0.0 (2025-07-22)

### Features

- **`QueryParser` class** — decomposes natural-language queries into semantic terms and structured metadata filters using an LLM
  - `parse(query_text) -> ParsedQuery` (synchronous)
  - `async_parse(query_text) -> ParsedQuery` (asynchronous)
  - Automatic Pydantic schema introspection for filterable field discovery
  - MongoDB-style filter operators (`$eq`, `$gte`, `$lte`, `$in`, etc.)
- **`ParsedQuery` dataclass** — immutable result with `semantic_terms`, `structured_filters`, `confidence`, and `explanation`
- **LiteLLM integration** — supports 100+ model providers out of the box
- **Robust JSON extraction** — handles raw JSON, Markdown code fences, and partial LLM responses with graceful degradation
- **Comprehensive test suite** — 54 tests covering schema introspection, JSON extraction, type coercion, sync/async parsing, and edge cases
