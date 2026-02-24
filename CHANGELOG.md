# CHANGELOG

<!-- version list -->

## v1.2.0 (2026-02-24)

### Features

- Implement LRU caching in QueryParser and add related tests
  ([`7fbc7cc`](https://github.com/IgnatG/langcore-rag/commit/7fbc7cc92b2e3504190d4fb87606cc745d661ea8))


## v1.1.0 (2026-02-24)

### Documentation

- Update README for clarity and structure; enhance installation and feature sections
  ([`6bc5fa5`](https://github.com/IgnatG/langcore-rag/commit/6bc5fa5cbd9679c4160f8b56e3be2f1a55e4a1ba))

### Features

- Add max_retries parameter to QueryParser for improved LLM response handling
  ([`65ac6da`](https://github.com/IgnatG/langcore-rag/commit/65ac6da84b159d474e21e2e849059639bea9ad8e))


## v1.0.4 (2026-02-23)

### Bug Fixes

- Update Python version in CI and refine langcore dependency in project files
  ([`876e634`](https://github.com/IgnatG/langcore-rag/commit/876e6348a1f7b24df24d4374e7b239fec6ab5827))


## v1.0.3 (2026-02-23)

### Bug Fixes

- Updated the folder structure
  ([`6206dcc`](https://github.com/IgnatG/langcore-rag/commit/6206dcc74fcc661d91bf31a673823b49560f68d1))


## v1.0.2 (2026-02-23)

### Bug Fixes

- Enhance error handling in QueryParser to return fallback ParsedQuery on invalid JSON responses
  ([`ab1ca3f`](https://github.com/IgnatG/langcore-rag/commit/ab1ca3fafa8a6777730b3c9f3612c2735275b9c5))


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
