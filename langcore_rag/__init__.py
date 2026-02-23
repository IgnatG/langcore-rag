"""langcore-rag — RAG query parsing for LangExtract.

Decomposes natural-language queries into semantic terms and
structured metadata filters using an LLM, enabling hybrid
vector + metadata retrieval in RAG pipelines.
"""

from langcore_rag.parser import ParsedQuery, QueryParser

__all__ = [
    "ParsedQuery",
    "QueryParser",
]
__version__ = "1.0.0"
