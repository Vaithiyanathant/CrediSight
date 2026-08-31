"""Local RAG index over the project's own documentation.

Embeddings come from sentence-transformers + FAISS when both are installed. When
they are not — which is the default in a 4 GiB container, where a 90 MB model
plus torch is a poor trade for a corpus of a few hundred chunks — the index
falls back to deterministic TF-IDF cosine retrieval from scikit-learn.

That fallback is a design decision, not a degradation: on a corpus this small
lexical retrieval over a controlled vocabulary of field names and rule IDs is
competitive, it adds no model download, no warm-up, and no nondeterminism, and
every retrieved passage still carries its citation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lpie.core.config import Settings, get_settings
from lpie.core.determinism import sha256_obj
from lpie.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    source_path: str
    kind: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "citation": self.citation(),
            "source_path": self.source_path,
            "kind": self.kind,
            "text": self.text,
            "metadata": self.metadata,
        }


def _split_markdown(text: str, doc_id: str, path: str, chunk_chars: int, overlap: int) -> list[Chunk]:
    """Split on headings first, then on size — so a field's row stays with its table."""
    sections: list[tuple[str, str]] = []
    current_title = doc_id
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if buffer:
                sections.append((current_title, "\n".join(buffer)))
                buffer = []
            current_title = line.lstrip("#").strip() or doc_id
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_title, "\n".join(buffer)))

    chunks: list[Chunk] = []
    for title, body in sections:
        body = body.strip()
        if not body:
            continue
        start = 0
        part = 0
        while start < len(body):
            piece = body[start : start + chunk_chars]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{_slug(title)}-{part}",
                    title=title,
                    text=piece.strip(),
                    source_path=path,
                    kind="markdown",
                )
            )
            part += 1
            start += max(chunk_chars - overlap, 1)
    return chunks


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "section"


def build_corpus(settings: Settings | None = None) -> list[Chunk]:
    """Index the data dictionary, the rules, the model card, and generated reports."""
    s = settings or get_settings()
    cfg = s.section("copilot").get("rag", {})
    chunk_chars = int(cfg.get("chunk_chars", 900))
    overlap = int(cfg.get("chunk_overlap", 150))

    chunks: list[Chunk] = []

    dictionary = s.dataset_file("data_dictionary")
    if dictionary.exists():
        chunks.extend(
            _split_markdown(dictionary.read_text(), "data_dictionary", str(dictionary), chunk_chars, overlap)
        )

    rules_path = s.root / "config" / "validation_rules.json"
    if not rules_path.exists():
        rules_path = s.dataset_file("validation_rules")
    if rules_path.exists():
        spec = json.loads(rules_path.read_text())
        for rule in spec.get("rules", []):
            chunks.append(
                Chunk(
                    doc_id="validation_rules",
                    chunk_id=rule["rule_id"],
                    title=f"{rule['rule_id']} {rule['name']}",
                    text=(
                        f"{rule['rule_id']} ({rule['name']}, severity {rule['severity']}, "
                        f"dimension {rule.get('dimension', 'n/a')}, exception type "
                        f"{rule['exception_type']}): {rule['description']} "
                        f"Condition: {rule.get('condition', '')}"
                    ),
                    source_path=str(rules_path),
                    kind="rule",
                    metadata={"rule_id": rule["rule_id"], "severity": rule["severity"]},
                )
            )

    for name in ("MODEL_CARD.md", "FEATURE_CONTRACT.md", "README.md", "SYSTEM_DESIGN.md"):
        path = s.root / name
        if path.exists():
            chunks.extend(
                _split_markdown(path.read_text(), path.stem.lower(), str(path), chunk_chars, overlap)
            )

    reports = s.path("reports_dir")
    if reports.exists():
        for path in sorted(reports.glob("*.md")):
            chunks.extend(
                _split_markdown(path.read_text(), f"report:{path.stem}", str(path), chunk_chars, overlap)
            )
        for path in sorted(reports.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            chunks.append(
                Chunk(
                    doc_id=f"artifact:{path.stem}",
                    chunk_id="summary",
                    title=path.stem,
                    text=json.dumps(payload, default=str)[:4000],
                    source_path=str(path),
                    kind="artifact",
                )
            )

    log.info("rag.corpus_built", n_chunks=len(chunks))
    return chunks


class RAGIndex:
    """Embedding index with a deterministic TF-IDF fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.section("copilot").get("rag", {})
        self.top_k = int(cfg.get("top_k", 6))
        self.embedder_name = str(cfg.get("embedder", ""))
        self.chunks: list[Chunk] = []
        self.backend: str = "none"
        self._vectoriser = None
        self._matrix = None
        self._faiss = None
        self._embedder = None
        self.corpus_hash: str = ""

    # ------------------------------------------------------------------ #
    def build(self, chunks: list[Chunk] | None = None, *, prefer_embeddings: bool = False) -> str:
        self.chunks = chunks if chunks is not None else build_corpus(self.settings)
        self.corpus_hash = sha256_obj([c.citation() for c in self.chunks])
        if not self.chunks:
            self.backend = "empty"
            return self.backend

        texts = [f"{c.title}\n{c.text}" for c in self.chunks]

        if prefer_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embedder_name)
                vectors = np.asarray(
                    self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                    dtype="float32",
                )
                import faiss

                index = faiss.IndexFlatIP(vectors.shape[1])
                index.add(vectors)
                self._faiss = index
                self.backend = f"faiss+{self.embedder_name}"
                log.info("rag.index_built", backend=self.backend, n=len(self.chunks))
                return self.backend
            except Exception as exc:
                log.warning("rag.embedding_unavailable", error=str(exc))

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectoriser = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\-\.]+\b",
        )
        self._matrix = self._vectoriser.fit_transform(texts)
        self.backend = "tfidf"
        log.info("rag.index_built", backend=self.backend, n=len(self.chunks))
        return self.backend

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        k = k or self.top_k
        if not self.chunks:
            return []

        if self._faiss is not None and self._embedder is not None:
            vector = np.asarray(
                self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )
            scores, indices = self._faiss.search(vector, min(k, len(self.chunks)))
            pairs = list(zip(indices[0].tolist(), scores[0].tolist(), strict=False))
        elif self._matrix is not None and self._vectoriser is not None:
            from sklearn.metrics.pairwise import linear_kernel

            vector = self._vectoriser.transform([query])
            similarities = linear_kernel(vector, self._matrix).ravel()
            order = np.argsort(-similarities)[:k]
            pairs = [(int(i), float(similarities[i])) for i in order]
        else:
            return []

        return [
            {**self.chunks[i].to_dict(), "score": round(float(score), 6)}
            for i, score in pairs
            if 0 <= i < len(self.chunks) and score > 0
        ]

    @property
    def is_built(self) -> bool:
        return bool(self.chunks) and self.backend not in ("none", "empty")

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_chunks": len(self.chunks),
            "documents": sorted({c.doc_id for c in self.chunks}),
            "corpus_hash": self.corpus_hash[:16] if self.corpus_hash else None,
            "top_k": self.top_k,
        }
