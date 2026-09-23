"""Phrase-to-item similarity from an encoder that ran outside the request path.

`tools/build_text_index.py` encodes the catalogue's own product text once, on the
machine with the GPU, and writes a flat float32 matrix plus a manifest saying
which corpus it was built from. This module reads that artifact back, refuses it
if it no longer matches the corpus being scored, and answers one question: how
close is this phrase to each product.

numpy is used when it is installed; the module stays importable and correct
without it, just slower, because a request path here must not require a model to
be present — the lexical route in `language.py` covers that case.
"""

from __future__ import annotations

from array import array
import hashlib
import json
from pathlib import Path

ARTIFACT_SCHEMA = 1
VECTOR_BYTES = 4


def _numpy():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Encoder:
    """Turns text into a unit vector with a sentence-embedding model."""

    def __init__(self, model: str, device: str = "auto"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the optional language extra: "
                "uv pip install --python .venv\\Scripts\\python.exe sentence-transformers") from error
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install the optional language extra: pip install torch") from error
        resolved = device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model
        self.device = resolved
        self._model = SentenceTransformer(model, device=resolved)
        self._torch = torch

    def vector(self, text: str) -> list[float]:
        return [float(value) for value in self.vectors([text])[0]]

    def vectors(self, texts) -> list[list[float]]:
        encoded = self._model.encode(list(texts), normalize_embeddings=True,
                                     convert_to_numpy=True, show_progress_bar=False)
        return [[float(value) for value in row] for row in encoded]


class EmbeddingIndex:
    """Unit vectors for every catalogue item, scored by cosine against a phrase."""

    def __init__(self, items, vectors, dimensions: int, encode, *, model: str = "",
                 corpus_sha256: str | None = None):
        self.dimensions = int(dimensions)
        if self.dimensions <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.items = list(items)
        if not self.items:
            raise ValueError("an embedding index needs at least one item")
        self.position = {item: row for row, item in enumerate(self.items)}
        self.encode = encode
        self.model = model
        self.corpus_sha256 = corpus_sha256
        self._cache: dict[str, list[float]] = {}
        numpy = _numpy()
        self._numpy = numpy
        self.matrix = None
        if numpy is not None:
            matrix = numpy.asarray(vectors, dtype=numpy.float32)
            if matrix.shape != (len(self.items), self.dimensions):
                raise ValueError(f"expected {len(self.items)}x{self.dimensions} vectors, "
                                 f"found {matrix.shape}")
            self.matrix = matrix
        else:
            flat = array("f")
            row_count = 0
            for row_count, row in enumerate(vectors, start=1):
                if len(row) != self.dimensions:
                    raise ValueError(f"expected {len(self.items)}x{self.dimensions} vectors, "
                                     f"found row {row_count} with {len(row)} dimensions")
                flat.extend(row)
            if row_count != len(self.items):
                raise ValueError(f"expected {len(self.items)}x{self.dimensions} vectors, "
                                 f"found {row_count} rows")
            self.flat = flat

    def _vector(self, text: str):
        key = text if len(text) <= 400 else text[:400]
        vector = self._cache.get(key)
        if vector is None:
            vector = self.encode(key)
            if len(vector) != self.dimensions:
                raise ValueError(f"the encoder returned {len(vector)} dimensions, "
                                 f"the index stores {self.dimensions}")
            if self._cache.get(key) is None and len(self._cache) < 4096:
                self._cache[key] = vector
        return vector

    def query_scores(self, text: str) -> dict[str, float]:
        """Cosine similarity between this phrase and every catalogue item.

        Every item is scored, not a shortlist, so a downstream ranker can lift an
        item it found by other means using its true similarity rather than an
        assumed one.
        """
        if not text.strip():
            return {}
        vector = self._vector(text)
        if self.matrix is not None:
            products = self.matrix @ self._numpy.asarray(vector, dtype=self._numpy.float32)
            return dict(zip(self.items, (float(value) for value in products)))
        score = []
        for start in range(0, len(self.flat), self.dimensions):
            row = self.flat[start:start + self.dimensions]
            score.append(sum(left * right for left, right in zip(row, vector)))
        return dict(zip(self.items, score))

    def top_scores(self, text: str, limit: int = 200,
                   restrict=None) -> list[tuple[str, float]]:
        """The closest `limit` products, which is all a capped re-order can use.

        Same answers as `query_scores`, without building a dictionary the size of
        the catalogue for a request that will only look at a few hundred.

        `restrict` narrows the search to one set of products *before* the cut. That
        matters: the nearest few hundred of the whole shelf are rarely among a couple
        of thousand behavioural candidates, so retrieving everything and filtering
        afterwards would leave a phrase nothing to choose between.
        """
        limit = int(limit)
        if not text.strip() or limit <= 0:
            return []
        vector = self._vector(text)
        wanted = None
        if restrict is not None:
            wanted = [self.position[item] for item in restrict if item in self.position]
            if not wanted:
                return []
        take = min(limit, len(self.items) if wanted is None else len(wanted))
        pairs = []
        if self.matrix is not None:
            products = self.matrix @ self._numpy.asarray(vector, dtype=self._numpy.float32)
            if wanted is not None:
                return [(self.items[row], float(products[row])) for row in sorted(
                    wanted, key=lambda row: (-float(products[row]), self.items[row]))[:take]]
            chosen = self._numpy.argpartition(-products, take - 1)[:take]
            pairs = [(self.items[int(index)], float(products[int(index)])) for index in chosen]
        else:
            offsets = (range(0, len(self.flat), self.dimensions) if wanted is None
                       else [row * self.dimensions for row in wanted])
            for start in offsets:
                row = self.flat[start:start + self.dimensions]
                pairs.append((self.items[start // self.dimensions],
                              sum(left * right for left, right in zip(row, vector))))
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs[:take]

    def closest(self, text: str, limit: int = 5) -> list[tuple[str, float]]:
        scores = self.query_scores(text)
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:max(1, int(limit))]

    def statistics(self) -> dict:
        return {"items": len(self.items), "dimensions": self.dimensions, "model": self.model,
                "dense": self.matrix is not None}


def load_embeddings(directory: Path, category: str, *, expected_corpus_sha256: str | None = None,
                    encode=None, device: str = "auto") -> tuple[EmbeddingIndex, dict]:
    """Read a built index, refusing one that was encoded from different product text."""
    manifest_path = directory / f"{category}.embeddings_manifest.json"
    blob_path = directory / f"{category}.embeddings.f32"
    if not manifest_path.exists() or not blob_path.exists():
        raise ValueError(f"no phrase index for {category} under {directory}; "
                         f"build it with tools/build_text_index.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"embedding manifest schema {manifest.get('schema')!r} "
                         f"is not {ARTIFACT_SCHEMA}")
    if manifest.get("category") != category:
        raise ValueError("embedding category mismatch")
    count, dimensions = int(manifest["count"]), int(manifest["dimensions"])
    if blob_path.stat().st_size != count * dimensions * VECTOR_BYTES:
        raise ValueError("the vector file does not match the shape recorded beside it")
    digest = _sha256(blob_path)
    if digest != manifest.get("vectors_sha256"):
        raise ValueError("the vector file does not match the hash recorded beside it")
    if expected_corpus_sha256 is not None and expected_corpus_sha256 != manifest.get("corpus_sha256"):
        raise ValueError("the index was encoded from different product text; rebuild it")
    with blob_path.open("rb") as stream:
        payload = stream.read()
    rows = array("f")
    rows.frombytes(payload)
    numpy = _numpy()
    if numpy is None:
        vectors = [rows[start:start + dimensions] for start in range(0, len(rows), dimensions)]
    else:
        vectors = numpy.asarray(rows, dtype=numpy.float32).reshape(count, dimensions)
    if encode is None:
        encode = Encoder(manifest["model"], device).vector
    index = EmbeddingIndex(manifest["items"], vectors, dimensions, encode,
                           model=manifest["model"], corpus_sha256=manifest.get("corpus_sha256"))
    return index, manifest


def write_embeddings(directory: Path, category: str, items, vectors, *, model: str,
                     corpus_sha256: str, document_sha256: str, dimensions: int,
                     extra: dict | None = None) -> dict:
    """Persist an encoded catalogue and record what it was encoded from."""
    directory.mkdir(parents=True, exist_ok=True)
    blob_path = directory / f"{category}.embeddings.f32"
    payload = array("f")
    for row in vectors:
        if len(row) != dimensions:
            raise ValueError("every embedding row needs the same dimensions")
        payload.extend(row)
    with blob_path.open("wb") as stream:
        stream.write(payload.tobytes())
    manifest = {"schema": ARTIFACT_SCHEMA, "category": category, "model": model,
                "dimensions": int(dimensions), "count": len(list(items)),
                "items": list(items), "corpus_sha256": corpus_sha256,
                "document_sha256": document_sha256, "vectors_sha256": _sha256(blob_path),
                "vector_bytes": blob_path.stat().st_size}
    manifest.update(extra or {})
    (directory / f"{category}.embeddings_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
