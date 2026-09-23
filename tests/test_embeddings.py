import json
import math
import unittest

from reliability import embeddings as module
from reliability.embeddings import (ARTIFACT_SCHEMA, EmbeddingIndex, load_embeddings,
                                    write_embeddings)
from pathlib import Path
from tempfile import TemporaryDirectory


ALPHABET = "abcd"


def letter_vector(text):
    """A stand-in encoder: a unit vector counting the first four letters."""
    counts = [float(text.count(letter)) for letter in ALPHABET]
    norm = math.sqrt(sum(value * value for value in counts))
    if not norm:
        counts[0] = 1.0
        norm = 1.0
    return [value / norm for value in counts]


def cosine(one, two):
    return sum(left * right for left, right in zip(one, two))


DOCUMENTS = {"a": "aaa b", "b": "a bb c", "c": "cc d", "d": ""}


def build(directory, category="Widget_World", model="stand-in", items=None):
    wanted = sorted(items if items is not None else DOCUMENTS)
    vectors = [letter_vector(DOCUMENTS[item]) for item in wanted]
    return write_embeddings(directory, category, wanted, vectors, model=model,
                            corpus_sha256="corpus-hash", document_sha256="document-hash",
                            dimensions=4)


class RoundTripTests(unittest.TestCase):
    def test_a_written_index_reads_back_and_scores_a_phrase(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            written = build(directory)
            self.assertEqual(written["count"], 4)
            self.assertEqual(written["schema"], ARTIFACT_SCHEMA)
            index, manifest = load_embeddings(directory, "Widget_World",
                                              expected_corpus_sha256="corpus-hash",
                                              encode=letter_vector)
            scores = index.query_scores("aaaa b")
            self.assertEqual(set(scores), set(DOCUMENTS))
            ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: -pair[1])]
            self.assertEqual(ranked[0], "a")
            self.assertAlmostEqual(scores["a"], cosine(letter_vector("aaaa b"),
                                                       letter_vector("aaa b")), places=6)
            self.assertEqual(manifest["model"], "stand-in")

    def test_a_phrase_is_only_encoded_once(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            calls = []

            def counting(text):
                calls.append(text)
                return letter_vector(text)

            index, _ = load_embeddings(directory, "Widget_World", encode=counting)
            index.query_scores("abc")
            index.query_scores("abc")
            index.closest("abc", limit=2)
            self.assertEqual(calls, ["abc"])

    def test_an_empty_phrase_never_reaches_the_encoder(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)

            def exploding(text):
                raise AssertionError("nothing should be encoded")

            index, _ = load_embeddings(directory, "Widget_World", encode=exploding)
            self.assertEqual(index.query_scores("   "), {})

    def test_closest_breaks_a_tie_by_item_identifier(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            index, _ = load_embeddings(directory, "Widget_World", encode=letter_vector)
            first, second, third = index.closest("cc dd", limit=3)
            self.assertEqual(first[0], "c")
            self.assertEqual(second[0], "b")
            self.assertEqual(third[0], "a")

    def test_top_scores_agrees_with_a_full_scoring_pass(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            index, _ = load_embeddings(directory, "Widget_World", encode=letter_vector)
            full = sorted(index.query_scores("aa bb").items(), key=lambda pair: (-pair[1], pair[0]))
            head = index.top_scores("aa bb", limit=2)
            self.assertEqual([item for item, _ in head], [item for item, _ in full[:2]])
            for (_, one), (_, other) in zip(head, full[:2]):
                self.assertAlmostEqual(one, other, places=6)
            self.assertEqual(len(index.top_scores("aa bb", limit=99)), len(DOCUMENTS))
            self.assertEqual(index.top_scores("", limit=3), [])
            self.assertEqual(index.top_scores("aa", limit=0), [])


class RefusalTests(unittest.TestCase):
    def test_an_index_from_other_product_text_is_refused(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            with self.assertRaisesRegex(ValueError, "different product text"):
                load_embeddings(directory, "Widget_World", expected_corpus_sha256="newer-corpus",
                                encode=letter_vector)

    def test_a_vector_file_that_does_not_match_its_hash_is_refused(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            blob = directory / "Widget_World.embeddings.f32"
            blob.write_bytes(blob.read_bytes() + blob.read_bytes()[:4])
            with self.assertRaisesRegex(ValueError, "does not match the shape"):
                load_embeddings(directory, "Widget_World", encode=letter_vector)

    def test_a_tampered_vector_is_refused_by_hash(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            blob = directory / "Widget_World.embeddings.f32"
            payload = bytearray(blob.read_bytes())
            manifest_path = directory / "Widget_World.embeddings_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["count"] += 1
            manifest["dimensions"] = 4
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            payload.extend(bytes(16))
            blob.write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "does not match the hash"):
                load_embeddings(directory, "Widget_World", encode=letter_vector)

    def test_a_missing_index_says_which_tool_builds_it(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaisesRegex(ValueError, "build_text_index"):
                load_embeddings(directory, "Widget_World")

    def test_a_manifest_from_another_schema_is_refused(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            path = directory / "Widget_World.embeddings_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["schema"] = ARTIFACT_SCHEMA + 1
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_embeddings(directory, "Widget_World", encode=letter_vector)

    def test_ragged_rows_are_refused_at_the_gate(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            with self.assertRaisesRegex(ValueError, "same dimensions"):
                write_embeddings(directory, "Widget_World", ["a", "b"], [[1.0], [1.0, 0.0]],
                                 model="stand-in", corpus_sha256="c", document_sha256="d",
                                 dimensions=1)


class NoNumpyTests(unittest.TestCase):
    def test_the_sparse_path_rejects_bad_shape(self):
        original = module._numpy
        module._numpy = lambda: None
        try:
            with self.assertRaisesRegex(ValueError, "expected 2x4 vectors"):
                EmbeddingIndex(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], 4,
                               letter_vector)
            with self.assertRaisesRegex(ValueError, "expected 2x4 vectors"):
                EmbeddingIndex(["a", "b"], [[1.0, 0.0, 0.0, 0.0]], 4,
                               letter_vector)
        finally:
            module._numpy = original

    def test_the_sparse_path_agrees_with_the_dense_one(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            build(directory)
            index, _ = load_embeddings(directory, "Widget_World", encode=letter_vector)
            dense = index.query_scores("aa bb")
            original = module._numpy
            module._numpy = lambda: None
            try:
                rebuild, _ = load_embeddings(directory, "Widget_World", encode=letter_vector)
                sparse = rebuild.query_scores("aa bb")
            finally:
                module._numpy = original
            self.assertFalse(rebuild.statistics()["dense"])
            for item, value in dense.items():
                self.assertAlmostEqual(sparse[item], value, places=5)


class ShapeTests(unittest.TestCase):
    def test_a_vector_of_the_wrong_width_is_refused(self):
        index = EmbeddingIndex(["a"], [[1.0, 0.0, 0.0, 0.0]], 4, lambda text: [1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "the encoder returned 3 dimensions"):
            index.query_scores("anything")

    def test_an_index_with_no_items_is_refused(self):
        with self.assertRaisesRegex(ValueError, "at least one item"):
            EmbeddingIndex([], [], 4, letter_vector)

    def test_a_ragged_matrix_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expected 2x4 vectors"):
            EmbeddingIndex(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], 4, letter_vector)
