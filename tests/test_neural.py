"""Neural training contracts; torch-specific checks run when installed."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from reliability.benchmark import T1
from reliability.neural import chronological_pairs


class ChronologyTests(unittest.TestCase):
    def test_pairs_use_only_prior_distinct_train_events(self):
        index = {"a": 1, "b": 2, "c": 3}
        events = [("u", "c", 30), ("u", "a", 10), ("u", "a", 20),
                  ("u", "b", 25), ("v", "b", 15)]
        self.assertEqual(chronological_pairs(events, index, 2),
                         [((1,), 2), ((1, 2), 3)])

    def test_train_cutoff_is_enforced(self):
        with self.assertRaises(ValueError):
            chronological_pairs([("u", "a", T1)], {"a": 1}, 20)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class CollisionTests(unittest.TestCase):
    def test_duplicate_targets_and_prior_positives_are_masked(self):
        import torch
        from reliability.neural import collision_mask

        histories = torch.tensor([[0, 1, 2], [0, 3, 4], [0, 5, 6]])
        targets = torch.tensor([7, 7, 2])
        mask = collision_mask(histories, targets)
        self.assertEqual(mask.tolist(), [[False, True, True],
                                          [True, False, False],
                                          [False, False, False]])

    def test_batched_exact_ranking_matches_served_blend_and_excludes_seen(self):
        from reliability.neural import NeuralConfig, NeuralRetriever, _architecture

        config = NeuralConfig(dimension=8, history_length=3)
        model = _architecture(4, config)
        catalog = ("a", "b", "c", "d")
        scores = {"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.0}
        retriever = NeuralRetriever(model, catalog, ("a", "b", "c"), config, "cpu")
        retriever.configure_blend(scores, 0.2)
        served = retriever.top_k(("a",), 2)
        grid, supported = retriever.rank_grid((("a",),), scores, (0.2,), 2)
        self.assertEqual(served, grid[0.2][0])
        self.assertEqual(supported, [True])
        self.assertNotIn("a", served)
        self.assertNotIn("d", served)
        self.assertIsNone(retriever.top_k((), 2))

    def test_checkpoint_rejects_weight_tampering(self):
        import torch
        from dataclasses import asdict
        from reliability.benchmark import file_hash
        from reliability.neural import NeuralConfig, _architecture, load_retriever

        config = NeuralConfig(dimension=8)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.csv.gz"
            train.write_bytes(b"source fixture")
            weights = root / "neural_weights.pt"
            torch.save({"state_dict": _architecture(2, config).state_dict(),
                        "catalog": ("a", "b"), "supported": ["a"],
                        "config": asdict(config), "train_sha256": file_hash(train)}, weights)
            (root / "neural_manifest.json").write_text(json.dumps({
                "schema": 1, "weights_sha256": file_hash(weights),
                "train_sha256": file_hash(train), "catalog_items": 2}))
            model, _ = load_retriever(weights, train, "cpu")
            self.assertEqual(len(model.catalog), 2)
            with weights.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_retriever(weights, train, "cpu")


if __name__ == "__main__":
    unittest.main()
