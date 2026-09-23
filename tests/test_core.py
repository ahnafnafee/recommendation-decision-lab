import random
import unittest

from reliability.benchmark import user_cluster_interval
from reliability.core import Recommender, last_distinct, ndcg_one_target


class RankingTests(unittest.TestCase):
    def test_sparse_retrieval_matches_full_catalog(self):
        generator = random.Random(7)
        items = [f"i{index:03d}" for index in range(80)]
        lifetime = {item: generator.random() for item in items}
        recent = {item: generator.random() for item in items}
        neighbors = {item: tuple((other, generator.random()) for other in generator.sample(items, 7))
                     for item in items}
        model = Recommender(lifetime, recent, neighbors)
        for baseline in ("lifetime", "recent"):
            base = lifetime if baseline == "lifetime" else recent
            for _ in range(12):
                history = tuple(generator.sample(items, 5))
                seen = set(history)
                personalized = {item: 0.0 for item in items}
                for source in last_distinct(history):
                    for item, similarity in neighbors[source]:
                        if item not in seen:
                            personalized[item] += similarity
                for alpha in (0.0, .25, .5, .75, 1.0):
                    expected = tuple(sorted((item for item in items if item not in seen),
                                            key=lambda item: (-(1-alpha)*base[item]-alpha*personalized[item],
                                                              -base[item], item))[:10])
                    self.assertEqual(model.top_k(history, baseline, alpha), expected)

    def test_train_cutoff_and_catalog(self):
        model = Recommender.train([("u1", "a", 1), ("u1", "b", 2),
                                   ("u2", "a", 2), ("u2", "b", 3)], {"a", "b", "c"}, 10)
        self.assertEqual(model.lifetime["c"], 0.0)
        self.assertEqual(model.top_k(("a",), k=2)[0], "b")
        self.assertEqual(model.neighbors["a"][0][0], "b")
        self.assertGreater(model.neighbors["a"][0][1], 0)
        with self.assertRaises(ValueError):
            Recommender.train([("u", "a", 10)], {"a"}, 10)

    def test_trained_neighbors_change_personalized_ranking(self):
        events = [(f"u{index}", "a", 1) for index in range(5)]
        events += [(f"u{index}", "b", 2) for index in range(5)]
        events += [(f"v{index}", "c", 3) for index in range(8)]
        model = Recommender.train(events, {"a", "b", "c"}, 10)
        self.assertEqual(model.top_k(("a",), alpha=0, k=1), ("c",))
        self.assertEqual(model.top_k(("a",), alpha=1, k=1), ("b",))

    def test_target_metric(self):
        self.assertEqual(ndcg_one_target(("a", "b"), "a"), 1.0)
        self.assertEqual(ndcg_one_target(("a", "b"), "c"), 0.0)

    def test_cluster_interval_uses_paired_queries(self):
        queries = [("u1", "a", ()), ("u1", "b", ()), ("u2", "a", ())]
        result = user_cluster_interval(queries, [0, 0, 0], [1, .5, 0], draws=100, seed=1)
        self.assertAlmostEqual(result["delta"], .5)
        self.assertEqual(result["users"], 2)


if __name__ == "__main__":
    unittest.main()
