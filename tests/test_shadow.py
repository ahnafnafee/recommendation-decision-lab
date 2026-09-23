import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from reliability.core import Recommender
from reliability.shadow import (ShadowRecommender, adjacent_projection, bucket_of,
                                interview_order, novice_prior, score_requests, shadow_queries)


CATALOG = tuple(f"i{n}" for n in range(10))


def tiny_shadow(records=None, oracle=None, questions=()):
    """Hand-built model so every score in the parity tests is checkable by hand."""
    lifetime = {item: 1.0 - 0.05 * index for index, item in enumerate(CATALOG)}
    recent = {item: 0.80 - 0.04 * index for index, item in enumerate(reversed(CATALOG))}
    neighbors = {"h1": (("i1", 0.50), ("i2", 0.40)),
                 "h2": (("i2", 0.30), ("i9", 0.90)),
                 "h3": (("i0", 0.70), ("i5", 0.20))}
    model = Recommender(lifetime, recent, neighbors)
    return ShadowRecommender(model, records or {}, lifetime, oracle, questions)


def full_scan(positive, negative, base, seen, alpha, gamma, k):
    scored = [(-(1 - alpha) * base[item] - alpha * (positive.get(item, 0.0) - gamma * negative.get(item, 0.0)),
               -base[item], item) for item in base if item not in seen]
    return tuple(item for *_, item in sorted(scored)[:k])


class ShadowEvidenceTests(unittest.TestCase):
    def test_repulsion_sums_do_not_raise_a_ranking_score(self):
        shadow = tiny_shadow()
        positive, negative = shadow.sums({"h1": 1.0, "h2": 1.0}, {"h3": 1.0})
        self.assertGreater(positive["i1"], 0)
        self.assertEqual(positive["i2"], 0.40 + 0.30)
        self.assertEqual(positive["i9"], 0.90)
        self.assertEqual(negative["i0"], 0.70)
        self.assertEqual(negative["i5"], 0.20)
        combined = {item: positive.get(item, 0.0) - negative.get(item, 0.0)
                    for item in set(positive) | set(negative)}
        for item, value in combined.items():
            self.assertLessEqual(value, positive.get(item, 0.0))

    def test_unknown_history_items_stay_presence_only(self):
        shadow = tiny_shadow()
        attraction, repulsion = shadow.evidence("u1", ("h1", "never-seen"), 10_000)
        self.assertEqual(attraction["h1"], 1.0)
        self.assertEqual(attraction["never-seen"], 1.0)
        self.assertEqual(repulsion, {})

    def test_negative_ratings_split_into_repulsion_but_keep_presence(self):
        shadow = tiny_shadow({"u1": {"h1": (2, 900), "h2": (5, 900)}})
        attraction, repulsion = shadow.evidence("u1", ("h1", "h2"), 10_000)
        self.assertEqual(repulsion, {"h1": 1.0})
        self.assertEqual(attraction["h1"], 1.0, "presence still counts as co-review evidence")
        self.assertEqual(attraction["h2"], 1.0)

    def test_graded_weights_and_age_decay_only_touch_observed_positives(self):
        shadow = tiny_shadow({"u1": {"h1": (4, 5_000), "h2": (5, 5_000), "h3": (2, 5_000)}})
        plain, _ = shadow.evidence("u1", ("h1", "h2"), 10_000)
        graded, _ = shadow.evidence("u1", ("h1", "h2"), 10_000, graded=True)
        self.assertEqual(plain, {"h1": 1.0, "h2": 1.0})
        self.assertLess(graded["h1"], graded["h2"], "a four-star history should weigh less than a five-star one")
        decayed, _ = shadow.evidence("u1", ("h1", "h2"), 10_000, half_life=1)
        self.assertLess(decayed["h2"], graded["h2"])
        _, negative = shadow.evidence("u1", ("h3",), 10_000, graded=True, half_life=1)
        self.assertEqual(negative, {"h3": 1.0}, "repulsion is reported, not decayed away")

    def test_novice_prior_uses_each_users_first_positive(self):
        prior = novice_prior([("u1", "a", 10), ("u1", "b", 5), ("u2", "c", 1)], ("a", "b", "c"))
        self.assertEqual(prior["b"], 1.0)
        self.assertEqual(prior["c"], 1.0)
        self.assertEqual(prior["a"], 0.0)

    def test_bucket_boundaries(self):
        self.assertEqual([bucket_of(size) for size in (0, 1, 2, 3, 5, 6, 19, 20, 500)],
                         ["0", "1-2", "1-2", "3-5", "3-5", "6-19", "6-19", "20+", "20+"])

    def test_interview_order_is_deterministic_and_capped(self):
        first = interview_order([("u", "a", 1), ("v", "a", 1), ("v", "b", 2)], depth=1)
        self.assertEqual(first, ("a",))

    def test_adjacent_projection_splits_credit_by_shared_users(self):
        projection = adjacent_projection({"u1": {"d1"}, "u2": {"d1", "d2"}},
                                         {"u1": {"t1"}, "u2": {"t1", "t2"}})
        self.assertEqual(dict(projection["d1"]), {"t1": 1.0, "t2": 0.5},
                         "d1 is shared by two users, so each contributes half")
        self.assertEqual(dict(projection["d2"]), {"t1": 1.0, "t2": 1.0},
                         "d2 has one overlapping user, who liked both targets")


class ShadowRankingTests(unittest.TestCase):
    def test_sparse_ranking_matches_a_full_catalog_scan(self):
        shadow = tiny_shadow()
        positive, negative = shadow.sums({"h1": 1.0, "h2": 1.0, "h3": 1.0}, {"h3": 1.0})
        heavy = dict(negative)
        heavy["i8"] = 5.0
        heavy["i7"] = 3.0
        for seen in (set(), {"i9"}, set(CATALOG[:6])):
            for negatives in (negative, heavy):
                for gamma in (0.0, 0.25, 1.0, 2.0):
                    for alpha in (0.25, 0.5, 0.75, 1.0):
                        for base_name, base in (("lifetime", shadow.bases["lifetime"]),
                                                ("recent", shadow.bases["recent"])):
                            exact = full_scan(positive, negatives, base, seen, alpha, gamma, 5)
                            sparse = shadow.rank(positive, negatives, base_name, (alpha,), (gamma,), seen, 5)
                            self.assertEqual(sparse[(alpha, gamma)], exact,
                                             f"alpha={alpha} gamma={gamma} base={base_name} seen={len(seen)}")

    def test_zero_repulsion_reproduces_the_committed_hybrid(self):
        shadow = tiny_shadow({"u1": {"h1": (2, 900)}})
        history = ("h1", "h2", "unknown-item")
        attraction, _ = shadow.evidence("u1", history, 10_000)
        positive, negative = shadow.sums(attraction, {})
        committed = shadow.model.rank_grid(history, "recent", (0.0, 0.25, 0.5, 0.75, 1.0), 5)
        reproduced = shadow.rank(positive, negative, "recent", tuple(committed), (0.0,), set(history), 5)
        for alpha, ranking in committed.items():
            self.assertEqual(reproduced[(alpha, 0.0)], ranking)

    def test_repulsion_pushes_a_popular_item_out_of_the_list(self):
        shadow = tiny_shadow()
        positive, _ = shadow.sums({"h3": 1.0}, {})
        base = shadow.bases["lifetime"]
        self.assertEqual(base["i0"], max(base.values()), "i0 is the most popular item")
        before = shadow.rank(positive, {}, "lifetime", (0.5,), (0.0,), set(), 3)[(0.5, 0.0)]
        after = shadow.rank(positive, {"i0": 40.0}, "lifetime", (0.5,), (1.0,), set(), 3)[(0.5, 1.0)]
        self.assertEqual(before[0], "i0")
        self.assertNotIn("i0", after)
        self.assertEqual(len(after), 3)

    def test_unknown_base_name_is_rejected(self):
        shadow = tiny_shadow()
        with self.assertRaises(ValueError):
            shadow.rank({}, {}, "collaborative", (0.5,), (0.0,), set())


class ShadowInterviewTests(unittest.TestCase):
    def test_interview_skips_target_and_known_items_and_counts_costs(self):
        shadow = tiny_shadow(questions=("q1", "q2", "q3", "q4"),
                             oracle={"u1": {"q1": (5, 1), "q2": (5, 2), "q3": (2, 3), "q4": (5, 4)}})
        asked, replies, costs = shadow.interview("u1", "q2", {"q1"}, (0, 1, 2, 5))
        self.assertEqual(asked, ["q3", "q4"])
        self.assertEqual(replies, [("q3", -1.0), ("q4", 1.0)])
        self.assertEqual(costs[0], (0, 0, 0, 0))
        self.assertEqual(costs[1], (1, 1, 0, 1))
        self.assertEqual(costs[2], (2, 2, 1, 1))
        self.assertEqual(costs[5], (2, 2, 1, 1), "a budget larger than the pool cannot buy more questions")

    def test_unanswerable_questions_still_cost(self):
        shadow = tiny_shadow(questions=("q1", "q2"), oracle={"u1": {"q2": (5, 1)}})
        asked, replies, costs = shadow.interview("u1", "target", set(), (1, 2))
        self.assertEqual(asked, ["q1", "q2"])
        self.assertEqual(replies, [None, ("q2", 1.0)])
        self.assertEqual(costs[1], (1, 0, 0, 0))
        self.assertEqual(costs[2], (2, 1, 1, 0))

    def test_interview_does_not_reask_about_items_beyond_profile_window(self):
        history = tuple(f"old{index}" for index in range(21))
        shadow = tiny_shadow(questions=("old0", "i1"),
                             oracle={"u1": {"old0": (5, 1), "i1": (5, 2)}})
        _, costs, _, _ = score_requests(shadow, [("u1", "i4", history, 10_000)],
                                        "lifetime", (0.5,), (0.0,), budgets=(0, 1))
        self.assertEqual(costs[1]["asked"], 1)
        self.assertEqual(costs[1]["answers"], 1)

    def test_budgets_share_evidence_and_keep_every_cell_scored(self):
        queries = [("u1", "i4", ("h1",), 10_000), ("u2", "i0", (), 10_000)]
        shadow = tiny_shadow(questions=("i1", "i2", "i9"),
                             oracle={"u1": {"i1": (1, 1), "i2": (5, 2)}, "u2": {"i9": (5, 1)}})
        outcomes, costs, stats, timings = score_requests(shadow, queries, "lifetime", (0.5,), (0.0,),
                                                         budgets=(0, 1, 2))
        for cell, values in outcomes.items():
            self.assertEqual(len(values), len(queries), cell)
        self.assertEqual(costs[0], {"asked": 0, "answers": 0, "positive": 0, "negative": 0})
        self.assertEqual(costs[1], {"asked": 2, "answers": 1, "positive": 0, "negative": 1})
        self.assertEqual(costs[2], {"asked": 4, "answers": 2, "positive": 1, "negative": 1})
        self.assertEqual(len(timings), len(queries))

    def test_novice_route_only_replaces_requests_without_evidence(self):
        queries = [("u1", "i4", ("h1",), 10_000), ("u2", "i0", ("zz", "zz"), 10_000)]
        shadow = tiny_shadow()
        _, _, plain_stats, _ = score_requests(shadow, queries, "lifetime", (0.5,), (0.0,))
        outcomes, _, routed_stats, _ = score_requests(shadow, queries, "lifetime", (0.5,), (0.0,),
                                                      novice_for_cold=True)
        self.assertEqual(plain_stats["novice_route_requests"], 0)
        self.assertEqual(plain_stats["unpersonalisable_requests"], 1)
        self.assertEqual(routed_stats["novice_route_requests"], 1)
        routed = outcomes[(0.5, 0.0, 0)][1]
        self.assertIn(routed, (0.0, 1.0))

    def test_elicited_dislike_can_change_the_recommendation(self):
        queries = [("u1", "i0", ("h1",), 10_000)]
        shadow = tiny_shadow(questions=("h3",), oracle={"u1": {"h3": (1, 1)}})
        without, _, _, _ = score_requests(shadow, queries, "lifetime", (1.0,), (1.0,), budgets=(0,))
        with_answer, _, _, _ = score_requests(shadow, queries, "lifetime", (1.0,), (1.0,), budgets=(1,))
        self.assertGreater(without[(1.0, 1.0, 0)][0], with_answer[(1.0, 1.0, 1)][0],
                           "a declared dislike should demote what it dislikes")

    def test_elicited_like_promotes_the_item_it_praises(self):
        queries = [("u1", "i0", ("h1",), 10_000)]
        shadow = tiny_shadow(questions=("h3",), oracle={"u1": {"h3": (5, 1)}})
        without, _, _, _ = score_requests(shadow, queries, "lifetime", (1.0,), (1.0,), budgets=(0,))
        with_answer, _, _, _ = score_requests(shadow, queries, "lifetime", (1.0,), (1.0,), budgets=(1,))
        self.assertLess(without[(1.0, 1.0, 0)][0], with_answer[(1.0, 1.0, 1)][0])
        self.assertEqual(with_answer[(1.0, 1.0, 1)][0], 1.0)


class ShadowCohortTests(unittest.TestCase):
    def write_valid(self, directory, rows):
        import csv
        import gzip
        path = Path(directory) / "valid.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, ["user_id", "parent_asin", "rating", "timestamp", "history"])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(zip(["user_id", "parent_asin", "rating", "timestamp", "history"], row)))
        return path

    def test_cohort_matches_the_committed_eligibility_rule_and_keeps_time(self):
        from reliability.benchmark import T1
        rows = [("u1", "B0", 5, T1 + 1, "A0"),
                ("u2", "B1", 3, T1 + 2, "A1"),
                ("u3", "B2", 5, T1 + 3, "B2 A2"),
                ("u4", "NEW", 4, T1 + 4, "")]
        with TemporaryDirectory() as temporary:
            queries, stats = shadow_queries(self.write_valid(temporary, rows), "valid", {"B0", "B1", "B2"})
        self.assertEqual([(user, target, history) for user, target, history, _ in queries],
                         [("u1", "B0", ("A0",)), ("u4", "NEW", ())])
        self.assertEqual([row[3] for row in queries], [T1 + 1, T1 + 4])
        self.assertEqual(stats["below_positive_threshold"], 1)
        self.assertEqual(stats["seen_target_excluded"], 1)
        self.assertEqual(stats["new_item_targets"], 1)
        self.assertEqual(stats["empty_history"], 1)
        self.assertEqual(stats["eligible_requests"], 2)


class SignalAuditTests(unittest.TestCase):
    def test_missing_user_rating_is_not_an_unseen_catalog_item(self):
        import csv
        import gzip
        from reliability.benchmark import T1, T2
        from tools.signal_audit import audit

        fields = ("user_id", "parent_asin", "rating", "timestamp", "history")
        rows = {
            "train": [("u1", "A", 2, T1 - 2, ""), ("u2", "B", 5, T1 - 1, "")],
            "valid": [("u1", "D", 5, T1 + 1, "A B C")],
            "test": [("u1", "E", 5, T2 + 1, "")],
        }
        with TemporaryDirectory() as temporary:
            for split, records in rows.items():
                path = Path(temporary) / f"Musical_Instruments.{split}.csv.gz"
                with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    writer.writerows(records)
            result, _ = audit(Path(temporary), "Musical_Instruments")

        valid = result["splits"]["valid"]
        self.assertEqual(valid["history_slots_rating_below_four"], 1)
        self.assertEqual(valid["history_slots_without_user_item_training_rating"], 2)
        self.assertEqual(valid["history_slots_item_absent_from_train_catalog"], 1)


if __name__ == "__main__":
    unittest.main()
