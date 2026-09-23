import gzip
import unittest
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from unittest.mock import patch

from reliability.benchmark import T1, T2, evaluation_data
from reliability.core import Recommender, ndcg_one_target
from reliability.language import build_index, parse_utterance
from reliability.phrasing import (BASE, CONDITIONS, CONFIGS, Phrases, agreement, cross_phrases,
                                  history_buckets, item_titles, outcomes_for,
                                  popularity_buckets, published_by_item, ranker_for,
                                  shortlist_report, spearman, stratified, summarise,
                                  timestamped_requests, utterances_for)

CATALOG = tuple(f"i{n}" for n in range(12))
DOCS = {
    "i0": "Fender Twin Reverb tube amplifier 100 watts reverb tremolo",
    "i1": "Fender Stratocaster electric guitar sunburst maple neck",
    "i2": "distortion pedal guitar effects overdrive compact",
    "i3": "acoustic guitar dreadnought spruce top natural",
    "i4": "distortion pedal metal high gain compact",
    "i5": "studio monitor speakers flat response pair",
    "i6": "midi controller keyboard 25 keys usb",
    "i7": "tube preamp rack warm vintage",
    "i8": "guitar strap leather wide padded",
    "i9": "compressor pedal dynamics sustain compact",
    "i10": "digital multi effects modeling amp simulation",
    "i11": "",
}


def tiny_model():
    lifetime = {item: 1.0 - 0.05 * index for index, item in enumerate(CATALOG)}
    recent = {item: 0.90 - 0.04 * index for index, item in enumerate(reversed(CATALOG))}
    neighbors = {"i1": (("i2", 0.50), ("i9", 0.20)), "i3": (("i2", 0.30), ("i8", 0.60))}
    return Recommender(lifetime, recent, neighbors)


def tiny_index(count=12):
    records = {item: {"title": DOCS[item]} for item in CATALOG}
    for number in range(count - len(CATALOG)):
        records[f"b{number}"] = {"title": "acoustic guitar spruce top natural finish"}
    index, _ = build_index(records, set(records))
    return index


class TimestampedRequestTests(unittest.TestCase):
    ROWS = {"valid": (("u2", "i2", 4.0, T1, "single"),
                      ("u3", "i5", 3.0, T1 + 10, "i1"),
                      ("u4", "i6", 5.0, T1 + 20, "i1 i3"),
                      ("u5", "i7", 5.0, T1 + 30, "i7 i3")),
            "test": (("u6", "i8", 5.0, T2, "i1"),
                     ("u7", "i9", 2.0, T2 + 5, "i1"),
                     ("u8", "i8", 5.0, T2 + 9, "i8 i3"))}

    def write(self, directory):
        paths = {}
        for split, rows in self.ROWS.items():
            path = Path(directory) / f"Widget_World.{split}.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("user_id,parent_asin,rating,timestamp,history\n")
                for row in rows:
                    stream.write(",".join(str(value) for value in row) + "\n")
            paths[split] = path
        return paths

    def test_the_timestamp_rides_along_without_changing_which_rows_count(self):
        with TemporaryDirectory() as folder:
            paths = self.write(folder)
            requests = timestamped_requests(paths["valid"], "valid", set(CATALOG))
            frozen, _ = evaluation_data(paths["valid"], "valid", set(CATALOG))
            self.assertEqual([row[:3] for row in requests], list(frozen))
            self.assertEqual([row[3] for row in requests], [T1, T1 + 20])
            self.assertEqual(requests[0][2], ("single",))
            self.assertEqual([row[3] for row in
                              timestamped_requests(paths["test"], "test", set(CATALOG))], [T2])

    def test_rows_that_stopped_matching_the_filter_are_reported_not_dropped(self):
        with TemporaryDirectory() as folder:
            paths = self.write(folder)
            with patch("reliability.phrasing.T1", T1 + 1):
                with self.assertRaises(ValueError):
                    timestamped_requests(paths["valid"], "valid", set(CATALOG))


class PhraseBuildingTests(unittest.TestCase):
    def setUp(self):
        self.index = tiny_index(213)
        self.phrases = Phrases({"u1": [[100, "i1", 5.0, "I want a warm tube sound"],
                                       [200, "i3", 5.0, "the dreadnought rang beautifully"],
                                       [300, "i2", 5.0, "this pedal hisses"]],
                                "u2": []},
                              DOCS,
                              published_by_item([("s:q1", "a high gain rock pedal", "i4", "s"),
                                                 ("s:q2", "unused", "i4", "s")]),
                              self.index, list(CATALOG))

    def test_only_words_spoken_before_the_request_and_about_something_else_are_read(self):
        spoken = self.phrases.for_request("u1", "i3", 250, Random(1))["own_words"]
        self.assertEqual(spoken, "I want a warm tube sound")
        self.assertNotIn("hisses", spoken)
        self.assertEqual(self.phrases.for_request("u1", "i2", 250, Random(1))["own_words"],
                         "the dreadnought rang beautifully I want a warm tube sound")
        self.assertEqual(self.phrases.for_request("u2", "i4", 10 ** 12, Random(1))["own_words"], "")

    def test_identifying_words_are_the_ones_this_product_alone_is_called_by(self):
        self.assertEqual(self.phrases.identifying("acoustic guitar dreadnought spruce top natural"),
                         {"dreadnought"})
        self.assertEqual(self.phrases.identifying("acoustic guitar spruce top natural"), set())
        text, removed = self.phrases.scrub("the dreadnought rang beautifully",
                                           "acoustic guitar dreadnought spruce top natural")
        self.assertEqual(text, "rang beautifully")
        self.assertEqual(removed, 1)
        self.assertEqual(self.phrases.scrub("", "anything"), ("", 0))

    def test_a_random_title_is_never_the_answer_and_a_published_phrase_is_the_first_one(self):
        seen = {self.phrases.for_request("u1", "i4", 10 ** 12, Random(seed))["random_title"]
                for seed in range(20)}
        self.assertNotIn(DOCS["i4"], seen)
        self.assertNotIn("", seen)
        self.assertEqual(self.phrases.for_request("u1", "i4", 10 ** 12, Random(1))["published"],
                         "a high gain rock pedal")

    def test_cross_phrases_hand_everyone_somebody_elses_words(self):
        bank = {f"u{n}": [[100, f"i{n}", 5.0, f"words about item {n}"]] for n in range(1, 6)}
        bank["u0"] = []
        phrases = Phrases(bank, DOCS, {}, self.index, list(CATALOG))
        rows = [phrases.for_request(f"u{n}", "i9", 10 ** 12, Random(n)) for n in range(6)]
        own = [row["own_words"] for row in rows]
        self.assertEqual(own[0], "")
        self.assertEqual(len({text for text in own if text}), 5)
        cross_phrases(rows, 7)
        for row, words in zip(rows, own):
            if not words:
                self.assertEqual(row["cross_words"], "")
                continue
            self.assertIn(row["cross_words"], own)
            self.assertNotEqual(row["cross_words"], words)

    def test_nothing_said_is_recorded_as_nothing_rather_than_an_empty_phrase(self):
        rows = [self.phrases.for_request("u1", "i4", 10 ** 12, Random(1))]
        rows[0]["random_title"] = "the"
        spoken = utterances_for(rows)
        self.assertEqual(set(spoken), set(CONDITIONS))
        self.assertIsNotNone(spoken["own_words"][0])
        self.assertIsNone(spoken["random_title"][0])

    def test_titles_come_from_the_corpus_and_empty_ones_stay_empty(self):
        corpus = {"i0": {"title": DOCS["i0"]}, "i11": {}}
        self.assertEqual(item_titles(corpus), {"i0": DOCS["i0"], "i11": ""})


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.index = tiny_index()
        self.requests = [("u2", "i4", (), 10 ** 12)]

    def score(self, phrase, **kwargs):
        rankers = {config: ranker_for(self.model, self.index, "recent", 0.75, config, **kwargs)
                   for config in CONFIGS[:2]}
        spoken = {condition: [parse_utterance(phrase) if (phrase and condition == "own_words")
                              else None for _ in self.requests] for condition in CONDITIONS}
        return outcomes_for(self.model, "recent", 0.75, rankers, self.requests, spoken)

    def test_a_request_with_nothing_to_say_is_scored_as_the_committed_route(self):
        results, surfaced, covered, _ = self.score("")
        target, history = self.requests[0][1], self.requests[0][2]
        quiet = self.model.top_k(history, "recent", 0.75, 10)
        for config in results:
            self.assertEqual(results[config]["own_words"], results[config][BASE])
            self.assertEqual(results[config][BASE], [ndcg_one_target(quiet, target)])
            self.assertEqual(surfaced[config]["own_words"], surfaced[config][BASE])
        self.assertEqual(covered[CONFIGS[0]]["own_words"], 0)

    def test_a_phrase_only_lifts_a_product_the_behavioural_route_was_already_considering(self):
        spoken = "high gain distortion pedal"
        wide, _, _, _ = self.score(spoken, scope="catalogue")
        held, _, _, shortlist = self.score(spoken, scope="behavioural", scope_depth=3)
        narrow = ranker_for(self.model, self.index, "recent", 0.75, CONFIGS[0], scope_depth=3)
        allowed = narrow.prepare(()).allowed
        self.assertEqual(allowed, frozenset({"i11", "i10", "i9"}))
        attraction, _, _ = narrow.phrase_scores(parse_utterance(spoken), frozenset(), allowed)
        self.assertTrue(set(attraction) <= allowed)
        self.assertNotIn("i4", set(narrow.rank((), spoken, 5)))
        free = ranker_for(self.model, self.index, "recent", 0.75, CONFIGS[0], scope="catalogue")
        self.assertIn("i4", set(free.rank((), spoken, 5)))
        self.assertEqual(shortlist["scope"], "behavioural")
        self.assertGreater(wide[CONFIGS[0]]["own_words"][0], held[CONFIGS[0]]["own_words"][0])

    def test_the_shortlist_report_says_how_much_room_the_words_had(self):
        _, _, _, shortlist = self.score("tube", scope="behavioural", scope_depth=3)
        report = shortlist_report(shortlist)
        self.assertEqual(report["requests"], shortlist["considered"])
        self.assertEqual(report["answer_was_in_scope"], shortlist["answer_was_in_scope"])
        self.assertEqual(report["coverage"],
                         round(shortlist["answer_was_in_scope"] / shortlist["considered"], 4))
        self.assertIsNone(shortlist_report({"scope": "catalogue", "considered": 5,
                                           "answer_was_in_scope": 5}))
        self.assertIsNone(shortlist_report({"scope": "behavioural", "considered": 0,
                                            "answer_was_in_scope": 0}))


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()

    def test_rank_correlation_needs_enough_common_ground_to_say_anything(self):
        left = list(CATALOG)
        self.assertEqual(spearman(left, list(left)), 1.0)
        self.assertIsNone(spearman(left[:5], list(CATALOG)))
        self.assertAlmostEqual(spearman(left, list(reversed(left))), -1.0)

    def test_ordering_that_agrees_at_different_depths_still_counts_as_agreement(self):
        agreed = [f"p{index}" for index in range(10)]
        shallow = agreed + [f"s{index}" for index in range(5)]
        deep = [f"d{index}" for index in range(100)] + agreed
        self.assertEqual(spearman(shallow, deep), 1.0)
        reversed_deep = [f"d{index}" for index in range(100)] + list(reversed(agreed))
        self.assertAlmostEqual(spearman(shallow, reversed_deep), -1.0)
        partial = agreed[:5] + [f"d{index}" for index in range(100)] + agreed[5:]
        self.assertTrue(-1.0 <= spearman(shallow, partial) <= 1.0)

    def test_buckets_follow_the_answer_s_popularity_and_the_person_s_history(self):
        requests = [("u", "i11", (), 0), ("u", "i0", ("i1", "i2"), 0),
                    ("u", "unknown", tuple(f"i{n}" for n in range(6)), 0)]
        self.assertEqual(popularity_buckets(self.model, requests, groups=4), [0, 3, 3])
        self.assertEqual(history_buckets(requests), [0, 1, 3])

    def test_the_summary_reports_the_paired_delta_and_its_gate(self):
        people = [(f"u{index}", f"i{index}", (), 0) for index in range(4)]
        results = {BASE: [0.1] * 4, "own_words": [0.2] * 4}
        surfaced = {BASE: [1, 1, 0, 0], "own_words": [1, 1, 1, 1]}
        summary = summarise(results, surfaced, {"own_words": 1}, people, "own_words",
                            results[BASE])
        self.assertEqual(summary["ndcg_at_10"], 0.2)
        self.assertEqual(summary["recall_at_10"], 1.0)
        self.assertEqual(summary["surfaced_in_top_three"], 1.0)
        self.assertEqual(summary["phrased_requests"], 1)
        self.assertEqual(summary["phrase_coverage"], 0.25)
        self.assertEqual(summary["ndcg_minus_base"]["delta"], 0.1)
        self.assertEqual(summary["ndcg_minus_base"]["users"], 4)
        self.assertGreater(summary["ndcg_minus_base"]["lower"], 0)
        self.assertTrue(summary["paired_gate_open"])

    def test_a_phrase_that_only_helps_a_few_people_does_not_clear_the_gate(self):
        people = [(f"u{index}", f"i{index}", (), 0) for index in range(30)]
        results = {BASE: [0.2] * 30,
                   "own_words": [1.7 if index < 2 else 0.2 for index in range(30)]}
        summary = summarise(results, {BASE: [0] * 30, "own_words": [0] * 30}, {"own_words": 2},
                            people, "own_words", results[BASE])
        self.assertGreater(summary["ndcg_minus_base"]["delta"], 0)
        self.assertFalse(summary["paired_gate_open"])

    def test_stratification_splits_on_the_bucket_it_was_given(self):
        results = {BASE: [0.5, 0.1, 0.9], "own_words": [0.4, 0.4, 0.9]}
        rows = stratified(results, [0, 0, 1], "own_words", "popularity")
        self.assertEqual(set(rows), {"popularity_0", "popularity_1"})
        self.assertEqual(rows["popularity_0"]["requests"], 2)
        self.assertEqual(rows["popularity_0"]["base_ndcg_at_10"], 0.3)
        self.assertEqual(rows["popularity_1"]["own_words_ndcg_at_10"], 0.9)

    def test_agreement_only_compares_requests_where_both_ways_of_wording_exist(self):
        rankers = {CONFIGS[0]: ranker_for(self.model, tiny_index(), "recent", 0.75, CONFIGS[0],
                                          scope="catalogue")}
        spoken = {"own_words": [None, parse_utterance("distortion pedal")],
                  "published": [parse_utterance("guitar"), parse_utterance("compact pedal")]}
        report = agreement([("u1", "i2", (), 0), ("u2", "i4", (), 0)], spoken, rankers, depth=12)
        self.assertEqual(report[0]["config"]["weight"], CONFIGS[0][0])
        self.assertEqual(report[0]["requests_compared"], 1)
        self.assertLessEqual(report[0]["mean_spearman"], 1.0)
        self.assertEqual(report[0]["median_overlap_at_depth"], 12)


if __name__ == "__main__":
    unittest.main()
