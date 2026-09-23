import json
import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from reliability.core import Recommender
from reliability.language import (LexicalIndex, TextRanker, Utterance, build_index, load_corpus,
                                  load_queries, load_voices, own_words, parse_utterance,
                                  price_ceiling, rank_normalised, tokenize,
                                  top_rank_normalised)


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
    """Hand-built scores so every blend in the parity tests is checkable by hand."""
    lifetime = {item: 1.0 - 0.05 * index for index, item in enumerate(CATALOG)}
    recent = {item: 0.90 - 0.04 * index for index, item in enumerate(reversed(CATALOG))}
    neighbors = {"i1": (("i2", 0.50), ("i9", 0.20)),
                 "i3": (("i2", 0.30), ("i8", 0.60)),
                 "i7": (("i0", 0.70), ("i10", 0.10))}
    return Recommender(lifetime, recent, neighbors)


class PhraseParsingTests(unittest.TestCase):
    def test_wanted_and_ruled_out_terms_split_on_one_cue(self):
        phrase = parse_utterance("a warm tube amp under $800, not digital")
        self.assertIn("tube", phrase.positive)
        self.assertIn("amp", phrase.positive)
        self.assertIn("warm", phrase.positive)
        self.assertEqual(phrase.negative, ("digital",))
        self.assertEqual(phrase.price_ceiling, 800.0)

    def test_negation_stops_at_a_restart_conjunction(self):
        phrase = parse_utterance("no leather strap but a good tremolo")
        self.assertEqual(phrase.negative, ("leather", "strap"))
        self.assertIn("tremolo", phrase.positive)
        self.assertNotIn("tremolo", phrase.negative)

    def test_comparative_cue_rules_out_only_the_following_thing(self):
        phrase = parse_utterance("a combo rather than a head cabinet")
        self.assertIn("combo", phrase.positive)
        self.assertEqual(phrase.negative, ("head", "cabinet"))

    def test_bare_rather_is_filler_not_a_cue(self):
        phrase = parse_utterance("I rather like tube amps")
        self.assertEqual(phrase.negative, ())
        self.assertIn("like", phrase.positive)

    def test_thousands_separator_is_not_read_as_a_decimal_point(self):
        self.assertEqual(price_ceiling("under $1,200"), 1200.0)
        self.assertEqual(price_ceiling("up to 29.99"), 29.99)
        self.assertEqual(price_ceiling("cheaper than $50"), 50.0)
        self.assertIsNone(price_ceiling("a guitar"))

    def test_apostrophes_are_folded_on_both_sides(self):
        self.assertEqual(tokenize("don't"), ("dont",))
        self.assertEqual(tokenize("the amp"), ("amp",))
        phrase = parse_utterance("I dont want digital modeling")
        self.assertIn("digital", phrase.negative)
        self.assertEqual(phrase.positive, ())

    def test_empty_phrase_is_the_parity_key(self):
        self.assertTrue(parse_utterance("").empty)
        self.assertTrue(parse_utterance("...").empty)
        self.assertFalse(parse_utterance("nothing with leather").empty)
        self.assertFalse(parse_utterance("under $50").empty)


class LexicalIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = LexicalIndex(DOCS, prices={"i0": 1499.0, "i2": 89.0, "i9": 129.0})

    def test_rare_term_outranks_a_term_that_appears_everywhere(self):
        rare = self.index.score(["reverb"])
        common = self.index.score(["guitar"])
        self.assertEqual(set(rare), {"i0"})
        self.assertIn("i1", common)
        self.assertGreater(rare["i0"], max(common.values()))

    def test_unmatched_terms_score_nothing_rather_than_everything(self):
        self.assertIsNone(self.index.score(["bagpipes"]))
        self.assertIsNone(self.index.score([]))

    def test_repeated_terms_are_counted_once(self):
        once = self.index.score(["reverb"])
        twice = self.index.score(["reverb", "reverb"])
        self.assertEqual(once, twice)

    def test_bm25_weight_matches_the_closed_form(self):
        from math import log

        weights = self.index.postings_score_dict("tremolo")
        self.assertEqual(list(weights), [0])
        doc_length, average = self.index.doc_length[0], self.index.average_length
        idf = log(1 + (len(self.index) - 1 + 0.5) / (1 + 0.5))
        expected = idf * (1 * (self.index.k1 + 1)
                          / (1 + self.index.k1 * (1 - self.index.b + self.index.b * doc_length / average)))
        self.assertAlmostEqual(weights[0], expected, places=12)

    def test_score_matches_summing_the_sparse_postings(self):
        terms = ["guitar", "compact", "tube"]
        summed = {}
        for term in terms:
            for position, value in self.index.postings_score_dict(term).items():
                summed[self.index.items[position]] = summed.get(self.index.items[position], 0.0) + value
        scored = self.index.score(terms)
        self.assertEqual(set(scored), set(summed))
        for item in summed:
            self.assertAlmostEqual(scored[item], summed[item], places=5)

    def test_documents_without_text_are_indexed_but_unmatched(self):
        self.assertIn("i11", self.index.items)
        self.assertEqual(self.index.statistics()["documents_with_zero_text"], 1)

    def test_price_is_reported_only_where_known(self):
        self.assertEqual(self.index.price_of("i2"), 89.0)
        self.assertIsNone(self.index.price_of("i1"))

    def test_coverage_counts_documents_holding_any_term(self):
        self.assertAlmostEqual(self.index.coverage(["guitar"]), 4 / 12)
        self.assertEqual(self.index.coverage(["bagpipes"]), 0.0)


class NormalisationTests(unittest.TestCase):
    def test_top_hit_maps_to_one_and_last_to_zero(self):
        mapped = rank_normalised({"a": 9.0, "b": 4.0, "c": 0.5})
        self.assertEqual(mapped["a"], 1.0)
        self.assertEqual(mapped["c"], 0.0)
        self.assertGreater(mapped["a"], mapped["b"], mapped["c"])

    def test_ties_break_on_identifier_and_single_hit_is_one(self):
        self.assertEqual(rank_normalised({"b": 1.0, "a": 1.0}), {"b": 1.0, "a": 1.0})
        self.assertEqual(rank_normalised({"only": 3.0}), {"only": 1.0})
        self.assertEqual(rank_normalised({}), {})

    def test_keeping_the_leaders_first_returns_exactly_those_leaders(self):
        scores = {f"i{n}": float(n) for n in range(60)}
        head = top_rank_normalised(scores, 5)
        self.assertEqual(set(head), {"i59", "i58", "i57", "i56", "i55"})
        self.assertEqual(head["i59"], 1.0)
        self.assertEqual(head["i55"], 0.2)

    def test_the_twentieth_match_is_not_scored_as_a_tie_with_the_first(self):
        scores = {f"i{n}": 12000.0 - n for n in range(12000)}
        head = top_rank_normalised(scores, 200)
        whole = rank_normalised(scores)
        self.assertEqual(len(head), 200)
        self.assertLess(min(head.values()), min(whole[item] for item in head))

    def test_an_empty_or_oversized_request_costs_nothing_extra(self):
        self.assertEqual(top_rank_normalised({}, 10), {})
        scores = {"a": 2.0, "b": 1.0}
        self.assertEqual(set(top_rank_normalised(scores, 200)), {"a", "b"})


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        records = {item: {"title": DOCS[item]} for item in CATALOG}
        records["i0"]["price"], records["i2"]["price"], records["i9"]["price"] = 1499.0, 89.0, 129.0
        self.index, _ = build_index(records, set(CATALOG))

    def brute(self, ranker, history, phrase, k):
        seen = set(history)
        base = ranker.model.lifetime if ranker.baseline == "lifetime" else ranker.model.recent
        order = ranker.model.lifetime_order if ranker.baseline == "lifetime" else ranker.model.recent_order
        personalised = ranker._personalised(history, seen)
        attraction, repulsion, overpriced = ranker.phrase_scores(parse_utterance(phrase), set(base))
        adjusted = {item: (ranker.weight * attraction.get(item, 0.0)
                           - ranker.repulsion * repulsion.get(item, 0.0)
                           - ranker.price_penalty * (1.0 if item in overpriced else 0.0))
                    for item in set(attraction) | set(repulsion) | set(overpriced)}
        scored = [(item, (1 - ranker.alpha) * base[item] + ranker.alpha * personalised.get(item, 0.0)
                   + adjusted.get(item, 0.0)) for item in order if item not in seen]
        return tuple(item for item, _ in sorted(scored, key=lambda pair: (-pair[1], -base[pair[0]], pair[0]))[:k])

    def test_zero_weight_reproduces_the_committed_hybrid_exactly(self):
        ranker = TextRanker(self.model, self.index, alpha=0.75, baseline="recent", weight=0.0)
        for history, phrase in ((("i1", "i3"), "tube amplifier"), ((), "distortion"),
                                (("i7",), "under $50")):
            self.assertEqual(ranker.rank(history, phrase, 5),
                             self.model.top_k(history, "recent", 0.75, 5))

    def test_empty_phrase_reproduces_the_committed_hybrid_exactly(self):
        ranker = TextRanker(self.model, self.index, weight=0.6, repulsion=0.4, price_penalty=0.3)
        self.assertEqual(ranker.rank(("i1", "i3"), "", 6), self.model.top_k(("i1", "i3"), "recent", 0.75, 6))

    def test_phrase_pulls_a_matching_item_above_a_popular_one(self):
        ranker = TextRanker(self.model, self.index, weight=0.9)
        quiet = ranker.rank((), "", 3)
        spoken = ranker.rank((), "high gain distortion pedal", 3)
        self.assertIn("i4", spoken)
        self.assertNotIn("i4", quiet)

    def test_pooled_search_is_exact_against_a_full_scan(self):
        cases = [("distortion pedal", 0.9, 0.0, 0.0), ("tube", 0.5, 0.5, 0.0),
                 ("guitar not leather", 0.7, 0.3, 0.0), ("under $100", 0.0, 0.0, 0.5),
                 ("amp no digital", 0.8, 0.6, 0.2), ("compact", 0.25, 0.9, 0.1)]
        for phrase, weight, repulsion, penalty in cases:
            for history in ((), ("i1",), ("i3", "i7", "i0")):
                ranker = TextRanker(self.model, self.index, weight=weight, repulsion=repulsion,
                                    price_penalty=penalty)
                self.assertEqual(ranker.rank(history, phrase, 5), self.brute(ranker, history, phrase, 5),
                                 f"mismatch for {phrase!r} with history {history}")

    def test_candidate_cap_limits_how_many_items_a_phrase_can_reach(self):
        broad = TextRanker(self.model, self.index, weight=0.9, candidate_cap=200)
        narrow = TextRanker(self.model, self.index, weight=0.9, candidate_cap=2)
        wide, _, _ = broad.phrase_scores(parse_utterance("guitar"), set(self.model.recent))
        tight, _, _ = narrow.phrase_scores(parse_utterance("guitar"), set(self.model.recent))
        self.assertEqual(len(wide), 4)
        self.assertEqual(len(tight), 2)
        self.assertTrue(set(tight) <= set(wide))

    def test_price_ceiling_demotes_priced_items_only(self):
        ranker = TextRanker(self.model, self.index, weight=0.0, price_penalty=1.0)
        utterance = parse_utterance("under $100")
        _, _, overpriced = ranker.phrase_scores(utterance, set(self.model.recent))
        self.assertEqual(overpriced, {"i0", "i9"})
        ranker = TextRanker(self.model, self.index, weight=0.0, price_penalty=1.0)
        demoted = ranker.rank((), "under $100", 5)
        self.assertNotIn("i0", demoted)

    def test_a_stated_budget_can_be_honoured_instead_of_scored(self):
        scoring = TextRanker(self.model, self.index, weight=0.0, price_penalty=0.5)
        honoured = TextRanker(self.model, self.index, weight=0.0, price_penalty=0.5,
                              enforce_ceiling=True)
        breadth = len(CATALOG) + 2
        self.assertIn("i0", set(scoring.rank((), "under $100", breadth)))
        eligible = set(honoured.prepare(()).eligible)
        self.assertEqual(set(honoured.rank((), "under $100", breadth)),
                         {item for item in eligible if item not in {"i0", "i9"}})
        self.assertIn("i2", honoured.rank((), "under $100", breadth))

    def test_a_budget_nothing_meets_answers_anyway(self):
        expensive = LexicalIndex(DOCS, prices={item: 900.0 for item in DOCS})
        ranker = TextRanker(self.model, expensive, weight=0.0, price_penalty=0.5,
                            enforce_ceiling=True)
        self.assertEqual(ranker.rank((), "under $100", 5), self.model.top_k((), "recent", 0.75, 5))

    def test_repulsion_cannot_raise_an_item(self):
        ranker = TextRanker(self.model, self.index, weight=0.0, repulsion=0.8)
        with_phrase = ranker.rank((), "no distortion", 5)
        without = self.model.top_k((), "recent", 0.75, 5)
        for item in with_phrase:
            self.assertLessEqual(with_phrase.index(item),
                                 without.index(item) if item in without else len(without))

    def test_history_items_never_come_back(self):
        ranker = TextRanker(self.model, self.index, weight=1.0)
        ranking = ranker.rank(("i2", "i1"), "distortion pedal guitar", 5)
        self.assertFalse({"i2", "i1"} & set(ranking))


class StandInEmbeddings:
    """A phrase index that returns recorded similarities, so fusion is checkable."""

    def __init__(self, scores):
        self.scores = dict(scores)
        self.queries = []

    def query_scores(self, text):
        self.queries.append(text)
        return dict(self.scores)


class EmbeddingFusionTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        records = {item: {"title": DOCS[item]} for item in CATALOG}
        self.index, _ = build_index(records, set(CATALOG))

    def ranker(self, scores, **options):
        return TextRanker(self.model, self.index, embeddings=StandInEmbeddings(scores), **options)

    def test_a_phrase_index_reaches_an_item_the_written_words_miss(self):
        far = {item: 0.05 for item in CATALOG}
        far["i5"] = 0.95
        ranker = self.ranker(far, weight=0.9)
        quiet = self.model.top_k((), "recent", 0.75, 3)
        spoken = ranker.rank((), "something for a live room", 3)
        self.assertIn("i5", spoken)
        self.assertNotIn("i5", quiet)

    def test_the_whole_phrase_is_encoded_not_a_bag_of_keywords(self):
        ranker = self.ranker({item: 0.1 for item in CATALOG})
        ranker.phrase_scores(parse_utterance("a warm tube amp under $800"), set(self.model.recent))
        self.assertEqual(ranker.embeddings.queries,
                         ["a warm tube amp under $800"])

    def test_each_route_is_put_on_its_own_scale_before_fusing(self):
        scores = {item: 0.4 - 0.02 * index for index, item in enumerate(CATALOG)}
        scores["i11"] = -0.3
        attraction, _, _ = self.ranker(scores, weight=0.9).phrase_scores(
            parse_utterance("distortion"), set(self.model.recent))
        self.assertNotIn("i11", attraction)
        self.assertEqual(max(attraction.values()), 1.0)
        self.assertEqual(attraction["i0"], 1.0)
        self.assertLess(attraction["i9"], 1.0)

    def test_a_phrase_only_index_reaches_the_items_the_ranker_would_lift(self):
        scores = {item: 0.9 - 0.05 * index for index, item in enumerate(CATALOG)}
        attraction, _, _ = self.ranker(scores, weight=0.9).phrase_scores(
            parse_utterance("warm"), set(self.model.recent))
        self.assertTrue(set(attraction) <= set(self.model.recent))
        self.assertEqual(len(attraction), len(CATALOG))

    def test_the_cap_applies_to_what_a_phrase_index_reaches(self):
        scores = {item: 0.9 - 0.05 * index for index, item in enumerate(CATALOG)}
        wide, _, _ = self.ranker(scores, candidate_cap=200).phrase_scores(
            parse_utterance("warm"), set(self.model.recent))
        tight, _, _ = self.ranker(scores, candidate_cap=3).phrase_scores(
            parse_utterance("warm"), set(self.model.recent))
        self.assertEqual(len(wide), len(CATALOG))
        self.assertEqual(len(tight), 3)
        self.assertTrue(set(tight) <= set(wide))

    def test_a_phrase_with_nothing_to_score_produces_no_attraction(self):
        ranker = self.ranker({item: 0.5 for item in CATALOG})
        attraction, _, _ = ranker.phrase_scores(parse_utterance(""), set(self.model.recent))
        self.assertEqual(attraction, {})
        self.assertEqual(ranker.embeddings.queries, [])


class CorpusLoadingTests(unittest.TestCase):
    def write_corpus(self, directory: Path, category="Widget_World", schema=3, records=None):
        records = records if records is not None else [
            {"asin": "a", "title": "alpha widget", "review_prose": ["solid build"]},
            {"asin": "b", "title": "beta gadget", "price": 12.5}]
        with gzip.open(directory / f"{category}.corpus.jsonl.gz", "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        (directory / f"{category}.corpus_manifest.json").write_text(json.dumps(
            {"schema": schema, "category": category, "catalogue_items": len(records)}), encoding="utf-8")
        return records

    def test_published_pairs_are_filtered_to_the_catalogue(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            rows = [{"query_id": "esci:1", "text": "a tuner for a seven string", "asin": "a",
                     "source": "esci"},
                    {"query_id": "c4:2", "text": "something for the stage", "asin": "gone",
                     "source": "c4"}]
            with gzip.open(directory / "Widget_World.queries.jsonl.gz", "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            self.assertEqual(load_queries(directory, "Widget_World", {"a", "b"}),
                             [("esci:1", "a tuner for a seven string", "a", "esci")])
            self.assertEqual(load_queries(directory, "Absent_Category"), [])

    def test_corpus_round_trip_and_index_build(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            self.write_corpus(directory)
            records, manifest = load_corpus(directory, "Widget_World", {"a", "b"})
            self.assertEqual(manifest["schema"], 3)
            index, stats = build_index(records, {"a", "b"})
            self.assertEqual(stats["indexed"], 2)
            self.assertEqual(stats["with_price"], 1)
            self.assertEqual(stats["without_any_text"], 0)
            self.assertEqual(set(index.score(["alpha"])), {"a"})

    def test_a_corpus_from_another_cut_or_another_category_is_refused(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            self.write_corpus(directory)
            with self.assertRaises(ValueError):
                load_corpus(directory, "Other_Place")
            self.write_corpus(directory, schema=1)
            with self.assertRaises(ValueError):
                load_corpus(directory, "Widget_World")

    def test_a_thin_corpus_is_refused_before_it_silently_starves_the_index(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            self.write_corpus(directory)
            with self.assertRaises(ValueError):
                load_corpus(directory, "Widget_World", {"a", "x", "y", "z"})

    def test_own_words_are_cut_by_time_and_can_drop_the_target(self):
        with TemporaryDirectory() as raw:
            directory = Path(raw)
            directory.joinpath("Widget_World.voices.jsonl.gz").with_suffix("")
            with gzip.open(directory / "Widget_World.voices.jsonl.gz", "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({"user_id": "u1", "entries": [
                    [300, "a", 5.0, "newest words"], [200, "b", 4.0, "middle words"],
                    [100, "c", 5.0, "oldest words"]]}) + "\n")
            bank = load_voices(directory, "Widget_World")
            self.assertEqual(len(bank["u1"]), 3)
            self.assertEqual(len(load_voices(directory, "Widget_World", before_ms=250)["u1"]), 2)
            self.assertEqual(own_words(bank, "u1", limit=1), "newest words")
            self.assertEqual(own_words(bank, "u1", exclude_item="a", limit=2), "middle words oldest words")
            self.assertEqual(own_words(bank, "nobody"), "")


class UtteranceValueTests(unittest.TestCase):
    def test_raw_text_is_carried_but_not_compared(self):
        self.assertEqual(Utterance(("amp",), (), None, "first"), Utterance(("amp",), (), None, "second"))
        self.assertEqual(Utterance((), (), None, "x").positive, ())


class PromotionScopeTests(unittest.TestCase):
    """Whether a phrase may resurrect a product nobody was going to suggest."""

    def setUp(self):
        self.model = tiny_model()
        records = {item: {"title": DOCS[item]} for item in CATALOG}
        self.index, _ = build_index(records, set(CATALOG))

    def test_the_default_still_lets_a_phrase_reach_the_whole_shelf(self):
        free = TextRanker(self.model, self.index, weight=0.9)
        prepared = free.prepare(())
        self.assertIsNone(prepared.allowed)
        self.assertIn("i4", free.rank((), "high gain distortion pedal", 5))

    def test_a_confined_phrase_only_promotes_what_was_already_being_considered(self):
        held = TextRanker(self.model, self.index, weight=0.9, scope="behavioural", scope_depth=3)
        prepared = held.prepare(())
        self.assertEqual(prepared.allowed, frozenset({"i11", "i10", "i9"}))
        spoken = parse_utterance("high gain distortion pedal")
        attraction, _, _ = held.phrase_scores(spoken, frozenset(), prepared.allowed)
        self.assertTrue(set(attraction) <= prepared.allowed)
        self.assertNotIn("i4", held.rank((), "high gain distortion pedal", 5))
        self.assertIn("i9", held.rank((), "distortion pedal", 5))

    def test_products_the_person_reached_through_are_always_promotable(self):
        held = TextRanker(self.model, self.index, weight=0.9, scope="behavioural", scope_depth=1)
        prepared = held.prepare(("i1",))
        self.assertEqual(prepared.allowed, frozenset({"i11", "i2", "i9"}))

    def test_a_phrase_still_names_products_outside_the_scope_it_is_allowed(self):
        held = TextRanker(self.model, self.index, weight=0.9, scope="behavioural", scope_depth=3)
        prepared = held.prepare(())
        open_route, _, _ = held.phrase_scores(parse_utterance("high gain distortion pedal"),
                                              frozenset(), None)
        confined, _, _ = held.phrase_scores(parse_utterance("high gain distortion pedal"),
                                            frozenset(), prepared.allowed)
        self.assertIn("i4", open_route)
        self.assertNotIn("i4", confined)


if __name__ == "__main__":
    unittest.main()
