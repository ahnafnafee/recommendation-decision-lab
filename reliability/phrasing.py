"""Recommendations from something the person said, scored on the frozen splits.

Six ways of putting one request into words, all scored against the same requests:

    base                 nothing said at all — the committed hybrid, on its own
    own_words            the person's own sentences, about *other* items, written
                         before this request was answered
    own_words_scrubbed   the same sentences with the target's identifying words cut
                         out, so a phrase used as a lookup key shows itself
    cross_words          another person's own sentences, same register and length,
                         wrong person
    target_title         the target item's own title: an oracle that already contains
                         the answer
    random_title         some unrelated item's title: a control for "any words at all"
    published            a phrase another person actually typed for that item

The last four are phrases this lab did not have to invent; the first two are what a
running system can assemble. `own_words`, `own_words_scrubbed`, `target_title` and
`random_title` are reported only as paired differences against `base`, never as
absolute scores, because the wording was chosen with the answer in view. The
`published` column is read on its own terms: independent of behaviour but small, so
it anchors phrase shape rather than carrying the result.

Alongside the metric, each condition reports how often the products the phrase
route reached actually contained the answer. That reach is the ceiling every later
stage works under, and it is the number that explains why the metric moves little
even when the phrase is doing real work.

Aggregate JSON only: no identifiers, no review text, nothing that leaves a file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from .benchmark import (ALPHAS, SPLITS, T1, T2, evaluation_data, file_hash,
                        percentile, quality, read_rows, training_data,
                        user_cluster_interval)
from .core import Recommender, ndcg_one_target
from .language import (LexicalIndex, STOPWORDS, TextRanker, build_index, load_corpus,
                       load_queries, load_voices, own_words, parse_utterance, tokenize)

BASE = "base"
CONDITIONS = ("own_words", "own_words_scrubbed", "cross_words", "target_title",
              "random_title", "published")
CONFIGS = ((0.25, 0.0, 0.5), (0.5, 0.0, 0.5), (0.75, 0.0, 0.5), (0.75, 0.5, 0.5),
           (0.75, 0.0, 0.0))
DEPTH = 50
AGREEMENT_REQUESTS = 300
PROGRESS = 5000
COMMON_SHARE = 0.01


def timestamped_requests(path: Path, split: str, catalog: set[str]):
    """The frozen evaluation rows with their timestamp, so words can be cut in time.

    `evaluation_data` is the authority on which rows are eligible. This reads the same
    file with the same predicates and then checks, row for row, that it agrees; a
    mismatch raises rather than quietly scoring a different request set.
    """
    requests = []
    for row in read_rows(path):
        timestamp = int(row["timestamp"])
        if (split == "valid" and not T1 <= timestamp < T2) or (split == "test" and timestamp < T2):
            continue
        if float(row["rating"]) < 4:
            continue
        user, target = row["user_id"], row["parent_asin"]
        history = tuple(row["history"].split())
        if target in history:
            continue
        requests.append((user, target, history, timestamp))
    frozen, _ = evaluation_data(path, split, catalog)
    if len(requests) != len(frozen) or any(left[:3] != right
                                           for left, right in zip(requests, frozen)):
        raise ValueError(f"{split} requests drifted from the frozen evaluation filter")
    return requests


def item_titles(corpus: dict) -> dict[str, str]:
    return {asin: str(record.get("title") or "") for asin, record in corpus.items()}


def published_by_item(pairs) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for identifier, text, asin, _source in pairs:
        grouped[asin].append((identifier, text))
    return grouped


@dataclass(frozen=True)
class Phrases:
    """Everything needed to put one request into words, from material that predates it."""

    voices: dict[str, list]
    titles: dict[str, str]
    published: dict[str, list[tuple[str, str]]]
    index: LexicalIndex | None
    pool: list[str]
    own_limit: int = 3

    def identifying(self, title: str) -> set[str]:
        """Words that point at this product, as opposed to words a whole category shares.

        Frequency in the catalogue is what separates them: "tube" and "pedal" appear on
        hundreds of listings, a model name appears on one. Cutting the rare ones out of
        a phrase leaves behind only what the person could have said without the box.
        """
        total = len(self.index) if self.index is not None else 0
        words = set()
        for token in tokenize(title):
            if token in STOPWORDS:
                continue
            if self.index is None or self.index.term_document_frequency(token) / total < COMMON_SHARE:
                words.add(token)
        return words

    def scrub(self, phrase: str, title: str) -> tuple[str, int]:
        words = self.identifying(title)
        if not phrase or not words:
            return phrase, 0
        kept = [token for token in tokenize(phrase) if token not in words]
        return " ".join(kept), len(tokenize(phrase)) - len(kept)

    def for_request(self, user: str, target: str, when: int, draw: random.Random) -> dict:
        """One phrase per condition, empty where no honest phrase exists."""
        found: dict[str, str] = {}
        ready = [entry for entry in self.voices.get(user, ())
                 if entry[0] < when and entry[1] != target]
        ready.sort(key=lambda entry: -entry[0])
        found["own_words"] = own_words({user: ready}, user, limit=self.own_limit).strip()
        found["own_words_scrubbed"], removed = self.scrub(found["own_words"],
                                                          self.titles.get(target, ""))
        found["cross_words"] = ""
        found["target_title"] = self.titles.get(target, "")
        found["random_title"] = ""
        for _ in range(8):
            candidate = self.pool[draw.randrange(len(self.pool))]
            if candidate != target and self.titles.get(candidate):
                found["random_title"] = self.titles[candidate]
                break
        options = self.published.get(target) or []
        found["published"] = options[0][1] if options else ""
        found["scrubbed_tokens"] = removed
        return found


def cross_phrases(phrases: list[dict], seed: int) -> None:
    """Give each request somebody else's own words, in place."""
    texts = [row["own_words"] for row in phrases]
    filled = [index for index in random.Random(seed).sample(range(len(phrases)), len(phrases))
              if texts[index]]
    for slot, position in enumerate(filled):
        for step in range(1, len(filled) + 1):
            other = filled[(slot + step) % len(filled)]
            if other != position:
                phrases[position]["cross_words"] = texts[other]
                break


def utterances_for(phrases: list[dict]) -> dict[str, list]:
    """Parse once; the same wording is scored under every configuration.

    A phrase with no usable words left — none given, or a scrub that took everything
    — is recorded as nothing said rather than as an empty phrase, so it is not counted
    as coverage and no scoring time is spent on it.
    """
    spoken = {}
    for condition in CONDITIONS:
        rows = []
        for row in phrases:
            utterance = parse_utterance(row[condition]) if row[condition] else None
            rows.append(None if utterance is None or utterance.empty else utterance)
        spoken[condition] = rows
    return spoken


def ranker_for(model: Recommender, index, baseline: str, alpha: float, config,
               embeddings=None, candidate_cap: int = 200,
               scope: str = "behavioural", scope_depth: int = 2000) -> TextRanker:
    weight, repulsion, price_penalty = config
    return TextRanker(model, index, alpha=alpha, baseline=baseline, weight=weight,
                      repulsion=repulsion, price_penalty=price_penalty,
                      candidate_cap=candidate_cap, embeddings=embeddings, scope=scope,
                      scope_depth=scope_depth)


def outcomes_for(model: Recommender, baseline: str, alpha: float,
                 rankers: dict[tuple, TextRanker], requests, utterances, k: int = 10):
    """Score every configuration in one pass, so each phrase is understood once.

    What a phrase means does not depend on how loudly the ranker is told to trust it,
    so the route runs once per request and every configuration re-uses it. An absent
    phrase is the base route: with nothing said there is nothing to adjust, so the
    committed order is returned untouched and no time is spent pretending.
    """
    results = {config: {condition: [0.0] * len(requests)
                        for condition in (BASE,) + CONDITIONS} for config in rankers}
    surfaced = {config: {condition: [0] * len(requests)
                         for condition in (BASE,) + CONDITIONS} for config in rankers}
    covered = {config: {condition: 0 for condition in CONDITIONS} for config in rankers}
    reference = next(iter(rankers.values()))
    shortlist = {"scope": reference.scope, "considered": 0, "answer_was_in_scope": 0}
    for position, (_, target, history, _) in enumerate(requests):
        quiet = model.top_k(history, baseline, alpha, k)
        quiet_score, quiet_flag = ndcg_one_target(quiet, target), int(target in quiet[:3])
        prepared = reference.prepare(history)
        if prepared.allowed is not None:
            shortlist["considered"] += 1
            shortlist["answer_was_in_scope"] += int(target in prepared.allowed)
        for config in rankers:
            results[config][BASE][position] = quiet_score
            surfaced[config][BASE][position] = quiet_flag
        for condition in CONDITIONS:
            utterance = utterances[condition][position]
            if utterance is None:
                for config in rankers:
                    results[config][condition][position] = quiet_score
                    surfaced[config][condition][position] = quiet_flag
                continue
            attraction, repulsion = reference.route(utterance, prepared)
            for config, ranker in rankers.items():
                covered[config][condition] += 1
                ranking = ranker.rank_from(prepared, attraction, repulsion,
                                           utterance.price_ceiling, k)
                results[config][condition][position] = ndcg_one_target(ranking, target)
                surfaced[config][condition][position] = int(target in ranking[:3])
        if PROGRESS and (position + 1) % PROGRESS == 0:
            print(f"[phrase] {position + 1:,}/{len(requests):,} requests")
    return results, surfaced, covered, shortlist


def shortlist_report(shortlist: dict) -> dict | None:
    """How often the products a phrase may promote already contained the answer.

    The other half of the ceiling. A phrase cannot surface a product the behavioural
    route never considered, so this says how much room the words had to work in — and
    on a catalogue-wide run, where nothing is confined, there is nothing to report.
    """
    if shortlist["scope"] != "behavioural" or not shortlist["considered"]:
        return None
    chances = shortlist["considered"]
    return {"requests": chances,
            "answer_was_in_scope": shortlist["answer_was_in_scope"],
            "coverage": round(shortlist["answer_was_in_scope"] / chances, 4)}


def reach_for(ranker: TextRanker, requests, utterances) -> dict[str, dict]:
    """How often the products a phrase reached contained the answer.

    This belongs to the route, not to a configuration: the weight only decides how
    far the ranker is allowed to follow what the route found, so one number serves
    every weight scored in this run.
    """
    reached = {}
    for condition in CONDITIONS:
        hits, chances = 0, 0
        for position, (_, target, history, _) in enumerate(requests):
            utterance = utterances[condition][position]
            if utterance is None:
                continue
            chances += 1
            attraction = ranker.route(utterance, ranker.prepare(history))[0]
            if target in attraction:
                hits += 1
        reached[condition] = {"chances": chances, "target_reached": hits,
                              "reach": round(hits / max(1, chances), 4)}
    return reached


def popularity_buckets(model: Recommender, requests, groups: int = 5) -> list[int]:
    """Bucket each request by how strong its target's train popularity already is."""
    order = {item: index for index, item in enumerate(model.recent_order)}
    span = max(1, len(order))
    buckets = []
    for _, target, _, _ in requests:
        rank = order.get(target, len(order))
        buckets.append(min(groups - 1, int(groups * rank / span)))
    return buckets


def history_buckets(requests, edges=(0, 2, 5, 20)) -> list[int]:
    """Bucket each request by how much behaviour the person arrives with."""
    buckets = []
    for _, _, history, _ in requests:
        distinct = len(set(history))
        bucket = len(edges)
        for position, edge in enumerate(edges):
            if distinct <= edge:
                bucket = position
                break
        buckets.append(bucket)
    return buckets


def spearman(left, right):
    """Rank correlation over the items both orderings actually reached.

    Only products the two lists share carry evidence, so each ordering is re-ranked
    across that shared set before the two rank vectors are correlated. Positions in
    the full lists cannot be subtracted directly: a product that appears tenth in a
    short list and fourteenth in a long one has not moved relative to anything the
    other ordering also saw, and treating that as disagreement would report a
    difference in depth as a difference in taste.
    """
    depth_of_right = {item: index for index, item in enumerate(right)}
    shared = [item for item in left if item in depth_of_right]
    if len(shared) < 10:
        return None
    rank_left = {item: index for index, item in enumerate(shared)}
    rank_right = {item: index for index, item in
                  enumerate(sorted(shared, key=lambda item: depth_of_right[item]))}
    count = len(shared)
    mean_left = (count - 1) / 2.0
    deviations_left = [rank_left[item] - mean_left for item in shared]
    deviations_right = [rank_right[item] - mean_left for item in shared]
    spread_left = sum(value * value for value in deviations_left)
    spread_right = sum(value * value for value in deviations_right)
    if spread_left <= 0.0 or spread_right <= 0.0:
        return None
    covariance = sum(a * b for a, b in zip(deviations_left, deviations_right))
    return covariance / ((spread_left * spread_right) ** 0.5)


def describe(config) -> dict:
    return {"weight": config[0], "repulsion": config[1], "price_penalty": config[2]}


def agreement(requests, utterances, rankers: dict[tuple, TextRanker], depth: int = DEPTH,
              seed: int = 20260922) -> list[dict]:
    """Do two independent ways of wording a request order products the same way?

    On requests where the person's own words and a published phrase both exist, rank
    deep with each and correlate the orderings. Agreement means the two sources of
    language point at the same products; disagreement says one of them is not
    carrying the meaning it is being credited with.
    """
    sample = [position for position in range(len(requests))
              if utterances["own_words"][position] and utterances["published"][position]]
    if len(sample) > AGREEMENT_REQUESTS:
        sample = random.Random(seed).sample(sample, AGREEMENT_REQUESTS)
    report = []
    for config, ranker in rankers.items():
        coefficients, overlap = [], []
        for position in sample:
            _, _, history, _ = requests[position]
            left = ranker.rank(history, utterances["own_words"][position], depth)
            right = ranker.rank(history, utterances["published"][position], depth)
            value = spearman(left, right)
            if value is not None:
                coefficients.append(value)
                overlap.append(len(set(left) & set(right)))
        report.append({"config": describe(config),
                       "requests_compared": len(coefficients),
                       "mean_spearman": round(mean(coefficients), 4) if coefficients else None,
                       "median_overlap_at_depth": int(sorted(overlap)[len(overlap) // 2])
                       if overlap else 0})
    return report


def summarise(results, surfaced, covered, requests, condition, base_outcomes) -> dict:
    interval = user_cluster_interval(requests, base_outcomes, results[condition])
    return {"requests": len(requests),
            "phrased_requests": covered[condition],
            "phrase_coverage": round(covered[condition] / max(1, len(requests)), 4),
            "ndcg_at_10": round(mean(results[condition]), 6),
            "recall_at_10": round(sum(1 for value in results[condition] if value > 0)
                                  / max(1, len(requests)), 6),
            "surfaced_in_top_three": round(mean(surfaced[condition]), 6),
            "ndcg_minus_base": {"delta": round(interval["delta"], 6),
                                "lower": round(interval["lower"], 6),
                                "upper": round(interval["upper"], 6),
                                "users": interval["users"], "draws": interval["draws"]},
            "paired_gate_open": interval["lower"] > 0}


def stratified(results, buckets, condition, name: str, groups: int = 5) -> dict:
    """Popularity and history length are the two confounds a phrase can ride on."""
    rows = {}
    for bucket in sorted(set(buckets)):
        chosen = [index for index, value in enumerate(buckets) if value == bucket]
        rows[f"{name}_{bucket}"] = {
            "requests": len(chosen),
            "base_ndcg_at_10": round(mean(results[BASE][index] for index in chosen), 6),
            "own_words_ndcg_at_10": round(mean(results[condition][index] for index in chosen), 6)}
    return rows


def timed_requests(ranker, requests, utterances, sample: int = 200) -> dict:
    """Local cost of answering one spoken request, route and all, on this machine."""
    timings = []
    for position, (_, _, history, _) in enumerate(requests[:sample]):
        utterance = utterances[position]
        began = perf_counter()
        ranker.rank(history, utterance if utterance is not None else "", 10)
        timings.append((perf_counter() - began) * 1000)
    return {"requests": len(timings), "p50": round(percentile(timings, .5), 3),
            "p95": round(percentile(timings, .95), 3),
            "scope": "one phrase and one unravelling request; no HTTP, model load, or disk"}


def run(data: Path, destination: Path, text: Path | None = None,
        category: str = "Musical_Instruments", use_embeddings: bool = False,
        own_limit: int = 3, review_limit: int = 4, limit: int = 0,
        device: str = "auto", seed: int = 20260922, scope: str = "behavioural") -> dict:
    if category not in ("Video_Games", "Musical_Instruments"):
        raise ValueError("category is outside the committed study design")
    if destination.exists():
        raise FileExistsError("run directory exists; retain prior evidence")
    paths = {split: data / f"{category}.{split}.csv.gz" for split in SPLITS}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"download all three official {category} split archives")
    directory = text or (data / "text")
    started = perf_counter()
    destination.mkdir(parents=True)
    hashes = {split: file_hash(paths[split]) for split in ("train", "valid")}

    positives, catalog, training = training_data(paths["train"])
    model = Recommender.train(positives, catalog, T1)
    del positives
    corpus, manifest = load_corpus(directory, category)
    index, index_stats = build_index(corpus, catalogue=set(catalog), review_limit=review_limit)
    voices = load_voices(directory, category, before_ms=T2)
    pairs = load_queries(directory, category, catalogue=set(catalog))
    source = Phrases(voices, item_titles(corpus), published_by_item(pairs), index,
                     sorted(catalog), own_limit)
    embeddings, phrase_index = None, "lexical only"
    if use_embeddings:
        from .embeddings import load_embeddings

        embeddings, embedding_manifest = load_embeddings(
            directory, category, expected_corpus_sha256=str(manifest.get("corpus_sha256")),
            device=device)
        phrase_index = f"{embedding_manifest['model']} encoded on {embedding_manifest['device']}"

    validation = timestamped_requests(paths["valid"], "valid", catalog)
    if not validation:
        raise ValueError("no validation requests")
    if limit:
        validation = validation[:limit]
    draw = random.Random(seed)
    validation_phrases = [source.for_request(user, target, when, draw)
                          for user, target, _, when in validation]
    cross_phrases(validation_phrases, seed)
    validation_utterances = utterances_for(validation_phrases)

    lifetime = [ndcg_one_target(model.top_k(history, "lifetime", 0.0), target)
                for _, target, history, _ in validation]
    recent = [ndcg_one_target(model.top_k(history, "recent", 0.0), target)
              for _, target, history, _ in validation]
    baseline = "recent" if mean(recent) > mean(lifetime) else "lifetime"
    hybrid = {alpha: [ndcg_one_target(model.top_k(history, baseline, alpha), target)
                      for _, target, history, _ in validation] for alpha in ALPHAS}
    alpha = max(ALPHAS, key=lambda value: (mean(hybrid[value]), -value))
    base_outcomes = hybrid[alpha]
    rankers = {config: ranker_for(model, index, baseline, alpha, config, embeddings,
                                  scope=scope)
               for config in CONFIGS}
    results, surfaced, covered, validation_shortlist = outcomes_for(
        model, baseline, alpha, rankers, validation, validation_utterances)
    chosen = max(CONFIGS, key=lambda config: (mean(results[config]["own_words"]),
                                              -config[0], -config[1]))
    best = rankers[chosen]
    # The ceiling question — could the words have found the product at all — is asked
    # of an unconstrained ranker, so that confining the phrase to the shortlist is
    # not allowed to flatter its own reach.
    probe = ranker_for(model, index, baseline, alpha, chosen, embeddings, scope="catalogue")
    scrubbed_tokens = [row["scrubbed_tokens"] for row in validation_phrases if row["own_words"]]

    decision = {"category": category,
                "source_sha256": {part: hashes[part] for part in ("train", "valid")},
                "training": training,
                "text": {"corpus_sha256": manifest.get("corpus_sha256"),
                         "voices_sha256": manifest.get("voices_sha256"),
                         "people_with_own_words": len(voices),
                         "published_pairs": len(pairs),
                         "items_with_titles": sum(1 for value in source.titles.values() if value),
                         "review_limit": review_limit, "phrase_index": phrase_index,
                         **index_stats},
                "baseline": baseline, "alpha": alpha,
                "baseline_ndcg_at_10": round(mean(base_outcomes), 6),
                "configs_scored": [describe(config) for config in CONFIGS],
                "own_words_ndcg_by_config": {json.dumps(describe(config)):
                                             round(mean(results[config]["own_words"]), 6)
                                             for config in CONFIGS},
                "selected_config": {**describe(chosen),
                                    "rule": "best validation own_words NDCG@10, then smaller weights"},
                "candidate_cap": best.candidate_cap,
                "phrase_scope": scope,
                "promotion_scope_depth": best.scope_depth if scope == "behavioural" else None,
                "own_words_scrubbed_tokens_mean": round(mean(scrubbed_tokens), 3)
                if scrubbed_tokens else 0,
                "validation": {"requests": len(validation), "base": quality(base_outcomes),
                               "conditions": {condition: summarise(results[chosen], surfaced[chosen],
                                                                   covered[chosen], validation,
                                                                   condition, base_outcomes)
                                              for condition in CONDITIONS},
                               "phrase_route_reach": reach_for(probe, validation,
                                                               validation_utterances),
                               "behavioural_shortlist": shortlist_report(validation_shortlist)},
                "agreement_own_words_vs_published": agreement(validation, validation_utterances,
                                                              dict(list(rankers.items())[:3]))}
    (destination / "validation_decision.json").write_text(json.dumps(decision, indent=2),
                                                          encoding="utf-8")

    # The test archive is first opened after the configuration is frozen on disk.
    hashes["test"] = file_hash(paths["test"])
    test = timestamped_requests(paths["test"], "test", catalog)
    if limit:
        test = test[:limit]
    draw = random.Random(seed + 1)
    test_phrases = [source.for_request(user, target, when, draw)
                    for user, target, _, when in test]
    cross_phrases(test_phrases, seed + 1)
    test_utterances = utterances_for(test_phrases)
    chosen_ranker = rankers[chosen]
    outcomes, surfaced_test, covered_test, test_shortlist = outcomes_for(
        model, baseline, alpha, {chosen: chosen_ranker}, test, test_utterances)
    test_results = outcomes[chosen]
    test_surfaced, test_covered = surfaced_test[chosen], covered_test[chosen]
    buckets = popularity_buckets(model, test)
    lengths = history_buckets(test)
    aggregate = {"experiment": f"{category} phrased recommendation arm",
                 "source": "frozen 5-core temporal split", "source_sha256": hashes,
                 "validation_decision": decision,
                 "test": {"requests": len(test),
                          "base": quality(test_results[BASE]),
                          "conditions": {condition: summarise(test_results, test_surfaced,
                                                              test_covered, test, condition,
                                                              test_results[BASE])
                                         for condition in CONDITIONS},
                          "phrase_route_reach": reach_for(probe, test, test_utterances),
                          "behavioural_shortlist": shortlist_report(test_shortlist),
                          "own_words_by_target_popularity": stratified(test_results, buckets,
                                                                       "own_words",
                                                                       "popularity_quintile"),
                          "own_words_by_history_length": stratified(test_results, lengths,
                                                                    "own_words", "history"),
                          "local_request_ms": timed_requests(chosen_ranker, test,
                                                             test_utterances["own_words"])},
                 "limitations": [
                     "own_words, own_words_scrubbed, target_title and random_title are worded with the "
                     "answer in view: only their paired difference to base is reportable, never their "
                     "absolute score",
                     "own_words skips the target item's own text, yet a person's earlier sentences can "
                     "still name the product they had already decided on; the scrubbed column measures "
                     "how much of any gain is that, and it is a diagnostic, not a fix",
                     "published phrases are few and were written for other purposes, so that column "
                     "anchors phrase shape rather than carrying the result",
                     "product prose is cut at the train cutoff, but a person's own sentences inside the "
                     "evaluation period can describe the item they went on to review",
                     "one fixed temporal split per category, aggregate output only",
                     "product text and published phrases are fetched locally under their own licences, "
                     "which this repository does not redistribute"]}
    (destination / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    own = aggregate["test"]["conditions"]["own_words"]
    print(json.dumps({"output": str(destination / "aggregate.json"), "alpha": alpha,
                      "config": describe(chosen), "phrase_index": phrase_index,
                      "scope": scope,
                      "own_words_delta": own["ndcg_minus_base"]["delta"],
                      "own_words_lower": own["ndcg_minus_base"]["lower"],
                      "gate_open": own["paired_gate_open"],
                      "shortlist": aggregate["test"]["behavioural_shortlist"],
                      "reach": aggregate["test"]["phrase_route_reach"],
                      "seconds": round(perf_counter() - started, 1)}, default=str))
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--text", type=Path, help="where the corpus lives, defaults to <data>/text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("Video_Games", "Musical_Instruments"),
                        default="Musical_Instruments")
    parser.add_argument("--embeddings", action="store_true",
                        help="add the encoded phrase index alongside the lexical route")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--own-limit", type=int, default=3,
                        help="how many of a person's own sentences to read back")
    parser.add_argument("--review-limit", type=int, default=4,
                        help="review excerpts per product in the lexical index")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N requests")
    parser.add_argument("--scope", choices=("behavioural", "catalogue"), default="behavioural",
                        help="how far a phrase may reach: only products the behavioural route "
                             "was already considering, or anything on the shelf")
    args = parser.parse_args()
    run(args.data, args.output, args.text, args.category, args.embeddings, args.own_limit,
        args.review_limit, args.limit, args.device, scope=args.scope)


if __name__ == "__main__":
    main()
