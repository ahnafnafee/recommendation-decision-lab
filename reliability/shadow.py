"""Shadow profiles for the recommender.

A shadow profile is whatever the system assembles beyond the history deliberately
attached to the current request. Some of it is already held and needs no new user
action: star ratings the positive-only pipeline discarded, history items missing from
the train-known catalog, the size of a history, first-purchase behaviour of analogous
users, and activity on an adjacent surface. The rest is elicited by a bounded
interview, priced in questions instead of being hidden inside a score.

All fitting uses training rows only. Elicited dislike can make a personalized score
negative, so the sparse candidate argument in ``core`` needs one extra step:
``ShadowRecommender.rank`` widens the base slice by the number of depressed
candidates, and ``tests/test_shadow.py`` checks the result against a full-catalog
scan.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from heapq import nsmallest
import json
from math import log1p
from pathlib import Path
from statistics import mean
from time import perf_counter

from .benchmark import (ALPHAS as HYBRID_ALPHAS, SOURCE, T1, T2, file_hash, quality,
                        training_data, user_cluster_interval)
from .core import Recommender, last_distinct, ndcg_one_target

NEGATIVE_RATING = 3
POSITIVE_RATING = 4
RATING_WEIGHT = {5: 1.0, 4: 0.6}
HALF_LIVES = (None, 1095, 365, 90)
GAMMAS = (0.0, 0.25, 0.5, 1.0)
REPULSION_USED = 1.0
QUESTION_BUDGETS = (0, 3, 10, 30, 100)
INTERVIEW_DEPTH = 200
BUCKET_BOUNDS = ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 19, "6-19"),
                 (20, 1 << 60, "20+"))
DAY_MS = 86_400_000


def bucket_of(size: int) -> str:
    for low, high, label in BUCKET_BOUNDS:
        if low <= size <= high:
            return label
    raise AssertionError("unbucketed history size")


def read_archive(path: Path):
    import csv
    import gzip
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"user_id", "parent_asin", "rating", "timestamp", "history"}:
            raise ValueError(f"unexpected fields in {path.name}")
        yield from reader


def observed_records(paths: dict[str, Path], splits=("train",)):
    """Ratings and timestamps the platform already holds, keyed by user then item.

    ``("train",)`` is what an inferred profile may read. Reading every split is allowed
    only as the answer oracle of the simulated interview, where the user declares an
    answer now about an item they will later review.
    """
    records: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for split in splits:
        for row in read_archive(paths[split]):
            timestamp = int(row["timestamp"])
            if split == "train" and timestamp >= T1:
                raise ValueError("training record crosses the global cutoff")
            stored = records[row["user_id"]]
            item = row["parent_asin"]
            previous = stored.get(item)
            if previous is None or timestamp >= previous[1]:
                stored[item] = (max(1, min(5, int(float(row["rating"])))), timestamp)
    return dict(records)


def novice_prior(positives, catalog):
    """Item distribution of each training user's first positive review.

    This is the derived prior for a request with nothing personal to rank from:
    the aggregate first-purchase shape of comparable users, normalized like the
    popularity controls so blend weights stay on the same scale.
    """
    earliest: dict[str, tuple[int, str]] = {}
    for user, item, timestamp in positives:
        current = earliest.get(user)
        if current is None or timestamp < current[0]:
            earliest[user] = (timestamp, item)
    counts: Counter[str] = Counter(item for _, item in earliest.values())
    if not counts:
        raise ValueError("no first positive reviews")
    scale = log1p(max(counts.values()))
    return {item: log1p(counts.get(item, 0)) / scale for item in catalog}


def interview_order(positives, depth: int = INTERVIEW_DEPTH):
    """Most-reviewed catalogue items first: the questions likeliest to land."""
    counts: Counter[str] = Counter(item for _, item, _ in positives)
    return tuple(item for item, _ in sorted(counts.items(), key=lambda row: (-row[1], row[0])))[:depth]


def adjacent_projection(donor_positives, target_positives, depth: int = 50):
    """Link donor-surface items to target items through the users who share both.

    ``donor_positives`` and ``target_positives`` map a user to positive items on each
    surface. A donor item is linked to the target items the overlapping users liked,
    weighted by one over the donor item's overlapping-user fan and truncated like a
    neighbourhood, so an adjacent-surface history can seed a profile at all.
    """
    donors: dict[str, set[str]] = defaultdict(set)
    for user, items in donor_positives.items():
        if target_positives.get(user):
            for item in items:
                donors[item].add(user)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for donor_item, users in donors.items():
        share = 1.0 / len(users)
        for user in users:
            for target in target_positives[user]:
                scores[donor_item][target] = scores[donor_item].get(target, 0.0) + share
    return {donor_item: tuple(sorted(values.items(), key=lambda row: (-row[1], row[0]))[:depth])
            for donor_item, values in scores.items()}


@dataclass(frozen=True)
class ShadowRecommender:
    model: Recommender
    records: dict[str, dict[str, tuple[int, int]]]
    novice: dict[str, float]
    oracle: dict[str, dict[str, tuple[int, int]]] | None = None
    questions: tuple[str, ...] = ()
    projection: dict[str, tuple[tuple[str, float], ...]] | None = None

    @cached_property
    def bases(self) -> dict[str, dict[str, float]]:
        return {"lifetime": self.model.lifetime, "recent": self.model.recent, "novice": self.novice}

    @cached_property
    def base_orders(self) -> dict[str, tuple[str, ...]]:
        return {"lifetime": self.model.lifetime_order, "recent": self.model.recent_order,
                "novice": tuple(sorted(self.novice, key=lambda item: (-self.novice[item], item)))}

    def evidence(self, user: str, history, request_ms: int, *, graded: bool = False,
                 half_life: int | None = None):
        """Split a history into attraction and repulsion using only held records."""
        attraction: dict[str, float] = {}
        repulsion: dict[str, float] = {}
        for item in last_distinct(history):
            record = self.records.get(user, {}).get(item)
            if record is None:
                # A history item the platform never scored stays a co-review signal:
                # presence only, no inferred sentiment, as in the committed arm.
                attraction[item] = max(attraction.get(item, 0.0), 1.0)
                continue
            rating, timestamp = record
            if rating <= NEGATIVE_RATING:
                # Presence is still co-review evidence, so it stays in the
                # attraction pool at full weight. The discarded star rating enters
                # only through repulsion, which keeps gamma=0 equal to the committed arm.
                attraction[item] = max(attraction.get(item, 0.0), 1.0)
                repulsion[item] = 1.0
                continue
            weight = RATING_WEIGHT[rating] if graded else 1.0
            if half_life:
                weight *= 0.5 ** (max(0, request_ms - timestamp) / DAY_MS / half_life)
            attraction[item] = max(attraction.get(item, 0.0), weight)
        return attraction, repulsion

    def interview(self, user: str, target: str, known, budgets):
        """Simulated interview over the most-reviewed catalogue items.

        Items already in the profile and the current target are never asked about, so
        an answer is never the label of the item being scored. An item the user has no
        observable rating for still costs a question and yields nothing. Returns one
        entry per asked item (``None`` when nothing was learned) plus per-budget cost
        accounting, where budgets are nested prefixes of the same question order.
        """
        oracle = (self.oracle or {}).get(user, {})
        known = set(known)
        ceiling = max(budgets)
        asked = [item for item in self.questions if item != target and item not in known][:ceiling]
        replies: list[tuple[str, float] | None] = []
        totals: dict[int, tuple[int, int, int, int]] = {0: (0, 0, 0, 0)}
        positives = negatives = 0
        for index, item in enumerate(asked, start=1):
            record = oracle.get(item)
            reply = None
            if record is not None and record[0] >= POSITIVE_RATING:
                reply, positives = (item, 1.0), positives + 1
            elif record is not None and record[0] <= NEGATIVE_RATING:
                reply, negatives = (item, -1.0), negatives + 1
            replies.append(reply)
            totals[index] = (index, positives + negatives, positives, negatives)
        costs = {budget: totals[min(budget, len(asked))] for budget in budgets}
        return asked, replies, costs

    def sums(self, attraction, repulsion, projected=None):
        """Expand profile items into catalogue scores; mirrors ``Recommender.rank_grid``."""
        neighbors, catalog = self.model.neighbors, self.bases["lifetime"]
        positive: dict[str, float] = defaultdict(float)
        negative: dict[str, float] = defaultdict(float)
        for source, weight in attraction.items():
            for item, similarity in neighbors.get(source, ()):
                if item in catalog:
                    positive[item] += weight * similarity
        for item, value in (projected or {}).items():
            positive[item] += value
        for source, weight in repulsion.items():
            for item, similarity in neighbors.get(source, ()):
                if item in catalog:
                    negative[item] += weight * similarity
        return positive, negative

    def rank(self, positive, negative, base_name: str, alphas, gammas, seen: set[str],
             k: int = 10) -> dict[tuple[float, float], tuple[str, ...]]:
        """Exact full-catalogue top-k for each (alpha, gamma) cell.

        Items outside ``positive`` and ``negative`` score as ``(1 - alpha) * base``,
        so they can only enter through the best base items. With repulsion some
        candidates fall below that floor, so the base slice is widened by exactly the
        number of depressed candidates; without repulsion this reduces to the
        candidate argument in ``core.Recommender.rank_grid``.
        """
        if base_name not in self.bases:
            raise ValueError("unknown score base")
        alphas, gammas = tuple(alphas), tuple(gammas)
        if (not alphas or not gammas or k < 1
                or any(not 0 <= alpha <= 1 for alpha in alphas) or any(gamma < 0 for gamma in gammas)):
            raise ValueError("blend or repulsion weight outside the committed grid")
        base, order = self.bases[base_name], self.base_orders[base_name]
        eligible = [item for item in order if item not in seen]
        candidates = set(positive) | set(negative)
        candidates.difference_update(seen)
        result = {}
        for alpha in alphas:
            for gamma in gammas:
                if alpha == 0:
                    result[(alpha, gamma)] = tuple(eligible[:k])
                    continue
                net = {item: positive.get(item, 0.0) - gamma * negative.get(item, 0.0)
                       for item in candidates}
                depressed = sum(1 for value in net.values() if value < 0)
                pool = set(eligible[:k + depressed])
                pool.update(net)
                result[(alpha, gamma)] = tuple(nsmallest(
                    k, pool,
                    key=lambda item: (-(1 - alpha) * base[item] - alpha * net.get(item, 0.0),
                                      -base[item], item)))
        return result


def shadow_queries(path: Path, split: str, catalog: set[str]):
    """Same cohort as ``benchmark.evaluation_data`` plus the request timestamp."""
    queries = []
    stats = {"rows": 0, "below_positive_threshold": 0, "seen_target_excluded": 0,
             "new_item_targets": 0, "empty_history": 0}
    for row in read_archive(path):
        timestamp = int(row["timestamp"])
        if (split == "valid" and not T1 <= timestamp < T2) or (split == "test" and timestamp < T2):
            raise ValueError("evaluation row outside source split")
        stats["rows"] += 1
        if float(row["rating"]) < POSITIVE_RATING:
            stats["below_positive_threshold"] += 1
            continue
        user, target = row["user_id"], row["parent_asin"]
        history = tuple(row["history"].split())
        if target in history:
            stats["seen_target_excluded"] += 1
            continue
        if target not in catalog:
            stats["new_item_targets"] += 1
        if not history:
            stats["empty_history"] += 1
        queries.append((user, target, history, timestamp))
    stats["eligible_requests"] = len(queries)
    stats["eligible_users"] = len({row[0] for row in queries})
    return queries, stats


def score_requests(shadow: ShadowRecommender, queries, base_name: str, alphas, gammas, *,
                   graded=False, half_life=None, novice_for_cold=False, budgets=(0,),
                   projection=None, k=10):
    """Per-request NDCG for every (alpha, gamma, question budget) cell.

    Question budgets are nested prefixes of one interview, so evidence is added
    incrementally rather than recomputed per budget. ``projection`` is an optional
    callable mapping a request to adjacent-surface evidence, used only when a request
    has nothing personalisable in its own history.
    """
    alphas, gammas = tuple(alphas), tuple(gammas)
    budgets = tuple(budgets)
    outcomes = {(alpha, gamma, budget): [] for alpha in alphas for gamma in gammas
                for budget in budgets}
    costs = {budget: {"asked": 0, "answers": 0, "positive": 0, "negative": 0} for budget in budgets}
    stats = {"unpersonalisable_requests": 0, "novice_route_requests": 0, "adjacent_route_requests": 0}
    timings_ms = []
    ceiling = max(budgets)
    neighbors = shadow.model.neighbors
    for user, target, history, timestamp in queries:
        started = perf_counter()
        attraction, repulsion = shadow.evidence(user, history, timestamp, graded=graded,
                                                half_life=half_life)
        positive, negative = shadow.sums(attraction, repulsion)
        personalisable = bool(positive)
        projected = {} if personalisable or projection is None else projection(user, history)
        if projected:
            positive = {**positive, **projected}
            stats["adjacent_route_requests"] += 1
        elif not personalisable:
            stats["unpersonalisable_requests"] += 1
        novice_route = novice_for_cold and not personalisable and not projected
        if novice_route:
            stats["novice_route_requests"] += 1
        base_for_request = "novice" if novice_route else base_name
        seen = set(history)
        asked, replies, interview_costs = (shadow.interview(user, target, attraction.keys() | repulsion.keys(),
                                                            budgets) if ceiling
                                           else ([], [], {0: (0, 0, 0, 0)}))
        cursor = 0
        for budget in sorted(budgets):
            while cursor < budget and cursor < len(asked):
                reply = replies[cursor]
                cursor += 1
                if reply is None:
                    continue
                item, sign = reply
                for neighbour, similarity in neighbors.get(item, ()):
                    if neighbour in seen:
                        continue
                    if sign > 0:
                        positive[neighbour] = positive.get(neighbour, 0.0) + similarity
                    else:
                        negative[neighbour] = negative.get(neighbour, 0.0) + similarity
            asked_count, answers, positives, negatives = interview_costs[budget]
            costs[budget]["asked"] += asked_count
            costs[budget]["answers"] += answers
            costs[budget]["positive"] += positives
            costs[budget]["negative"] += negatives
            rankings = shadow.rank(positive, negative, base_for_request, alphas, gammas, seen, k)
            for alpha in alphas:
                for gamma in gammas:
                    outcomes[(alpha, gamma, budget)].append(
                        ndcg_one_target(rankings[(alpha, gamma)], target))
        timings_ms.append((perf_counter() - started) * 1000)
    return outcomes, costs, stats, timings_ms


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))] if ordered else None


def report(stage: str, started: float):
    from sys import stderr
    print(f"[shadow] {stage} ({perf_counter() - started:.0f}s)", file=stderr, flush=True)


def run(data: Path, destination: Path, category="Musical_Instruments", include_test=False,
        donor_category=None):
    if category not in ("Video_Games", "Musical_Instruments"):
        raise ValueError("category is outside the committed study design")
    if destination.exists():
        raise FileExistsError("run directory exists; retain prior evidence")
    paths = {split: data / f"{category}.{split}.csv.gz" for split in ("train", "valid", "test")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"download all three official {category} timestamp split archives")
    destination.mkdir(parents=True)
    started = perf_counter()
    positives, catalog, train_stats = training_data(paths["train"])
    model = Recommender.train(positives, catalog, T1)
    records = observed_records(paths, splits=("train",))
    oracle = observed_records(paths, splits=("train", "valid", "test"))
    novice = novice_prior(positives, catalog)
    questions = interview_order(positives)
    shadow = ShadowRecommender(model, records, novice, oracle, questions)

    report("model, inferred records, answer oracle and priors built", started)
    validation, validation_stats = shadow_queries(paths["valid"], "valid", catalog)
    if not validation:
        raise ValueError("no validation queries")
    lifetime = [ndcg_one_target(model.top_k(history, "lifetime", 0.0), target)
                for _, target, history, _ in validation]
    recent = [ndcg_one_target(model.top_k(history, "recent", 0.0), target)
              for _, target, history, _ in validation]
    lifetime_quality, recent_quality = quality(lifetime), quality(recent)
    baseline = "recent" if recent_quality["ndcg_at_10"] > lifetime_quality["ndcg_at_10"] else "lifetime"
    base_outcomes = recent if baseline == "recent" else lifetime

    hybrid_cells, _, hybrid_stats, hybrid_timing = score_requests(
        shadow, validation, baseline, (0.0,) + HYBRID_ALPHAS, (0.0,))
    hybrid_alpha = max(HYBRID_ALPHAS, key=lambda alpha: (mean(hybrid_cells[(alpha, 0.0, 0)]), -alpha))
    hybrid_outcomes = hybrid_cells[(hybrid_alpha, 0.0, 0)]
    hybrid_gate = user_cluster_interval(validation, base_outcomes, hybrid_outcomes)
    active_name = "hybrid" if hybrid_gate["lower"] > 0 else baseline
    active_outcomes = hybrid_outcomes if active_name == "hybrid" else base_outcomes

    arms: dict[str, dict] = {}

    def best_cell(cells, exclude_budget=True):
        candidates = [cell for cell in cells if not exclude_budget or cell[2] == 0]
        return max(candidates, key=lambda cell: (mean(cells[cell]), -cell[0], -cell[1]))

    def register(name, cells, note="", extra=None, timing=None):
        alpha, gamma, _ = best_cell(cells)
        outcomes = cells[(alpha, gamma, 0)]
        arms[name] = {"ndcg_at_10": quality(outcomes)["ndcg_at_10"],
                      "recall_at_10": quality(outcomes)["recall_at_10"],
                      "alpha": alpha, "gamma": gamma,
                      "delta_versus_active": user_cluster_interval(validation, active_outcomes, outcomes),
                      "grid": {f"{a}|{g}": round(quality(v)["ndcg_at_10"], 6)
                               for (a, g, _), v in sorted(cells.items())},
                      "note": note}
        if timing is not None:
            arms[name]["local_cpu_request_ms_p95"] = percentile(timing, .95)
        if extra:
            arms[name].update(extra)
        report(f"{name} scored", started)

    repulsion_cells, _, _, repulsion_timing = score_requests(shadow, validation, baseline, (hybrid_alpha,), GAMMAS)
    register("C1_repulsion", repulsion_cells,
             "gamma=0 reproduces the committed hybrid; larger gamma penalises items the user rated at most 3",
             timing=repulsion_timing)

    weighting_grid, weighting_cells, weighting_timing = {}, {}, {}
    for half_life in HALF_LIVES:
        for graded in (False, True):
            cells, _, _, cells_timing = score_requests(shadow, validation, baseline, (hybrid_alpha,), (0.0,),
                                                       graded=graded, half_life=half_life)
            key = f"half_life={half_life}|graded={graded}"
            weighting_cells[key] = cells[(hybrid_alpha, 0.0, 0)]
            weighting_timing[key] = cells_timing
            weighting_grid[key] = round(quality(weighting_cells[key])["ndcg_at_10"], 6)
    best_weight = max(weighting_grid, key=weighting_grid.get)
    register("C2_graded_decay", {(hybrid_alpha, 0.0, 0): weighting_cells[best_weight]},
             f"selected configuration {best_weight}: graded rating weights and exponential age decay, "
             "repulsion off; the configuration was chosen on this cohort, so its interval is optimistic",
             {"configurations": weighting_grid}, timing=weighting_timing[best_weight])

    novice_cells, _, novice_stats, novice_timing = score_requests(shadow, validation, baseline, (hybrid_alpha,),
                                                                  (0.0,), novice_for_cold=True)
    register("C3_novice_prior", novice_cells,
             "first-purchase prior substituted where no personalisable history exists", timing=novice_timing)

    interview_cells, interview_costs, _, interview_timing = score_requests(
        shadow, validation, baseline, (hybrid_alpha,), (0.0, REPULSION_USED),
        budgets=QUESTION_BUDGETS)
    requests = len(validation)
    arms["C5_interview"] = {
        "budgets": {
            str(budget): {
                "questions_per_request": interview_costs[budget]["asked"] / requests,
                "answers_per_request": interview_costs[budget]["answers"] / requests,
                "declared_positive_per_request": interview_costs[budget]["positive"] / requests,
                "declared_negative_per_request": interview_costs[budget]["negative"] / requests,
                "questions_per_answer": (round(interview_costs[budget]["asked"] / interview_costs[budget]["answers"], 2)
                                         if interview_costs[budget]["answers"] else None),
                "columns": {f"gamma={gamma}": {
                    "ndcg_at_10": quality(interview_cells[(hybrid_alpha, gamma, budget)])["ndcg_at_10"],
                    "recall_at_10": quality(interview_cells[(hybrid_alpha, gamma, budget)])["recall_at_10"],
                    "delta_versus_active": user_cluster_interval(
                        validation, active_outcomes, interview_cells[(hybrid_alpha, gamma, budget)]),
                } for gamma in (0.0, REPULSION_USED)},
            } for budget in QUESTION_BUDGETS},
        "local_cpu_request_ms_p95": percentile(interview_timing, .95),
        "note": ("answers come from the user's own observable rating of the asked item, and the target is never "
                 "asked; gamma=0 uses only declared likes, gamma=1 also uses declared dislikes; the reported p95 "
                 "covers one request scored across all ten budget-by-gamma cells")}

    bucket_cells, _, _, bucket_timing = score_requests(shadow, validation, baseline,
                                                       (0.0,) + HYBRID_ALPHAS, (0.0,))
    indices_by_bucket: dict[str, list[int]] = defaultdict(list)
    for index, (_, _, history, _) in enumerate(validation):
        indices_by_bucket[bucket_of(len(history))].append(index)
    composed = list(active_outcomes)
    bucket_report = {}
    for _, _, label in BUCKET_BOUNDS:
        indices = indices_by_bucket.get(label) or []
        if not indices:
            continue
        choice = max((0.0,) + HYBRID_ALPHAS,
                     key=lambda alpha: (mean(bucket_cells[(alpha, 0.0, 0)][i] for i in indices), -alpha))
        for index in indices:
            composed[index] = bucket_cells[(choice, 0.0, 0)][index]
        bucket_report[label] = {
            "requests": len(indices),
            "active_ndcg_at_10": mean(active_outcomes[i] for i in indices),
            "selected_alpha": choice,
            "selected_ndcg_at_10": mean(bucket_cells[(choice, 0.0, 0)][i] for i in indices)}
    arms["C4_experience_blend"] = {
        "ndcg_at_10": quality(composed)["ndcg_at_10"],
        "recall_at_10": quality(composed)["recall_at_10"],
        "delta_versus_active": user_cluster_interval(validation, active_outcomes, composed),
        "per_bucket": bucket_report,
        "local_cpu_request_ms_p95": percentile(bucket_timing, .95),
        "note": "blend weight inferred from history size alone; alpha chosen per bucket on validation, so this arm is optimistic"}

    if donor_category:
        if donor_category == category:
            raise ValueError("an adjacent surface must be a different category")
        donor_path = data / f"{donor_category}.train.csv.gz"
        if not donor_path.is_file():
            raise FileNotFoundError(f"missing adjacent surface archive {donor_path}")
        donor_records = observed_records({"train": donor_path})
        positive_of = lambda rows: {item for item, (rating, _) in rows.items() if rating >= POSITIVE_RATING}
        donor_positives = {user: positive_of(rows) for user, rows in donor_records.items()}
        target_positives = {user: positive_of(rows) for user, rows in records.items()}
        links = adjacent_projection(donor_positives, target_positives)

        def projection(user, history):
            totals = {}
            for item in donor_positives.get(user, ()):
                for neighbour, value in links.get(item, ()):
                    totals[neighbour] = totals.get(neighbour, 0.0) + value
            return totals

        adjacent_cells, _, adjacent_stats, adjacent_timing = score_requests(
            shadow, validation, baseline, (hybrid_alpha,), (0.0,), projection=projection)
        arms["C6_adjacent_surface"] = {
            "donor_category": donor_category,
            "ndcg_at_10": quality(adjacent_cells[(hybrid_alpha, 0.0, 0)])["ndcg_at_10"],
            "delta_versus_active": user_cluster_interval(
                validation, active_outcomes, adjacent_cells[(hybrid_alpha, 0.0, 0)]),
            "requests_with_adjacent_activity": adjacent_stats["adjacent_route_requests"],
            "local_cpu_request_ms_p95": percentile(adjacent_timing, .95),
            "share_of_requests": adjacent_stats["adjacent_route_requests"] / len(validation),
            "note": ("the donor surface is scored only where the request has nothing personalisable in its own "
                     "history; a small cohort measures feasibility, not effect size")}

    decision = {"category": category, "baseline": baseline, "hybrid_alpha": hybrid_alpha,
                "hybrid_gate": hybrid_gate, "active_route": active_name,
                "active_ndcg_at_10": quality(active_outcomes)["ndcg_at_10"],
                "popularity_controls": {"lifetime": lifetime_quality, "recent": recent_quality},
                "arms": arms, "training": train_stats, "validation": validation_stats,
                "routing_stats": {"measured_by_arms": "route selection and C3, which score without a projection",
                                  **hybrid_stats, **novice_stats},
                "blend_selection_request_ms_p95": percentile(hybrid_timing, .95),
                "timing_note": ("every p95 is the local wall-clock time of scoring one validation request through "
                                "that arm's own grid, single process, no concurrency; grid sizes differ between "
                                "arms, so a figure compares an arm with its own grid, not arms with each other"),
                "elapsed_seconds": round(perf_counter() - started, 1),
                "source_sha256": {split: file_hash(paths[split]) for split in ("train", "valid")}}
    (destination / "validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    aggregate = {"source": SOURCE, "experiment": f"{category} shadow-profile pilot",
                 "status": "exploratory; this category's test period was already opened by earlier studies",
                 "validation_decision": decision}
    if include_test:
        test, test_stats = shadow_queries(paths["test"], "test", catalog)
        if active_name == "hybrid":
            test_active = [ndcg_one_target(model.top_k(history, baseline, hybrid_alpha), target)
                           for _, target, history, _ in test]
        else:
            test_active = [ndcg_one_target(model.top_k(history, baseline, 0.0), target)
                           for _, target, history, _ in test]
        test_cells, test_costs, test_routing, _ = score_requests(
            shadow, test, baseline, (hybrid_alpha,), GAMMAS, novice_for_cold=True,
            budgets=(0, 10, 30))
        test_report = {"cohort": {**test_stats, **test_routing},
                       "active_ndcg_at_10": quality(test_active)["ndcg_at_10"],
                       "arms": {}}
        for alpha, gamma, budget in sorted(test_cells):
            reference = test_active if budget == 0 else test_cells[(hybrid_alpha, 0.0, 0)]
            test_report["arms"][f"{alpha}|{gamma}|{budget}"] = {
                "ndcg_at_10": quality(test_cells[(alpha, gamma, budget)])["ndcg_at_10"],
                "delta_versus_reference": user_cluster_interval(
                    test, reference, test_cells[(alpha, gamma, budget)])}
        aggregate["test"] = test_report
    (destination / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    interview = arms.get("C5_interview")
    deepest = (interview["budgets"][str(max(QUESTION_BUDGETS))]["columns"][f"gamma={REPULSION_USED}"]
               if interview else None)
    headline = {name: arm.get("ndcg_at_10") for name, arm in arms.items()}
    if deepest:
        headline["C5_interview"] = deepest["ndcg_at_10"]
    print(json.dumps({"output": str(destination / "aggregate.json"), "active_route": active_name,
                      "active_ndcg_at_10": decision["active_ndcg_at_10"], "arms": headline}, indent=2))
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="exploratory shadow-profile evaluation")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("Video_Games", "Musical_Instruments"),
                        default="Musical_Instruments")
    parser.add_argument("--include-test", action="store_true",
                        help="also score the already-exposed test period; keeps exploratory framing")
    parser.add_argument("--donor-category", choices=("Video_Games", "Musical_Instruments"),
                        help="adjacent surface to project when a request has nothing personalisable")
    args = parser.parse_args()
    run(args.data, args.output, args.category, include_test=args.include_test,
        donor_category=args.donor_category)


if __name__ == "__main__":
    main()
