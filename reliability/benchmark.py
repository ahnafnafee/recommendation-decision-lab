"""Frozen Video_Games benchmark; public output contains aggregates only."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from .core import Recommender, ndcg_one_target


T1 = 1628643414042
T2 = 1658002729837
ALPHAS = (0.25, 0.5, 0.75, 1.0)
FILES = {split: f"Video_Games.{split}.csv.gz" for split in ("train", "valid", "test")}
SOURCE = "https://amazon-reviews-2023.github.io/data_processing/5core.html"


def read_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        expected = {"user_id", "parent_asin", "rating", "timestamp", "history"}
        if set(rows.fieldnames or ()) != expected:
            raise ValueError(f"unexpected fields in {path.name}")
        yield from rows


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def training_data(path: Path):
    catalog = set()
    positives = []
    total = 0
    users = set()
    for row in read_rows(path):
        timestamp = int(row["timestamp"])
        if timestamp >= T1:
            raise ValueError("training row crosses t1")
        user, item = row["user_id"], row["parent_asin"]
        if not user or not item:
            raise ValueError("empty training ID")
        catalog.add(item)
        users.add(user)
        total += 1
        if float(row["rating"]) >= 4:
            positives.append((user, item, timestamp))
    return positives, catalog, {"rows": total, "users": len(users), "items": len(catalog),
                                "positive_rows": len(positives)}


def evaluation_data(path: Path, split: str, catalog: set[str]):
    if split not in ("valid", "test"):
        raise ValueError("invalid evaluation split")
    queries = []
    stats = {"rows": 0, "below_positive_threshold": 0, "seen_target_excluded": 0,
             "new_item_targets": 0, "empty_history": 0}
    for row in read_rows(path):
        timestamp = int(row["timestamp"])
        if (split == "valid" and not T1 <= timestamp < T2) or (split == "test" and timestamp < T2):
            raise ValueError("evaluation row outside source split")
        stats["rows"] += 1
        if float(row["rating"]) < 4:
            stats["below_positive_threshold"] += 1
            continue
        user, target = row["user_id"], row["parent_asin"]
        history = tuple(row["history"].split())
        if not user or not target:
            raise ValueError("empty evaluation ID")
        if target in history:
            stats["seen_target_excluded"] += 1
            continue
        if target not in catalog:
            stats["new_item_targets"] += 1
        if not history:
            stats["empty_history"] += 1
        queries.append((user, target, history))
    stats["eligible_requests"] = len(queries)
    stats["eligible_users"] = len({query[0] for query in queries})
    return queries, stats


def score_methods(model: Recommender, queries, baseline: str, alphas=(0.0,)):
    outcomes = {alpha: [] for alpha in alphas}
    timings_ms = []
    for _, target, history in queries:
        started = perf_counter()
        rankings = model.rank_grid(history, baseline, alphas)
        timings_ms.append((perf_counter() - started) * 1000)
        for alpha in alphas:
            outcomes[alpha].append(ndcg_one_target(rankings[alpha], target))
    return outcomes, timings_ms


def user_cluster_interval(queries, first, second, draws=1000, seed=20260922):
    if len(queries) != len(first) or len(first) != len(second) or not queries:
        raise ValueError("invalid paired outcomes")
    by_user = defaultdict(lambda: [0.0, 0])
    for query, left, right in zip(queries, first, second):
        record = by_user[query[0]]
        record[0] += right - left
        record[1] += 1
    clusters = list(by_user.values())
    point = sum(record[0] for record in clusters) / len(queries)
    generator = random.Random(seed)
    samples = []
    for _ in range(draws):
        total, count = 0.0, 0
        for _ in clusters:
            record = clusters[generator.randrange(len(clusters))]
            total += record[0]
            count += record[1]
        samples.append(total / count)
    samples.sort()
    return {"delta": point, "lower": samples[int(.025 * (draws - 1))],
            "upper": samples[int(.975 * (draws - 1))], "users": len(clusters),
            "draws": draws}


def quality(outcomes):
    return {"ndcg_at_10": mean(outcomes), "recall_at_10": sum(value > 0 for value in outcomes) / len(outcomes)}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def run(data: Path, destination: Path):
    if destination.exists():
        raise FileExistsError("run directory exists; retain prior evidence")
    paths = {split: data / name for split, name in FILES.items()}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("download all three official Video_Games timestamp split archives")
    destination.mkdir(parents=True)
    hashes = {split: file_hash(paths[split]) for split in ("train", "valid")}
    positives, catalog, train_stats = training_data(paths["train"])
    model = Recommender.train(positives, catalog, T1)
    del positives
    validation, validation_stats = evaluation_data(paths["valid"], "valid", catalog)
    if not validation:
        raise ValueError("no validation queries")
    lifetime, _ = score_methods(model, validation, "lifetime")
    recent, _ = score_methods(model, validation, "recent")
    lifetime_quality, recent_quality = quality(lifetime[0.0]), quality(recent[0.0])
    baseline = "recent" if recent_quality["ndcg_at_10"] > lifetime_quality["ndcg_at_10"] else "lifetime"
    baseline_outcomes = recent[0.0] if baseline == "recent" else lifetime[0.0]
    hybrid, _ = score_methods(model, validation, baseline, ALPHAS)
    challenger_alpha = max(ALPHAS, key=lambda alpha: (mean(hybrid[alpha]), -alpha))
    validation_interval = user_cluster_interval(validation, baseline_outcomes, hybrid[challenger_alpha])
    gate_open = validation_interval["lower"] > 0
    decision = {"baseline": baseline, "challenger_alpha": challenger_alpha,
                "gate_open": gate_open, "validation_interval": validation_interval,
                "candidate_ndcg": {str(alpha): quality(hybrid[alpha]) for alpha in ALPHAS},
                "lifetime": lifetime_quality, "recent": recent_quality,
                "training": train_stats, "validation": validation_stats,
                "source_sha256": dict(hashes), "protocol": "protocol/video_games_v1.md"}
    (destination / "validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    # Test rows are first opened only after validation choice and gate are persisted.
    hashes["test"] = file_hash(paths["test"])
    test, test_stats = evaluation_data(paths["test"], "test", catalog)
    if not test:
        raise ValueError("no test queries")
    test_alphas = (0.0, challenger_alpha)
    test_outcomes, timing_ms = score_methods(model, test, baseline, test_alphas)
    paired = user_cluster_interval(test, test_outcomes[0.0], test_outcomes[challenger_alpha])
    test_results = {"baseline": quality(test_outcomes[0.0]),
                    "challenger": quality(test_outcomes[challenger_alpha]),
                    "active_route": quality(test_outcomes[challenger_alpha] if gate_open else test_outcomes[0.0]),
                    "challenger_minus_baseline": paired, "cohort": test_stats,
                    "local_cpu_request_ms": {"p50": percentile(timing_ms, .5),
                                             "p95": percentile(timing_ms, .95),
                                             "scope": "Python ranking of baseline and challenger; no HTTP, initialization, or disk loading"}}
    aggregate = {"source": SOURCE, "experiment": "Video_Games 5-core global temporal split",
                 "source_sha256": hashes,
                 "validation_decision": decision, "test": test_results,
                 "limitations": ["5-core selection uses full-corpus support", "review activity is not exposure or online utility",
                                 "one fixed temporal split", "user-cluster interval conditions on fitted model and fixed period"]}
    (destination / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination / "aggregate.json"), "baseline": baseline,
                      "alpha": challenger_alpha, "gate_open": gate_open,
                      "test_delta": paired["delta"]}))
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
