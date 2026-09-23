"""Validation-only calibration of a trained neural retriever against popularity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from .benchmark import (ALPHAS as HYBRID_ALPHAS, T1, evaluation_data, file_hash, quality, training_data,
                        user_cluster_interval)
from .core import Recommender, ndcg_one_target
from .neural import load_retriever


ALPHAS = (0.05, 0.1, 0.2, 0.4, 1.0)


def score_grid(retriever, queries, fallback, baseline, alphas=ALPHAS, batch_size=128):
    outcomes = {alpha: [] for alpha in alphas}
    fallback_requests = 0
    elapsed = []
    scores = fallback.recent if baseline == "recent" else fallback.lifetime
    for offset in range(0, len(queries), batch_size):
        batch = queries[offset:offset + batch_size]
        started = perf_counter()
        rankings, supported = retriever.rank_grid([row[2] for row in batch], scores, alphas)
        elapsed.append((perf_counter() - started) * 1000 / len(batch))
        for index, (_, target, history) in enumerate(batch):
            if not supported[index]:
                fallback_requests += 1
                chosen = fallback.top_k(history, baseline, 0.0)
                value = ndcg_one_target(chosen, target)
                for alpha in alphas:
                    outcomes[alpha].append(value)
            else:
                for alpha in alphas:
                    outcomes[alpha].append(ndcg_one_target(rankings[alpha][index], target))
    return outcomes, fallback_requests, elapsed


def run(data: Path, weights: Path, destination: Path, category="Musical_Instruments"):
    if destination.exists():
        raise FileExistsError("run directory exists; retain prior evidence")
    paths = {split: data / f"{category}.{split}.csv.gz" for split in ("train", "valid", "test")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("missing source archive")
    retriever, manifest = load_retriever(weights, paths["train"])
    destination.mkdir(parents=True)
    positives, catalog, _ = training_data(paths["train"])
    fallback = Recommender.train(positives, catalog, T1)
    del positives
    valid, valid_cohort = evaluation_data(paths["valid"], "valid", catalog)
    baseline_scores = {}
    for method in ("lifetime", "recent"):
        baseline_scores[method] = [ndcg_one_target(fallback.top_k(history, method, 0.0), target)
                                   for _, target, history in valid]
    baseline = max(("lifetime", "recent"), key=lambda method: (mean(baseline_scores[method]), method == "recent"))
    baseline_valid = baseline_scores[baseline]
    hybrid_candidates = {value: [] for value in HYBRID_ALPHAS}
    for _, target, history in valid:
        rankings = fallback.rank_grid(history, baseline, HYBRID_ALPHAS)
        for value in HYBRID_ALPHAS:
            hybrid_candidates[value].append(ndcg_one_target(rankings[value], target))
    hybrid_alpha = max(HYBRID_ALPHAS, key=lambda value: (mean(hybrid_candidates[value]), -value))
    hybrid_valid = hybrid_candidates[hybrid_alpha]
    hybrid_gate = user_cluster_interval(valid, baseline_valid, hybrid_valid)["lower"] > 0
    active_valid = hybrid_valid if hybrid_gate else baseline_valid
    grid, valid_fallbacks, _ = score_grid(retriever, valid, fallback, baseline)
    alpha = max(ALPHAS, key=lambda value: (mean(grid[value]), -value))
    paired = user_cluster_interval(valid, baseline_valid, grid[alpha])
    versus_active = user_cluster_interval(valid, active_valid, grid[alpha])
    gate_open = versus_active["lower"] > 0
    decision = {"category": category, "baseline_method": baseline, "blend_alpha": alpha,
                "candidate_ndcg": {str(value): quality(grid[value]) for value in ALPHAS},
                "baseline": quality(baseline_valid), "neural_with_fallback": quality(grid[alpha]),
                "neural_minus_baseline": paired, "gate_open": gate_open,
                "hybrid_alpha": hybrid_alpha, "hybrid_gate_open": hybrid_gate,
                "hybrid": quality(hybrid_valid), "neural_minus_active": versus_active,
                "validation": valid_cohort, "fallback_requests": valid_fallbacks,
                "weights_sha256": manifest["weights_sha256"],
                "source_sha256": {"train": manifest["train_sha256"], "valid": file_hash(paths["valid"])}}
    (destination / "validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    # This category's test period was already read by the earlier hybrid study;
    # this extension is exploratory even though its calibration is validation-only.
    test_hash = file_hash(paths["test"])
    test, test_cohort = evaluation_data(paths["test"], "test", catalog)
    baseline_test = [ndcg_one_target(fallback.top_k(history, baseline, 0.0), target)
                     for _, target, history in test]
    hybrid_test = [ndcg_one_target(fallback.top_k(history, baseline, hybrid_alpha), target)
                   for _, target, history in test]
    active_test = hybrid_test if hybrid_gate else baseline_test
    neural_test, test_fallbacks, timings = score_grid(retriever, test, fallback, baseline, (alpha,))
    neural_test = neural_test[alpha]
    result = {"experiment": f"{category} calibrated neural retrieval (exploratory)",
              "source_sha256": {**decision["source_sha256"], "test": test_hash},
              "validation_decision": decision,
              "test": {"cohort": test_cohort, "baseline": quality(baseline_test),
                       "hybrid": quality(hybrid_test),
                       "neural_with_fallback": quality(neural_test),
                       "active_route": quality(neural_test if gate_open else active_test),
                       "neural_minus_baseline": user_cluster_interval(test, baseline_test, neural_test),
                       "neural_minus_prior_active": user_cluster_interval(test, active_test, neural_test),
                       "fallback_requests": test_fallbacks,
                       "local_batch_rank_ms_per_request": {
                           "p50": sorted(timings)[len(timings)//2],
                           "p95": sorted(timings)[int(.95 * (len(timings)-1))],
                           "scope": "batched exact score calculation; excludes model load and HTTP"}},
              "limitations": ["Exploratory: model design followed prior exposure to this category test period",
                              "5-core selection uses full-corpus support",
                              "review activity is not exposure or online utility",
                              "one fixed temporal split"]}
    (destination / "aggregate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination / "aggregate.json"), "alpha": alpha,
                      "gate_open": gate_open, "test_delta": result["test"]["neural_minus_baseline"]["delta"]}))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("Video_Games", "Musical_Instruments"), default="Musical_Instruments")
    args = parser.parse_args()
    run(args.data, args.weights, args.output, args.category)


if __name__ == "__main__":
    main()
