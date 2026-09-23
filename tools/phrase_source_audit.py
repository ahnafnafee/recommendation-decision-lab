"""Decompose the fixed published-pair diagnostic by query provenance.

The saved phrase experiment groups ESCI search queries with Amazon-C4 review-derived
rewrites. This audit uses the exact frozen model and selected lexical configuration,
then reports source contributions without writing request-level data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from reliability.benchmark import T1, file_hash, training_data
from reliability.core import Recommender, ndcg_one_target
from reliability.language import build_index, load_corpus, load_queries
from reliability.phrasing import ranker_for, timestamped_requests


REPORTS = {
    "Musical_Instruments": "musical_phrase_exploratory.json",
    "Video_Games": "video_games_phrase_exploratory.json",
}


def audit(category: str, data: Path, text: Path, report: Path) -> dict:
    saved = json.loads(report.read_text(encoding="utf-8"))
    decision = saved["validation_decision"]
    if decision["category"] != category or decision["phrase_scope"] != "behavioural":
        raise ValueError("report is not the fixed confined phrase run for this category")
    if "lexical" not in decision["text"]["phrase_index"]:
        raise ValueError("source audit expects the lexical phrase route")

    train_path = data / f"{category}.train.csv.gz"
    test_path = data / f"{category}.test.csv.gz"
    for split, path in (("train", train_path), ("test", test_path)):
        if file_hash(path) != saved["source_sha256"][split]:
            raise ValueError(f"{split} archive differs from the saved result")

    positives, catalog, _ = training_data(train_path)
    model = Recommender.train(positives, catalog, T1)
    corpus, manifest = load_corpus(text, category)
    if manifest["corpus_sha256"] != decision["text"]["corpus_sha256"]:
        raise ValueError("product corpus differs from the saved result")
    query_manifest = json.loads((text / f"{category}.queries.json").read_text(encoding="utf-8"))
    if file_hash(text / f"{category}.queries.jsonl.gz") != query_manifest["queries_sha256"]:
        raise ValueError("query pairs differ from their source manifest")

    index, _ = build_index(corpus, catalogue=set(catalog),
                           review_limit=decision["text"]["review_limit"])
    pairs = load_queries(text, category, catalogue=set(catalog))
    first_by_item = {}
    for _, phrase, item, source in pairs:
        first_by_item.setdefault(item, (phrase, source))

    config = decision["selected_config"]
    chosen = (config["weight"], config["repulsion"], config["price_penalty"])
    ranker = ranker_for(model, index, decision["baseline"], decision["alpha"],
                        chosen, scope="behavioural",
                        scope_depth=decision["promotion_scope_depth"])
    requests = timestamped_requests(test_path, "test", catalog)
    sources = defaultdict(lambda: {"requests": 0, "delta_sum": 0.0,
                                   "wins": 0, "losses": 0, "targets": set()})
    for _, target, history, _ in requests:
        pair = first_by_item.get(target)
        if pair is None:
            continue
        phrase, source = pair
        baseline = ndcg_one_target(model.top_k(history, decision["baseline"],
                                               decision["alpha"], 10), target)
        phrased = ndcg_one_target(ranker.rank(history, phrase, 10), target)
        delta = phrased - baseline
        row = sources[source]
        row["requests"] += 1
        row["delta_sum"] += delta
        row["wins"] += delta > 0
        row["losses"] += delta < 0
        row["targets"].add(target)

    combined = sum(row["delta_sum"] for row in sources.values()) / len(requests)
    reported = saved["test"]["conditions"]["published"]["ndcg_minus_base"]["delta"]
    if round(combined, 6) != reported:
        raise ValueError(f"source contributions {combined:.6f} disagree with {reported:.6f}")
    return {
        "category": category,
        "scope": "fixed lexical diagnostic, full test denominator; source pairs attach by known target",
        "test_requests": len(requests),
        "combined_delta": round(combined, 6),
        "sources": {
            source: {
                "source_pairs_in_catalog": query_manifest["sources"][source]["rows_kept"],
                "matched_test_requests": row["requests"],
                "distinct_test_targets": len(row["targets"]),
                "full_split_delta_contribution": round(row["delta_sum"] / len(requests), 6),
                "improved_requests": row["wins"],
                "worsened_requests": row["losses"],
            }
            for source, row in sorted(sources.items())
        },
        "interpretation": (
            "Descriptive decomposition of a target-linked diagnostic, not evidence of "
            "live-query routing or independent confirmation. Repeated requests share "
            "a small set of query-item pairs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--text", type=Path, default=Path("data/text"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--category", choices=tuple(REPORTS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.category, args.data, args.text,
                   args.reports / REPORTS[args.category])
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
