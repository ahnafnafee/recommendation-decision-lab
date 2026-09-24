"""Decompose the fixed published-pair diagnostic by query provenance.

The saved phrase experiment groups ESCI search queries with Amazon-C4 review-derived
rewrites. This audit uses the exact frozen model and selected lexical configuration,
then reports source contributions without writing request-level data.

When the hash-pinned ESCI parquet is present in the cache, the audit additionally
splits the ESCI contribution by judgement (exact match vs acceptable substitute)
and reports how often the phrase route reaches a product judged an acceptable
substitute for the staged query — the many-valid-answers credit that a strict
target-reach count cannot show. Both extensions are descriptive: the frozen gate
and its reported interval are untouched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from reliability.benchmark import T1, file_hash, training_data
from reliability.core import Recommender, ndcg_one_target
from reliability.language import build_index, load_corpus, load_queries
from reliability.phrasing import parse_utterance, ranker_for, timestamped_requests
from tools.text_corpus import ESCI_EXAMPLES_SHA256


REPORTS = {
    "Musical_Instruments": "musical_phrase_exploratory.json",
    "Video_Games": "video_games_phrase_exploratory.json",
}


def esci_label_view(cache: Path):
    """Judgement label and substitute set for every staged ESCI pair.

    Reads the same hash-verified parquet the bank was staged from. The substitute
    set of a staged pair is every US-locale product judged an acceptable substitute
    for the same query, whether or not it sits in this catalogue. Returns
    (None, None) when the cached parquet is absent, so the source split still
    runs without it.
    """
    target = cache / "shopping_queries_dataset_examples.parquet"
    if not target.exists():
        return None, None
    if file_hash(target) != ESCI_EXAMPLES_SHA256:
        raise ValueError("cached ESCI parquet differs from the pinned source")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('ESCI source audit requires pip install -e ".[query-data]"') from exc

    table = pq.read_table(target, columns=["example_id", "query_id", "product_id",
                                           "product_locale", "esci_label"])
    columns = [table.column(name).to_pylist() for name in
               ("example_id", "query_id", "product_id", "product_locale", "esci_label")]
    label_of: dict[int, str] = {}
    query_of: dict[int, str] = {}
    subs_of_query: dict[str, set[str]] = defaultdict(set)
    for example_id, query_id, item, locale, label in zip(*columns):
        if locale != "us":
            continue
        if label in ("E", "S"):
            label_of[example_id] = label
            query_of[example_id] = query_id
        if label == "S":
            subs_of_query[query_id].add(item)
    subs_of = {eid: frozenset(subs_of_query[qid]) for eid, qid in query_of.items()}
    return label_of, subs_of


def audit(category: str, data: Path, text: Path, report: Path,
          cache: Path | None = None) -> dict:
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
    for identifier, phrase, item, source in pairs:
        first_by_item.setdefault(item, (phrase, source, identifier))

    config = decision["selected_config"]
    chosen = (config["weight"], config["repulsion"], config["price_penalty"])
    ranker = ranker_for(model, index, decision["baseline"], decision["alpha"],
                        chosen, scope="behavioural",
                        scope_depth=decision["promotion_scope_depth"])

    label_of, subs_of = esci_label_view(cache) if cache is not None else (None, None)
    probe = None
    reach_saved = None
    if label_of is not None:
        probe = ranker_for(model, index, decision["baseline"], decision["alpha"],
                           chosen, scope="catalogue")
        reach_saved = saved["test"]["phrase_route_reach"]["published"]

    requests = timestamped_requests(test_path, "test", catalog)
    sources = defaultdict(lambda: {"requests": 0, "delta_sum": 0.0,
                                   "wins": 0, "losses": 0, "targets": set()})
    labels = defaultdict(lambda: {"requests": 0, "delta_sum": 0.0,
                                  "wins": 0, "losses": 0, "targets": set()})
    reach = defaultdict(lambda: {"requests": 0, "target_reached": 0,
                                 "reached_by_substitute_only": 0})
    for _, target, history, _ in requests:
        entry = first_by_item.get(target)
        if entry is None:
            continue
        phrase, source, identifier = entry
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

        if label_of is None:
            continue
        if source == "esci":
            example_id = int(identifier.split(":", 1)[1])
            label = label_of.get(example_id)
            if label is None:
                raise ValueError(f"staged pair {identifier!r} is not a US-locale E/S row")
            lrow = labels[label]
            lrow["requests"] += 1
            lrow["delta_sum"] += delta
            lrow["wins"] += delta > 0
            lrow["losses"] += delta < 0
            lrow["targets"].add(target)
            substitutes = subs_of.get(example_id, frozenset())
        else:
            substitutes = frozenset()
        utterance = parse_utterance(phrase)
        if utterance is None or utterance.empty:
            continue
        attraction, _ = probe.route(utterance, probe.prepare(history))
        rrow = reach["overall"]
        rrow["requests"] += 1
        if target in attraction:
            rrow["target_reached"] += 1
        elif substitutes and not substitutes.isdisjoint(attraction):
            rrow["reached_by_substitute_only"] += 1

    combined = sum(row["delta_sum"] for row in sources.values()) / len(requests)
    reported = saved["test"]["conditions"]["published"]["ndcg_minus_base"]["delta"]
    if round(combined, 6) != reported:
        raise ValueError(f"source contributions {combined:.6f} disagree with {reported:.6f}")

    result = {
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
    if label_of is not None:
        result["esci_label_split"] = {
            "exact" if label == "E" else "substitute": {
                "matched_test_requests": row["requests"],
                "distinct_test_targets": len(row["targets"]),
                "full_split_delta_contribution": round(row["delta_sum"] / len(requests), 6),
                "improved_requests": row["wins"],
                "worsened_requests": row["losses"],
            }
            for label, row in sorted(labels.items())
        }
        total = reach["overall"]
        if (total["requests"] != reach_saved["chances"]
                or total["target_reached"] != reach_saved["target_reached"]):
            raise ValueError(
                "strict reach from the catalogue probe "
                f"({total['target_reached']}/{total['requests']}) disagrees with the "
                f"saved run ({reach_saved['target_reached']}/{reach_saved['chances']})")
        result["substitute_inclusive_reach"] = {
            "matched_requests": total["requests"],
            "strict_target_reached": total["target_reached"],
            "reached_by_substitute_only": total["reached_by_substitute_only"],
            "substitute_inclusive_reached": (total["target_reached"]
                                             + total["reached_by_substitute_only"]),
            "note": (
                "strict counts equal the saved run's phrase_route_reach by construction "
                "check; the substitute credit asks whether the route reached any product "
                "judged an acceptable substitute for that query. Descriptive, not a gate."
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--text", type=Path, default=Path("data/text"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--category", choices=tuple(REPORTS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.category, args.data, args.text,
                   args.reports / REPORTS[args.category], cache=args.cache)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
