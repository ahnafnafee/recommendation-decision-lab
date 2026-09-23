"""Aggregate-only audit of covert and derivable user signals in a category archive.

Emits no user IDs, item IDs or row-level values. Used to decide whether a
shadow-profile extension is measurable on the existing archives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from reliability.benchmark import T1, T2, read_rows


def audit(data: Path, category: str):
    paths = {part: data / f"{category}.{part}.csv.gz" for part in ("train", "valid", "test")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("download all three official category archives")

    train_rating: dict[str, float] = {}
    train_user_items: dict[str, list[tuple[int, str]]] = defaultdict(list)
    train_rating_by_item: dict[str, list[float]] = defaultdict(list)
    train_stats = Counter()
    for row in read_rows(paths["train"]):
        user, item = row["user_id"], row["parent_asin"]
        rating = float(row["rating"])
        timestamp = int(row["timestamp"])
        train_stats["rows"] += 1
        train_stats["train_positive_rows"] += int(rating >= 4)
        train_rating[f"{user}|{item}"] = rating
        train_rating_by_item[item].append(rating)
        train_user_items[user].append((timestamp, item))

    histogram: Counter[str] = Counter()
    for ratings in train_rating_by_item.values():
        for rating in ratings:
            histogram[str(int(rating))] += 1
    result = {"category": category, "train": dict(train_stats),
              "train_users": len(train_user_items),
              "train_items": len(train_rating_by_item),
              "train_rating_histogram": dict(sorted(histogram.items())),
              "splits": {}}

    # Fraction of history slots whose rating the platform already observed
    # (covert label available without asking the user anything).
    for split, lower, upper in (("valid", T1, T2), ("test", T2, None)):
        stats = Counter()
        history_length: Counter[str] = Counter()
        low_rated_in_history = 0
        history_slots = 0
        known_in_history = 0
        unseen_in_train_catalog = 0
        for row in read_rows(paths[split]):
            timestamp = int(row["timestamp"])
            if timestamp < lower or (upper is not None and timestamp >= upper):
                continue
            stats["rows"] += 1
            rating = float(row["rating"])
            stats["positive_rows"] += int(rating >= 4)
            user, target = row["user_id"], row["parent_asin"]
            history = row["history"].split()
            if not history:
                stats["empty_history"] += 1
            stats["first_seen_user"] += int(user not in train_user_items)
            returning = train_user_items.get(user, [])
            stats["positive_history_len_sum"] += sum(
                1 for item in history if train_rating.get(f"{user}|{item}", 4) >= 4)
            stats["repeat_target"] += int(target in history)
            for bucket_low, bucket_high, label in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"),
                                                    (6, 19, "6-19"), (20, 10**9, "20+")):
                if bucket_low <= len(history) <= bucket_high:
                    history_length[label] += 1
                    break
            for item in history:
                history_slots += 1
                observed = train_rating.get(f"{user}|{item}")
                if observed is None:
                    unseen_in_train_catalog += 1
                else:
                    known_in_history += 1
                    low_rated_in_history += int(observed < 4)
            if target not in train_rating_by_item:
                stats["new_item_target"] += 1
        stats["history_slots"] = history_slots
        stats["history_slots_rating_known"] = known_in_history
        stats["history_slots_rating_below_four"] = low_rated_in_history
        stats["history_slots_not_in_train"] = unseen_in_train_catalog
        stats["history_len_buckets"] = dict(sorted(history_length.items()))
        stats["mean_history_len_all"] = history_slots / max(1, stats["rows"])
        result["splits"][split] = dict(stats)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--categories", nargs="+", default=["Musical_Instruments", "Video_Games"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {category: audit(args.data, category) for category in args.categories}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
