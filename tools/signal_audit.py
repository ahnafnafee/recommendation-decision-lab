"""Aggregate-only audit of inferred and derivable user signals in a category archive.

Emits no user IDs, item IDs or row-level values. Used to decide whether a
shadow-profile extension is measurable on the existing archives.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from reliability.benchmark import T1, T2, read_rows
from reliability.core import last_distinct

# Number of most recent distinct history items the committed profile reads.
PROFILE_CAP = 20


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
    # (label available without asking the user anything).
    for split, lower, upper in (("valid", T1, T2), ("test", T2, None)):
        stats = Counter()
        history_length: Counter[str] = Counter()
        low_rated_in_history = 0
        history_slots = 0
        known_in_history = 0
        unseen_in_train_catalog = 0
        first_moment: dict[str, int] = {}
        first_history: dict[str, set[str]] = {}
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
            # The recommender reads the most recent PROFILE_CAP distinct items.
            # Everything past that window is still held and still rated, so count
            # what the window discards: it is evidence available at no user cost.
            kept = set(last_distinct(history, PROFILE_CAP))
            for item in dict.fromkeys(history):
                side = "in" if item in kept else "beyond"
                stats[f"distinct_items_{side}_profile_cap"] += 1
                observed = train_rating.get(f"{user}|{item}")
                if observed is not None:
                    stats[f"{side}_profile_cap_rating_known"] += 1
                    stats[f"{side}_profile_cap_rating_below_four"] += int(observed < 4)
            if user not in first_moment or timestamp < first_moment[user]:
                first_moment[user] = timestamp
                first_history[user] = set(dict.fromkeys(history))
            if target not in train_rating_by_item:
                stats["new_item_target"] += 1
        covered = 0
        recalled = 0.0
        exact = 0
        checked = 0
        # Does the history column carry the user's whole prior record, or only part
        # of it? Checked once per user, at their first request in this window.
        for user, moment in first_moment.items():
            prior = {item for when, item in train_user_items.get(user, ()) if when < moment}
            if not prior:
                continue
            checked += 1
            held = first_history[user]
            recalled += len(held & prior) / len(prior)
            covered += int(prior <= held)
            exact += int(held == prior)
        stats["profile_cap"] = PROFILE_CAP
        stats["history_column_users_checked"] = checked
        stats["history_column_exact_matches"] = exact
        stats["history_column_covers_all_prior_items"] = covered
        stats["history_column_mean_prior_items_recalled"] = recalled / max(1, checked)
        stats["history_slots"] = history_slots
        stats["history_slots_rating_known"] = known_in_history
        stats["history_slots_rating_below_four"] = low_rated_in_history
        stats["history_slots_not_in_train"] = unseen_in_train_catalog
        stats["history_len_buckets"] = dict(sorted(history_length.items()))
        stats["mean_history_len_all"] = history_slots / max(1, stats["rows"])
        result["splits"][split] = dict(stats)
    return result, set(train_user_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--categories", nargs="+", default=["Musical_Instruments", "Video_Games"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, users = {}, {}
    for category in args.categories:
        report[category], users[category] = audit(args.data, category)
    # Cross-surface profiles are only possible where one account's activity is
    # visible in more than one archive, so measure the overlap directly.
    for category in args.categories:
        report[category]["shared_train_users"] = {
            other: len(users[category] & users[other])
            for other in args.categories if other != category
        }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
