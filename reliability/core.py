"""Train-only neighborhood and exact top-k retrieval for sparse adjustments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cached_property
from heapq import heappush, heapreplace, nsmallest
from math import log1p, sqrt
from typing import Iterable


def last_distinct(items: Iterable[str], limit: int = 20) -> tuple[str, ...]:
    seen = set()
    result = []
    for item in reversed(tuple(items)):
        if item not in seen:
            seen.add(item)
            result.append(item)
            if len(result) == limit:
                break
    return tuple(result)


def top_neighbors(user_positives: dict[str, list[tuple[int, str]]], limit: int = 20,
                  neighbors_per_item: int = 50) -> tuple[dict[str, tuple[tuple[str, float], ...]], dict[str, int]]:
    """Cosine co-review similarity from training users only."""
    item_users: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    for events in user_positives.values():
        ordered = [item for _, item in sorted(events, key=lambda row: (row[0], row[1]))]
        basket = sorted(last_distinct(ordered, limit))
        item_users.update(basket)
        for index, left in enumerate(basket):
            for right in basket[index + 1:]:
                pairs[left, right] += 1
    heaps: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (left, right), common in pairs.items():
        similarity = common / sqrt(item_users[left] * item_users[right])
        for source, destination in ((left, right), (right, left)):
            heap = heaps[source]
            entry = (similarity, destination)
            if len(heap) < neighbors_per_item:
                heappush(heap, entry)
            elif entry > heap[0]:
                heapreplace(heap, entry)
    return ({item: tuple((destination, similarity)
                         for similarity, destination in sorted(heap, key=lambda pair: (-pair[0], pair[1])))
             for item, heap in heaps.items()}, dict(item_users))


@dataclass(frozen=True)
class Recommender:
    lifetime: dict[str, float]
    recent: dict[str, float]
    neighbors: dict[str, tuple[tuple[str, float], ...]]

    @cached_property
    def lifetime_order(self) -> tuple[str, ...]:
        return tuple(sorted(self.lifetime, key=lambda item: (-self.lifetime[item], item)))

    @cached_property
    def recent_order(self) -> tuple[str, ...]:
        return tuple(sorted(self.recent, key=lambda item: (-self.recent[item], item)))

    @classmethod
    def train(cls, positive_events: Iterable[tuple[str, str, int]], catalog: Iterable[str],
              cutoff_ms: int, recent_days: int = 365) -> "Recommender":
        lifetime_counts: Counter[str] = Counter()
        recent_counts: Counter[str] = Counter()
        by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
        recent_start = cutoff_ms - recent_days * 86_400_000
        for user, item, timestamp in positive_events:
            if timestamp >= cutoff_ms:
                raise ValueError("training event crosses the global cutoff")
            lifetime_counts[item] += 1
            if timestamp >= recent_start:
                recent_counts[item] += 1
            by_user[user].append((timestamp, item))
        if not lifetime_counts:
            raise ValueError("empty training catalog")
        neighbors, _ = top_neighbors(by_user)
        max_lifetime = log1p(max(lifetime_counts.values()))
        max_recent = log1p(max(recent_counts.values(), default=1))
        catalog = set(catalog)
        if not catalog.issuperset(lifetime_counts):
            raise ValueError("positive item absent from catalog")
        lifetime = {item: log1p(lifetime_counts.get(item, 0)) / max_lifetime for item in catalog}
        recent = {item: log1p(recent_counts.get(item, 0)) / max_recent for item in catalog}
        return cls(lifetime, recent, neighbors)

    def top_k(self, history: Iterable[str], baseline: str = "lifetime", alpha: float = 0.0,
              k: int = 10) -> tuple[str, ...]:
        return self.rank_grid(history, baseline, (alpha,), k)[alpha]

    def rank_grid(self, history: Iterable[str], baseline: str,
                  alphas: Iterable[float], k: int = 10) -> dict[float, tuple[str, ...]]:
        alphas = tuple(alphas)
        if (baseline not in ("lifetime", "recent") or not alphas
                or any(not 0 <= alpha <= 1 for alpha in alphas) or k < 1):
            raise ValueError("invalid ranking request")
        history = tuple(history)
        seen = set(history)
        base = self.lifetime if baseline == "lifetime" else self.recent
        # Every item outside the neighbor set has personalized score zero.
        # Its top-k can only come from the first k eligible items in base order.
        base_order = self.lifetime_order if baseline == "lifetime" else self.recent_order
        candidates = set()
        for item in base_order:
            if item not in seen:
                candidates.add(item)
                if len(candidates) == k:
                    break
        base_candidates = candidates.copy()
        personalized: dict[str, float] = defaultdict(float)
        if any(alpha > 0 for alpha in alphas):
            for source in last_distinct(history):
                for item, similarity in self.neighbors.get(source, ()):
                    if item in base and item not in seen:
                        personalized[item] += similarity
            candidates.update(personalized)
        result = {}
        for alpha in alphas:
            pool = candidates if alpha else base_candidates
            result[alpha] = tuple(nsmallest(k, pool,
                                          key=lambda item: (-(1 - alpha) * base[item] - alpha * personalized.get(item, 0.0),
                                                            -base[item], item)))
        return result


def ndcg_one_target(ranking: tuple[str, ...], target: str) -> float:
    from math import log2
    try:
        return 1.0 / log2(ranking.index(target) + 2)
    except ValueError:
        return 0.0
