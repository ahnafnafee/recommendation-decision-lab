"""Train-only two-tower retrieval with exact full-catalog evaluation.

PyTorch is optional for the standard-library baseline. Neural runs write fitted
weights only to ignored local directories; aggregate JSON contains no IDs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from .benchmark import (T1, evaluation_data, file_hash, quality,
                        training_data, user_cluster_interval)
from .core import Recommender, ndcg_one_target


@dataclass(frozen=True)
class NeuralConfig:
    dimension: int = 64
    history_length: int = 20
    batch_size: int = 256
    epochs: int = 3
    learning_rate: float = 0.002
    temperature: float = 0.15
    seed: int = 20260922


def _torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install the optional neural dependency: pip install torch") from error
    return torch


def chronological_pairs(positive_events, item_index, history_length):
    """Only previous train interactions can form a query for a train target."""
    by_user = defaultdict(list)
    for user, item, timestamp in positive_events:
        if timestamp >= T1:
            raise ValueError("training event crosses the global cutoff")
        by_user[user].append((timestamp, item))
    pairs = []
    for events in by_user.values():
        prior = []
        seen = set()
        for _, item in sorted(events, key=lambda value: (value[0], value[1])):
            if item not in seen:
                if prior:
                    pairs.append((tuple(prior[-history_length:]), item_index[item]))
                seen.add(item)
                prior.append(item_index[item])
    return pairs


def collision_mask(history, target):
    """Do not teach a query that another known positive is a negative."""
    torch = _torch()
    same_target = target[:, None].eq(target[None, :])
    known_positive = history[:, :, None].eq(target[None, None, :]).any(dim=1)
    mask = same_target | known_positive
    mask.fill_diagonal_(False)
    return mask


def _architecture(item_count, config):
    torch = _torch()
    from torch import nn
    from torch.nn import functional as F

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.history_embedding = nn.Embedding(item_count + 1, config.dimension, padding_idx=0)
            self.item_embedding = nn.Embedding(item_count + 1, config.dimension, padding_idx=0)
            self.history_projection = nn.Sequential(
                nn.Linear(config.dimension, config.dimension), nn.GELU(),
                nn.Linear(config.dimension, config.dimension))
            self.item_projection = nn.Linear(config.dimension, config.dimension)

        def query(self, history):
            active = history.ne(0).unsqueeze(-1)
            pooled = (self.history_embedding(history) * active).sum(dim=1)
            pooled = pooled / active.sum(dim=1).clamp(min=1)
            return F.normalize(pooled + self.history_projection(pooled), dim=-1)

        def items(self, item_ids):
            return F.normalize(self.item_projection(self.item_embedding(item_ids)), dim=-1)

    return TwoTower()


def _matrix(pairs, length, device):
    torch = _torch()
    history = torch.zeros((len(pairs), length), dtype=torch.long)
    targets = torch.empty(len(pairs), dtype=torch.long)
    for row, (prior, target) in enumerate(pairs):
        history[row, -len(prior):] = torch.tensor(prior, dtype=torch.long)
        targets[row] = target
    return history.to(device), targets.to(device)


class NeuralRetriever:
    def __init__(self, model, catalog, supported, config, device):
        self.model = model.eval()
        self.catalog = tuple(catalog)
        self.item_index = {item: index + 1 for index, item in enumerate(catalog)}
        self.supported = set(supported)
        self.config = config
        self.device = device
        self.torch = _torch()
        with self.torch.inference_mode():
            ids = self.torch.arange(1, len(catalog) + 1, device=device)
            self.item_vectors = model.items(ids)
            available = self.torch.tensor([item in self.supported for item in catalog], device=device)
            self.available = available
        self.blend_alpha = 1.0
        self.baseline_scores = None

    def configure_blend(self, scores, alpha):
        if not 0 < alpha <= 1 or set(scores) != set(self.catalog):
            raise ValueError("invalid neural blend")
        self.blend_alpha = alpha
        self.baseline_scores = self.torch.tensor([scores[item] for item in self.catalog],
                                                 dtype=self.torch.float32, device=self.device)

    def top_k(self, history, k=10):
        if k < 1:
            raise ValueError("k must be positive")
        encoded = [self.item_index[item] for item in history
                   if item in self.item_index and item in self.supported]
        if not encoded:
            return None
        ids = encoded[-self.config.history_length:]
        tensor = self.torch.zeros((1, self.config.history_length), dtype=self.torch.long, device=self.device)
        tensor[0, -len(ids):] = self.torch.tensor(ids, device=self.device)
        with self.torch.inference_mode():
            scores = self.model.query(tensor) @ self.item_vectors.T
            if self.baseline_scores is not None:
                scores = (1 - self.blend_alpha) * self.baseline_scores + self.blend_alpha * (scores + 1) / 2
            scores[:, ~self.available] = -float("inf")
            scores[:, self.torch.tensor([index - 1 for index in set(encoded)], device=self.device)] = -float("inf")
            count = min(k, max(0, int(self.available.sum()) - len(set(encoded))))
            chosen = self.torch.topk(scores, count).indices[0].tolist()
        return tuple(self.catalog[index] for index in chosen)

    def rank_grid(self, histories, baseline_scores, alphas, k=10):
        """Score a batch against every trained item; no sampled candidate pool."""
        torch = self.torch
        encoded = [[self.item_index[item] for item in history
                    if item in self.item_index and item in self.supported] for history in histories]
        matrix = torch.zeros((len(histories), self.config.history_length),
                             dtype=torch.long, device=self.device)
        for row, ids in enumerate(encoded):
            if ids:
                tail = ids[-self.config.history_length:]
                matrix[row, -len(tail):] = torch.tensor(tail, device=self.device)
        base = torch.tensor([baseline_scores[item] for item in self.catalog],
                            dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            neural = (self.model.query(matrix) @ self.item_vectors.T + 1) / 2
            invalid = ~self.available.expand(len(histories), -1).clone()
            for row, history in enumerate(histories):
                excluded = [self.item_index[item] - 1 for item in set(history) if item in self.item_index]
                if excluded:
                    invalid[row, torch.tensor(excluded, device=self.device)] = True
            result = {}
            for alpha in alphas:
                scores = ((1 - alpha) * base + alpha * neural).masked_fill(invalid, -float("inf"))
                indices = torch.topk(scores, min(k, len(self.catalog)), dim=1).indices.tolist()
                result[alpha] = [tuple(self.catalog[index] for index in row) for row in indices]
        return result, [bool(ids) for ids in encoded]


def load_retriever(weights_path: Path, train_path: Path, device: str = "auto"):
    """Load a local fitted model after checking both model and source hashes."""
    torch = _torch()
    manifest_path = weights_path.with_name("neural_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema", "weights_sha256", "train_sha256", "catalog_items"} or manifest["schema"] != 1:
        raise ValueError("neural manifest schema mismatch")
    if file_hash(weights_path) != manifest["weights_sha256"]:
        raise ValueError("neural weights integrity mismatch")
    if file_hash(train_path) != manifest["train_sha256"]:
        raise ValueError("neural training source mismatch")
    payload = torch.load(weights_path, map_location="cpu", weights_only=True)
    if set(payload) != {"state_dict", "catalog", "supported", "config", "train_sha256"}:
        raise ValueError("neural checkpoint schema mismatch")
    if (payload["train_sha256"] != manifest["train_sha256"] or
            len(payload["catalog"]) != manifest["catalog_items"] or
            len(set(payload["catalog"])) != len(payload["catalog"]) or
            not set(payload["supported"]).issubset(payload["catalog"])):
        raise ValueError("neural checkpoint metadata mismatch")
    config = NeuralConfig(**payload["config"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _architecture(len(payload["catalog"]), config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return NeuralRetriever(model.to(device), payload["catalog"], payload["supported"], config, device), manifest


def score_neural(retriever, queries, fallback, baseline):
    outcomes = []
    fallback_count = 0
    elapsed = []
    for _, target, history in queries:
        started = perf_counter()
        ranking = retriever.top_k(history)
        if ranking is None:
            fallback_count += 1
            ranking = fallback.top_k(history, baseline, 0.0)
        elapsed.append((perf_counter() - started) * 1000)
        outcomes.append(ndcg_one_target(ranking, target))
    return outcomes, fallback_count, elapsed


def run(data: Path, destination: Path, category: str = "Musical_Instruments",
        config: NeuralConfig = NeuralConfig(), device: str = "auto"):
    torch = _torch()
    if category not in ("Video_Games", "Musical_Instruments"):
        raise ValueError("category outside available study archives")
    if destination.exists():
        raise FileExistsError("run directory exists; retain prior evidence")
    paths = {part: data / f"{category}.{part}.csv.gz" for part in ("train", "valid", "test")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("download all three official category archives")
    destination.mkdir(parents=True)
    started = perf_counter()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available() and device == "auto":
        device = "cuda"
    elif device == "auto":
        device = "cpu"
    train_hash = file_hash(paths["train"])
    valid_hash = file_hash(paths["valid"])
    positives, catalog, training = training_data(paths["train"])
    fallback = Recommender.train(positives, catalog, T1)
    catalog = tuple(sorted(catalog))
    item_index = {item: index + 1 for index, item in enumerate(catalog)}
    supported = {item for _, item, _ in positives}
    pairs = chronological_pairs(positives, item_index, config.history_length)
    del positives
    if not pairs:
        raise ValueError("no chronological training pairs")
    model = _architecture(len(catalog), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    histories_all, targets_all = _matrix(pairs, config.history_length, device)
    epoch_losses = []
    for _ in range(config.epochs):
        model.train()
        shuffled = torch.randperm(len(pairs), generator=generator).to(device)
        losses = []
        for offset in range(0, len(pairs), config.batch_size):
            batch_indices = shuffled[offset:offset + config.batch_size]
            if len(batch_indices) < 2:
                continue
            histories, targets = histories_all[batch_indices], targets_all[batch_indices]
            logits = model.query(histories) @ model.items(targets).T / config.temperature
            logits = logits.masked_fill(collision_mask(histories, targets), -1e9)
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch_indices), device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(mean(losses))
    training_seconds = perf_counter() - started
    retriever = NeuralRetriever(model, catalog, supported, config, device)
    validation, validation_stats = evaluation_data(paths["valid"], "valid", set(catalog))
    lifetime_validation = [ndcg_one_target(fallback.top_k(history, "lifetime", 0.0), target)
                           for _, target, history in validation]
    recent_validation = [ndcg_one_target(fallback.top_k(history, "recent", 0.0), target)
                         for _, target, history in validation]
    baseline = "recent" if mean(recent_validation) > mean(lifetime_validation) else "lifetime"
    baseline_validation = [ndcg_one_target(fallback.top_k(history, baseline, 0.0), target)
                           for _, target, history in validation]
    neural_validation, valid_fallbacks, _ = score_neural(retriever, validation, fallback, baseline)
    interval = user_cluster_interval(validation, baseline_validation, neural_validation)
    gate_open = interval["lower"] > 0
    decision = {"category": category, "config": asdict(config), "device": device,
                "torch_version": torch.__version__, "source_sha256": {"train": train_hash, "valid": valid_hash},
                "training": training, "training_pairs": len(pairs), "training_seconds": training_seconds,
                "epoch_losses": epoch_losses, "validation": validation_stats,
                "baseline_method": baseline, "baseline": quality(baseline_validation),
                "neural_with_fallback": quality(neural_validation),
                "fallback_requests": valid_fallbacks, "neural_minus_baseline": interval,
                "gate_open": gate_open, "selection": "fixed architecture and epochs; gate uses validation lower interval"}
    (destination / "validation_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.cpu().state_dict(), "catalog": catalog,
                "supported": sorted(supported), "config": asdict(config),
                "train_sha256": train_hash}, destination / "neural_weights.pt")
    (destination / "neural_manifest.json").write_text(json.dumps({
        "schema": 1, "weights_sha256": file_hash(destination / "neural_weights.pt"),
        "train_sha256": train_hash, "catalog_items": len(catalog)}, indent=2), encoding="utf-8")
    # The test archive is first opened after the validation decision is persisted.
    test_hash = file_hash(paths["test"])
    test, cohort = evaluation_data(paths["test"], "test", set(catalog))
    retriever = NeuralRetriever(model.to(device), catalog, supported, config, device)
    baseline_test = [ndcg_one_target(fallback.top_k(history, baseline, 0.0), target)
                     for _, target, history in test]
    neural_test, test_fallbacks, timings = score_neural(retriever, test, fallback, baseline)
    paired = user_cluster_interval(test, baseline_test, neural_test)
    aggregate = {"experiment": f"{category} neural retrieval extension",
                 "source_sha256": {"train": train_hash, "valid": valid_hash, "test": test_hash},
                 "validation_decision": decision,
                 "test": {"cohort": cohort, "baseline": quality(baseline_test),
                          "neural_with_fallback": quality(neural_test),
                          "active_route": quality(neural_test if gate_open else baseline_test),
                          "neural_minus_baseline": paired, "fallback_requests": test_fallbacks,
                          "local_neural_request_ms": {"p50": sorted(timings)[len(timings)//2],
                                                      "p95": sorted(timings)[int(.95 * (len(timings)-1))],
                                                      "scope": "single request; exact item matrix; excludes HTTP and loading"}},
                 "limitations": ["architecture selected after prior category results were known",
                                 "review activity is not exposure or online utility",
                                 "5-core selection uses full-corpus support",
                                 "single fixed temporal split", "no new independent confirmation"]}
    (destination / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination / "aggregate.json"), "gate_open": gate_open,
                      "test_delta": paired["delta"]}))
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("Video_Games", "Musical_Instruments"), default="Musical_Instruments")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run(args.data, args.output, args.category, device=args.device)


if __name__ == "__main__":
    main()
