"""Local, hash-checked model bundles; no user histories are serialized."""

from __future__ import annotations

import argparse
import gzip
import json
from math import isfinite
from pathlib import Path

from .benchmark import T1, file_hash, training_data
from .core import Recommender


def save_bundle(model: Recommender, directory: Path, category: str, source_sha256: str):
    if directory.exists():
        raise FileExistsError("bundle directory exists")
    directory.mkdir(parents=True)
    payload = {"lifetime": model.lifetime, "recent": model.recent,
               "neighbors": {item: [[neighbor, similarity] for neighbor, similarity in values]
                             for item, values in model.neighbors.items()}}
    model_path = directory / "model.json.gz"
    with gzip.open(model_path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
    manifest = {"schema": 1, "category": category, "catalog_items": len(model.lifetime),
                "source_sha256": source_sha256, "model_sha256": file_hash(model_path)}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_bundle(directory: Path) -> tuple[Recommender, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {"schema", "category", "catalog_items", "source_sha256", "model_sha256"} or manifest["schema"] != 1:
        raise ValueError("bundle manifest schema mismatch")
    model_path = directory / "model.json.gz"
    if file_hash(model_path) != manifest["model_sha256"]:
        raise ValueError("bundle integrity mismatch")
    with gzip.open(model_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if set(payload) != {"lifetime", "recent", "neighbors"}:
        raise ValueError("model schema mismatch")
    lifetime, recent = payload["lifetime"], payload["recent"]
    if (set(lifetime) != set(recent) or len(lifetime) != manifest["catalog_items"]
            or not all(isinstance(value, (float, int)) and isfinite(value) and 0 <= value <= 1
                       for scores in (lifetime, recent) for value in scores.values())):
        raise ValueError("invalid popularity scores")
    neighbors = {}
    for source, entries in payload["neighbors"].items():
        if source not in lifetime or len(entries) > 50:
            raise ValueError("invalid neighborhood source")
        parsed = []
        for entry in entries:
            if (not isinstance(entry, list) or len(entry) != 2 or entry[0] not in lifetime
                    or not isinstance(entry[1], (int, float)) or not isfinite(entry[1])
                    or not 0 <= entry[1] <= 1):
                raise ValueError("invalid neighbor score")
            parsed.append((entry[0], float(entry[1])))
        neighbors[source] = tuple(parsed)
    return Recommender(lifetime, recent, neighbors), manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--category", choices=("Video_Games", "Musical_Instruments"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.data / f"{args.category}.train.csv.gz"
    positives, catalog, _ = training_data(train_path)
    model = Recommender.train(positives, catalog, T1)
    manifest = save_bundle(model, args.output, args.category, file_hash(train_path))
    print(json.dumps({"bundle": str(args.output), "category": manifest["category"],
                      "catalog_items": manifest["catalog_items"]}))


if __name__ == "__main__":
    main()
