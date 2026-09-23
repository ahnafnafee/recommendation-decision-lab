"""Encode the catalogue's own product text so a phrase can be matched to it.

Run this where the GPU is (`.venv\\Scripts\\python.exe`, which has the language
extra installed). It writes a flat float32 file and a manifest next to the corpus
it read, so a later request can score a phrase without touching a model at all —
and so a stale index is caught by hash rather than by memory.

    $env:PYTHONPATH="$PWD"
    .venv\\Scripts\\python.exe tools/build_text_index.py --category Musical_Instruments
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliability.embeddings import Encoder, write_embeddings
from reliability.language import document_text, load_corpus


def character_report(documents) -> dict:
    lengths = sorted(len(text) for text in documents)
    if not lengths:
        return {"documents": 0}
    return {"documents": len(lengths),
            "chars_p50": lengths[len(lengths) // 2],
            "chars_p90": lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))],
            "chars_max": lengths[-1],
            "without_any_text": sum(1 for length in lengths if not length)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path,
                        help="where to write, defaults to the corpus directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="sentence-embedding model, encoded and queried with the same one")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--review-limit", type=int, default=4,
                        help="review excerpts per item, matched to the lexical index")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="encode only the first N items")
    parser.add_argument("--force", action="store_true", help="re-encode over an existing index")
    args = parser.parse_args(argv)

    directory = args.output or args.data / "text"
    manifest_path = directory / f"{args.category}.embeddings_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[index] {args.category}: already encoded, pass --force to redo it")
        return 0

    records, corpus = load_corpus(directory, args.category)
    items = sorted(records)
    if args.limit:
        items = items[: args.limit]
    documents = [document_text(records[item], args.review_limit) for item in items]
    print(f"[index] {args.category}: encoding {len(items):,} items with {args.model}")
    print(f"[index] document characters {character_report(documents)}")

    encoder = Encoder(args.model, args.device)
    print(f"[index] device: {encoder.device}")
    vectors = []
    started = perf_counter()
    for start in range(0, len(items), args.batch_size):
        vectors.extend(encoder.vectors(documents[start:start + args.batch_size]))
        done = min(len(items), start + args.batch_size)
        if done == len(items) or done % (20 * args.batch_size) == 0:
            print(f"[index] {done:,}/{len(items):,} items in {perf_counter() - started:.1f}s")

    dimensions = len(vectors[0])
    if any(len(row) != dimensions for row in vectors):
        raise ValueError("the encoder returned ragged vectors")
    if encoder.device == "cuda":
        import torch

        torch.cuda.empty_cache()
    written = write_embeddings(
        directory, args.category, items, vectors, model=args.model,
        corpus_sha256=str(corpus.get("corpus_sha256")),
        document_sha256=sha256("\n".join(documents).encode("utf-8")).hexdigest(),
        dimensions=dimensions,
        extra={"device": encoder.device, "review_limit": args.review_limit,
               "encode_seconds": round(perf_counter() - started, 1),
               "documents": character_report(documents)})
    print(json.dumps({key: value for key, value in written.items() if key != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
