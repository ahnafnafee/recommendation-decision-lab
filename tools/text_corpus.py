"""Build a local product-text corpus for one evaluated category.

The interaction archives in this repository carry ratings only, so every word
that describes a product comes from the official Amazon Reviews'23 snapshot.
Three streams are combined:

  review prose    reviewer-written headline and body, kept for the people and
                  items this lab evaluates, before the period end;
  listing text    the item's own title, brand, categories, features, description
                  and price, where the snapshot still holds the listing;
  query pairs     external phrase-to-item pairs restricted to catalogue items:
                  observed ESCI shopping queries and Amazon-C4 review-derived
                  rewrites, the only phrases in this lab written by somebody else.

Each stream is a stage that writes its own filtered artifact, so an interrupted
download costs only that stage. Merge turns the two into one corpus per item.
Nothing here is committed: product text stays under the Git-ignored data
directory, separate from this repository's MIT licence.
"""

from __future__ import annotations

import argparse
import gzip
from gzip import GzipFile
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from reliability.benchmark import T1, T2, file_hash, read_rows, training_data

SNAPSHOT = ("https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
            "resolve/main/raw/{kind}/{name}")
ORIGIN = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/{kind}/{name}.gz"
SOURCES = {"hf": SNAPSHOT, "origin": ORIGIN}
HF_FILE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
LISTING_FIELDS = ("title", "features", "description", "categories", "details", "brand", "price")
REVIEW_LIMIT = 12
REVIEW_CHARS = 240
# External phrase-to-item pairs whose items are catalogue ASINs. ESCI contains
# observed search queries; Amazon-C4 contains review-derived query rewrites.
# The ESCI source is the full official Shopping Queries Dataset, both splits and
# all locales in one hash-pinned parquet that is fetched once into the
# Git-ignored cache; the US locale and the exact or substitute judgements are
# the pairs kept, so a kept phrase is one a real shopper typed that was judged
# to find the item.
ESCI_EXAMPLES = ("https://media.githubusercontent.com/media/amazon-science/esci-data/"
                 "main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet")
ESCI_EXAMPLES_SHA256 = "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"
ESCI_COLUMNS = ("example_id", "query", "product_id", "product_locale", "esci_label", "split")
QUERY_SOURCES = (
    {"source": "esci", "format": "parquet", "url": ESCI_EXAMPLES,
     "sha256": ESCI_EXAMPLES_SHA256, "cache_name": "shopping_queries_dataset_examples.parquet",
     "locale": "us", "labels": ("E", "S"),
     "note": "the full official ESCI Shopping Queries Dataset (both splits), US locale, "
             "judged an exact match or acceptable substitute for the item"},
    {"source": "c4", "format": "csv", "repo": "McAuley-Lab/Amazon-C4", "path": "test.csv",
     "note": "first-person queries rewritten from a review the person wrote about that item"},
)
QUERY_FIELDS = {"identifier": "qid", "query": "query", "item": "item_id"}


def report(stage: str) -> None:
    print(f"[corpus] {stage}", file=sys.stderr, flush=True)


def record_stream(kind: str, name: str, timeout: int = 300, max_bytes: int | None = None,
                  source: str = "hf"):
    """Yield (status, declared_bytes) and then one parsed JSON record per line."""
    url = SOURCES[source].format(kind=kind, name=name)
    request = urllib.request.Request(url, headers={"User-Agent": "recommendation-decision-lab/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        declared = response.headers.get("Content-Length")
        yield response.status, int(declared) if declared else None
        reader = GzipFile(fileobj=response) if url.endswith(".gz") else response
        buffer = bytearray()
        delivered = 0
        while chunk := reader.read(1 << 20):
            delivered += len(chunk)
            buffer += chunk
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                if line.strip():
                    yield json.loads(line)
            if max_bytes and delivered >= max_bytes:
                report(f"stopped after {delivered / 1e6:.0f} MB as requested")
                return


def participants(data: Path, category: str):
    """People and items the lab actually evaluates, plus the training catalogue."""
    train_path = data / f"{category}.train.csv.gz"
    _, catalogue, training = training_data(train_path)
    people = set()
    for split in ("train", "valid", "test"):
        for row in read_rows(data / f"{category}.{split}.csv.gz"):
            people.add(row["user_id"])
    return catalogue, people, train_path, training


def collect_reviews(data: Path, category: str, destination: Path, max_bytes=None,
                    limit: int = REVIEW_LIMIT, force: bool = False, source: str = "hf") -> Path:
    out_path = destination / f"{category}.review_prose.jsonl.gz"
    if out_path.exists() and not force:
        report(f"review prose already staged at {out_path.name}; --force to refetch")
        return out_path
    catalogue, people, _, _ = participants(data, category)
    report(f"{category}: {len(catalogue):,} catalogue items, {len(people):,} evaluated people")
    destination.mkdir(parents=True, exist_ok=True)
    stats = {"rows_scanned": 0, "rows_kept": 0, "declared_bytes": None, "stopped_early": False}
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as sink:
        for entry in record_stream("review_categories", f"{category}.jsonl", max_bytes=max_bytes,
                                   source=source):
            if isinstance(entry, tuple):
                stats["declared_bytes"] = entry[1]
                report(f"review stream HTTP {entry[0]}, "
                       + (f"{entry[1] / 1e9:.2f} GB declared" if entry[1] else "length unknown"))
                continue
            stats["rows_scanned"] += 1
            person, item = entry.get("user_id"), entry.get("parent_asin")
            stamp = entry.get("timestamp")
            if person not in people or item not in catalogue:
                continue
            if not isinstance(stamp, (int, float)) or stamp >= T2:
                continue
            headline = " ".join((entry.get("title") or "").split())
            body = " ".join((entry.get("text") or "").split())
            if not headline and not body:
                continue
            rating = entry.get("rating")
            sink.write(json.dumps({"user_id": person, "asin": item, "timestamp": int(stamp),
                                   "rating": rating if isinstance(rating, (int, float)) else None,
                                   "text": (f"{headline}. {body}".strip(". "))[:REVIEW_CHARS]},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["rows_kept"] += 1
            if stats["rows_scanned"] % 400_000 == 0:
                report(f"{stats['rows_scanned']:,} review rows scanned, {stats['rows_kept']:,} kept")
        else:
            stats["stopped_early"] = max_bytes is not None
    stats["prose_sha256"] = file_hash(out_path)
    (destination / f"{category}.review_prose.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report(f"review prose staged: {stats['rows_kept']:,} rows -> {out_path.name}")
    return out_path


def collect_listings(data: Path, category: str, destination: Path, max_bytes=None,
                     force: bool = False, source: str = "hf") -> Path:
    out_path = destination / f"{category}.listings.jsonl.gz"
    if out_path.exists() and not force:
        report(f"listing text already staged at {out_path.name}; --force to refetch")
        return out_path
    catalogue, _, _, _ = participants(data, category)
    destination.mkdir(parents=True, exist_ok=True)
    stats = {"rows_scanned": 0, "rows_kept": 0, "declared_bytes": None, "stopped_early": False}
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as sink:
        for entry in record_stream("meta_categories", f"meta_{category}.jsonl", max_bytes=max_bytes,
                                   source=source):
            if isinstance(entry, tuple):
                stats["declared_bytes"] = entry[1]
                report(f"listing stream HTTP {entry[0]}, "
                       + (f"{entry[1] / 1e6:.0f} MB declared" if entry[1] else "length unknown"))
                continue
            stats["rows_scanned"] += 1
            key = entry.get("parent_asin")
            if not isinstance(key, str) or key not in catalogue:
                continue
            record = {field: entry.get(field) for field in LISTING_FIELDS if entry.get(field)}
            if not record.get("title"):
                continue
            record["asin"] = key
            sink.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            stats["rows_kept"] += 1
            if stats["rows_scanned"] % 250_000 == 0:
                report(f"{stats['rows_scanned']:,} listing rows scanned, {stats['rows_kept']:,} matched")
        else:
            stats["stopped_early"] = max_bytes is not None
    stats["listings_sha256"] = file_hash(out_path)
    (destination / f"{category}.listings.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report(f"listing text staged: {stats['rows_kept']:,} items -> {out_path.name}")
    return out_path


def _percentiles(values, points=(0.5, 0.9)):
    if not values:
        return {}
    ordered = sorted(values)
    return {f"p{int(point * 100)}": ordered[min(len(ordered) - 1, int(point * len(ordered)))]
            for point in points}


def _fetch_query_source(origin: dict, cache: Path) -> tuple:
    """One external query source, as an iterable of (identifier, text, item) rows.

    A CSV source is the published test table, streamed row by row. The parquet
    source is the full official ESCI file, fetched once into the Git-ignored
    cache and verified against the hash published in the repository's LFS
    pointer, then narrowed to the locale and judgements the entry declares.
    Returns the rows and extras that the manifest records for transparency.
    """
    import csv
    import io
    from collections import Counter

    if origin["format"] == "csv":
        url = HF_FILE.format(repo=origin["repo"], path=origin["path"])
        request = urllib.request.Request(url, headers={"User-Agent": "recommendation-decision-lab/0.1"})
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read().decode("utf-8-sig")
        rows = ((row.get(QUERY_FIELDS["identifier"]), row.get(QUERY_FIELDS["query"]),
                 row.get(QUERY_FIELDS["item"]))
                for row in csv.DictReader(io.StringIO(body)))
        return rows, {}
    if origin["format"] != "parquet":
        raise ValueError(f"unknown query source format {origin['format']!r}")
    import pyarrow.parquet as pq

    cache.mkdir(parents=True, exist_ok=True)
    target = cache / origin["cache_name"]
    if not target.exists():
        request = urllib.request.Request(origin["url"], headers={"User-Agent": "recommendation-decision-lab/0.1"})
        with urllib.request.urlopen(request, timeout=600) as response:
            with open(target, "wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
        report(f"{origin['source']}: fetched {target.stat().st_size:,} bytes")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != origin["sha256"]:
        raise ValueError(f"{origin['source']} cached file hash {digest} does not match {origin['sha256']}")
    report(f"{origin['source']}: cached file matches the published LFS object")
    table = pq.read_table(target, columns=list(ESCI_COLUMNS))
    columns = [table.column(name).to_pylist() for name in ESCI_COLUMNS]
    identifiers, texts, items, locales, labels, splits = columns
    keep_locale = origin.get("locale")
    keep_labels = set(origin.get("labels", ()))
    rows = []
    for identifier, text, item, locale, label in zip(
            identifiers, texts, items, locales, labels):
        if (keep_locale is None or locale == keep_locale) and label in keep_labels:
            rows.append((str(identifier), text, item))
    extras = {
        "rows_in_file": len(identifiers),
        "rows_after_filter": len(rows),
        "filter": {"locale": keep_locale, "labels": sorted(keep_labels)},
        "locale_distribution": dict(sorted(Counter(locales).items())),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "split_distribution": dict(sorted(Counter(splits).items())),
    }
    return rows, extras


def collect_queries(data: Path, category: str, destination: Path, force: bool = False) -> Path:
    """Phrase-to-item pairs written outside this lab, restricted to our catalogue.

    Their length spread and phrasing are the reference shape for phrases this lab
    derives from its own evidence. A pair is kept when its item is a catalogue
    ASIN and its source-specific filter passes, so a kept ESCI phrase is one a
    real shopper typed that was judged to find the item. The first staged pair
    per (source, item) is the one a request reads.
    """
    cache = Path(__file__).resolve().parents[1] / "data" / "cache"
    out_path = destination / f"{category}.queries.jsonl.gz"
    if out_path.exists() and not force:
        report(f"queries already staged at {out_path.name}; --force to refetch")
        return out_path
    catalogue, _, _, _ = participants(data, category)
    destination.mkdir(parents=True, exist_ok=True)
    stats = {"sources": {}, "rows_scanned": 0, "rows_kept": 0}
    seen, lengths = set(), []
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as sink:
        for origin in QUERY_SOURCES:
            rows, extras = _fetch_query_source(origin, cache)
            counted = kept = 0
            for identifier, text, item in rows:
                counted += 1
                text = " ".join((text or "").split())
                key = (origin["source"], text, item)
                if not text or item not in catalogue or key in seen:
                    continue
                seen.add(key)
                lengths.append(len(text.split()))
                sink.write(json.dumps({"query_id": f"{origin['source']}:{identifier}",
                                       "text": text, "asin": item, "source": origin["source"]},
                                      ensure_ascii=False, separators=(",", ":")) + "\n")
                kept += 1
            entry = {
                "rows_scanned": extras.get("rows_in_file", counted),
                "rows_kept": kept,
                "url": origin.get("url") or HF_FILE.format(repo=origin["repo"], path=origin["path"]),
                "note": origin["note"],
            }
            entry.update(extras)
            stats["sources"][origin["source"]] = entry
            stats["rows_scanned"] += entry["rows_scanned"]
            stats["rows_kept"] += kept
            eligible = entry.get("rows_after_filter", counted)
            report(f"{origin['source']}: {kept:,} of {eligible:,} eligible rows point at this catalogue")
    stats["distinct_queries"] = len({text for _, text, _ in seen})
    stats["distinct_items"] = len({item for _, _, item in seen})
    stats["catalogue_items"] = len(catalogue)
    stats["query_word_count"] = _percentiles(lengths)
    stats["queries_sha256"] = file_hash(out_path)
    (destination / f"{category}.queries.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    report(f"queries staged: {stats['rows_kept']:,} pairs -> {out_path.name}")
    return out_path


def read_staged(path: Path):
    if not path.exists():
        return
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def merge(data: Path, category: str, destination: Path, limit: int = REVIEW_LIMIT,
          prose_cutoff: int = T1, source: str = "hf") -> dict:
    """One record per catalogue item, plus a bank of what people wrote themselves.

    Item prose is cut at the model cutoff, so a phrase is matched only against
    words written before the evaluated period. A person's own words are kept to
    the period end because each request filters them by its own timestamp.
    """
    from collections import defaultdict

    catalogue, _, train_path, training = participants(data, category)
    listings = {record["asin"]: record for record in read_staged(destination / f"{category}.listings.jsonl.gz") or ()}
    prose: dict[str, list] = defaultdict(list)
    voices: dict[str, list] = defaultdict(list)
    for record in read_staged(destination / f"{category}.review_prose.jsonl.gz") or ():
        if record["timestamp"] < prose_cutoff:
            prose[record["asin"]].append(record)
        voices[record["user_id"]].append(record)
    for entries in prose.values():
        entries.sort(key=lambda record: -record["timestamp"])
    for entries in voices.values():
        entries.sort(key=lambda record: -record["timestamp"])

    trimmed = {item: [entry["text"] for entry in entries[:limit]] for item, entries in prose.items()}
    out_path = destination / f"{category}.corpus.jsonl.gz"
    destination.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as sink:
        for item in sorted(catalogue):
            record = {"asin": item}
            record.update(listings.get(item, {}))
            if item in trimmed:
                record["review_prose"] = trimmed[item]
            sink.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    voice_path = destination / f"{category}.voices.jsonl.gz"
    with gzip.open(voice_path, "wt", encoding="utf-8", compresslevel=6) as sink:
        for person in sorted(voices):
            sink.write(json.dumps({"user_id": person,
                                   "entries": [[entry["timestamp"], entry["asin"], entry["rating"],
                                                entry["text"]] for entry in voices[person]]},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")

    titled = sum(1 for item in catalogue if item in listings)
    priced = sum(1 for item in catalogue if isinstance(listings.get(item, {}).get("price"), (int, float)))
    manifest = {"schema": 3, "category": category, "snapshot": SOURCES[source], "mirror": source,
                "cutoff_ms": T1, "period_end_ms": T2, "prose_cutoff_ms": prose_cutoff,
                "train_sha256": file_hash(train_path), "train_rows": training["rows"],
                "catalogue_items": len(catalogue),
                "items_with_review_prose": len(trimmed), "items_with_listing_title": titled,
                "items_with_price": priced, "people_with_own_words": len(voices),
                "review_limit": limit, "review_chars": REVIEW_CHARS,
                "corpus_sha256": file_hash(out_path), "voices_sha256": file_hash(voice_path),
                "stages": {"review_prose": _stage(destination, category, "review_prose"),
                           "listings": _stage(destination, category, "listings"),
                           "queries": _stage(destination, category, "queries")}}
    (destination / f"{category}.corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _stage(destination: Path, category: str, kind: str) -> dict:
    path = destination / f"{category}.{kind}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"missing": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--category", choices=("Video_Games", "Musical_Instruments"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stage", choices=("reviews", "listings", "queries", "merge", "all"),
                        default="all")
    parser.add_argument("--source", choices=("hf", "origin"), default="hf",
                        help="Snapshot mirror. The origin serves compressed per-category files.")
    parser.add_argument("--max-mb", type=int, default=None, help="Stop a stream after this many MB.")
    parser.add_argument("--review-limit", type=int, default=REVIEW_LIMIT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = args.output or args.data / "text"
    cap = args.max_mb * 1_000_000 if args.max_mb else None
    if args.stage in ("reviews", "all"):
        collect_reviews(args.data, args.category, destination, max_bytes=cap,
                        limit=args.review_limit, force=args.force, source=args.source)
    if args.stage in ("listings", "all"):
        collect_listings(args.data, args.category, destination, max_bytes=cap,
                         force=args.force, source=args.source)
    if args.stage in ("queries", "all"):
        collect_queries(args.data, args.category, destination, force=args.force)
    if args.stage in ("merge", "all"):
        print(json.dumps(merge(args.data, args.category, destination, args.review_limit,
                               source=args.source), indent=2))


if __name__ == "__main__":
    main()
