# Architecture

The repository is a measurement harness and a small local service that shares its code.
Everything below is the shape a contributor will meet in the source; the claims the
measurements support are in [README.md](README.md) and `reports/`.

## Shape of the system

```
data/*.csv.gz                        official 5-core archives, fetched, Git-ignored
data/text/                           listings, review prose and query pairs you fetch, Git-ignored
        |
        +---> reliability.benchmark   frozen train / validation / test replay
        +---> reliability.bundle      one training pass written once, hash-checked
        +---> reliability.shadow      inferred and elicited profile arms
        +---> reliability.phrasing    the wording-driven arm, scored against the same gate
        |                                    |
        +---> reliability.server  <---------+  loopback HTTP, same ranking code
        |          |
        +---> web/                  static page, works offline from web/demo_bundle.json
        +---> reports/*.json        aggregate numbers only, committed
```

No component talks to a network at run time except the two fetchers that fill `data/`
and `data/text/`. There is no database, no cache layer and no background worker: model
state lives in one process, and each run writes its own directory and refuses to
overwrite an earlier one.

## Module map

| Module | Contract |
| --- | --- |
| `reliability/core.py` | `Recommender.train(positive_events, catalog, cutoff_ms, recent_days=365)` plus `top_k`, `rank_grid`, `last_distinct`, `top_neighbors` and `ndcg_one_target`. Standard library only, deterministic, and the exact order any other component reports. |
| `reliability/benchmark.py` | The frozen boundaries `T1`, `T2`, the score grid `ALPHAS`, split membership checks, and `user_cluster_interval(queries, first, second, draws=1000, seed=20260922)`. `run` hashes the archives before reading them, scores validation, writes `validation_decision.json`, and only then opens the test rows. |
| `reliability/bundle.py` | `save_bundle` / `load_bundle`. `manifest.json` holds `schema`, `category`, `catalog_items`, `source_sha256` and `model_sha256`; loading re-hashes the payload and rejects a file that disagrees. |
| `reliability/neural.py` | Two-tower challenger. `torch` is imported inside `neural._torch()`, so nothing in the core path pays for it. `load_retriever` binds a weights file to the training archive it was fitted on. |
| `reliability/neural_calibrate.py` | Validation-only calibration of a fitted challenger: a blend grid against whichever route is currently active, with its own `validation_decision.json`. The run is labelled exploratory because the model design followed prior exposure to that category's test period. |
| `reliability/shadow.py` | Inferred and elicited profile arms: `novice_prior`, `interview_order`, `ShadowRecommender`, `shadow_queries`, `score_requests`. Tuning values sit beside it (`NEGATIVE_RATING = 3`, `RATING_WEIGHT`, `HALF_LIVES`, `GAMMAS`, `QUESTION_BUDGETS`, `INTERVIEW_DEPTH = 200`). |
| `reliability/language.py` | Phrase parsing, the product-text corpus and BM25 ranking. See the phrase route below. |
| `reliability/embeddings.py` | Optional encoded retrieval over the same corpus. `Encoder` imports `sentence_transformers` and `torch` inside `__init__`; `EmbeddingIndex` scores unit vectors by cosine and keeps a small cache of encoded phrases. `load_embeddings` refuses an artifact whose recorded corpus hash disagrees with the corpus it is being asked about. |
| `reliability/phrasing.py` | The measured arm. Six wording conditions, a fixed weight grid, `validation_decision.json` written before the test split is opened. |
| `reliability/server.py` | `RecommendationApp` (route, gate, counters, held sentences) and a `ThreadingHTTPServer` bound to `127.0.0.1`. [API.md](API.md) documents the interface. |
| `tools/text_corpus.py` | Fetcher that writes `data/text/`. Runs as a plain script or as a module. |
| `tools/build_text_index.py` | Builds the sentence-embedding artifact from that corpus, using the optional `language` extra. |
| `tools/signal_audit.py` | Aggregate bound on how many elicited answers are reachable on a frozen boundary. |
| `tools/release_check.py` | The published-source boundary: allowlist of root files, forbidden JSON keys, size caps. |
| `web/` | No build step and no dependency; falls back to `web/demo_bundle.json` when the service is not running. |
| `tests/` | Standard-library `unittest`, no network, one class that skips when `torch` is absent. |

## Evaluation order, and why it is fixed

`reliability.benchmark.run` is the pattern every measurement command follows: hash the
archives, train on `train` only up to `T1`, score the validation split, choose the
baseline and the challenger's alpha from validation, compute the user-cluster paired
bootstrap, persist `validation_decision.json`, and only then read the test split. The
challenger is active only when the interval's `lower` bound exceeds zero, so an empty
interval is a decline rather than a tie. Nothing downstream re-derives that choice: the
service reads the persisted decision, which is why a started server answers the same way
after a restart.

`validation_decision.json` is the seam. It is written before the test split is opened, so
a decision cannot be moved to fit the number it will be judged on, and a reader can check
the choice without reading any test result.

## Phrase route

`reliability/language.py` turns a sentence into a score adjustment, using only the
standard library.

- `parse_utterance` returns `Utterance(positive, negative, price_ceiling, raw)`. Negation
  is scoped to the words after a cue and ends at the next cue or at a word that restarts
  the request (`but`, `only`, `prefer`); `rather than X` and `instead of X` rule `X` out,
  while a bare `rather` is filler. `price_ceiling` reads `under $1,200`, `up to 300`,
  `< 50`.
- `LexicalIndex` is BM25 written against sorted item IDs: postings, `k1 = 1.2`, `b = 0.75`
  and `idf = log(1 + (N - df + 0.5) / (df + 0.5))`. Deterministic order, no dependency, and
  `numpy` is used only when already importable — the lexical path keeps the core
  dependency-free.
- `load_corpus` reads a `data/text/` directory and refuses it unless the manifest declares
  `CORPUS_SCHEMA = 3`, names the requested category, and covers at least half the
  catalogue the model was trained on.
- `TextRanker` fuses the committed route with the phrase. Its `scope` field decides what a
  phrase may promote at all, and the repository uses two deliberately different settings
  because they answer different questions: `"catalogue"` (serving — a phrase may reach any
  listing, phrase weight `1.0`) and `"behavioural"` (the measured arm — confined to the head
  of the fallback order plus this person's co-review neighbours, weight taken from a fixed
  validation grid). `rank_from` keeps the result exact without re-scoring the catalogue:
  an item with no adjustment holds its base place, and the candidate pool is extended by
  exactly as many items as were depressed. With `weight = 0` or an empty phrase the ranking
  reproduces `Recommender.top_k` item for item. A stated figure is split the same way: the
  measured arm scores it through `price_penalty`, so a costlier product can still place,
  while the served route sets `enforce_ceiling`, which drops products whose recorded price is
  above the figure, keeps products with no recorded price, and gives up the figure rather than
  returning an empty answer.

`reliability/embeddings.py` adds a dense route over the same documents. It is optional
twice over: the package needs the `language` extra to encode, and a served route works
without it because `TextRanker` fuses whichever routes exist and takes the better of the
two. An artifact records the model, dimensions, item list, document hash and corpus hash;
`load_embeddings` re-hashes the vector file and refuses one encoded from different product
text rather than scoring against stale vectors.

`reliability/phrasing.py` scores the arm. It reads the corpus, the review prose of earlier
reviewers and published query pairs from the same archives, builds a scrubbed variant that
keeps only product words appearing in under 1% of listings, scores six conditions against
five weight configurations, and selects its configuration with a rule recorded in the
decision file: best validation `own_words` NDCG@10, then the smaller weights. It writes
`validation_decision.json` before the test split is opened, and reports two ceilings beside
the headline number — how often the answer sat inside the shortlist a confined phrase could
promote at all, and how often each wording reached the answer at all. Four of the six
conditions are informed by the answer, so they are read as paired deltas rather than
absolutes.

`tools/text_corpus.py` writes the corpus: a gzip line per product, a gzip line per person
who reviewed earlier at that boundary, and `{category}.corpus_manifest.json` with `schema`
3, the frozen boundary timestamps, source hashes, coverage counts, and the `corpus_sha256`
and `voices_sha256` that every later artifact is checked against.

## Heavy code, loaded lazily

The dependency-free rule is kept by import placement, not by configuration: `torch`
arrives inside `neural._torch()` and inside `embeddings.Encoder.__init__`, and `numpy`
only behind an `ImportError` fallback in `language.py`. Installing the package pulls in
nothing; the `neural` and `language` extras in `pyproject.toml` unlock the two-tower
challenger and the encoded phrase route, and the lexical route, the behavioural route and
the test suite behave the same either way.

## Published boundary

`data/`, `data/text/`, `runs/` and `build/` are Git-ignored. `tools/release_check.py`
enforces the rest at the last step before a commit: a fixed allowlist of root files, the
keys `user_id`, `parent_asin`, `history`, `target`, `ranked`, `weights` and `model` absent
from every committed JSON, 200,000-byte and 100-element caps on report arrays, and a size
cap on the manuscript PDF.
