# Contributing

The bar here is unusual in one respect: the numbers already committed are reproducible from
this source, so a change that silently moves one of them is a change to the findings, not
just to the code. Everything below follows from that.

## Environment

Python 3.11 or newer, and nothing else is required for the core, the service or the tests.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Two optional extras unlock the heavier routes. Install them into the same environment you
run the tests with, so the classes that skip without them actually run:

```powershell
pip install -e ".[neural]"      # torch, numpy: the two-tower challenger
pip install -e ".[language]"    # sentence-transformers, numpy: encoded phrase retrieval
```

## Running things

Commands are written as modules and run from the repository root, which puts the packages
on `sys.path`:

```powershell
python -m unittest discover -s tests -v
python -m reliability.benchmark --data data --category Video_Games --output runs/video-games-replay
python -m reliability.phrasing --data data --output runs/musical-phrasing --category Musical_Instruments
```

The files in `tools/` are plain scripts, not modules, so running one that imports
`reliability` needs the root on the path:

```powershell
$env:PYTHONPATH="$PWD"
python tools/text_corpus.py --data data --category Musical_Instruments --output data/text --source origin
python tools/signal_audit.py --data data --categories Musical_Instruments Video_Games --output runs/signal_audit.json
```

`tools/build_text_index.py` and `tools/release_check.py` add the root themselves and run
either way. `python -m tools.<name>` also works for all of them.

## Data layout

`data/` holds the official Amazon Reviews 2023 5-core archives for `Video_Games` and
`Musical_Instruments` and is Git-ignored; `DATA_LICENSE.md` covers their terms. `data/text/`
is also Git-ignored and holds what the wording route reads: the fetched product listings,
review prose and published query pairs, and the embedding artifacts built from them. Keep
them out of commits — `tools/release_check.py` rejects a stray root file and any committed
JSON carrying per-request keys.

`runs/` is Git-ignored and holds every measurement. Each run writes its own directory and
refuses to overwrite an earlier one, so evidence is retained rather than replaced: name the
directory for the category and the variant, and start a new one for a new question instead
of reusing a name.

## Conventions the measurements depend on

- **The core stays standard-library-only.** A heavy dependency belongs behind a lazy import
  inside the function that needs it — as `neural._torch()` and `embeddings.Encoder.__init__`
  do — and the code must keep a working path without it. Adding a hard dependency to
  `reliability/core.py` needs a reason in the pull request.
- **Frozen boundaries stay frozen.** `T1`, `T2`, `ALPHAS`, the split membership rules and
  the bootstrap seed in `reliability/benchmark.py` define what the committed numbers mean.
  Changing one invalidates every figure beside it; extend with a new run instead.
- **Decide on validation, then look at the test split.** `validation_decision.json` is
  written before the test rows are opened, in `benchmark.py` and in `phrasing.py`. Keep that
  order in anything new that measures.
- **Artifacts carry a schema version and a sha256 of what they were built from**, and the
  loader refuses a disagreement. Bundle manifests, corpora and embedding artifacts all do
  this; a new artifact should too.
- **Aggregate JSON holds aggregates.** No identifiers, no histories, no rankings. The
  forbidden-key list in `tools/release_check.py` is the floor, not the target.
- **Phrase serving and phrase measurement are different configurations on purpose** — see
  the phrase section of [ARCHITECTURE.md](ARCHITECTURE.md). Do not unify them to make a
  comparison convenient.
- **Wording about people.** Describe behaviour: what a route used is "inferred" or "already
  held", what it asked for is "elicited", wording that arrives on a surface other than the
  search box is "cross-surface", and state where it is kept ("held in process memory"). Stay
  with those terms in reports and comments, and do not reach for a legal or emotional framing
  of the same behaviour.

## Before a commit

```powershell
$env:PYTHONPATH="$PWD"
python -m unittest discover -s tests
python tools/release_check.py
```

Both are what CI runs, on Python 3.11 and 3.14. A test class skips when `torch` is not
installed; that is the only skip, and a change to the neural or encoded phrase paths should
be run with the extras in place. If you add a file at the repository root, it must be added
to the allowlist in `tools/release_check.py`, which is the file that enforces the published
source boundary.

Documentation moves with the code: [README.md](README.md) for what a first-time reader runs
and gets, [ARCHITECTURE.md](ARCHITECTURE.md) for module contracts, [API.md](API.md) for any
request or response field, [CHANGELOG.md](CHANGELOG.md) under `Unreleased`. Numbers quoted
in prose must match the committed JSON in `reports/` — when they disagree, the JSON is
right.
