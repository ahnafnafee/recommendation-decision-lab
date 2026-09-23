# Recommendation Decision Lab

A runnable recommendation service built around a simple question: when does a personalized challenger earn the right to replace a reliable popularity baseline? The project combines a full-catalog temporal evaluation on two Amazon Reviews'23 product categories, a validation-only route gate, a local HTTP API, an interactive synthetic demo, explicit fallback behavior, and a wording-driven arm that was measured against the same gate and declined. It is a companion engineering project to an earlier MovieLens measurement study; the two studies have separate code, data and evidence boundaries.

**Measured result:** On a fresh Musical Instruments category, the fixed hybrid scored **0.008681 NDCG@10** against **0.008007** for validation-selected recent popularity across 35,905 eligible positive-review requests. The paired difference was **+0.000674 [0.000040, 0.001338]**. This is a small offline gain with low absolute recovery, not online impact or a new algorithm. The corrected Video Games run is exploratory because an earlier replay exposed a tuple-order bug in personalization. Read the [transfer note](reports/transfer_note.md) for the protocol and limitations.

## Try the demo

Python 3.11 or newer and the standard library are sufficient. From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m reliability.server --port 8899
```

Open `http://127.0.0.1:8899/`. The browser profiles and item names are invented. Compare baseline, challenger and active rankings, then click **Simulate challenger failure** to inspect the fallback. The synthetic gate stays closed because no real validation decision applies to those invented users. Below the comparison there is a phrase box: **Answer in words** scores a typed sentence, and **Offer without asking** re-answers from a sentence the service is already holding, with no wording in the request. Started this way the service holds no product listings, so the wording stands down and the behavioural route keeps answering; see the [phrase-to-products extension](#phrase-to-products-extension) for the run that answers in words. The service binds to loopback only. The same `web/` directory also runs as a static preview with no backend. Its GitHub Pages workflow requires manual dispatch and has not been deployed.

The API exposes `GET /api/health`, `GET /api/metrics`, `GET /api/ambient`, `POST /api/recommend` and `POST /api/utterance`, the last two with JSON bodies such as `{"history":["atlas","comet"],"k":5}` and `{"phrase":"a warm tube amp for a small room, not a pedalboard"}`. Responses show the active route, shadow and baseline lists, fallback reason and local ranking time. No submitted history is logged or persisted. [API.md](API.md) documents every field, counter and startup flag. This is a local demonstrator rather than a hardened production service.

## Reproduce the data experiment

The source is the [McAuley Lab Amazon Reviews'23 5-core absolute-timestamp benchmark](https://amazon-reviews-2023.github.io/data_processing/5core.html). Review the owner's current page and terms before acquisition. Download the **Video_Games** and **Musical_Instruments** `timestamp_w_his` train/valid/test CSV gzip archives into `data/` with the filenames `Video_Games.train.csv.gz`, `Video_Games.valid.csv.gz`, and so on. The source paths follow this pattern:

```text
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/timestamp_w_his/<Category>.<split>.csv.gz
```

The Video Games replay is exploratory after a corrected implementation defect. The fixed method was then evaluated on Musical Instruments as a fresh category. The [research note](reports/transfer_note.md) describes the evaluation design and its limits. Re-running the commands below verifies the implementation; it does not create another untouched test result.

```powershell
python -m reliability.benchmark --data data --category Video_Games --output runs/video-games-replay
python -m reliability.benchmark --data data --category Musical_Instruments --output runs/musical-replay
```

Every run refuses an existing output directory. Its `validation_decision.json` is written before the test archive is opened. `aggregate.json` contains counts, metrics, paired intervals, source hashes and limitations, with no row-level predictions or user IDs. The reviewed public aggregate copies are [Video Games](reports/video_games_exploratory.json) and [Musical Instruments](reports/musical_instruments_confirmation.json).

To serve a locally trained Musical Instruments model, build an ignored, hash-checked bundle and pair it with the exact aggregate from its run:

```powershell
python -m reliability.bundle --data data --category Musical_Instruments --output runs/musical-bundle
python -m reliability.server --bundle runs/musical-bundle --aggregate runs/musical-replay/aggregate.json --port 8899
```

The real-data API accepts item IDs in `history`; the browser demo's invented profiles are unavailable in that mode. Neither the archive, model bundle nor review-derived weights are part of the public repository.

## Neural retrieval extension

The optional PyTorch path trains a two-tower retriever on chronological pairs from the training archive. A masked mean of up to 20 prior positive items passes through a small MLP to form the query vector; a separate item tower embeds train-known products. In-batch softmax supplies negatives, with duplicate targets and already reviewed products masked so they are not treated as negatives. Training uses a fixed seed and three epochs. Exact full-catalog scoring excludes prior items and keeps future items outside the train catalog as misses.

Install PyTorch for your platform using the [official selector](https://pytorch.org/get-started/locally/), then run these commands from the repository root:

```powershell
.venv\Scripts\python.exe -m reliability.neural --data data --category Musical_Instruments --output runs/musical-neural
.venv\Scripts\python.exe -m reliability.neural_calibrate --data data --weights runs/musical-neural/neural_weights.pt --output runs/musical-neural-calibrated
```

The first command saves fitted weights and a hash-checked manifest only in ignored `runs/`. The second selects a neural/popularity blend on validation and compares it with the already active hybrid before opening test rows. Each command refuses an existing output directory. To inspect the neural shadow through the local API, add `--neural-weights runs/musical-neural/neural_weights.pt --neural-train data/Musical_Instruments.train.csv.gz --neural-aggregate runs/musical-neural-calibrated/aggregate.json` to the real-data server command above. The response includes `neural_shadow`, its gate decision and fallback reason; only a validation-approved challenger can become active.

The [exploratory neural extension](reports/neural_extension.md) records the measured result. On Musical Instruments, the calibrated neural blend beat recent popularity but did not beat the existing co-review hybrid, so its stronger-challenger gate stayed closed. This category's test period was already known from the earlier study; the extension is a model-development comparison, not a new independent confirmation. The synthetic browser demo remains separate from fitted review-derived weights.

## Shadow-profile extension

The committed route personalises from the last twenty distinct reviewed items attached to a request, regardless of their star rating. It does not use the rating or review time, and it leaves older history items and adjacent-category activity outside the scorer. This exploratory extension builds user profiles from that remainder and keeps the two acquisition modes apart because their costs are not comparable: already-held evidence needs no new prompts, while elicitation would require prompts whose real response rate is not measured here.

```powershell
python -m tools.signal_audit --data data --categories Musical_Instruments Video_Games --output runs/signal_audit.json
python -m reliability.shadow --data data --output runs/musical-shadow-v2 --category Musical_Instruments --donor-category Video_Games
```

The audit answers how much discarded evidence exists: in Musical Instruments, 47,459 validation history slots have a train-recorded rating below four (35,444 inside the scorer's window), while 1,288 eligible requests have no history to infer from. Another 96,984 slots lack a user–item rating in training; 20,310 of those refer to products absent from the training catalog. For all 14,175 validation users with a prior training record, the request's own `history` column reproduced that record exactly. The last-twenty-items window is therefore a scorer choice; 23% of distinct history items and 25% of the low ratings in those records sit outside it. The pilot scores six challengers against the route validation already chose, using the same full-catalog evaluation and paired user-cluster gate as the earlier study. It refuses an existing output directory and publishes aggregates only. Omit `--donor-category` to skip the cross-surface probe, or add `--include-test` to report the selected configuration on the already-exposed test period.

The [shadow-profile design note](reports/shadow_profile_design.md) records the measured result and its limits; the public-safe [shadow aggregate](reports/musical_shadow_exploratory.json) and [signal audit](reports/signal_audit.json) contain the exact figures. On Musical Instruments two arms using already-held information had positive same-cohort differences: repulsion from sub-four ratings gained `+0.000449` NDCG@10 (descriptive 95% interval `0.000266` to `0.000626`), and weighting observed four- and five-star reviews by grade and age gained `+0.000466` (`0.000165` to `0.000750`). Both configurations were selected and scored on the same validation cohort, so these intervals do not establish a fresh improvement. The published eight-cell ablation says where the second gain lives: almost all of it is the age decay, a three-year half-life beating no decay by `+0.000412` while the graded weights add `+0.000054` on top, and a 90-day half-life loses most of the benefit. For comparison, a simulated 100-prompt interview using recorded ratings from all splits as a hindsight answer oracle reached `+0.000871` (`0.000537` to `0.001172`) over the active route, including the already-held repulsion effect. The oracle found about one recorded answer per 440 prompts. This is neither a measured user response rate nor a live-study gain. A prior borrowed from training users' first positive reviews lost instead, and borrowing a profile from the adjacent category could not be measured at all: a donor record was reachable for 47 of the 33,993 requests. The history-size arm also met the nominal gate on that same cohort; its bucket results suggest the committed blend gives too much weight to long histories. The bucket weights need a separate evaluation. Every number is exploratory: both of this repository's test periods were already opened by earlier studies.

## Phrase-to-products extension

The committed route reads the `history` column attached to the current request and nothing else. It cannot answer someone who says what they want: a sentence with a negation in it and a price ceiling attached. This extension matches a typed sentence against product listings held locally, and reports what the wording selected beside the unchanged behavioural answer rather than merging the two into one number. Negation is scoped to the words after the cue and ends at a restart, so "not a pedalboard" costs a pedalboard its place, and a stated "under $300" keeps out products whose recorded price is above it, while products with no recorded price stay. Product text is fetched by you into the Git-ignored `data/text/`; this repository does not redistribute it.

```powershell
python -m tools.text_corpus --data data --category Musical_Instruments --output data/text --source origin
```

Each stage — review prose, listing fields, published query pairs, merge — writes its own filtered artifact and a schema-3 manifest carrying the corpus and voices hashes, so a later reader can tell which cut of the publishers' snapshot produced a number. Matching is BM25 over the listings and stays inside the standard library. An optional second route encodes the same listings once, outside the request path, under an environment that has the `language` extra:

```powershell
pip install -e ".[language]"
python -m tools.build_text_index --data data --category Musical_Instruments --output data/text
```

Serving pairs the corpus with the validated bundle, and the measured arm runs on the same frozen requests the earlier studies used:

```powershell
python -m reliability.server --bundle runs/musical-bundle --aggregate runs/musical-replay/aggregate.json --port 8899 --text data/text
python -m reliability.phrasing --data data --output runs/musical-phrasing --category Musical_Instruments
```

Add `--embeddings --device cuda` to the server command to fold the encoded route into the phrase answer. The server refuses `--embeddings` without `--text`, and refuses `--text` without `--bundle`, because invented items have no listings to read. A sentence the person already left with the service answers `GET /api/ambient` without being repeated: up to eight of them live in process memory and nothing is written to a log.

Serving and measurement are deliberately different configurations, because they answer different questions. An interactive box is phrase-first and may reach any listing, since someone who asks for something specific expects to be answered. The measured arm asks whether ambient wording is worth more than a feed, so it confines the phrase to the head of the behavioural fallback order plus co-review neighbours, and takes its weight from a fixed validation grid.

At 400 validation requests the scored arm had not cleared the gate under either scope. The confined primary configuration (weight 0.75, repulsion 0.5, price penalty 0.5) lost `−0.00240` NDCG@10 against the same frozen base, 95% lower bound `−0.00878`; the catalogue-wide fusion lost `−0.00337` (`−0.00963`). The decomposition says why, and the two ceilings multiply: at 400 validation requests the frozen behavioural route had never considered the eventual answer in 78.5% of requests, so a confined phrase had nothing to promote; and read back unconstrained, a person's own earlier words named the eventual answer's product in 1.41% of requests, where the exact product title reached it in 100%. Every figure in this section is a pilot reading taken at 400 validation requests; the full Musical Instruments run is still writing to `runs/phrase-mi-lexical`, so these numbers are not final. The [phrase-to-products note](reports/phrase_to_products.md) holds the protocol, the six wording conditions and the reach table; the [literature trail](reports/phrase_to_products_survey.md) holds the sources behind the design.

## Manuscript

The [paper](paper/main.pdf) presents the temporal comparison, neural challenger, guarded service, and limits in one standalone article. It is also available as a [ResearchGate preprint](https://www.researchgate.net/publication/414634917_When_Does_Personalization_Earn_the_Route_A_Full-Catalog_Temporal_Study_of_Guarded_Recommendation). Its [LaTeX source](paper/main.tex) and [build instructions](paper/README.md) are included for inspection.

## What this demonstrates

- A validation-selected transparent control and a challenger evaluated over the entire train-known catalog, with unseen future items retained as misses.
- A gate that declines a challenger without sufficient validation evidence, plus runtime fallback when the challenger fails or a history is unsupported.
- Train-only popularity and item co-review modeling, deterministic retrieval, a hash-checked local bundle, an HTTP interface, aggregate request metrics and end-to-end service checks.
- A trained, hash-checked two-tower retrieval path that can be compared with the existing route and inspected as a shadow without displacing a stronger challenger.
- A measured comparison between evidence the platform already holds and evidence it must ask for: two exploratory arms using already-held ratings or review times had positive same-cohort differences; a hindsight-oracle questioning probe found roughly one recorded answer per 440 simulated prompts.
- An arm that was measured and declined: in a pilot at 400 validation requests a wording-driven challenger lost against the same frozen behavioural base under both the confined and the whole-shelf scope, so the sentence answers a request on its own terms instead of displacing a route that has not earned the change.
- An auditable failure: the first exploratory replay silently dropped personalization because neighbor tuple fields were reversed. A training-to-ranking regression test now catches that fault.

The 5-core dataset was retrospectively filtered using the full corpus; a review is not an exposure or engagement label. A second product category is useful transfer evidence but remains on the same platform and one temporal split. Local timing excludes HTTP, initialization, concurrency and network effects. These limits are explained in the [research note](reports/transfer_note.md).

## Research notes and literature

Every part of this system ships with a design note that states what was measured and
a separate literature survey that records the published work the design draws on. The
surveys name only sources whose abstract or full text was read, give the arXiv
identifier beside each claim, and repeat every URL in a Sources list at the end.

| Component | Design note | Literature survey |
| --- | --- | --- |
| Full-catalog temporal evaluation and the route decision | [transfer_note.md](reports/transfer_note.md) | [survey_evaluation_method.md](reports/survey_evaluation_method.md) |
| Two-tower challenger and the calibration gate | [neural_extension.md](reports/neural_extension.md) | [survey_dense_retrieval.md](reports/survey_dense_retrieval.md) |
| Evidence already held versus evidence elicited | [shadow_profile_design.md](reports/shadow_profile_design.md) | [survey_shadow_profiles.md](reports/survey_shadow_profiles.md) |
| Answering a sentence | [phrase_to_products.md](reports/phrase_to_products.md) | [phrase_to_products_survey.md](reports/phrase_to_products_survey.md) |
| Product text and the encoded route | [phrase_to_products.md](reports/phrase_to_products.md) | [survey_dense_retrieval.md](reports/survey_dense_retrieval.md) |
| Guarded serving, fallback and reproducibility | [ARCHITECTURE.md](ARCHITECTURE.md) | [survey_reliable_serving.md](reports/survey_reliable_serving.md) |

## Repository map and publication status

| Path | Purpose |
| --- | --- |
| `reliability/` | Training, full-catalog evaluation, bundle verification, local service, shadow-profile arms, and the phrase route (`language.py` for parsing and BM25, `embeddings.py` for the encoded route, `phrasing.py` for the measured arm) |
| `reports/` | Public-safe aggregates and research interpretation |
| `paper/` | Manuscript PDF, source and build instructions |
| `tools/` | Release check, the aggregate signal audit that bounds the shadow-profile questions, and the corpus and embedding-artifact builders that fill `data/text/` |
| `data/text/` | Product listings, review prose and published query pairs you fetch locally, plus the embedding artifacts built from them; Git-ignored |
| `web/` | Dependency-free synthetic browser demo with static fallback |
| `tests/` | Exact-ranking, training-path, bundle, HTTP-fallback, phrase-parsing, negative-evidence retrieval and corpus-integrity checks |
| `.github/` | CI workflow and issue and pull request templates |

[ARCHITECTURE.md](ARCHITECTURE.md) explains the module contracts and where the optional dependencies load, [API.md](API.md) documents every request field, response field, counter and startup flag, [CONTRIBUTING.md](CONTRIBUTING.md) covers environment setup and the conventions the committed numbers depend on, [CHANGELOG.md](CHANGELOG.md) records what changed, [SECURITY.md](SECURITY.md) states what the demo service does and does not send, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) covers how contributions land.

Original source, tests and documentation use the [MIT license](LICENSE). Dataset records, fitted weights and owner metadata are outside that license; see [data permissions](DATA_LICENSE.md). The [public GitHub repository](https://github.com/ahnafnafee/recommendation-decision-lab) contains source, the manuscript and aggregate results, but no review archives or fitted weights. The paper has no claimed conference or journal acceptance. `CITATION.cff` names Ahnaf An Nafee for software citation.
