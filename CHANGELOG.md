# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project does not tag
releases, so sections are dated by the commits they describe and unreleased work is
collected at the top.

## Unreleased

### Added

- Phrase-to-products route. `POST /api/utterance` answers a typed sentence against
  locally held product listings: `reliability/language.py` parses the sentence into the
  terms to look for, the terms it rules out and any stated price ceiling, scores them with
  a standard-library BM25 index over titles, brands, categories, features and descriptions,
  and fuses that with the committed behavioural route — at phrase weight 1.0 when served, at
  a validation-chosen weight when measured. Negation is honoured, a stated ceiling keeps out
  products recorded above it on the serving path while products with no recorded price stay,
  and an empty phrase reproduces the committed order exactly.
- `GET /api/ambient`, which re-answers from a sentence the service is already holding, with
  no wording in the request. Up to eight such sentences are held in process memory, and
  nothing is written to a log.
- `reliability/embeddings.py` and `tools/build_text_index.py`, which build and load a
  sentence-embedding artifact over the same product text. An artifact records the corpus
  hash it was encoded from, and loading refuses one that disagrees. Available through the
  server's `--embeddings` and `--device` flags.
- `tools/text_corpus.py`, which fetches product listings, earlier review prose and published
  query pairs into Git-ignored `data/text/` and writes a schema-3 manifest holding the
  corpus and voices hashes every later artifact is checked against.
- `reliability/phrasing.py`, the measured wording arm: six wording conditions scored against
  five weight configurations, with `validation_decision.json` written before the test split
  is opened and the configuration chosen by a fixed recorded rule. In a pilot at 400
  validation requests the arm had not cleared the activation gate under either promotion
  scope, and the full run is still writing; it answers in the demo and does not displace the
  behavioural route.
- `language` optional extra (`sentence-transformers`, `numpy`) for the encoded route.
- Demo controls for the route: a sentence field and an *Offer without asking* button.
- `product_text` in `GET /api/health`, and the `phrase_requests`, `phrase_answers`,
  `sentences_held` and `offerings` counters in `GET /api/metrics`.
- Tests for utterance parsing and the corpus contract, the embedding artifact contract, and
  the measured arm.
- `ARCHITECTURE.md`, `API.md`, `CONTRIBUTING.md` and this file.
- `reports/phrase_to_products.md` (protocol, wording conditions, reach table) and
  `reports/phrase_to_products_survey.md` (the literature trail behind the design).
- `SECURITY.md`, which states what the demo service binds to, what it writes and its one
  outbound call; `CODE_OF_CONDUCT.md`; issue and pull request templates; and `[project.urls]`
  in `pyproject.toml` naming the repository, the issue tracker and this changelog.
- Four further literature surveys, one per remaining component, each listing only sources
  whose arXiv page or published page was opened and repeating every URL in a Sources list:
  `reports/survey_dense_retrieval.md` (the encoded route and the calibration gate),
  `reports/survey_shadow_profiles.md` (already-held versus elicited evidence),
  `reports/survey_evaluation_method.md` (the frozen temporal boundary and the activation
  gate), and `reports/survey_reliable_serving.md` (fallback, artifact integrity and
  reproducibility).
- `reports/survey_data_provenance.md`, the sixth literature survey, for the data layer:
  where the corpus and its text sidecars come from, where the 5-core / leave-last-out /
  absolute-timestamp protocol comes from, why the split is time-ordered, what is published
  on cold targets and catalogue churn, and what the dataset authors state about usage
  terms. It records every identifier and DOI beside each claim and lists the candidates it
  dropped rather than cited, with the reason.

### Changed

- `paper/main.tex`: the manuscript now reports every challenger family in the repository —
  the committed co-review hybrid, the gated two-tower retriever, the two arms that re-read
  evidence the platform already holds, the arm that asks instead, the size-conditioned prior,
  and the wording arm — together with the reach decomposition behind the wording result. Its
  bibliography grew to 27 entries, adding the evaluation-methodology sources on split choice,
  full-catalogue scoring, baseline tuning and uncertainty.
- `pyproject.toml` gained the `language` extra; the core still installs no dependencies.
- `tools/release_check.py` allowlists the new root documents.
- The repository map and the demo section of `README.md` cover the wording route and the
  locally fetched corpus.

### Fixed

- The shadow audit had counted one number for two different things: history slots with no
  user–item rating in training and history slots whose item is absent from the training
  catalog. `tools/signal_audit.py` now reports them separately — on Musical Instruments
  validation, 96,984 slots lack a user–item training rating, 20,310 of those absent from the
  catalog — and `reliability/shadow.py` no longer offers to ask about items the supplied
  history already carries. The Musical Instruments validation probe was rerun with the
  corrected simulator: the two already-held arms beat the active route by 0.000449
  (repulsion) and 0.000466 (graded decay and age) NDCG@10, with intervals that are
  post-selection and exploratory; the first-positive-review prior loses (−0.000145), and the
  simulated interview is reported as a hindsight-oracle sensitivity analysis that needs
  about 440 questions per recorded answer at its largest budget, replacing the 248 in the
  entry below. `reports/musical_shadow_exploratory.json` and `reports/signal_audit.json` are
  the rerun aggregates, and the manuscript describes the interview accordingly.

## 2026-09-23

### Added

- `reliability/shadow.py`: two exploratory profile arms measured on the frozen boundary — one
  reading evidence the platform already holds (ratings below four that were never ranked),
  one asking the person instead — with a per-person prior fitted only on earlier rows, and a
  donor-category arm for wording that arrives on another surface.
- `tools/signal_audit.py`, an aggregate bound on how many elicited answers are reachable, and
  `reports/shadow_profile_design.md` with the resulting reading: both already-held arms
  cleared the activation gate at zero user cost, while asking needed roughly 248 questions
  per usable answer. Reported as exploratory, without opening the test split.

### Removed

- The standalone research protocol directory; its protocol now lives in `reports/`.

## 2026-09-22

### Added

- `reliability/core.py` and `reliability/benchmark.py`: full-catalogue temporal
  leave-future-out replay on the Amazon Reviews 2023 5-core archives, with frozen boundaries,
  validation-selected scoring, a user-cluster paired bootstrap, and a challenger active only
  when its interval clears zero. Video Games and Musical Instruments replays, with the
  aggregate reports committed.
- `reliability/bundle.py`: a one-training-pass, hash-checked local model bundle.
- `reliability/server.py`: loopback-only HTTP service with the persisted validation gate,
  explicit fallback reasons, aggregate counters, and no access log.
- `reliability/neural.py` and `reliability/neural_calibrate.py`: a gated two-tower retrieval
  challenger with a validation-selected blend weight and its own gate, served as a shadow that
  does not displace a stronger challenger. `torch` loads lazily so the core stays
  dependency-free.
- `web/`: a dependency-free static preview for GitHub Pages with a static fallback bundle,
  and a workflow to publish it.
- `paper/`: the manuscript, its source and build instructions, with the published preprint
  linked.
- `tools/release_check.py` and CI, enforcing the published-source boundary: root-file
  allowlist, forbidden per-request JSON keys, and size caps on reports and the manuscript.

### Fixed

- Personalised neighbour scoring was silently dropped in the first replay because the tuple
  fields were reversed, so the hybrid behaved like popularity. Restored, and a
  training-to-ranking regression test now covers that path; the corrected Video Games numbers
  replaced the earlier ones.
