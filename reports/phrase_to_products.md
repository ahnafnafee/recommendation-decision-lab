# Answering a sentence: phrase-to-products design and evaluation

Design note for the wording-driven route. Evaluation definitions live here and in
the implementation, in the same place as the earlier research notes. The
literature trail is in [phrase_to_products_survey.md](phrase_to_products_survey.md).

## What ships

`POST /api/utterance` takes `{phrase, history, k, remember}` and answers a sentence
in the person's own words against locally held product text. The response separates
what the wording did from what the behavioural route did:

- `said`, `heard`, `ruled_out`, `budget` — the sentence back, the terms that were
  actually used, what was negated away, and a price ceiling if one was stated.
- `spoken` — products the wording selected, with real product names.
- `active` — the validated behavioural answer, unchanged from `/api/recommend`.
- `combined`, `answering`, `phrase_fallback` — which route answered, and if the
  wording could not, why.

`GET /api/ambient` answers without being asked again. The request carries no
wording at all: it returns what a sentence held from earlier in the same session
selects, which is the standing-utterance behaviour the feature is meant to
demonstrate. Up to `HELD_SENTENCES = 8` sentences are kept in process memory, under
the same rule as everything else this service submits — nothing written to a log.
No microphone is involved; the sentence is what was typed.

Degradation is explicit. Without product text the phrase lane reports
`phrase_fallback: "text_corpus_unavailable"` and the behavioural route keeps
answering, so the demo never fails because an optional local file is absent.

## Two configurations, because two questions are being asked

|  | serving (`reliability/server.py`) | measured arm (`reliability/phrasing.py`) |
| --- | --- | --- |
| question | what did this person just ask for? | is ambient wording worth more than a feed? |
| phrase scope | `catalogue`: the wording may reach any product | `behavioural`: the head of the fallback order plus co-review neighbours |
| phrase weight | 1.0, phrase-first | chosen on validation from a fixed grid |
| stated price | enforced: products whose recorded price is above the figure are dropped, products with no recorded price stay, and if nothing is affordable the figure is dropped rather than the answer | scored: `price_penalty` lowers the rank of an over-budget product without removing it |
| reported | product names and the words that matched | paired NDCG@10 delta against the frozen base |

Someone who states a figure while typing expects it to be respected, so the
interactive route treats it as a limit on what may be returned
(`TextRanker.enforce_ceiling = True`). The measured arm treats the same figure as one
more term in the score, because a request answered with a pricier product is still a
correct answer under the frozen metric and removing it would change what is being
compared. The split is deliberate; `tests/test_language.py` fixes both behaviours.

Someone who types a request expects to be answered, so the interactive route lets
the sentence pick from the whole shelf. The measured question is different, and a
catalogue-wide fusion is the wrong instrument for it: in the 400-request
validation pilot it *lost* 0.0034 NDCG@10 against the same base, and the
full-scale measurement confines the phrase to the behavioural shortlist for
exactly that reason. Production systems put the lexical and dense
route inside candidate generation rather than in front of the ranking
([LIGER, arXiv:2411.18814](https://arxiv.org/abs/2411.18814);
[Facebook EBR, arXiv:2006.11632](https://arxiv.org/abs/2006.11632)), so the frozen
primary configuration confines the phrase to what the behavioural route was already
considering (`TextRanker.scope = "behavioural"`, `scope_depth = 2000`). Both
configurations are run and reported separately; neither is presented as the other.

## Scoring discipline

- Requests come from the frozen temporal splits. `timestamped_requests` recomputes
  the frozen filter locally and raises if it drifts from
  `reliability.benchmark.evaluation_data` row for row.
- Six phrase conditions. `own_words` (a person's earlier review prose),
  `own_words_scrubbed`, `cross_words` (somebody else's prose, register matched),
  `target_title`, `random_title`, `published` (an externally sourced pair for the
  known target: an observed ESCI search query or a review-derived Amazon-C4
  rewrite). `own_words`, `own_words_scrubbed`, `target_title` and `random_title` are
  answer-informed: they are reported as **paired deltas only**, never as absolutes.
  `cross_words` is the generalisation control.
- The control bar is free behavioural features, not random. A random profile scores
  0.0006 NDCG@20 against 0.0039 for most-popular in
  [LangPTune, arXiv:2410.18870](https://arxiv.org/abs/2410.18870), and in the same
  paper a variant that skips the written profile and trains the embedding model
  directly on the item-metadata list reaches 0.0543, above the language pipeline's
  0.0494 — beating nothing is easy.
- Answer-side leakage is measured, not assumed. `identifying()` keeps title tokens
  whose document frequency is under 1% of the catalogue, the threshold motivated by
  [Z3 concept intrusion, arXiv:2608.25245](https://arxiv.org/abs/2608.25245) (7.40%
  of non-generic concepts intrude across 97 of 100 topics; over-generate and take
  the minimum-Z3 list falls to 0.06%). `own_words_scrubbed` is the scrubbed variant.
- `P(answer was reachable)` is reported as its own diagnostic.
  [Cold-start retrieval ceilings, arXiv:2606.29947](https://arxiv.org/abs/2606.29947)
  put single retrievers at 4.6–22.9%, so a low reach is expected ground truth rather
  than a bug to hide.
- Synthetic-wording results are paired deltas plus rank-agreement across scorer
  configurations, because a synthetic-phrase lift demonstrates phrase handling and
  not recommendation quality ([arXiv:2209.04973](https://arxiv.org/abs/2209.04973)).
- Activation is the same gate as every other challenger here: paired user-cluster
  bootstrap, 1000 draws, seed 20260922, and the route activates only if the 95%
  lower percentile is above zero.

## What has been measured

Full test splits, read after the configuration was fixed on validation:
35,905 Musical Instruments requests and 35,562 Video Games requests, scored
with the lexical route in both categories and with the encoded route on
Musical Instruments, selected confined configuration (phrase weight 0.25,
repulsion 0, price penalty 0.5) chosen by the fixed rule from five scored
configurations on the validation cohort (33,993 Musical Instruments requests;
26,706 for Video Games; the encoded route selected the same configuration on
its own validation pass) and recorded in `validation_decision.json` before
the test split was read.
Base NDCG@10 is 0.008681 (Musical Instruments) and 0.010172 (Video Games), so
the deltas below are of the same order as the metric itself.

| category, split | own-words delta | 95% interval | gate |
| --- | --- | --- | --- |
| Musical Instruments, test, lexical | −0.001678 | −0.002318 to −0.000980 | shut |
| Musical Instruments, test, encoded | −0.001282 | −0.001944 to −0.000577 | shut |
| Video Games, test, lexical | −0.002858 | −0.003598 to −0.002195 | shut |

The encoded variant (all-MiniLM-L6-v2 over the same shortlist, Musical
Instruments only, where the embedding artifact exists) is directionally
positive on the validation cohort (+0.001048, interval −0.000062 to
+0.002109) but loses on the full test split (−0.001282, interval −0.001944 to
−0.000577), so the gate stays closed on both routes. Among phrase-like
conditions, only the mixed external-pair condition has a positive full-split paired
difference: +0.004931 by the lexical route and +0.004232 by the encoded route
on Musical Instruments (intervals 0.004261 to 0.005611 and 0.003578 to
0.004884), and +0.006167 on Video Games (0.005505 to 0.006885). These
differences average over all test requests, although an external pair is
available for only 1,983 Musical Instruments requests and 2,115 Video Games
requests. The selected pairs do not represent live query traffic. The lexical
contribution decomposes as follows; each entry uses the entire category test
split as its denominator, so the source contributions add to the mixed delta.

| category | ESCI observed-query contribution | Amazon-C4 review-rewrite contribution | mixed external-pair delta |
| --- | ---: | ---: | ---: |
| Musical Instruments | +0.002500 (1,512 requests, 553 targets) | +0.002431 (471 requests, 41 targets) | +0.004931 |
| Video Games | +0.003883 (1,568 requests, 615 targets) | +0.002284 (547 requests, 47 targets) | +0.006167 |

The C4 pair counts in the catalog are 53 and 65; the ESCI counts are 1,913 and
3,886, so the catalog carries 1,966 in-catalogue Musical Instruments pairs and
3,951 Video Games pairs. The request counts are larger because the same target-linked phrase is used
for multiple later review requests. The [source audit](musical_phrase_sources.json)
and [Video Games counterpart](video_games_phrase_sources.json) are reproduced by
`python -m tools.phrase_source_audit --category Musical_Instruments` and the same
command for `Video_Games`. This is a descriptive decomposition, not a separate
confirmatory experiment. At full scale both sources contribute positively: the
official ESCI queries are the larger contributor in both categories (slightly
so on Musical Instruments, clearly so on Video Games), while the rewrites keep
the larger per-request effect (mean +0.1853 versus +0.0594 per matched request
on Musical Instruments, +0.1485 versus +0.0881 on Video Games); the small
observed-query slice that read the ESCI contribution as negative was a sample
artifact.

The audit also separates the ESCI pairs by their recorded judgement and, separately,
asks whether the route reached a product judged an acceptable substitute for the
staged query. Exact-match pairs contribute +0.002107 on 1,242 Musical Instruments
requests and +0.003537 on 1,209 Video Games requests; acceptable-substitute pairs
contribute +0.000393 on 270 and +0.000346 on 359 requests, and they almost never
worsen the request they attach to (0 of 270 on Musical Instruments, 6 of 359 on
Video Games). Counting a reached substitute as a hit lifts the strict mixed-set
reach from 61.57% to 65.25% on Musical Instruments (73 requests) and from 53.90%
to 58.49% on Video Games (97 requests). The paired intervals above still score
the exact target only; the substitute credit is a descriptive reading of the
many-valid-answers gap, not a second gate.

Two separate constraints limit the confined arm (full reach table in the
manuscript):

| diagnostic | Musical Instruments, test | Video Games, test | reading |
| --- | --- | --- | --- |
| behavioural shortlist coverage | 11,925 / 35,905 = 33.21% | 8,267 / 35,562 = 23.25% | in most requests the frozen behavioural route never considered the answer, so a confined phrase cannot promote it |
| unconstrained phrase reach, `own_words` | 627 / 25,926 = 2.42% | 390 / 22,320 = 1.75% | the person's earlier words named the eventual answer's product almost never |
| `cross_words` reach | 0.74% | 0.56% | somebody else's words are worse, as expected |
| `target_title` reach | 99.98% | 99.97% | target-informed exact wording confirms the index can recover a known title, but does not test natural-query retrieval |
| `random_title` reach | 0.73% | 0.42% | and that is not an accident of a permissive index |
| mixed external-pair reach | 61.57% | 53.90% | target-linked ESCI queries and Amazon-C4 rewrites reach more often than earlier review prose; this is not a live-request comparison |

The gap between `own_words` at 2.42% and the mixed external set at 61.57%
indicates that earlier review prose is a poor proxy for target-linked query
wording, but does not isolate the effect of real search intent. The
33.21% behavioural shortlist coverage is another constraint: even a useful
phrase cannot promote a target outside that shortlist. These are marginal
rates with different denominators, so their product is not a measured joint
ceiling. Exact-title reach checks the index on target-informed wording; the
mixed-set reach, still far short of the exact-title probe, shows that
retrieval can still fail even when words are linked to the target. Under the frozen protocol, the negative own-words result is
reported rather than tuned away, and neither diagnostic establishes the
quality of live typed requests.

In-process timing on the scored requests (one phrase and one unravelling
request; no HTTP, model load, or disk): 23.1 ms median and 30.3 ms at the 95th
percentile for Musical Instruments, 22.1 ms and 36.2 ms for Video Games, and
27.6 ms and 42.3 ms for the encoded route on Musical Instruments.

## Data and artifacts

Product text is fetched locally into the Git-ignored `data/text/` directory from the
 publishers' snapshot
([McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023))
or the origin mirror, with `tools/text_corpus.py`: 22,753 Musical Instruments
listings (55,981 people with their own review prose) and 22,976 Video Games
listings (92,807 people), corpus manifest schema 3.
External query pairs come from the full official
[ESCI Shopping Queries Dataset](https://github.com/amazon-science/esci-data)
release (US locale, judged an exact match or an acceptable substitute for the
item, both splits) plus `Amazon-C4`, giving 1,966 in-catalogue Musical
Instruments pairs and 3,951 for Video Games. ESCI contains shopping queries
people typed that were judged against catalogue items, while Amazon-C4 contains
semi-synthetic rewrites of reviews written about their target products. The two
sources are measured separately and must not be presented together as real
searcher wording; this is why `published` is a diagnostic and not the headline
condition.
`tools/build_text_index.py` encodes each listing with `all-MiniLM-L6-v2` (384
dimensions, 2 reviews per product so the document fits the model's 256-token
window). Everything heavier than the standard library sits behind an optional
extra: `pip install -e ".[language]"`.

## Limits

- No condition carries a live typed request. Worded-by-somebody-else phrasing is
  available at scale — the full official ESCI release links typed queries to
  catalogue items — but it arrives target-linked, so `published` is read as a
  diagnostic. The primary condition therefore uses a person's earlier review
  prose as a proxy for a spoken request: the closest available
  non-answer-informed substitute, not the same thing.
- NDCG@10 with a single target scores one product. A sentence that legitimately
  matches five products is credited for one.
- The frozen splits are not reordered or re-filtered. Published splits leak up to
  150% and non-monotonically ([arXiv:2602.13626](https://arxiv.org/abs/2602.13626)),
  and phrasing pipelines tolerate corruption up to about 30%
  ([arXiv:2604.02527](https://arxiv.org/abs/2604.02527)), so any change here would
  need its own protocol rather than a tweak.
- Report the paired delta, never the phrase lane's absolute score: four of the six
  conditions are constructed with the answer in hand.
