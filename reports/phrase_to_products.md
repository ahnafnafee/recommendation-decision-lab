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
catalogue-wide fusion is the wrong instrument for it: at 400 requests it *lost*
0.0034 NDCG@10 against the same base. Production systems put the lexical and dense
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
  `target_title`, `random_title`, `published` (a real searcher's phrase for that
  product). `own_words`, `own_words_scrubbed`, `target_title` and `random_title` are
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

Musical Instruments, validation split, 400 requests, lexical index over 22,753
product listings. Absolute NDCG@10 is around 0.008, so the deltas below are of the
same order as the metric itself.

| configuration | chosen weights | own-words delta | 95% lower | gate |
| --- | --- | --- | --- | --- |
| `behavioural` (primary) | weight 0.75, repulsion 0.5, price 0.5 | −0.00240 | −0.00878 | shut |
| `catalogue` (naive comparison) | weight 0.25, repulsion 0, price 0.5 | −0.00337 | −0.00963 | shut |

Two ceilings explain the sign, and they multiply:

| diagnostic | value | reading |
| --- | --- | --- |
| behavioural shortlist coverage | 86 / 400 = 21.5% | in 78.5% of requests the frozen behavioural route never considered the answer, so a confined phrase cannot promote it |
| unconstrained phrase reach, `own_words` | 5 / 355 = 1.41% | the person's earlier words named the eventual answer's product almost never |
| `cross_words` reach | 0.28% | somebody else's words are worse, as expected |
| `target_title` reach | 177 / 177 = 100% | the matcher is not the weak link: the exact words of the product always find it |
| `random_title` reach | 1 / 400 = 0.25% | and that is not an accident of a permissive index |
| `published` reach | 2 / 5 = 40% | a real searcher's phrase for the product reaches far more often than the person's own prose |

The gap between `own_words` at 1.41% and `published` at 40% is the interesting
number: retrieval is not the bottleneck, expression is. A person reviewing a
keyboard six months ago wrote about their hands, not about the model they would
later be shown. Under the frozen protocol that is reported as what language is
worth on this data, not fixed by tuning the phrase weight until the sign flips.

## Data and artifacts

Product text is fetched locally into the Git-ignored `data/text/` directory from the
 publishers' snapshot
([McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023))
or the origin mirror, with `tools/text_corpus.py`: 22,753 Musical Instruments
listings, 55,981 people with their own review prose, corpus manifest schema 3.
Published query pairs are scarce — `blair-bench` (ESCI) plus `Amazon-C4` give 157
in-catalogue Musical Instruments pairs and 271 for Video Games — which is why
`published` is a diagnostic and not the headline condition.
`tools/build_text_index.py` encodes each listing with `all-MiniLM-L6-v2` (384
dimensions, 2 reviews per product so the document fits the model's 256-token
window). Everything heavier than the standard library sits behind an optional
extra: `pip install -e ".[language]"`.

## Limits

- No public dataset links free-form wording to these ASINs at scale, so the primary
  condition uses a person's earlier review prose as a proxy for a spoken request.
  It is the closest available substitute, not the same thing.
- NDCG@10 with a single target scores one product. A sentence that legitimately
  matches five products is credited for one.
- The frozen splits are not reordered or re-filtered. Published splits leak up to
  150% and non-monotonically ([arXiv:2602.13626](https://arxiv.org/abs/2602.13626)),
  and phrasing pipelines tolerate corruption up to about 30%
  ([arXiv:2604.02527](https://arxiv.org/abs/2604.02527)), so any change here would
  need its own protocol rather than a tweak.
- Report the paired delta, never the phrase lane's absolute score: four of the six
  conditions are constructed with the answer in hand.
