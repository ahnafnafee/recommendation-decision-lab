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
−0.000577), so the gate stays closed on both routes. Among conditions worded
like something a person would type, only the published search phrase beats the
base on the test split: +0.002394 by the lexical route and +0.002132 by the
encoded route on Musical Instruments (intervals 0.001892 to 0.002875 and
0.001669 to 0.002622, the same 681 requests) and +0.003115 on Video Games
(0.002714 to 0.003545, 824 requests).

Two ceilings explain the sign, and they multiply (full reach table in the
manuscript):

| diagnostic | Musical Instruments, test | Video Games, test | reading |
| --- | --- | --- | --- |
| behavioural shortlist coverage | 11,925 / 35,905 = 33.21% | 8,267 / 35,562 = 23.25% | in most requests the frozen behavioural route never considered the answer, so a confined phrase cannot promote it |
| unconstrained phrase reach, `own_words` | 627 / 25,926 = 2.42% | 390 / 22,320 = 1.75% | the person's earlier words named the eventual answer's product almost never |
| `cross_words` reach | 0.74% | 0.56% | somebody else's words are worse, as expected |
| `target_title` reach | 99.98% | 99.97% | the matcher is not the weak link: the exact words of the product always find it |
| `random_title` reach | 0.73% | 0.42% | and that is not an accident of a permissive index |
| `published` reach | 47.72% | 33.50% | a real searcher's phrase for the product reaches far more often than the person's own prose |

The gap between `own_words` at 2.42% and `published` at 47.72% is the
interesting number: retrieval is not the bottleneck, expression is. A person
reviewing a keyboard six months ago wrote about their hands, not about the
model they would later be shown. Under the frozen protocol that is reported as
what language is worth on this data, not fixed by tuning the phrase weight
until the sign flips.

In-process timing on the scored requests (one phrase and one unravelling
request; no HTTP, model load, or disk): 38.3 ms median and 48.1 ms at the 95th
percentile for Musical Instruments, 21.2 ms and 29.3 ms for Video Games, and
26.9 ms and 62.2 ms for the encoded route on Musical Instruments.

## Data and artifacts

Product text is fetched locally into the Git-ignored `data/text/` directory from the
 publishers' snapshot
([McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023))
or the origin mirror, with `tools/text_corpus.py`: 22,753 Musical Instruments
listings (55,981 people with their own review prose) and 22,976 Video Games
listings (92,807 people), corpus manifest schema 3.
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
