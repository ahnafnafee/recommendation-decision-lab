# When a simple recommender earns a challenger

Ahnaf An Nafee · 22 September 2026

## Question and result

The companion MovieLens study showed that a learned ranker could improve when its training candidates resembled the retrieval task, yet still trail a transparent popularity baseline. This separate experiment asks whether a small, train-only personalization layer can outperform a validation-selected popularity control on product-review activity. It is an engineering contribution and a bounded transfer check; cosine co-review scoring and blending are established techniques, not a new algorithmic principle.

The corrected Video Games replay was exploratory after an implementation defect made its first test run invalid. Its challenger scored 0.010172 NDCG@10 versus 0.007151 for recent popularity on 35,562 eligible test requests. The paired user-cluster difference was +0.003020 [0.002216, 0.003881]. The method and code were then fixed before acquiring Musical Instruments as an untouched category. There, the validation-selected recent-popularity baseline scored 0.008007 and the 0.75-weight hybrid scored 0.008681 on 35,905 eligible test requests. The paired difference was +0.000674 [0.000040, 0.001338]. The validation gate opened in both corrected runs. The confirmation interval's lower endpoint is close to zero, so the result is modest and should not be described as a robust product lift.

| Category | Evidence status | Test requests | Train-known items | Baseline NDCG@10 | Hybrid NDCG@10 | Paired difference [95% interval] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Video Games | Exploratory after defect | 35,562 | 22,976 | 0.007151 | 0.010172 | +0.003020 [0.002216, 0.003881] |
| Musical Instruments | Fresh category, fixed method | 35,905 | 22,753 | 0.008007 | 0.008681 | +0.000674 [0.000040, 0.001338] |

The absolute scores matter. On Musical Instruments, 12,305 of the 35,905 eligible positive test reviews targeted items absent from the train-known catalog, so those requests could not be recovered by either method. Video Games had 22,051 such targets among 35,562 eligible requests. This is a catalog-coverage constraint, not a ranking failure that a different ordering alone can solve.

## Design

The [McAuley Lab Amazon Reviews'23 5-core benchmark](https://amazon-reviews-2023.github.io/data_processing/5core.html) provides global train/validation/test time cutoffs and prior-item histories. Train-only positive reviews (ratings at least four) supply lifetime counts, 365-day recent counts, and a top-50 cosine co-review neighborhood per item. Every evaluated positive review is a request to rank the full train-known catalog, excluding the user's prior reviewed items; an unknown target remains a miss. A target already in that history is excluded and counted. The primary metric is per-request NDCG@10; Recall@10 is secondary. The baseline is the better of lifetime and recent popularity on validation. The challenger blends that baseline with co-review scores using a validation-selected weight from {0.25, 0.5, 0.75, 1.0}. A user-cluster bootstrap on validation controls whether the challenger is active: its lower 95% percentile endpoint must exceed zero. The selected choice is persisted before the test rows are opened. The reported test interval uses 1,000 user-cluster draws with seed 20260922. These definitions are implemented in `reliability/benchmark.py` and `reliability/core.py`.

The category test uses the same platform and dataset construction. It tests a change of product category, not transfer to another site or to live traffic. The owner-generated 5-core subset was filtered using the full corpus; item and user support therefore carry retrospective selection. A review is not an exposure log, a purchase target, or an unbiased preference label. The paired intervals resample users within one fixed time period, leaving shared-item dependence, model refitting, selection optimism and future regimes outside their coverage. No online effect follows from these scores.

## Implementation failure and safeguard

The first Video Games run returned the same ranking for all hybrid weights because the trained neighborhood serializer emitted `(similarity, item)` while retrieval expected `(item, similarity)`. The training-to-ranking regression test now requires a learned neighbor to change a ranking. The defective aggregate stays in ignored local runs and is not part of the public evidence. The corrected Video Games replay is exploratory; Musical Instruments is the only fresh-category check for this fixed method. The evaluation code and aggregate reports document the corrected result.

## System demonstration

The accompanying loopback HTTP service uses the same ranking function and a locally built, hash-checked model bundle. It exposes active, baseline and shadow rankings, route and fallback reasons, health, and aggregate request counters. A synthetic browser example uses invented titles and events; the owner data and fitted review-derived weights never ship with the repository. This makes the failure analysis inspectable while keeping the offline result separate from a production deployment claim. The measured local ranking p95 was 0.611 ms on corrected Video Games and 0.760 ms on Musical Instruments, for in-process baseline-plus-challenger scoring only. Those timings exclude HTTP, initialization, disk loading, network, concurrency and sustained load.

The reusable lesson is practical: a small personalized layer can earn a route on validation and show a positive but narrow gain in a second category, while a simple fallback remains valuable. Neither the method nor the evidence yet supports a distinguished-conference novelty claim. The next scientific step would test raw, unfiltered traffic-like data or an exposure-logged dataset, plus independent temporal periods, without changing this held-out result after the fact.
