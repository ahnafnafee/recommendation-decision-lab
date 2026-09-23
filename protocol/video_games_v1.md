# Video-game recommendation transfer study: protocol v1

Frozen design, 22 September 2026. The follow-up MovieLens outcomes motivated this separate-domain test. This protocol is written before opening the Amazon Reviews'23 Video_Games rows. This is a local commitment, not an external preregistration.

## Data and boundary

Use the McAuley Lab Amazon Reviews'23 Video_Games 5-core `timestamp_w_his` train, valid and test CSV archives at the published owner URLs. The owner defines global cutoffs t1=1628643414042 and t2=1658002729837 milliseconds. Train only on train rows. Treat valid and test histories as the user's prior interactions at the row timestamp; global item statistics and co-occurrence remain frozen at t1. The 5-core subset was filtered using the full corpus, so user/item support is retrospectively selected. This is an offline review-activity benchmark, not a claim about unfiltered traffic. Keep all rows, model state and user-level outputs local. Publish only code and aggregates.

## Task

For each valid/test review with rating >=4, rank the entire train-known item catalog excluding every item in its supplied prior history. If the target item was already seen, exclude the row and count it. If the target is absent from the train catalog, retain a zero outcome and count it. A positive review is a proxy target, not proof of preference or a user exposure opportunity. Report eligible request and distinct-user counts, new-item misses, excluded repeats, and ratings below threshold. The primary metric is macro average per-request NDCG@10 with one target (1/log2(rank+1) if rank <=10, else 0). Recall@10 is secondary.

## Methods

Use training positives (rating >=4) for item counts and co-occurrence. Lifetime popularity ranks by positive count. Recent popularity counts positives in the final 365 days before t1. Both are normalized with log1p count divided by log1p maximum count; deterministic item-ID ties. The personalized score sums cosine-normalized co-occurrence with the user's last 20 distinct positive historical items; pair construction uses each training user's last 20 distinct positives and retains the top 50 neighbors per item. Histories at serving time include earlier reviewed items even when their rating is unavailable, so personalized evidence means co-review, not inferred positive sentiment. Do not use target-row labels in any feature.

The validation-selected baseline is the better of lifetime and recent popularity by NDCG@10 (tie: lifetime). For each alpha in {0.25, 0.5, 0.75, 1.0}, compute (1-alpha)*baseline normalized score + alpha*personalized score, with popularity resolving score ties. Select the best alpha on validation (tie: smaller alpha). A conservative gate activates the challenger only when the user-cluster bootstrap lower 95% percentile endpoint for challenger-minus-baseline validation NDCG is above zero. Otherwise, deploy the baseline. The gate does not assert a post-selection coverage guarantee.

## Test and uncertainty

Evaluate the fixed baseline, fixed challenger and validation-gated route on the untouched test split once. Report the paired request NDCG delta and a 95% user-cluster bootstrap percentile interval using 1,000 draws and seed 20260922. This interval conditions on the fixed temporal periods and model state; it does not establish causal engagement effects. Do not tune anything after test access and call the same split untouched again. Report all candidate methods on validation, only the fixed three on test. If the challenger loses or its interval includes zero, retain that result. No conference or novelty claim follows automatically from a lift.

## System evidence

Provide an independently runnable local HTTP demo on invented item names and interactions, using the same ranking and fallback functions. Show baseline, shadow challenger, active route, reasons, request counts and local request-time percentiles. No real review rows or model weights ship. Test endpoint behavior, parity, fallback, unknown/seen items, and release exclusions. Local timings exclude a network deployment and real traffic.
