# Shadow profiles: what a recommender can infer, and what it has to ask for

Ahnaf An Nafee · 23 September 2026

## Question

The committed route personalises from the last twenty distinct reviewed items attached to a request. It does not filter those items by star rating: an item rated below four still contributes attraction, while the rating and review time are discarded. The rest of the supplied history sits outside the scoring window. This exploratory extension asks three bounded questions about the discarded remainder:

1. **What can be inferred** from records the platform already holds and the current pipeline throws away — a star rating below four on an item the user reviewed, an older history item, a missing user–item training rating, or the size of a history?
2. **What can be elicited** from a short interview, and how much recorded evidence a fixed question budget would reveal under a hindsight answer oracle?
3. **What can be derived** for a user who arrives with nothing to infer from, when the system must substitute a prior borrowed from comparable users?

A **shadow profile** here is any user representation the system assembles beyond the history deliberately attached to the request. The two acquisition modes are kept apart throughout because their costs are not comparable: already-held evidence requires no new prompt, while elicitation would, so a simulated elicitation result is always reported with its prompt budget. Nothing in this study measures whether a person notices this or finds it acceptable — that needs a user study, which is out of scope. The term is used here for a registered customer whose profile is assembled from records the platform already has; in the social-network literature it names something adjacent but different, [a profile built for someone with no account at all](https://doi.org/10.1126/sciadv.1701172).

## What the archives actually contain

The audit that bounds all three questions is in `tools/signal_audit.py`. Musical Instruments, all validation rows unless stated:

| Quantity | Musical Instruments | Video Games |
| --- | ---: | ---: |
| Train rows / positive rows | 427,957 / 367,558 | 736,827 / 592,599 |
| Train rows rated below four (discarded today) | 60,399 | 144,228 |
| Validation history slots | 416,816 | 316,774 |
| …of which the platform already holds a rating | 319,832 | 254,713 |
| …and that rating is below four | 47,459 | 53,348 |
| …without a user–item rating in training | 96,984 | 62,061 |
| …whose product is absent from the training catalog | 20,310 | 16,388 |
| Validation rows with an empty history | 1,573 | 1,416 |
| Distinct history items sitting outside the 20-item window every arm reads | 23% | 25% |
| …of the ratings below four the archive holds, share sitting outside that window | 25% | 30% |
| Training users shared with the other category | 2,802 | 2,802 |
| Test rows whose user is unseen in train | 15,408 of 42,595 | 17,736 of 43,249 |

Two facts set the shape of the study. Train-recorded low ratings are plentiful — 47,459 validation history slots carry one, including 35,444 inside the scoring window — so the repulsion arm has evidence to consume. And the newest users have none: within the eligible positive validation cohort, 1,288 requests have no history at all, 4,379 have one or two items, and 10,919 have three to five, out of 33,993. Below one catalog-known history item there is nothing to personalise from, which is exactly where the current route falls back to popularity.

One further check decided how the arms are written. For every one of the 14,175 validation users who had a prior training record, the history column attached to their first validation request reproduced that record *exactly* — same items, nothing missing, all 14,175 users. So the 20-item window that every arm in this repository reads is a truncation the scorer imposes, not a limit of what the archive holds: 23% of the distinct items a validation record lists, and 25% of the ratings below four it carries, sit outside it. Widening the window could help or hurt; no arm here tests it.

The adjacent-surface idea was measured rather than assumed. The two categories share 2,802 training users, which is the only bridge between them, and it turns out to be narrow: a donor record was reachable for 47 of the 33,993 eligible validation requests, so that arm is reported as a feasibility probe and not as a powered comparison.

## Arms

All arms rank the whole train-known catalog, exclude the request's history, use the same sparse-to-exact retrieval argument, are selected on validation only, and are gated against whichever route validation already chose — the co-review hybrid at α = 0.75 here, not the popularity baseline. A challenger wins only if the paired user-cluster bootstrap's lower 95% endpoint exceeds zero.

| Arm | Evidence it uses | User cost |
| --- | --- | --- |
| C1 repulsion | `positive − γ·negative` co-review attraction, negatives being history items the platform observed at rating ≤ 3 | none |
| C2 grading and decay | observed star weight on history items, exponential age decay by half-life | none |
| C3 analogous first positive review | distribution of first positive reviews of comparable training users, substituted where nothing personalisable exists | none |
| C4 history-conditional blend | blend weight chosen per history-size bucket, the bucket inferred from history length alone | none |
| C5 interview | declared likes and dislikes over the most often positively reviewed catalog items, at budgets 0 / 3 / 10 / 30 / 100 | questions asked |
| C6 adjacent surface | the same user identifier's activity in the other category archive, projected through co-likers | none, cross-surface |

## What the literature already knows

Four bodies of work carry this design, and each plays a different role.

**Inferred behaviour is real signal, and is biased in known ways.** [Hu, Koren and Volinsky](https://doi.org/10.1109/ICDM.2008.22) is the canonical statement that observed is not the same as liked and unobserved is not the same as disliked. [Steck](https://doi.org/10.1145/1835804.1835895) and [Marlin and Zemel](https://doi.org/10.1145/1639714.1639717) model the observation process itself rather than treating missingness as noise — which is what an inferred profile is. The passive-negative line is the closest empirical support for the repulsion arm: [SINE models skips of served content as a supervision signal](https://arxiv.org/abs/2308.04086), [a "not-to-recommend" loss with a responsiveness measure](https://arxiv.org/abs/2308.12256) shows how to score such a system on more than top-N accuracy, and [Yi et al.](https://doi.org/10.1145/2645710.2645724) demonstrate that positive-only implicit feedback misleads in app stores. [Schnabel et al.](https://arxiv.org/abs/1602.05352) and [Joachims et al.](https://doi.org/10.1145/3018661.3018699) supply the caution: exposure is caused by your own policy, so a behavioural feature partly measures the past recommender.

**Elicitation is the control arm, and it has an accounting system.** The [active-learning chapter of the handbook](https://doi.org/10.1007/978-1-4899-7637-6_24) and [Elahi, Ricci and Rubens](https://doi.org/10.1016/j.cosrev.2016.05.002) define which item to ask about and what a question is worth; [Rashid, Karypis and Riedl](https://doi.org/10.1145/1540276.1540302) quantify accuracy per collected rating for new users; [Anava et al.](https://doi.org/10.1145/2736277.2741109) spend a hard probe budget with optimal design. These studies motivate reporting the number of prompts next to a quality change, although our hindsight oracle does not reproduce a live elicitation session.

**Deriving a profile for someone who told you nothing is a named line of work.** The closest published precedent is [Golbandi, Koren and Lempel](https://doi.org/10.1145/1871437.1871734): cluster users by their trajectory through item space and hand a rating-less user the cluster profile, extended [adaptively with decision trees](https://doi.org/10.1145/1935826.1935910). [Zhou and Brunskill](https://arxiv.org/abs/1604.06743) and [Young and Leith](https://arxiv.org/abs/2305.18305) make the borrowed-class prior an online algorithm; [Deezer's semi-personalized system](https://arxiv.org/abs/2106.03819) is the deployed analogue, validated offline *and* online. [Gantner et al.](https://doi.org/10.1109/ICDM.2010.129) learn a mapping from observable attributes into the model's own feature space — the architectural template for any inferred profile. [Schein et al.](https://doi.org/10.1145/564376.564421) is the reason empty-history users get their own cohort rather than being averaged into one aggregate. Borrowing a profile from an adjacent surface is a different mechanism from borrowing a prior across users: it first has to establish that two records are the same person, which [record linkage](https://doi.org/10.1109/tkde.2011.127) and the [cross-network identity review](https://doi.org/10.1145/3068777.3068781) treat as noisy approximate matching rather than a key join.

**Two honesty checks the gate must survive.** [Ji et al.](https://arxiv.org/abs/2005.13829) show that a popularity baseline as usually implemented ignores which items were popular at the moment of interaction, and independently find that low-activity users follow the crowd more than heavy users do — which is both a threat to this comparison and independent support for the premise. [The popularity-bias survey](https://arxiv.org/abs/2308.01118) supplies the metrics to rule out that a "better profile" is just re-ranking toward hits. Borrowed cross-surface profiles come with [negative transfer](https://arxiv.org/abs/2211.11964) as a documented failure mode, and [Fernández-Tobías et al.](https://doi.org/10.1145/2959100.2959175) measure the accuracy/diversity trade of exactly that construction under positive-only feedback. The dataset itself is [Amazon Reviews 2023](https://arxiv.org/2403.03952), and full-catalog evaluation of implicit-feedback models follows [Steck](https://doi.org/10.1145/3298689.3347069).

Where this sits: the verified literature has inference papers that assume the behaviour stream already exists and elicitation papers that assume the interview happens. The present same-harness comparison reports the prompt budget alongside inferred signals, but the simulated interview uses a hindsight rating oracle; it cannot establish the cost or benefit of a real elicitation session. Similarly, the passive-negative work studies *skips of served items*; the reviewed sources do not directly test the case here, where someone chose to review a product and rated it poorly; that selection process differs from passive skips. The validation-only bootstrap gate is an implementation choice here, not a novelty claim; the nearest neighbours are [logged-policy off-policy evaluation](https://arxiv.org/abs/2008.07146) and the responsiveness framework above. Two citation notes kept from the survey: Gantner et al.'s published title is "Attribute-to-*Feature* Mappings", and the Rashid "Getting to know you" paper has conflicting IUI 2002 / 2008 venue trails, so this note cites only the [2008 SIGKDD Explorations](https://doi.org/10.1145/1540276.1540302) version.

## What the validation split shows

The current evaluator scores each arm on the Musical Instruments validation cohort. The simulated interview excludes the current target and every item in the supplied history, including items outside the scorer's twenty-item window. It uses later ratings solely as a hindsight answer oracle. All results remain exploratory because configuration selection and interval estimation use the same validation cohort.

The cohort is 33,993 eligible requests from 13,908 users, against 427,957 training rows over 54,483 users and 22,753 items. Validation selects lifetime popularity, recent popularity, or the co-review blend; it selects the hybrid at α = 0.75, which clears recent popularity by 0.0012538 NDCG@10 with a bootstrap interval of [0.0004220, 0.0020491]. Everything below is therefore measured against the hybrid, not against popularity.

| Route | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| Lifetime popularity | 0.008173 | 0.014709 |
| Recent popularity | 0.010715 | 0.021504 |
| Co-review hybrid at α = 0.75, active route | 0.011969 | 0.023152 |

### The inferred arms

| Arm | NDCG@10 | Recall@10 | Δ vs active route | Descriptive 95% interval | Same-cohort gate |
| --- | ---: | ---: | ---: | --- | --- |
| C1 repulsion from discarded ratings, γ = 1 | 0.012418 | 0.023799 | +0.000449 | [0.000266, 0.000626] | clears |
| C2 graded weights with 1095-day decay | 0.012435 | 0.023887 | +0.000466 | [0.000165, 0.000750] | clears |
| C3 first-positive-review prior | 0.011824 | 0.022652 | −0.000145 | [−0.000270, −0.000028] | loses |
| C4 blend weight by history length | 0.012523 | 0.024593 | +0.000554 | [0.000062, 0.001057] | clears, optimistically |

C1 requires no new prompt. It changes one term: a history item rated below four now contributes attraction at weight 1.0 *and* repulsion at weight γ. Setting γ = 0 reproduces the active route exactly, and on this data that cell returns 0.011969, the active route's own figure, which is the parity check passing on real records rather than only in a unit test. Raising γ moves NDCG@10 monotonically across the tested grid — 0.011969, 0.012227, 0.012345, 0.012418 — with the best tested value at its boundary; larger values were not evaluated. The audit counts 47,459 low-rated history slots across all validation rows, including 35,444 inside the scoring window. This audit denominator includes rows excluded from the positive-review evaluation cohort.

C2 reaches a slightly larger gain by a different route, weighting history stars at 1.0 / 0.6 and damping old evidence with a three-year half-life. Its configuration was chosen by the maximum over eight cells on the same cohort it is then scored on. C1's γ was also selected on this cohort, so neither interval accounts for configuration selection. The run publishes all eight C2 cells rather than only the winner, and the ablation is more informative than the headline:

| Age half-life | Flat rating weights | Graded rating weights |
| --- | ---: | ---: |
| none | 0.011969 | 0.012074 |
| 1,095 days | 0.012381 | **0.012435** |
| 365 days | 0.012323 | 0.012309 |
| 90 days | 0.012117 | 0.012075 |

Almost all of the measured gain is the decay, not the grading: switching on a three-year half-life with flat weights moves NDCG@10 from 0.011969 to 0.012381, while grading on top of it adds 0.000054. Among the tested half-lives, three years scores highest; a 90-day half-life leaves less of the gain (0.012117). This pattern suggests longer-lived signal in this validation period, but does not establish an optimal half-life for future traffic.

The two arms are not independent. C1 uses recorded ratings below four for repulsion. C2 instead grades and ages observed four- and five-star reviews; it leaves low-rated items at the committed attraction weight. Their interaction has not been measured, because no combination was fitted after results were open.

### The derived arm

C3 has a negative difference on this validation cohort, with an interval below zero. It fires on 1,618 of the 33,993 requests, the ones with no catalog-known history item, and changes nothing elsewhere, so its overall −0.000145 rescales to about −0.00305 on the slice it actually touches. Replacing popularity with the aggregate first-positive-review distribution of comparable training users ranks the eventual target *worse* than the current crowd does. That is consistent with the low-activity finding in [Ji et al.'s popularity-bias work](https://arxiv.org/abs/2005.13829): new and low-activity users follow the crowd, so the crowd worth copying is the current one. A derived profile needs a conditioning variable that actually conditions, and "users who reviewed this category" does not. Nor was there adjacent-surface evidence waiting to be borrowed instead: the bridge between the two archives covers a small fraction of users and an even smaller fraction of requests, which the next subsection measures.

### The cross-surface probe

C6 tries to borrow a profile rather than build one: for a request with nothing personalisable in its own history, it scores candidates by what the same person did on the adjacent archive, matching on user ID across the two categories. Its small reachable cohort makes this a feasibility probe rather than a powered quality comparison.

| | |
| --- | ---: |
| Requests where the donor surface is reachable | 47 of 33,993 (0.14%) |
| NDCG@10 with the borrowed profile | 0.011960 |
| Δ versus the active route | −0.000009 [−0.000028, 0.000000] |

The count above is the arm's own record inside `validation_decision.json`. The file's top-level `routing_stats` block is written by the two arms that score without a projection, so its `adjacent_route_requests: 0` describes those runs and not this one.

The design set 5% of requests as the line below which this arm could not be read as a comparison; the measured share is 0.14%, thirty-six times below it. The arithmetic behind that is visible in the audit: of Musical Instruments' 54,483 training users, 2,802 — about one in twenty — also appear in the donor archive, and the arm only fires on the 1,618 requests that have nothing personalisable at home, of which 47 belonged to those users. Two category logs simply do not contain enough of the same person. What few requests qualify give no sign of helping: the interval's upper bound is exactly zero. A platform that wanted this arm would need broader linked activity records. No effect size is claimed.

### The elicited arm, and what a question is worth

C5 was scored at both γ = 0 (positive oracle answers only) and γ = 1 (repulsion from low oracle answers and already-held low ratings). The latter setting changes both sources of negative evidence.

| Simulated prompts per request | Recorded answers per request | Prompts per recorded answer | Δ, positive answers only | Δ, repulsion enabled |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.030 | 98.5 | +0.000111 [−0.000023, +0.000267] | +0.000535 [+0.000303, +0.000789] |
| 10 | 0.054 | 184.9 | +0.000111 [−0.000045, +0.000292] | +0.000522 [+0.000288, +0.000789] |
| 30 | 0.099 | 304.3 | +0.000137 [−0.000040, +0.000323] | +0.000551 [+0.000295, +0.000825] |
| 100 | 0.227 | 439.9 | +0.000461 [+0.000224, +0.000700] | +0.000871 [+0.000537, +0.001172] |

At the deepest simulated budget, 100 prompts per request recover 0.227 recorded ratings, about one per 440 prompts. Most asked products have no recorded rating for that user; this does not show that the user would decline to answer in a live interview. The recovered ratings average 0.201 positive and 0.027 low ratings per request. The score difference between repulsion settings cannot be attributed to those low answers alone because the switch also activates the low ratings already held in training.

C1's +0.000449, with no new prompts, is close to the +0.000461 from the 100-prompt cell that uses only positive oracle answers. With repulsion enabled, three prompts reach +0.000535 over the active route, but that includes C1's existing repulsion signal. At 100 prompts the difference is +0.000871 over the active route and +0.000422 over C1. The oracle draws ratings from all three archive splits, excludes the current target and all prior history items, and may read a review written after the simulated request. These contrasts are therefore sensitivity analyses about available labels, not evidence that asking real users produces the same gain.

### The experience curve

C4's value is less its aggregate +0.000554, which is selection-optimistic, than the bucket table underneath it.

| History length | Requests | Active route NDCG@10 | Blend selected | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,288 | 0.011332 | 0.0 | 0.011332 |
| 1–2 | 4,379 | 0.014825 | 0.75 | 0.014825 |
| 3–5 | 10,919 | 0.013806 | 0.75 | 0.013806 |
| 6–19 | 13,468 | 0.011128 | 0.5 | 0.011990 |
| 20+ | 3,939 | 0.006784 | 0.5 | 0.008615 |

Two things fall out. Accuracy is *not* monotone in experience: it peaks at one or two recorded items and then declines steadily, so the hardest requests in this archive come from the users who told the platform the most, not the least. And where the arm differs from the active route, it differs by *reducing* the personalised weight — 0.5 instead of 0.75 for anyone past five items, which lifts the 20+ bucket by a quarter in relative terms, and 0.0 for the empty history, confirming that with nothing personalisable the blend contributes nothing but noise. This cohort suggests that the committed blend gives too much weight to long histories. The bucket weights were chosen on these same records, so this routing pattern needs a separate evaluation.

### Cost of computing the profile

Each figure below is the local wall-clock time of scoring one validation request through that arm's own grid, single process, no concurrency. Grid sizes differ between arms, so a number compares an arm with its own grid rather than with another arm. The committed route's own selection step was timed in the same run, on the same cohort, and came to 3.21 ms at p95 (3.67 ms in the earlier run); C1's 3.41 ms includes its own grid search and is not an incremental serving cost.

| Step | One pass over the validation cohort (p95 ms per request) |
| --- | ---: |
| Committed hybrid, blend-weight selection | 3.21 |
| C3 first-positive-review prior | 1.73 |
| C2 graded weights with 1095-day decay | 2.15 |
| C6 cross-surface projection | 2.17 |
| C4 blend weight by history length | 3.23 |
| C1 repulsion from discarded ratings | 3.41 |
| C5 interview at the 100-question budget | 10.56 |

This is a validation-time artefact of the harness, not a serving estimate. Every row is the cost of *choosing* profile parameters per request by re-scoring a grid, which is what a search does and what a deployed system would not: a production profile is computed once per user and read per request, and nothing here measures that shape.


## What this does not establish

- Both test periods in this repository were opened by earlier studies, so every number in this note is exploratory. A confirmation needs an untouched category and a fixed arm chosen before its test period is opened.
- Every arm reads at most the last twenty distinct history items, a cap inherited from the committed route. The audit above shows the archive holds more than that per request, but the effect of widening the window is untested.
- The interview is simulated with hindsight. Answers come from the user's recorded reviews across train, validation, and test; the current target is never asked. A user may not remember or answer this way in a live interview, and an unrated item is treated as a non-answer even though the user might have an opinion. The simulated answer rate is therefore not an estimate of real response behavior.
- C4 selects its per-bucket blend weight on the same cohort it is scored on. That is selection optimism, declared rather than corrected.
- A review is not an exposure or a purchase. The repulsion arm uses low ratings only from items users chose to review, so it inherits selection bias that this benchmark cannot correct.
- The popularity control is the committed one. Its popularity terms are fixed at split time; a moment-of-interaction control would change the comparison for every challenger.
- No claim about users. Nothing here measures whether an inferred profile is welcome, noticeable or fair; that requires people, and this study has none. The C5 question counts are simulated prompt budgets, not observed effort or experience. If someone wants the missing half, the instruments already exist: [how bothersome people find information collection](https://doi.org/10.1080/00913367.2002.10673665) as the denominator, [transparency and control over a profile](https://doi.org/10.1007/s11257-011-9118-4) and [disclosure by inference](https://doi.org/10.1007/978-3-030-42504-3_16) as the mechanisms, measured on [inferred interest models specifically](https://doi.org/10.1145/2559206.2581141).
- The cross-surface arm assumes two archives refer to the same person because a string matches. Identity resolution is treated as solved there, which the [record-linkage](https://doi.org/10.1109/tkde.2011.127) and [re-identification](https://doi.org/10.1126/science.1256297) literature would not permit in a real deployment, where the join itself carries an error rate that sets a ceiling on the borrowed profile.

## Reproduce

```powershell
python -m unittest discover -s tests
python -m tools.signal_audit --data data --output runs/signal_audit.json --categories Musical_Instruments Video_Games
python -m reliability.shadow --data data --output runs/musical-shadow-v2 --category Musical_Instruments --donor-category Video_Games
```

The last command writes `validation_decision.json` and `aggregate.json` into a directory it must create; omit `--donor-category` to skip the cross-surface probe, and add `--include-test` to report the selected configuration on the already-exposed test period. Unit checks pin the two things most likely to fail silently: that γ = 0 reproduces the committed hybrid exactly, and that sparse candidate retrieval with negative evidence matches a full-catalogue scan.
