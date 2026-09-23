# Shadow profiles: what a recommender can infer, and what it has to ask for

Ahnaf An Nafee · 23 September 2026

## Question

The committed route in this repository personalises from one thing: the positive-review history attached to the current request, read as at most the last twenty items with a rating of at least four. Everything else the platform holds about that person is discarded before ranking. This exploratory extension asks three bounded questions about the discarded remainder:

1. **What can be inferred** from records the platform already holds and the current pipeline throws away — a star rating below four on an item the user actually bought, an item in the history that is absent from the train-known catalog, the mere size of a history?
2. **What can be elicited** from a short interview, and what does each unit of quality cost the user in questions asked?
3. **What can be derived** for a user who arrives with nothing to infer from, when the system must substitute a prior borrowed from comparable users?

A **shadow profile** here is any user representation the system assembles beyond the history deliberately attached to the request. The two acquisition modes are kept apart throughout because their costs are not comparable: inferred evidence is free to the user and elicited evidence is not, so an elicited result is never reported without the number of questions that produced it. Nothing in this study measures awareness, consent or acceptability — those need a user study, which is out of scope. The term is used here for a registered customer whose profile is assembled from records the platform already has; in the social-network literature it names something adjacent but different, [a profile built for someone with no account at all](https://doi.org/10.1126/sciadv.1701172).

## What the archives actually contain

The audit that bounds all three questions is in `tools/signal_audit.py`. Musical Instruments, all validation rows unless stated:

| Quantity | Musical Instruments | Video Games |
| --- | ---: | ---: |
| Train rows / positive rows | 427,957 / 367,558 | 736,827 / 592,599 |
| Train rows rated below four (discarded today) | 60,399 | 144,228 |
| Validation history slots | 416,816 | 316,774 |
| …of which the platform already holds a rating | 319,832 | 254,713 |
| …and that rating is below four | 47,459 | 53,348 |
| …not present in the train-known catalog at all | 96,984 | 62,061 |
| Validation rows with an empty history | 1,573 | 1,416 |
| Distinct history items sitting outside the 20-item window every arm reads | 23% | 25% |
| …of the ratings below four the archive holds, share sitting outside that window | 25% | 30% |
| Training users shared with the other category | 2,802 | 2,802 |
| Test rows whose user is unseen in train | 15,408 of 42,595 | 17,736 of 43,249 |

Two facts set the shape of the study. Discarded low ratings are plentiful — 47,459 validation history slots carry one — so the inferred arm has real evidence to consume. And the newest users have none: within the eligible positive validation cohort, 1,288 requests have no history at all, 4,379 have one or two items, and 10,919 have three to five, out of 33,993. Below one catalog-known history item there is nothing to personalise from, which is exactly where the current route falls back to popularity.

One further check decided how the arms are written. For every one of the 14,175 validation users who had a prior training record, the history column attached to their first validation request reproduced that record *exactly* — same items, nothing missing, all 14,175 users. So the 20-item window that every arm in this repository reads is a truncation the scorer imposes, not a limit of what the archive holds: 23% of the distinct items a validation record lists, and 25% of the ratings below four it carries, sit outside it. Widening the window could help or hurt; no arm here tests it.

The adjacent-surface idea was measured rather than assumed. The two categories share 2,802 training users, which is the only bridge between them, and it turns out to be narrow: a donor record was reachable for 47 of the 33,993 eligible validation requests, so that arm is reported as a feasibility probe and not as a powered comparison.

## Arms

All arms rank the whole train-known catalog, exclude the request's history, use the same sparse-to-exact retrieval argument, are selected on validation only, and are gated against whichever route validation already chose — the co-review hybrid at α = 0.75 here, not the popularity baseline. A challenger wins only if the paired user-cluster bootstrap's lower 95% endpoint exceeds zero.

| Arm | Evidence it uses | User cost |
| --- | --- | --- |
| C1 repulsion | `positive − γ·negative` co-review attraction, negatives being history items the platform observed at rating ≤ 3 | none |
| C2 grading and decay | observed star weight on history items, exponential age decay by half-life | none |
| C3 analogous first purchase | distribution of first positive reviews of comparable training users, substituted where nothing personalisable exists | none |
| C4 history-conditional blend | blend weight chosen per history-size bucket, the bucket inferred from history length alone | none |
| C5 interview | declared likes and dislikes over the most-reviewed catalog items, at budgets 0 / 3 / 10 / 30 / 100 | questions asked |
| C6 adjacent surface | the same user identifier's activity in the other category archive, projected through co-likers | none, cross-surface |

## What the literature already knows

Four bodies of work carry this design, and each plays a different role.

**Inferred behaviour is real signal, and is biased in known ways.** [Hu, Koren and Volinsky](https://doi.org/10.1109/ICDM.2008.22) is the canonical statement that observed is not the same as liked and unobserved is not the same as disliked. [Steck](https://doi.org/10.1145/1835804.1835895) and [Marlin and Zemel](https://doi.org/10.1145/1639714.1639717) model the observation process itself rather than treating missingness as noise — which is what an inferred profile is. The passive-negative line is the closest empirical support for the repulsion arm: [SINE models skips of served content as a supervision signal](https://arxiv.org/abs/2308.04086), [a "not-to-recommend" loss with a responsiveness measure](https://arxiv.org/abs/2308.12256) shows how to score such a system on more than top-N accuracy, and [Yi et al.](https://doi.org/10.1145/2645710.2645724) demonstrate that positive-only implicit feedback misleads in app stores. [Schnabel et al.](https://arxiv.org/abs/1602.05352) and [Joachims et al.](https://doi.org/10.1145/3018661.3018699) supply the caution: exposure is caused by your own policy, so a behavioural feature partly measures the past recommender.

**Elicitation is the control arm, and it has an accounting system.** The [active-learning chapter of the handbook](https://doi.org/10.1007/978-1-4899-7637-6_24) and [Elahi, Ricci and Rubens](https://doi.org/10.1016/j.cosrev.2016.05.002) define which item to ask about and what a question is worth; [Rashid, Karypis and Riedl](https://doi.org/10.1145/1540276.1540302) quantify accuracy per collected rating for new users; [Anava et al.](https://doi.org/10.1145/2736277.2741109) spend a hard probe budget with optimal design. This is the only literature that treats interactions spent learning who you are as an explicit budget, so it is the only fair yardstick for the interview arm.

**Deriving a profile for someone who told you nothing is a named line of work.** The closest published precedent is [Golbandi, Koren and Lempel](https://doi.org/10.1145/1871437.1871734): cluster users by their trajectory through item space and hand a rating-less user the cluster profile, extended [adaptively with decision trees](https://doi.org/10.1145/1935826.1935910). [Zhou and Brunskill](https://arxiv.org/abs/1604.06743) and [Young and Leith](https://arxiv.org/abs/2305.18305) make the borrowed-class prior an online algorithm; [Deezer's semi-personalized system](https://arxiv.org/abs/2106.03819) is the deployed analogue, validated offline *and* online. [Gantner et al.](https://doi.org/10.1109/ICDM.2010.129) learn a mapping from observable attributes into the model's own feature space — the architectural template for any inferred profile. [Schein et al.](https://doi.org/10.1145/564376.564421) is the reason empty-history users get their own cohort rather than being averaged into one aggregate. Borrowing a profile from an adjacent surface is a different mechanism from borrowing a prior across users: it first has to establish that two records are the same person, which [record linkage](https://doi.org/10.1109/tkde.2011.127) and the [cross-network identity review](https://doi.org/10.1145/3068777.3068781) treat as noisy approximate matching rather than a key join.

**Two honesty checks the gate must survive.** [Ji et al.](https://arxiv.org/abs/2005.13829) show that a popularity baseline as usually implemented ignores which items were popular at the moment of interaction, and independently find that low-activity users follow the crowd more than heavy users do — which is both a threat to this comparison and independent support for the premise. [The popularity-bias survey](https://arxiv.org/abs/2308.01118) supplies the metrics to rule out that a "better profile" is just re-ranking toward hits. Borrowed cross-surface profiles come with [negative transfer](https://arxiv.org/abs/2211.11964) as a documented failure mode, and [Fernández-Tobías et al.](https://doi.org/10.1145/2959100.2959175) measure the accuracy/diversity trade of exactly that construction under positive-only feedback. The dataset itself is [Amazon Reviews 2023](https://arxiv.org/2403.03952), and full-catalog evaluation of implicit-feedback models follows [Steck](https://doi.org/10.1145/3298689.3347069).

Where this sits: the verified literature has inference papers that assume the behaviour stream already exists and elicitation papers that assume the interview happens. A fixed-budget, same-harness comparison of the two at matched information cost is not in it — which is the honest claim here, a controlled comparison plus a serving rule, not a new capability. Similarly, the passive-negative work studies *skips of served items*; nobody in the verified set models the review-corpus case where a user selected an item, consumed it, and then rated it badly — a two-stage self-selection. And no verified paper uses a validation-only bootstrap gate as the sole mechanism deciding whether a challenger may leave shadow mode; the nearest neighbours are [logged-policy off-policy evaluation](https://arxiv.org/abs/2008.07146) and the responsiveness framework above. Two citation notes kept from the survey: Gantner et al.'s published title is "Attribute-to-*Feature* Mappings", and the Rashid "Getting to know you" paper has conflicting IUI 2002 / 2008 venue trails, so this note cites only the [2008 SIGKDD Explorations](https://doi.org/10.1145/1540276.1540302) version.

## What the validation split shows

The shadow-profile evaluator was run twice on Musical Instruments. The first run produced the arm numbers below. The second run added a configuration ablation for C2, the cross-surface arm, and a local timing figure per arm; every arm number came back digit-for-digit the same. Both runs are validation-only, so nothing here is a confirmation.

The cohort is 33,993 eligible requests from 13,908 users, against 427,957 training rows over 54,483 users and 22,753 items. Validation selects lifetime popularity, recent popularity, or the co-review blend; it selects the hybrid at α = 0.75, which clears recent popularity by 0.0012538 NDCG@10 with a bootstrap interval of [0.0004220, 0.0020491]. Everything below is therefore measured against the hybrid, not against popularity.

| Route | NDCG@10 | Recall@10 |
| --- | ---: | ---: |
| Lifetime popularity | 0.008173 | 0.014709 |
| Recent popularity | 0.010715 | 0.021504 |
| Co-review hybrid at α = 0.75, active route | 0.011969 | 0.023152 |

### The inferred arms

| Arm | NDCG@10 | Recall@10 | Δ vs active route | 95% interval | Gate |
| --- | ---: | ---: | ---: | --- | --- |
| C1 repulsion from discarded ratings, γ = 1 | 0.012418 | 0.023799 | +0.000449 | [0.000266, 0.000626] | clears |
| C2 graded weights with 1095-day decay | 0.012435 | 0.023887 | +0.000466 | [0.000165, 0.000750] | clears |
| C3 first-purchase prior | 0.011824 | 0.022652 | −0.000145 | [−0.000270, −0.000028] | loses |
| C4 blend weight by history length | 0.012523 | 0.024593 | +0.000554 | [0.000062, 0.001057] | clears, optimistically |

C1 asks nothing of anyone. It changes one term: a history item rated below four now contributes attraction at weight 1.0 *and* repulsion at weight γ. Setting γ = 0 reproduces the active route exactly, and on this data that cell returns 0.011969, the active route's own figure, which is the parity check passing on real records rather than only in a unit test. Raising γ moves NDCG@10 monotonically across the tested grid — 0.011969, 0.012227, 0.012345, 0.012418 — with the best tested value at its boundary; larger values were not evaluated. What the arm is consuming is visible in the audit: 47,459 history slots in this validation split carry a rating below four, about 1.4 usable low-rating records per request.

C2 reaches a slightly larger gain by a different route, weighting history stars at 1.0 / 0.6 and damping old evidence with a three-year half-life. Its configuration was chosen by the maximum over eight cells on the same cohort it is then scored on. C1's γ was also selected on this cohort, so neither interval accounts for configuration selection. The run publishes all eight C2 cells rather than only the winner, and the ablation is more informative than the headline:

| Age half-life | Flat rating weights | Graded rating weights |
| --- | ---: | ---: |
| none | 0.011969 | 0.012074 |
| 1,095 days | 0.012381 | **0.012435** |
| 365 days | 0.012323 | 0.012309 |
| 90 days | 0.012117 | 0.012075 |

Almost all of the measured gain is the decay, not the grading: switching on a three-year half-life with flat weights moves NDCG@10 from 0.011969 to 0.012381, while grading on top of it adds 0.000054. Among the tested half-lives, three years scores highest; a 90-day half-life leaves less of the gain (0.012117). This pattern suggests longer-lived signal in this validation period, but does not establish an optimal half-life for future traffic.

The two arms are not independent. Both read the same discarded low ratings — one subtracts them, one stops over-trusting them — so their gains overlap and the combination was deliberately not fitted after results were open.

### The derived arm

C3 lost, and lost on the whole cohort rather than being noise. It fires on 1,618 of the 33,993 requests, the ones with no catalog-known history item, and changes nothing elsewhere, so its overall −0.000145 rescales to about −0.00305 on the slice it actually touches. Replacing popularity with the aggregate first-purchase distribution of comparable training users ranks the eventual target *worse* than the current crowd does. That is consistent with the low-activity finding in [Ji et al.'s popularity-bias work](https://arxiv.org/abs/2005.13829): new and low-activity users follow the crowd, so the crowd worth copying is the current one. A derived profile needs a conditioning variable that actually conditions, and "users who bought this category" does not. Nor was there adjacent-surface evidence waiting to be borrowed instead: the bridge between the two archives covers a small fraction of users and an even smaller fraction of requests, which the next subsection measures.

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

C5 was scored at both γ = 0 (declared likes only) and γ = 1 (declared likes and declared dislikes), so the same budget buys two different amounts of information.

| Questions asked | Answers returned | Questions per answer | Δ, likes only | Δ, likes and dislikes |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.050 | 59.9 | +0.000139 [−0.000003, 0.000298] | +0.000540 [0.000303, 0.000793] |
| 10 | 0.098 | 102.1 | +0.000174 [0.000002, 0.000359] | +0.000569 [0.000311, 0.000853] |
| 30 | 0.187 | 160.8 | +0.000196 [0.000005, 0.000400] | +0.000602 [0.000334, 0.000892] |
| 100 | 0.403 | 248.4 | +0.000525 [0.000281, 0.000770] | +0.000940 [0.000613, 0.001259] |

Cost is the interesting column. At the deepest budget the platform asks 100 questions per request and gets back 0.40 usable answers, one for every 248 questions, because it is asking about items this user simply never touched. And the answers it does get are lopsided: 0.356 are agreements and 0.047 are refusals. Just under half of the total gain at that budget — 44% — comes from the refusals, which are 12% of the answers.

Put against the inferred arms, the ranking is uncomfortable for the interview. C1's +0.000449, at no cost and no questions, is 86% of what 100 questions buy when only likes are allowed (+0.000525). Three questions with dislikes allowed (+0.000540) barely beat it. What inference cannot supply is the *refusal*: the archive contains roughly 1.4 sub-four ratings per request for free, against 0.047 elicited ones at a cost of 248 questions each, and the two mechanisms overlap almost exactly. Since the interview never asks about an item already in the profile, its remaining gain does stack on top of C1 (+0.000940 against C1's +0.000449 at the same γ), which is the one place where asking genuinely beats inferring — for items nobody has told you about.

### The experience curve

C4's value is less its aggregate +0.000554, which is selection-optimistic, than the bucket table underneath it.

| History length | Requests | Active route NDCG@10 | Blend selected | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,288 | 0.011332 | 0.0 | 0.011332 |
| 1–2 | 4,379 | 0.014825 | 0.75 | 0.014825 |
| 3–5 | 10,919 | 0.013806 | 0.75 | 0.013806 |
| 6–19 | 13,468 | 0.011128 | 0.5 | 0.011990 |
| 20+ | 3,939 | 0.006784 | 0.5 | 0.008615 |

Two things fall out. Accuracy is *not* monotone in experience: it peaks at one or two recorded items and then declines steadily, so the hardest requests in this archive come from the users who told the platform the most, not the least. And where the arm differs from the active route, it differs by *reducing* the personalised weight — 0.5 instead of 0.75 for anyone past five items, which lifts the 20+ bucket by a quarter in relative terms, and 0.0 for the empty history, confirming that with nothing personalisable the blend contributes nothing but noise. The committed route over-trusts long histories. That is a routing finding, not a shadow-profile one, and it came out of an arm designed for a different question.

### Cost of computing the profile

Each figure below is the local wall-clock time of scoring one validation request through that arm's own grid, single process, no concurrency. Grid sizes differ between arms, so a number compares an arm with its own grid rather than with another arm. The committed route's own selection step was timed in the same run, on the same cohort, and came to 3.21 ms at p95 (3.67 ms in the earlier run); C1's 3.41 ms includes its own grid search and is not an incremental serving cost.

| Step | One pass over the validation cohort (p95 ms per request) |
| --- | ---: |
| Committed hybrid, blend-weight selection | 3.21 |
| C3 first-purchase prior | 1.73 |
| C2 graded weights with 1095-day decay | 2.15 |
| C6 cross-surface projection | 2.17 |
| C4 blend weight by history length | 3.23 |
| C1 repulsion from discarded ratings | 3.41 |
| C5 interview at the 100-question budget | 10.56 |

This is a validation-time artefact of the harness, not a serving estimate. Every row is the cost of *choosing* profile parameters per request by re-scoring a grid, which is what a search does and what a deployed system would not: a production profile is computed once per user and read per request, and nothing here measures that shape.


## What this does not establish

- Both test periods in this repository were opened by earlier studies, so every number in this note is exploratory. A confirmation needs an untouched category.
- Every arm reads at most the last twenty distinct history items, a cap inherited from the committed route. The audit above shows the archive holds more than that per request, but the effect of widening the window is untested.
- The interview is simulated. Answers come from the user's own observable review of the asked item and the target is never asked, which flatters the elicited arm in one direction (a real respondent may not answer) and penalises it in another (questions about unrated items cost and return nothing).
- C4 selects its per-bucket blend weight on the same cohort it is scored on. That is selection optimism, declared rather than corrected.
- A review is not an exposure or a purchase. The repulsion arm uses low ratings only from items users chose to review, so it inherits selection bias that this benchmark cannot correct.
- The popularity control is the committed one. Its popularity terms are fixed at split time; a moment-of-interaction control would change the comparison for every challenger.
- No claim about users. Nothing here measures whether an inferred profile is welcome, noticeable or fair; that requires people, and this study has none. The measured quantities on the elicited side are questions asked, not discomfort felt, so the cost columns in C5 are effort, not experience. If someone wants the missing half, the instruments already exist: [how bothersome people find information collection](https://doi.org/10.1080/00913367.2002.10673665) as the denominator, [transparency and control over a profile](https://doi.org/10.1007/s11257-011-9118-4) and [disclosure by inference](https://doi.org/10.1007/978-3-030-42504-3_16) as the mechanisms, measured on [inferred interest models specifically](https://doi.org/10.1145/2559206.2581141).
- The cross-surface arm assumes two archives refer to the same person because a string matches. Identity resolution is treated as solved there, which the [record-linkage](https://doi.org/10.1109/tkde.2011.127) and [re-identification](https://doi.org/10.1126/science.1256297) literature would not permit in a real deployment, where the join itself carries an error rate that sets a ceiling on the borrowed profile.

## Reproduce

```powershell
python -m unittest discover -s tests
python -m tools.signal_audit --data data --output runs/signal_audit.json --categories Musical_Instruments Video_Games
python -m reliability.shadow --data data --output runs/musical-shadow-v2 --category Musical_Instruments --donor-category Video_Games
```

The last command writes `validation_decision.json` and `aggregate.json` into a directory it must create; omit `--donor-category` to skip the cross-surface probe, and add `--include-test` to report the selected configuration on the already-exposed test period. Unit checks pin the two things most likely to fail silently: that γ = 0 reproduces the committed hybrid exactly, and that sparse candidate retrieval with negative evidence matches a full-catalogue scan.
