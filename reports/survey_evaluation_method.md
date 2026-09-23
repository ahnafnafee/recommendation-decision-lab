# Evaluation method: what the published record says this design should produce

Scope: the measurement apparatus — frozen temporal splits, full-catalogue ranking, one headline metric with one
secondary, one interval, one gate, one validation selection, and the diagnostics printed beside it. Every source was
opened during this session; repository numbers are copied from `README.md`,
`reports/musical_instruments_confirmation.json`, `reports/transfer_note.md` and `reports/phrase_to_products.md`.

## 1. What this repository does

`reliability/benchmark.py` fixes two absolute cutoffs, `T1 = 1628643414042` and `T2 = 1658002729837`, over the
`timestamp_w_his` archives of Amazon Reviews 2023 (`SOURCE`), and treats boundary crossings as errors:
`training_data` raises `"training row crosses t1"`, `Recommender.train` raises
`"training event crosses the global cutoff"`, `evaluation_data` raises `"evaluation row outside source split"`.
`file_hash` records sha256 digests of the training and validation archives before the test archive is read, and
`validation_decision.json` is written first — test rows open only after the validation choice and gate persist.
`run` refuses categories `"outside the committed study design"` and re-running over `"run directory exists; retain
prior evidence"`.

Ranking is whole-catalogue and exact. `Recommender.rank_grid(history, baseline, alphas, k=10)` scores the union of the
first `k` eligible items in base order and every item with a non-zero personalised score, because — in the code's own
words — items outside the neighbour set score zero, so their top-k can only come from the first k eligible items in
base order; neighbours come from `top_neighbors`, a cosine co-review similarity
`common / sqrt(item_users[left] * item_users[right])`, over histories built by `last_distinct`. No sampled negatives
appear anywhere between rows and metric. The metric has one target per request:
`ndcg_one_target(ranking, target)` returns `1 / log2(rank + 2)` if the target is inside `k` and zero otherwise, and
`quality` reports `ndcg_at_10` and `recall_at_10` as macro averages over requests. `user_cluster_interval` aggregates
the paired difference `right - left` per user (`query[0]`) and resamples user clusters for `draws = 1000` at
`seed = 20260922`, returning `delta`, `lower`, `upper`, `users`, `draws`; the challenger becomes `active_route` only
when `validation_interval["lower"] > 0`. Two selections run on validation and persist before test: the baseline is
`"recent"` if its validation NDCG beats `"lifetime"` popularity, and `challenger_alpha` is the best of
`ALPHAS = (0.25, 0.5, 0.75, 1.0)` with `-alpha` as tiebreaker.

The other arms measure the same cohort under different evidence. `reliability/phrasing.py` rebuilds it in
`timestamped_requests` and raises if it disagrees row-for-row with `evaluation_data`; the four answer-informed of seven
`CONDITIONS` (`own_words`, `own_words_scrubbed`, `target_title`, `random_title`) reach the reader through `summarise`
as paired deltas against `base` only. `reach_for` reports how often each route's own candidates contained the answer;
`agreement` scores rank agreement between scorers over `AGREEMENT_REQUESTS = 300` requests at `DEPTH = 50` with
`spearman`; `popularity_buckets`, `history_buckets` and `stratified` cut the same numbers by popularity and history
length; `timed_requests` measures latency. `reliability/shadow.py` runs the elicited arm: `QUESTION_BUDGETS = (0, 3,
10, 30, 100)` questions with `0` as the nothing-elicited control, `ShadowRecommender` over `novice_prior`,
`interview_order` and `adjacent_projection`, already-held evidence weighted by `RATING_WEIGHT = {5: 1.0, 4: 0.6}` and
decayed over `HALF_LIVES = (None, 1095, 365, 90)`, `score_requests` sweeping `alphas` against `GAMMAS`, and `run`
defaulting to `include_test=False`.

## 2. The decisions, and the literature behind each

### 2.1 Freeze time, then verify the freeze held

Gusak, Volodkevich, Klenitskiy, Vasilev and Frolov write that leave-one-out "permits the overlap between training and
test periods, which leads to temporal leakage", and outcomes "vary significantly across splitting strategies,
influencing model rankings and practical deployment decisions". Taken from this: the split changes which systems look
deployable. Ji, Sun, Zhang and Li find BPR, NeuMF, SASRec and LightGCN recommending "future items" to a past test
instance, and the direction is the load-bearing part: "The impact of data leakage on recommender models is
*unpredictable*." Taken from this: inflation is not monotone in leaked volume, so a drift check cannot be corrected
with a factor — it has to raise, which is what `training_data` and `train` do.

Meng, McCreadie, Macdonald and Ounis find the split strategy "is an important confounding variable that can markedly
alter the ranking of state-of-the-art recommender systems", and recommend temporal global splitting as the default.
Taken from this: the frozen `T1`/`T2` design is the accepted default, and the cohort shrinkage it causes is a priced
cost. Hidasi and Czapp measure concept drift as the share of test transitions already shared with training, and show
overlapping splits "provide an easier, unrealistic problem for which less generalization is needed". Taken from this:
what the cutoff guards detect is measurable and large. Sun shows why "the popularity baseline is poorly defined"
without temporal separation. Taken from this: `train(..., recent_days=365)` deriving `lifetime` and `recent` from
training rows only is the correction, and choosing between them on validation is its minimum honest form.
Benchmark-side leakage adds a channel no split guards, and its sign is conditional — "substantial but spurious
performance gains" from domain-relevant exposure, "typically degrades recommendation accuracy" otherwise. Taken from
this: leakage has no single sign, which removes any temptation to treat a measured gap as inflated by a known amount.

### 2.2 Rank the whole catalogue

Rendle: sampled metrics "do not persist relative statements, e.g., 'algorithm A is better than B', not even in
expectation", and "the smaller the sampling size, the less difference between metrics"; Krichene and Rendle state it
for the metric family as used in practice. Taken from this: every claim here is a relative statement, so a sampled
metric could not have supported it at any noise level.

Hidasi and Czapp list "negative item sampling" among four severe flaws and quantify it: an "8.1% improvement in
recall@20 … turns into -4.7%" on Coveo, a "14.3% uplift … vanishes and drops to 1.5%" on Retailrocket, and a reordering
on Rees46. The mechanism is catalogue size — uniform sampling cannot supply hard negatives when the catalogue dwarfs
the sample, and a target ranked 1,490th among 10,000 still reaches a top-20 list most of the time — and their
conclusion is that "Ranking over all items is how real-life recommenders work". Taken from this: `rank_grid`'s exact
top-k is what makes rank 12 distinguishable from rank 140, and it is cheap here: p95 0.760 ms per request in Musical
Instruments, 0.611 ms in Video Games. Dallmann, Zoller and Hotho confirm it at the usual target-set size: "both
sampling strategies can produce inconsistent rankings compared with the full ranking of the models"; Li, Jin, Liu, Ren,
Gao and Liu formalise the estimator problem as a "blind spot" remedyable only by larger samples at proportional cost.
Taken from this: the sampled-metric literature trades resolution for compute, and changing the sampling distribution
is not an exit. LIGER states what full scoring means mechanically — dense retrieval ranks by "inner product computation
between the user and all item representations" — and scopes its hybrid to "small-scale benchmarks". Taken from this:
full-catalogue scoring is the reference condition the phrase lane must survive, and a published hybrid is not evidence
about this catalogue.

### 2.3 One target, one cut-off, and what the metric can see

Jeunen, Potapov and Ustimenko: "even when DCG is unbiased, ranking competing methods by their normalised DCG can invert
their relative order", and the correlation with online reward holds for unbiased DCG but "This statement no longer
holds for its normalised variant". Taken from this: NDCG@10 is a defensible headline but its normalisation makes it
ordinal per configuration, so it is reported beside `recall_at_10` and as paired deltas rather than as a percentage of
anything.

Diaz, Ekstrand and Mitra formalise "recall-orientation" as a metric's sensitivity to a user who wants every relevant
item, with discriminative power and stability changing "in the presence of missing labels". Taken from this: with one
judged item per request the second metric answers a different question, and the document states which one is primary.
Meng, McCreadie, Macdonald and Ounis report the shape is itself design-dependent — leave-one-last-item splitting
produces "a wider distribution of NDCG@10 and Recall@10 scores" than temporal splitting. Taken from this: the spread of
single-target NDCG@10 is a property of the design, so the interval must be built over users. Parajuli, Vaez Barenji and
Ekstrand vary filtering thresholds and candidate-set construction against dense ground truth and find "there is no
uniformly best offline evaluation design". Taken from this: the 5-core cohort is a validity trade with no cost-free
alternative, which is why the filter appears in the limitations.

### 2.4 Intervals that treat the user as the unit

Deng, Knoblich and Lu separate the randomization unit from the "analysis unit — the aggregation level of metric
computation", give `rho = tau^2/(sigma^2+tau^2)`, and give clustered variance with the inflation term `1 + (m-1)rho`.
Taken from this: a request is not an independent observation, and a request-level interval understates variance by that
factor; `user_cluster_interval` resamples clusters keyed by `query[0]`, one draw per user. Schnabel reports that "many
results on smaller datasets are likely not statistically significant" and that better uncertainty quantification "rules
out some reported differences between linear and neural methods". Taken from this: the `lower > 0` gate substitutes for
the testing whose absence that study documents, and the methods most at risk of over-claiming are neighbour-based ones
— which is what the challenger is. Dror, Baumer, Shlomov and Reichart supply the protocol: match the significance test
to the structure of the data, with resampling tests as the general fallback. Taken from this: the resampling scheme is
itself a design decision to be stated, which is why `draws` and `seed` are written into the artifact.

### 2.5 Choosing the fusion weight without spending the test set

Rodriguez, Lesota and Tommasel fix the partition and vary only the training seed, finding seed effects detectable at
the user-score level, that different seeds select different validation configurations, and that on Amazon "several
validation winners do not remain better on test" — "validation selection can be stable, but the selected configuration
may not transfer as well to test". Taken from this: a validation-chosen `alpha` is expected to under-deliver on test,
and a thin gap between adjacent grid points is a warning about the selection rather than a preference between them.
Schnabel puts hyperparameter search inside the uncertainty loop with an explicit budget and asks for well-tuned
baselines; Ferrari Dacrema et al. supply the matched-baseline half: "11 out of the 12 reproducible neural approaches can
be outperformed by conceptually simple methods, e.g., based on the nearest-neighbor heuristic or linear models", traced
to "the choice and optimization of the baselines used for comparison". Taken from this: the four-point `ALPHAS` grid is
far inside that budget, so selection pressure is modest and the residual belongs in a limitation string; and because
the challenger is neighbour-based, the only comparison that means anything is against a tuned popularity baseline,
`recent` and `lifetime`, selected on validation.

### 2.6 Matched controls for evidence the model already holds

LangPTune (Gao, Zhou, Dai and Joachims) prices a constructed profile against matched controls: on Amazon-Movie-TV,
0.0006 NDCG@20 for a random no-profile and 0.0039 for the most-popular no-profile control, against 0.0494 for the
trained language-based profile at 512 tokens, with zero-shot profile "quality remains insufficient". Taken from this: a
control that keeps the format and destroys the content is what makes a lift interpretable — which is why `random_title`
and `target_title` exist here as paired-delta conditions rather than as headline numbers.

### 2.7 Reach ceilings and rank agreement beside the headline number

Dong, Qin, Shah and Wang separate reranking quality from retrieval coverage over five domains: single retrievers place
the gold item in a 200-item pool only 4.6-22.9% of the time because 32-91% of cold-start targets are new, and their
validation-trained fusion "recovers 17-61% of oracle coverage headroom on content-rich domains, but only 5-7% on
collaboratively strong domains", while prompt-level reranking "often degrades" the better pool. Taken from this:
`reach_for` is the same diagnostic; a lane cannot rank an answer its pool does not contain, fusion gains concentrate
where collaborative signal is weak, and a larger candidate set is not automatically worth its cost. Facebook's
embedding retrieval gives the production shape — semantic retrieval served alongside an inverted index, optimised
"end-to-end … including ANN parameter tuning and full-stack optimization", with gains "observed in online A/B
experiments". Taken from this: candidate generation is where a semantic lane earns its place, and the published gains
rest on an online measurement this repository lacks.

### 2.8 Constructed phrases: what a lift demonstrates

Ma, Wanyan, Hettiachchi, Xu and Chan audit 77,004 simulated queries over 100 UQV100 topics, 8 models and 5 prompt
conditions: answer-side concepts are 7.40 percent of non-generic concepts, appear in 97 of 100 topics, and moving them
changes results more than random deletion (d = -0.47 versus -0.34) yet they "explain less than 2% of aggregate
evaluation variance"; "no prompt condition eliminates" them. Taken from this: evidence the answer already supplies is
detectable per request and small in aggregate, so `own_words`, `own_words_scrubbed`, `target_title` and `random_title`
must be read as paired deltas — how `summarise` treats them.

Rahmani, Ramineni, Yilmaz, Craswell and Mitra validate synthetic-data bias "using a linear mixed-effects model" and
conclude it "could be significant, for e.g. computing absolute system performance" but "may not be as significant in
comparing relative system performance". Taken from this: report a constructed lane against a common base and never its
absolute level as achievable performance — the split between the behavioural and `--scope catalogue` arms; the
companion study adds that fully synthetic collections can be used reliably for retrieval evaluation. He, Kim, Diaz,
Arguello and Mitra validate simulated queries against human-elicited ones by "high correlations with how CQA-based TOT
queries rank TOT retrieval systems" plus linguistic similarity; CASTLE (Wei et al.) instead makes relevance labels
follow "by construction", "achieving near-zero false positives without LLM judgment", in daily deployment. Taken from
this: the honest validity claim for an elicited phrase is a rank correlation against a human-elicited reference, while
`published`, whose relevance follows from what the review itself says, needs no such argument. REGEN (Su et al.) adds
user "steering" queries and narratives to this dataset family, which "primarily focus on sequential item prediction".
Taken from this: extra text here is a resource for constructing requests, not an evaluation of recommendation quality.

### 2.9 Tolerance to damaged text

Bayley, Zhu, Aoki, Cao and Wilson measure how much corrupted language-model guidance survives, reporting that
"warm-starting remains effective up to 30% corruption, loses its advantage around 40%, and degrades performance
beyond 50%", and that under systematic misalignment the priors "can lead to higher regret than a cold-start bandit".
Taken from this: the 30/40/50 ladder is the published analogue for `own_words_scrubbed`, and it warns that systematic
distortion — unlike random noise — can leave a lane worse than not using it at all.

## 3. What the literature says to expect here

| Design element | Measured here | What the published record predicts |
| --- | --- | --- |
| Frozen splits, guards that raise | `T1`/`T2`; `"training row crosses t1"`; `seen_target_excluded = 0` | Unguarded splits let models recommend future items; the effect is not monotone in leaked volume |
| Full-catalogue exact top-k | `rank_grid`; p95 0.760 ms (MI), 0.611 ms (VG) | Sampled negatives hide deep-rank differences and reorder models |
| One target; NDCG@10 macro plus Recall@10 | `ndcg_one_target`, `quality` | Single-target distributions are wide and design-dependent; normalisation blocks interval-scale reading |
| Paired user-cluster bootstrap, 1000 draws, seed 20260922, gate `lower > 0` | +0.000674 [0.000040, 0.001338] over 11,648 users | Ungated single runs at this size are often not significant; request-level variance understates user-level variance |
| Validation-selected baseline and alpha | `recent` 0.010715246 over `lifetime` 0.008173450; alpha 0.75 at 0.011969058 | Selection gains shrink or vanish on test; thin margins between adjacent grid points are unstable |
| Matched controls | `base`, `random_title`, `recent`/`lifetime`, `QUESTION_BUDGETS` 0 | Controls, not levels, are what make a lift interpretable |
| Reach ceiling and rank agreement | `reach_for`; `agreement` over 300 requests at `DEPTH = 50` | A fusion lane is capped by how often its pool contains the answer |
| Constructed phrases | `published` in absolute terms; four answer-informed conditions as paired deltas | Constructed collections support relative claims; answer-side evidence is per-request detectable and aggregate-small |
| Elicited evidence with a budget ladder | `QUESTION_BUDGETS` 3-100; `RATING_WEIGHT`, `HALF_LIVES` | Tolerable near 30% corruption, worse than the unaided control under systematic distortion |

**Validation-to-test shrinkage is the expected result.** On validation the selected challenger beat the selected
baseline by 0.0012538119668424027 with lower percentile 0.0004220061257854118; on test the same pair differs by
0.0006741059717609642 with lower percentile 4.002188538339936e-05. About half the validation gap survived and the lower
bound fell by an order of magnitude while still clearing zero — the pattern Rodriguez et al. report for Amazon.

**The selected grid point rests on a thin margin, and the cohort bounds the effect.** Alpha 0.75 scores 0.011969058 on
validation and alpha 0.5 scores 0.011862047 — about 0.0001 apart — while alpha 1.0 collapses to 0.004460087, below the
`recent` baseline at 0.010715246: the phrase lane alone is worse than the baseline. Of 42,595 test rows, 6,690 fall
below the rating threshold and `empty_history = 1186` arrive with no usable history; 12,305 of the 35,905 eligible
requests — 34.3% — have an answer item absent from the training-known catalogue, inside the 32-91% band Dong et al.
identify as the reason retrieval pools miss the answer. Video Games behaves the same way: 22,051 of 35,562 targets
absent from the train-known catalogue, and a paired gain of +0.003020 [0.002216, 0.003881] over a 0.007151 baseline —
four times the Musical Instruments effect, from a cohort of the same structure.

## 4. Open questions

- How much of the test lower bound (4.0e-05) is cohort luck? The gate opens by a margin far smaller than the
  validation-to-test shrinkage above; a second frozen period would say whether the gate is repeatable.
- Should `alpha` be selected on stability rather than argmax? Choosing between 0.5 and 0.75 on a 0.0001 validation gap
  is the documented failure mode; repeated fits selecting on the lower percentile across repeats is the analogue of the
  interval already computed.
- What does `reach_for` actually cap? If phrase-route candidates contain the answer materially more often than the
  34.3% of requests whose target is catalog-new, item novelty is not the binding constraint.
- Does rank agreement between scorers fall as `alpha` rises? `agreement` reports it over 300 requests; widening it to
  list-overlap measures over all requests costs nothing at sub-millisecond ranking.
- At which `QUESTION_BUDGETS` point does elicited evidence separate from evidence the model already holds, and does that
  point survive systematic distortion rather than random noise?
- Is any of this portable without retuning? `run` restricts `category` to the committed design, and the two categories
  measured so far differ by more than four times in effect size.

## Sources

- Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders — Danil Gusak, Anna Volodkevich, Anton Klenitskiy, Alexey Vasilev, Evgeny Frolov; arXiv:2507.16289, RecSys 2025 author's version. [arxiv.org/abs/2507.16289](https://arxiv.org/abs/2507.16289) — leave-one-out "permits the overlap between training and test periods, which leads to temporal leakage and unrealistically long test horizon"; outcomes "vary significantly across splitting strategies, influencing model rankings and practical deployment decisions".
- A Critical Study on Data Leakage in Recommender System Offline Evaluation — Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li; arXiv:2010.11060; journal version ACM Transactions on Information Systems 41(3), Article 75 (2023). [arxiv.org/abs/2010.11060](https://arxiv.org/abs/2010.11060) — leave-one-out makes BPR, NeuMF, SASRec and LightGCN recommend future items; "The impact of data leakage on recommender models is unpredictable. It is not true that more future data in training leads to higher accuracy", and model rankings can move.
- Exploring Data Splitting Strategies for the Evaluation of Recommendation Models — Zaiqiao Meng, Richard McCreadie, Craig Macdonald, Iadh Ounis; arXiv:2007.13237 (2020/2021). [arxiv.org/abs/2007.13237](https://arxiv.org/abs/2007.13237) — the split strategy "is an important confounding variable that can markedly alter the ranking of state-of-the-art recommender systems"; "Report performance under temporal global splitting: This is generally seen as the most realistic setting, and so should be the default splitting strategy used"; "leave one last item data splitting is producing a wider distribution of NDCG@10 and Recall@10 scores, while the temporal and leave one last basket splitting seems to group systems in a more stratified manner".
- Take a Fresh Look at Recommender Systems from an Evaluation Standpoint — Aixin Sun; arXiv:2210.04149 (2022/2023). [arxiv.org/abs/2210.04149](https://arxiv.org/abs/2210.04149) — under random or leave-one-out splits "the popularity baseline is poorly defined"; ignoring a global timeline causes both leakage and oversimplified preference modelling.
- Benchmark Leakage Trap: Can We Trust LLM-based Recommendation? — Mingqiao Zhang et al.; arXiv:2602.13626 (2026). [arxiv.org/abs/2602.13626](https://arxiv.org/abs/2602.13626) — domain-relevant benchmark exposure gives "substantial but spurious performance gains" while "domain-irrelevant leakage typically degrades recommendation accuracy".
- Evaluation Metrics for Item Recommendation under Sampling — Steffen Rendle; arXiv:1912.02263 (Dec 2019). [arxiv.org/abs/1912.02263](https://arxiv.org/abs/1912.02263) — "Sampled metrics do not persist relative statements, e.g., 'algorithm A is better than B', not even in expectation. Moreover the smaller the sampling size, the less difference between metrics, and for very small sampling size, all metrics collapse to the AUC metric".
- On Sampled Metrics for Item Recommendation (Extended Abstract) — Walid Krichene, Steffen Rendle; Proc. 30th IJCAI, Sister Conferences track, pp. 4784–4788, 2021, DOI 10.24963/ijcai.2021/651. [ijcai.org/proceedings/2021/651](https://www.ijcai.org/proceedings/2021/651) — sampled metrics are inconsistent with their exact counterpart and do not persist relative statements.
- A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models — Alexander Dallmann, Daniel Zoller, Andreas Hotho; RecSys 2021, DOI 10.1145/3460231.3475943. [arxiv.org/abs/2107.13045](https://arxiv.org/abs/2107.13045) — at target-set size 100, "both sampling strategies can produce inconsistent rankings compared with the full ranking of the models".
- Widespread Flaws in Offline Evaluation of Recommender Systems — Balázs Hidasi, Ádám Tibor Czapp; RecSys '23, DOI 10.1145/3604915.3608839. [arxiv.org/abs/2307.14951](https://arxiv.org/abs/2307.14951) — names negative item sampling among four severe flaws (§2.4 "Negative sampling during testing"); Coveo "8.1% improvement in recall@20 … turns into -4.7%", Retailrocket "14.3% uplift in recall@20 vanishes and drops to 1.5%", Rees46 true order "A > C >> B, but with 100 samples it reads as B > A >> C"; "Uniform sampling is not able to provide strong samples if the size of the item catalog is significantly larger than the number of samples", with a target ranked 1,490th among 10,000 reaching a top-20 list over 90% of the time; overlapping splits "lessen the need of modeling the concept drift"; "Ranking over all items is how real-life recommenders work".
- Towards Reliable Item Sampling for Recommendation Evaluation — Dong Li, Ruoming Jin, Zhenming Liu, Bin Ren, Jing Gao, Zhi Liu; arXiv:2211.15743 (AAAI 2023). [arxiv.org/abs/2211.15743](https://arxiv.org/abs/2211.15743) — item-sampling estimators suffer a "blind spot" at small K, with larger samples as the costly remedy.
- Unifying Generative and Dense Retrieval for Sequential Recommendation (LIGER) — Liu Yang et al.; arXiv:2411.18814 (2024). [arxiv.org/abs/2411.18814](https://arxiv.org/abs/2411.18814) — dense retrieval ranks by "inner product computation between the user and all item representations"; the hybrid narrows the generative/dense gap and aids cold-start items, scoped to small-scale benchmarks.
- On (Normalised) Discounted Cumulative Gain as an Off-Policy Evaluation Metric for Top-n Recommendation — Olivier Jeunen, Ivan Potapov, Aleksei Ustimenko; arXiv:2307.15053; accepted at KDD '24. [arxiv.org/abs/2307.15053](https://arxiv.org/abs/2307.15053) — "even when DCG is unbiased, ranking competing methods by their normalised DCG can invert their relative order"; on a large platform unbiased DCG correlates with online reward, and "This statement no longer holds for its normalised variant".
- Recall, Robustness, and Lexicographic Evaluation — Fernando Diaz, Michael D. Ekstrand, Bhaskar Mitra; arXiv:2302.11370 (2023/2024). [arxiv.org/abs/2302.11370](https://arxiv.org/abs/2302.11370) — defines "recall-orientation" as a metric's sensitivity to a user seeking every relevant item, and ties discriminative power and stability to the presence of missing labels.
- On the Convergent Validity of Offline Evaluation Designs for Recommender Systems — Sushobhan Parajuli, Samira Vaez Barenji, Michael D. Ekstrand; arXiv:2607.25097, DOI 10.1145/3773078.3831818 (2026). [arxiv.org/abs/2607.25097](https://arxiv.org/abs/2607.25097) — sparse-log evaluation validity depends on dataset and dense targets; "there is no uniformly best offline evaluation design".
- Where Do We Go From Here? Guidelines For Offline Recommender Evaluation — Tobias Schnabel; arXiv:2211.01261 (2022). [arxiv.org/abs/2211.01261](https://arxiv.org/abs/2211.01261) — "many results on smaller datasets are likely not statistically significant"; "improved uncertainty quantification (via nested CV and statistical testing) rules out some reported differences between linear and neural methods"; prescribes a hyperparameter search budget (50 iterations in its own runs) and strong tuned baselines.
- Applying the Delta method in metric analytics: A practical guide with novel ideas — Alex Deng, Ulf Knoblich, Jiannan Lu; arXiv:1803.06336 (2018). [arxiv.org/abs/1803.06336](https://arxiv.org/abs/1803.06336) — separates the randomization unit from the "analysis unit — the aggregation level of metric computation", gives rho = tau^2/(sigma^2+tau^2), and gives cluster-randomised variance with the inflation term 1 + (m−1)ρ.
- The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing — Dror, Baumer, Shlomov, Reichart; ACL 2018, pp. 1383–1392, DOI 10.18653/v1/P18-1128. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/) — a protocol for choosing a significance test to match the structure of the data, with resampling tests as the general fallback.
- Training seeds and model-selection stability in recommender-system evaluation — Juan Manuel Rodriguez, Oleg Lesota, Antonela Tommasel; arXiv:2609.02499, DOI 10.1145/3773078.3841289 (2026). [arxiv.org/abs/2609.02499](https://arxiv.org/abs/2609.02499) — with users treated as repeated observations, seed variation is often detectable, "validation selection can be stable, but the selected configuration may not transfer as well to test", and on Amazon "several validation winners do not remain better on test".
- A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research — Maurizio Ferrari Dacrema, Simone Boglio, Paolo Cremonesi, Dietmar Jannach; arXiv:1911.07698 (ACM TOIS 2021). [arxiv.org/abs/1911.07698](https://arxiv.org/abs/1911.07698) — "11 out of the 12 reproducible neural approaches can be outperformed by conceptually simple methods, e.g., based on the nearest-neighbor heuristic or linear models", traced to "the choice and optimization of the baselines used for comparison".
- End-to-end Training for Recommendation with Language-based User Profiles (LangPTune) — Zhaolin Gao, Joyce Zhou, Yijia Dai, Thorsten Joachims; arXiv:2410.18870 (2024/2025). [arxiv.org/abs/2410.18870](https://arxiv.org/abs/2410.18870) — Table 3, Amazon-Movie-TV NDCG@20: 0.0006 for the random no-profile, 0.0039 for the most-popular no-profile control, 0.0494 for the trained language-based profile at 512 tokens; zero-shot profile "quality remains insufficient, leading to suboptimal recommendation performance".
- Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation — Zhe Dong, Fang Qin, Manish Shah, Yicheng Wang; arXiv:2606.29947 (2026). [arxiv.org/abs/2606.29947](https://arxiv.org/abs/2606.29947) — "standard single retrievers place the gold item in a 200-item pool only 4.6-22.9% of the time, largely because 32-91% of cold-start targets are brand-new items with no training interactions"; learned fusion "recovers 17-61% of oracle coverage headroom on content-rich domains, but only 5-7% on collaboratively strong domains", and prompt-level reranking "often degrades" the better pool.
- Embedding-based Retrieval in Facebook Search — Jui-Ting Huang et al.; arXiv:2006.11632 (KDD 2020). [arxiv.org/abs/2006.11632](https://arxiv.org/abs/2006.11632) — semantic retrieval served inside an inverted-index system, optimised "end-to-end … including ANN parameter tuning and full-stack optimization", with gains "observed in online A/B experiments".
- The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion — Chenglong Ma, Xinye Wanyan, Danula Hettiachchi, Ziqi Xu, Jeffrey Chan; arXiv:2608.25245, CIKM '26, DOI 10.1145/3799682.3840922. [arxiv.org/abs/2608.25245](https://arxiv.org/abs/2608.25245) — over 77,004 simulated queries on 100 UQV100 topics, answer-side concepts are "7.40 percent of non-generic concepts" appearing in 97 of 100 topics; deleting them moves results more than random deletion (d = −0.47 vs −0.34) but they "explain less than 2% of aggregate evaluation variance", and "no prompt condition eliminates" them.
- CASTLE: Contrastive and Seed-Guided Training for Cold-Start Natural Language Search — Wendy Ran Wei et al.; arXiv:2605.21812, CIKM 2026, DOI 10.1145/3799682.3841071. [arxiv.org/abs/2605.21812](https://arxiv.org/abs/2605.21812) — synthetic queries whose relevance labels hold "by construction via contrastive listing pairs derived from booking sessions, achieving near-zero false positives without LLM judgment", KL 1.01 against 9.33 for the best baseline, in daily deployment.
- Towards Understanding Bias in Synthetic Data for Evaluation — Hossein A. Rahmani, Varsha Ramineni, Emine Yilmaz, Nick Craswell, Bhaskar Mitra; arXiv:2506.10301 (2025). [arxiv.org/abs/2506.10301](https://arxiv.org/abs/2506.10301) — bias effects validated "using a linear mixed-effects model": "the effect of bias present in evaluation results obtained using synthetic test collections could be significant, for e.g. computing absolute system performance, its effect may not be as significant in comparing relative system performance".
- Synthetic Test Collections for Retrieval Evaluation — Hossein A. Rahmani, Nick Craswell, Emine Yilmaz, Bhaskar Mitra, Daniel Campos; arXiv:2405.07767 (2024). [arxiv.org/abs/2405.07767](https://arxiv.org/abs/2405.07767) — fully synthetic collections can be used reliably for retrieval evaluation, with stated bias risks.
- Tip of the Tongue Query Elicitation for Simulated Evaluation — He, Kim, Diaz, Arguello, Mitra; arXiv:2502.17776 (2025). [arxiv.org/abs/2502.17776](https://arxiv.org/abs/2502.17776) — simulated queries validated by "high correlations with how CQA-based TOT queries rank TOT retrieval systems" plus linguistic similarity, and released as TREC 2024/2025 TOT track queries.
- REGEN: A Dataset and Benchmarks with Natural Language Critiques and Narratives — Kun Su et al.; arXiv:2503.11924 (2025). [arxiv.org/abs/2503.11924](https://arxiv.org/abs/2503.11924) — adds user critiques ("user 'steering' queries that lead to the selection of a subsequent item") and narratives to Amazon Reviews, against datasets that "primarily focus on sequential item prediction".
- Jump Start or False Start? A Theoretical and Empirical Evaluation of LLM-initialized Bandits — Adam Bayley, Xiaodan Zhu, Raquel Aoki, Yanshuai Cao, Kevin H. Wilson; arXiv:2604.02527 (2026). [arxiv.org/abs/2604.02527](https://arxiv.org/abs/2604.02527) — the paper reports that "warm-starting remains effective up to 30% corruption, loses its advantage around 40%, and degrades performance beyond 50%"; under systematic misalignment LLM priors "can lead to higher regret than a cold-start bandit".

Dropped rather than cited: arXiv:2209.04973, which opened as a field study of peer recommendation for health support
rather than split contamination; arXiv:1701.09600, which resolved to no arXiv record, so the ACL Anthology tutorial
carries the resampling point instead; Cañamares and Castells' SIGIR 2020 sampling study, whose only reachable host
returned HTTP 403 (Dallmann et al. carries that result here); and one deployment-scale offline-versus-online study
whose URL was not opened in this session and which therefore supports no sentence here.
