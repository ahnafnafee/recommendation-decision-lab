# Dense retrieval and fusion: the literature behind the heavy-inference half

Prior work behind the two optional, heavier routes: the collaboratively trained
two-tower retriever ([neural_extension.md](neural_extension.md)) and the
sentence-embedding lane ([phrase_to_products.md](phrase_to_products.md)). The
wording-side trail lives in [phrase_to_products_survey.md](phrase_to_products_survey.md),
cross-linked where the halves meet rather than re-surveyed.

## What this repository does

The active route is behavioural and needs no deep-learning framework.
`reliability/core.py` fits lifetime and recent popularity (`Recommender.train`,
log-counts normalised to [0, 1]) and a co-review neighbour table (`top_neighbors`,
cosine of item co-occurrence); `Recommender.top_k` blends neighbour scores into the
base order at weight `alpha`, and `reliability/bundle.py` ships that state as a hash-checked local artifact.

Two challengers train on top of it, each gated on validation before it can serve.
The two-tower retriever (`reliability/neural.py`) pairs a history tower
(mean-pooled item embeddings plus a residual `history_projection`, then
`F.normalize`) with an item tower, so scoring is cosine; `chronological_pairs`
builds train pairs in time order and `collision_mask` removes duplicate targets and
history items from the in-batch negative pool. `reliability/neural_calibrate.py`
scores `rank_grid` over `ALPHAS = (0.05, 0.1, 0.2, 0.4, 1.0)` against a
validation-selected base, and the gate opens only if the paired user-cluster
bootstrap lower bound beats the incumbent. The embedding lane
(`reliability/embeddings.py`) wraps a sentence-transformers `Encoder` with
`normalize_embeddings=True`, and `EmbeddingIndex` scores by cosine.
`reliability/phrasing.py` fuses phrase scores with the behavioural order; its
primary configuration passes `scope="behavioural", scope_depth=2000` to
`ranker_for`, so wording re-orders only the behavioural shortlist.

## The literature, grouped by the decision it informed

### Dense retrieval as candidate generation

- [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)
  splits the system into a deep candidate-generation stage and a separate ranking
  stage, "the classic two-stage information retrieval dichotomy". Taken from this:
  the place of `TwoTower` — a retrieval stage to feed a ranker, not replace one.
- [MIND](https://arxiv.org/abs/1904.08030) names the same matching/ranking division
  at Tmall, replacing one pooled user vector with capsule-routed interest vectors.
  Taken from this: the compression loss in `TwoTower.query`'s single mean-pooled vector.
- [TDM](https://arxiv.org/abs/1801.02294) traverses a learned tree in logarithmic
  steps for full-corpus retrieval at Taobao. Taken from this: the cost model — at
  22,753 listings the repo scores the catalogue directly, so tree/ANN machinery is
  a scale answer this data does not need yet.
- [Facebook EBR](https://arxiv.org/abs/2006.11632) trains one unified embedding and
  serves it inside an existing inverted-index pipeline, with online gains. Taken
  from this: the deployment shape — a dense route entering through candidate
  generation and merging with lexical routes, mirrored by the confined phrase scope.
- [DLRM](https://arxiv.org/abs/1906.00091) shows production recommenders dominated
  by embedding tables over categorical features. Taken from this: scale context —
  the 64-dimensional model in `neural.py` sits at the small end by choice.

### Two-tower training and negative sampling

- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) replaces the
  factorisation inner product with an MLP over embeddings. Taken from this: keeping
  cosine scoring for retrieval, and a learned interaction layer as an untried lever.
- [Cross-Batch Negative Sampling](https://arxiv.org/abs/2110.15154) caches item
  embeddings from recent mini-batches to enlarge the negative pool. Taken from
  this: the ceiling of in-batch softmax at `batch_size=256` (~255 negatives per
  step) and the cheapest documented way to raise it.
- [SimCLR](https://arxiv.org/abs/2002.05709) formalises NT-Xent with its
  temperature and ties contrastive quality to batch size. Taken from this:
  `temperature=0.15` in `NeuralConfig`, and the small-batch penalty.
- [Sampled softmax loss for item recommendation](https://arxiv.org/abs/2201.02327)
  analyses when sampled softmax / InfoNCE faithfully proxies catalogue-wide softmax.
  Taken from this: `collision_mask` masking repeated targets and history items
  addresses exactly the false-negative contamination it warns about.

### Fusing a learned score with a non-personalised base

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
  shows modern networks are poorly calibrated and one-parameter post-hoc scaling
  recovers much of it. Taken from this: raw cosine is not commensurable with a
  [0, 1] popularity score; the post-hoc correction here acts on the blend weight,
  chosen on validation, instead of on the model.
- [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)
  compares convex combination (CC) with reciprocal-rank fusion: CC is scale
  sensitive but robust once its single weight is tuned on a small sample, beating
  RRF in- and out-of-domain. Taken from this: the fixed blend
  `base * (1 - a) + learned * a` over `ALPHAS`, selected on validation only.

### Dense scoring confined to a candidate set

- [LIGER](https://arxiv.org/abs/2411.18814) unifies dense and generative retrieval
  so each covers the other's weakness and improves cold-start. Taken from this: the
  division of labour behind `scope="behavioural"` — the dense component re-orders a
  behaviourally built candidate set instead of retrieving freely.
- [COIL](https://arxiv.org/abs/2104.07186) stores contextualised token embeddings
  inside inverted lists, keeping exact-match efficiency with semantic power. Taken
  from this: dense and index-structured retrieval merge at the data-structure level,
  so the confined lane is a production pattern, not a hedge.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) makes BERT produce fixed-size
  sentence vectors comparable by cosine, putting large-set similarity search within
  reach. Taken from this: the design of `Encoder` / `EmbeddingIndex` — precomputed
  normalised vectors, and the 256-token document window as the known constraint.

### Sequential transducers over purchase history

- [SASRec](https://arxiv.org/abs/1808.09781) positions self-attention between
  Markov chains (sparse-data tolerant) and RNNs (dense-data capable). Taken from
  this: a 20-window mean-pool sits below both; this is the lineage to upgrade to.
- [BERT4Rec](https://arxiv.org/abs/1904.06690) trains bidirectionally with a cloze
  objective, arguing left-to-right encoding restricts history representations. Taken
  from this: the frozen temporal protocol leaves no future context at prediction
  time, so causal `chronological_pairs` is the honest construction. HSTU and TIGER
  are covered in [phrase_to_products_survey.md](phrase_to_products_survey.md).

### On-device and LLM-based recommenders

- [On-Device Recommender Systems: A Tutorial](https://arxiv.org/abs/2312.10864)
  surveys lightweight models running on the user's device over local data, shipped
  at Taobao, Google and Kuaishou. Taken from this: the heavy half as optional
  infrastructure; `bundle.py` and the offline demo server are the first leg.
- [CoDA](https://arxiv.org/abs/2201.10382) finds on-device training overfits on a
  user's few local samples and augments them with cloud-retrieved similar samples.
  Taken from this: keeping training central while serving goes local.
- [TALLRec](https://arxiv.org/abs/2305.00447) shows a general LLM is weak at
  recommendation until instruction-tuned on recommendation data. Taken from this:
  expecting any LLM route to need supervised adaptation on these splits first.
- [P5](https://arxiv.org/abs/2203.13366) unifies recommendation tasks as
  text-to-text in one model. Taken from this: the outer bound the repo does not
  enter — everything-through-language against a fixed behavioural core.

## What the literature says to expect here

Numbers below are quoted from [neural_extension.md](neural_extension.md) and
[phrase_to_products.md](phrase_to_products.md); none is a new measurement.

| measured (validation) | value | literature reading |
| --- | --- | --- |
| unblended two-tower NDCG@10 | 0.001411 | a retrieval stage alone is not expected to rank (Covington, MIND); 256-batch negatives weaken its training (CBNS, SimCLR) |
| recent popularity | 0.010715 | the behavioural base is a strong head-start scorer in sparse data (SASRec) |
| blend at 40% neural | 0.011774 | one validation-tuned weight on a complementary scorer beats either component (fusion analysis) |
| co-review hybrid | 0.011969 | neighbour scores are already dense collaborative similarity; the neural route is partly redundant (NCF, LIGER) |
| neural − hybrid | −0.000195 [−0.000985, 0.000611] | an interval spanning zero is expected when the dense route's gain sits in cold-start cases (LIGER) |

A small contrastive retriever losing to popularity before fusion is the published
shape: two-tower systems pair the retriever with a separate ranker, and this recipe
— 64 dimensions, 256-item batches, three epochs over 313,522 pairs — sits under the
batch sizes contrastive work treats as load-bearing. The rescue by fusion at weight
0.4 reproduces that analysis's central claim at toy scale; cosine carries no
comparability with log-normalised popularity, so the weight must be chosen post hoc
and the validation gate is the calibration step. The gate closing against the
co-review hybrid matches where dense routes are reported to pay — recall of items
the incumbent misses (EBR) and cold-start (LIGER) — while the incumbent already
spends its budget on collaborative neighbours, leaving an ID-tower on the same
statistics to mostly re-rank what is already promoted. Test-period order holds
(blend 0.008357, hybrid 0.008681, popularity 0.008007), and the README's
+0.000674 [0.000040, 0.001338] lead of hybrid over popularity across 35,905
requests stands with the neural route shut out.

On the embedding lane, confinement reproduces the EBR/COIL/LIGER pattern, and the
lane's negative paired delta (−0.00240 under `behavioural`, −0.00337 catalogue-wide,
400 requests) is a coverage fact, not a retrieval failure: the behavioural shortlist
contains the answer in 21.5% of requests and own wording names it in 1.41%, while
exact titles reach 100%; no retrieval machinery exceeds those ceilings. The encoder
inherits SBERT's constraint — `all-MiniLM-L6-v2` is 384-dimensional with a 256-token
window, so product documents are truncated by construction. The on-device and LLM
groups explain why this half ships optional: inference can move to the device but
training should not (CoDA), and language-native recommenders need supervised
adaptation (TALLRec) or an all-language architecture (P5).

## Open questions

- Would a cross-batch embedding cache (CBNS) at unchanged batch size lift the
  unblended 0.001411 enough to make the gate's decision closer?
- Should cosine be calibrated post hoc (temperature scaling, per Guo) before the
  blend, or is the validation weight grid the cheaper equivalent at this scale?
- Can multi-interest pooling (MIND) move the confined lane's 21.5% coverage, which
  currently caps everything downstream?
- `scope_depth = 2000` is a constant; could validation select when the dense score
  overrides behavioural order, as LIGER's role division suggests?
- The two-tower is far smaller than the behavioural bundle — which decisions are
  worth scoring on-device (ODS tutorial) and which must stay central (CoDA)?

## Sources

- Deep Neural Networks for YouTube Recommendations — Covington, Adams, Belin, *ACM RecSys*, 2016. <https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/> — two-stage candidate-generation-then-ranking architecture at YouTube scale.
- Embedding-based Retrieval in Facebook Search — Huang et al., *KDD*, 2020. <https://arxiv.org/abs/2006.11632> — a unified dense embedding served inside an inverted-index search system, with online gains.
- Unifying Generative and Dense Retrieval for Sequential Recommendation (LIGER) — arXiv, 2024. <https://arxiv.org/abs/2411.18814> — dense and generative retrieval cover each other's weaknesses; cold-start improves.
- Multi-Interest Network with Dynamic Routing for Recommendation at Tmall (MIND) — arXiv, 2019. <https://arxiv.org/abs/1904.08030> — matching-then-ranking at Tmall with capsule-routed multiple interest vectors per user.
- Learning Tree-based Deep Model for Recommender Systems (TDM) — *KDD*, 2018. <https://arxiv.org/abs/1801.02294> — learned tree traversal gives logarithmic-cost full-corpus candidate generation at Taobao.
- Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM) — arXiv, 2019. <https://arxiv.org/abs/1906.00091> — production recommender memory dominated by embedding tables over categorical features.
- Neural Collaborative Filtering (NCF) — arXiv, 2017. <https://arxiv.org/abs/1708.05031> — an MLP interaction layer generalises matrix factorisation's inner product.
- Cross-Batch Negative Sampling for Training Two-Tower Recommenders — *SIGIR*, 2021. <https://arxiv.org/abs/2110.15154> — cached embeddings from recent batches enlarge the contrastive negative pool.
- A Simple Framework for Contrastive Learning of Visual Representations (SimCLR) — *ICML*, 2020. <https://arxiv.org/abs/2002.05709> — NT-Xent objective with temperature; contrastive quality improves with batch size.
- On the Effectiveness of Sampled Softmax Loss for Item Recommendation — arXiv, 2022. <https://arxiv.org/abs/2201.02327> — sampled softmax as a surrogate for catalogue-wide softmax, and when the surrogate holds.
- On Calibration of Modern Neural Networks — *ICML*, 2017. <https://arxiv.org/abs/1706.04599> — modern networks are poorly calibrated; one-parameter post-hoc scaling recovers much of it.
- An Analysis of Fusion Functions for Hybrid Retrieval — arXiv, 2022. <https://arxiv.org/abs/2210.11934> — convex combination with one validation-tuned weight outperforms reciprocal-rank fusion in- and out-of-domain.
- COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List — arXiv, 2021. <https://arxiv.org/abs/2104.07186> — contextualised token embeddings inside inverted lists keep exact-match speed with semantic power.
- Self-Attentive Sequential Recommendation (SASRec) — arXiv, 2018. <https://arxiv.org/abs/1808.09781> — self-attention balances Markov-chain sparsity tolerance against RNN capacity on Amazon-style sequences.
- BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer — arXiv, 2019. <https://arxiv.org/abs/1904.06690> — bidirectional cloze training beats left-to-right encoders when future context exists.
- Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks — arXiv, 2019. <https://arxiv.org/abs/1908.10084> — siamese BERT yields fixed-size sentence vectors comparable by cosine at large-set scale.
- On-Device Recommender Systems: A Tutorial on The New-Generation Recommendation Paradigm — arXiv, 2023. <https://arxiv.org/abs/2312.10864> — taxonomy of lightweight, device-local recommendation shipped at Taobao, Google, Kuaishou.
- CoDA: Device-Cloud Collaborative Learning for Recommendation — arXiv, 2022. <https://arxiv.org/abs/2201.10382> — on-device training overfits; cloud-retrieved similar samples repair it.
- TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation — arXiv, 2023. <https://arxiv.org/abs/2305.00447> — general LLMs underperform at recommendation until instruction-tuned on recommendation data.
- Recommendation as Language Processing (P5): A Unifying Text-to-Text Framework for Recommendation Models — arXiv, 2022. <https://arxiv.org/abs/2203.13366> — many recommendation tasks expressed as a single text-to-text model.
