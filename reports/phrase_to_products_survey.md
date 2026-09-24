# Phrase to products: what the published record says about answering a sentence

Scope: text-conditioned recommendation, semantic item representations that bridge cold-start, e-commerce query
understanding, the measured value of asking, evaluation when the corpus has no typed queries,
and speech input. Every arXiv identifier below was resolved through the arXiv API (`export.arxiv.org`) and every
DOI through doi.org or the publisher before being cited. The abstract of every entry was read, and the full text was
opened for the works whose evaluation design this note describes in detail.

Repository constraints that gate relevance: full catalogue (tens of thousands of items), temporal train/val/test on implicit +
explicit signals, per-request NDCG@10, paired-bootstrap activation gate, stdlib-only core (torch optional), no typed
queries anywhere in the corpus, no-overclaim policy. Current reference numbers: fixed hybrid 0.008681 vs validation-
selected recent popularity 0.008007 NDCG@10 over 35,905 eligible requests; paired +0.000674 [0.000040, 0.001338].

---

## 1. Text-conditioned / instruction-driven recommendation

- InstructRec — Recommendation as Instruction Following (arXiv:2305.07001, Zhang et al.).
  Task: recsys as instruction following. Mechanism: instruction-tune an open LLM on 252K auto-generated
  personalised instructions from 39 templates (behaviour→instruction, not human-written). Metric: beats several
  competitive baselines incl. GPT-3.5 on their tasks. Eval: offline, generated-instruction benchmarks. Limitation: instructions and evaluation both come from the same generative recipe, and it never ranks a full real
  catalogue under a temporal split — the exact gap for this repository, which has real logs and no instructions.
- Chat-REC (arXiv:2303.14524). Conversational LLM recommender: user profile + history → prompts,
  in-context learning, cross-domain + cold-start claims. Limitation: prototype; no full-catalogue ranking
  metric, no A/B; interactive quality is argued, not measured with an effect size.
- KAR — Open-World Recommendation with Knowledge Augmentation from LLMs (arXiv:2306.10933).
  Mechanism: LLM produces *reasoning knowledge* (user preferences) + *factual knowledge* (item facts) via factorization
  prompting, converted to lightweight features consumed by a conventional CTR ranker. Headline: +7% and +1.7%
  online A/B gains (Huawei). Limitation: language never conditions retrieval online; it is offline feature
  distillation, so it says nothing about handling an unseen utterance at request time.
- RecGPT Technical Report (arXiv:2507.22879, Taobao/Alibaba, 50+ authors). Replaces log-fitting with
  intent-centric design: LLMs do interest mining, item retrieval by predicted intent, and explanation generation;
  fully deployed on the Taobao App. Limitation: no public offline protocol; the reported wins are platform A/B,
  so the intent→retrieval mapping is not independently reproducible.
- RecGPT-V2 Technical Report (arXiv:2512.14503). Hierarchical multi-agent intent reasoning,
  constrained RL, Agent-as-a-Judge evaluation; +2.98% CTR, +3.71% IPV online. Limitation: LLM-judge
  metrics are themselves model-generated, i.e. the eval inherits the generator's priors.
- Aligning LLMs for Controllable Recommendations (Controllable-Rec, arXiv:2403.05063, ACL 2024).
  Explicitly documents that LLM recommenders are bad at *following instructions*, and fixes it with SFT labels
  distilled from a conventional recommender + RL alignment. Limitation, and what we take from it: the instruction
  signal is bootstrapped from the CF model, so instruction-following gains are bounded by the CF model it imitates.
- WhisperLite — Contrastive Learning for Interactive Recommendation in Fashion (arXiv:2207.12033, Amazon authors).
  Closest classical analogue of what this repository is building: user-provided free-form text request → personalised
  items, CLIP-style text encoder + personalization layers, composite BCE + contrastive loss. Eval: retrieval metrics on
  a real online-retail fashion dataset plus Amazon restaurant/movie/TV/clothing/shoe review sets, plus a user study.
  Limitation: text requests are paired to items users already engaged with (log-mined pairs), so intent coverage
  is whatever the logs already contain — a selection bias any synthetic-phrase pipeline must acknowledge.
- FilterLLM — Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation (arXiv:2502.16924).
  Instead of LLM-as-judge ("would this user like this?"), one inference predicts an item's interaction
  probability distribution over the whole user set — engineered for catalogue scale. Limitation: distribution is
  over *users*, not over a free-form utterance space; the phrase never enters the model.

## 2. Semantic item representations bridging cold-start

- Ask the GRU (arXiv:1609.02116, Bansal et al.). The canonical text→CF bridge: encode item text into
  a latent vector that stands in for a missing ID embedding, multi-task-trained jointly with CF; explicit cold-start
  gains. Limitation: 2016 encoder; semantic granularity far below a modern product catalogue.
- TIGER — Recommender Systems with Generative Retrieval (arXiv:2305.05065). RQ-VAE semantic IDs
  (codeword tuples) replace atomic item IDs; seq2seq generates the next item's SID; better generalisation for items with
  no interaction history. Limitation: generation is over SID space, so a novel *utterance* must first be
  projected into that space; retrieval quality for unseen-item phrasings is not the property that paper measures.
- LEARN (arXiv:2405.03988, AAAI 2025). LLM→item encoder adapted by a recommendation-supervised
  twin-tower while keeping open-world knowledge intact; state-of-the-art results on Amazon Review plus online A/B. Limitation:
  the tower is trained to reproduce behavioural similarity, so language is used to *warm* CF representations, not to be
  conditioned on at query time.
- Let It Go? Not Quite (arXiv:2507.19473). Item cold start in SASRec/BERT4Rec via content-based
  initialisation plus trainable deltas on frozen embeddings, so new items don't drift away from semantic structure.
  Limitation: sequential-recommendation setting only; no language query interface.
- Actions Speak Louder than Words (HSTU, arXiv:2402.17152, ICML 2024, Meta). Recommendation as
  sequential transduction; up to +65.8% NDCG, 5.3–15.2× faster than FlashAttention2 transformers, deployed at
  trillion-parameter scale. Limitation: the title is the thesis — behaviour tokens, not text, carry the signal;
  HSTU itself is a strong argument that language must *earn* its place against a behavioural sequence model.
- FLUID — From Ephemeral IDs to Multimodal Semantic Codes (arXiv:2605.21832, Kuaishou livestreaming).
  Cross-domain multimodal encoder whose semantic codes replace ephemeral item IDs inside production
  ranking; +0.55% Quality Watch Duration on a billion-user platform. Limitation: content codes help most
  where IDs are noisy/ephemeral; on a stable 40K-item catalogue the marginal gain may be much smaller than 0.55%.
- Can Generative Recommendation Reach Cold Items? (arXiv:2607.21101). Temporal protocol; reads SID
  generation as hierarchical semantic bucketing and measures whether cold items are actually reachable. Limitation, and what we take from it: the reachability ceiling is a property of the tokenizer, so "semantic ID solves cold start" is
  only true within the buckets the codebook can express.
- (Related, checked but not cited elsewhere: Cold-Starts in Generative Recommendation: A Reproducibility Study,
  arXiv:2603.29845 — unified cold-start protocols across GR models; OneRec-V2, arXiv:2508.20900; LLMTreeRec,
  arXiv:2404.00702, cold-start LLM recommendation in a tree over candidates, deployed with an A/B win on Huawei.)

## 3. E-commerce query understanding (the ESCI / Shopping Queries axis)

- Shopping Queries Dataset (ESCI) (arXiv:2206.06588). ~130K unique queries, ~2.6M human relevance
  judgements, EN/JA/ES, three tasks (ranking, ESCI classification, substitute detection); queries are deliberately
  *difficult*, i.e. chosen where lexical matching struggles. Limitation: judgements are per (query, product)
  without session/behaviour context, so a high ESCI score is not evidence of a better recommender.
- KDD Cup 2022 Task-1 winner (arXiv:2208.02958). Multilingual (m)BERT-style query–product
  alignment, augmentation + adversarial training + self-distillation + ensemble; NDCG 0.9043.
- A Boring-yet-effective Approach (arXiv:2208.06264). mT5 with minimal task-specific adaptation came
  within 0.004 nDCG@20 of the top submission; authors state the top-20 teams all clustered near 0.90 and argue the
  benchmark is too easy to discriminate retrieval methods. This is the ceiling number to calibrate expectations with.
- Beyond Relevance: Structured Semantic Supervision for Product Search (arXiv:2609.23646).
  Structured LLM-generated query + product attributes with human-validated signals on ESCI-style pairs;
  oracle 0.9382 nDCG@10, trained systems 0.9258. Limitation: oracle configuration uses LLM attributes at
  inference, so the gap to 0.9258 is mostly annotation cost, not modelling.
- Domain-Adaptive Dense Retrieval for Content-Based Recommendation (arXiv:2602.00899). Two-tower
  bi-encoder fine-tuned on Amazon Reviews Fashion: Recall@10 0.26 → 0.66, 6.1 ms CPU inference. Motivating
  finding: generic BM25 fails on vocabulary mismatch between casual descriptions and catalogue text — the practitioner
  statement of *why* you need a semantic encoder even when lexical search "works".
- LEAPS (arXiv:2601.05513, Taobao AI Search). The production answer to multi-constraint natural
  language queries ("broaden-and-refine": upstream LLM query expander, downstream relevance verifier), deployed
  since Aug 2025 serving hundreds of millions monthly. Limitation: the verifier is a second LLM call, so the
  cost is latency and a second failure mode, which the paper's headline omits.
- SMART (arXiv:2607.23121, Snap, dynamic product ads). Hybrid retrieval mixing rule-generated
  (lexical) and LLM-generated queries with adaptive routing; +27.6% ad conversion in A/B. The practitioner
  lexical-vs-embedding result: lexical/rule queries still win where intent is exact-match/retargeting, LLM queries win on
  vague intent — routing, not replacement. Limitation: the +27.6% is on the routed slice, not sitewide.
- Semantic Retrieval for Product Search in E-Commerce (arXiv:2606.01504, Kothari et al.). Siamese LLM
  dual-encoder in two stages: contrastive training with a false-negative margin mask, then ROAR, a Bradley-Terry
  extension over "variable-sized graded relevance groups via consecutive odds-ratio margins". The stage that
  matters here: "substitute query-product pairs provide coarse semantic supervision in Stage 1" — external
  confirmation that ESCI Substitute judgements function as target-reachable positive pairs for retrieval
  training, which is the inclusion this repository's query bank makes. Gains "confirmed across query-frequency
  strata and business verticals" and "validated through live A/B deployment at scale". Limitation: industrial
  setting, no public benchmark numbers in the abstract; the graded-relevance ordering is a search-ranking result,
  not a comparison against a behavioural recommender under a temporal split.
- ClardRec — Counterfactual Learning-Driven Representation Disentanglement for Search-Enhanced
  Recommendation (arXiv:2411.18631, Cui et al.). The transfer caution for the query→item axis: "search
  activities provide additional insights into user interest through query-click interactions", but interacted
  items "also have features related to users' unique intents in the search domain", so direct incorporation
  "introduce[s] partial negative transfer"; counterfactual signals built from search queries disentangle
  query-independent item features that then augment CF and sequential recommenders. Limitation: the query
  signal is logged query-click behaviour, not request-time wording, and the abstract names no datasets or
  backbones — the transfer claim cannot be checked beyond "real datasets".
- OSPO — Owen-Shapley Policy Optimization (arXiv:2601.08403, Nath et al.). Token-level credit assignment
  (Owen/Shapley attribution as potential-based reward shaping) for RL-trained generative search LLMs doing
  "personalized recommendation" from under-specified language; evaluated "on Amazon ESCI and H&M Fashion
  datasets". Limitation: preprint with no venue; ESCI serves here as evaluation currency for a generative
  retriever, not as a temporal recommender comparison, so it does not touch the paired-gate question.
- (Also checked but not cited: BEQUE long-tail query rewriting deployed on Taobao since Oct 2023, arXiv:2311.03758;
  JD's multi-task multi-stage LLM query rewriting grounded in relevance, ICDE 2026, arXiv:2603.02555; GRIT + the
  task-oriented-query benchmark over ESCI, +6.3% recall, arXiv:2504.05310.)

## 4. Preference elicitation — the measured value of asking

- Combating the Cold Start User Problem in Model-Based CF (arXiv:1703.00397). Formalises the exact
  question this repository needs answered: *which b items should we ask about* to learn a cold profile, proves NP-hardness, gives
  heuristics and measures NDCG/recommendation gain as a function of b. Limitation: offline simulation on MovieLens-
  style data; no cost model for asking, and user willingness is assumed.
- Minimizing Live Experiments in Recommender Systems (arXiv:2409.17436). Evaluates onboarding
  preference-elicitation policies for YouTube Music with counterfactually-robust user-simulation behaviour models
  instead of live A/B. Limitation: policy ranking is only as trustworthy as the simulator's counterfactuals.
- Selection Bias in Preference Elicitation (arXiv:2405.00554). First study of bias introduced by the
  elicitation interaction itself, with a simulation-based evaluation framework and debiasing methods. Limitation:
  bias is quantified in simulation, not with real onboarding traffic.
- (Checked but not cited, same theme: personalized embedding-region elicitation at cold start, UAI 2024,
  arXiv:2406.00973; hierarchical key-term vs item questions with bandit feedback, arXiv:2209.06129; explicit question
  budget for cold-start personalization, arXiv:2602.15012; LLM clarifying-question generation, arXiv:2510.12015.
  No 2024–2026 paper was found that reports a real A/B effect size for elicitation on a production recommender;
  the field has moved to simulated evaluation, which is itself a finding.)

## 5. Evaluating language-conditioned recommendation when the corpus has no typed queries

- CASTLE (arXiv:2605.21812, CIKM 2026, Airbnb, DOI 10.1145/3799682.3841071). The template for a query-less launch. Mechanism: structure-guided prompting over structured catalogue data + ~500
  curated survey seed queries, ~10,000 synthetic queries/day (20× amplification), contrastive pairs mined from real
  booking sessions (~5,000 pairs) as the supervision source; ~2 h for 10K queries. Headline: query-length
  KL divergence to real users 1.01 vs real, a 9.2× improvement over the best baseline (InPars and Promptagator
  collapse to near-zero length spread σ≈0.99; a contrastive-only variant degenerates to 97.7% 9+-word queries,
  σ=5.28); human evaluation 91–93% annotator agreement on labels. Accepted mitigations it names explicitly:
  leakage guardrails + difficulty buckets + hierarchical pair sampling with controlled similarity ("uniformly high
  accuracy provides no signal"), KL-divergence QC gate vs the seed/real query distribution, attribute-coverage floor,
  LLM-as-judge label-consistency sampling, cosine-distance dedup, and a cold→warm transition gated on KL(real‖synthetic).
  Limitation: it needed ~500 real survey-seeded queries as an anchor — a corpus with *zero* real queries has no
  distribution to align to and no KL gate to run.
- Tip-of-the-Tongue Query Elicitation for Simulated Evaluation (arXiv:2502.17776, SIGIR 2025). The accepted validation instrument: generate queries from the target entity (LLM prompting
  sweep) and from humans placed in a ToT state with visual stimuli, then validate by system rank correlation —
  rank 40 retrieval models on CQA-derived queries vs elicited queries (MRR@1000, NDCG@1000) and check agreement with
  Kendall's τ / Pearson. Queries released for TREC 2024/2025 ToT tracks (Movie, Landmark, Person). Limitation: validity is *relative-system ranking agreement* — it licenses "A beats B", never "absolute NDCG = x", and
  ToT queries are descriptive-entity queries, a different distribution from shopping intent.
- "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion
  (arXiv:2608.25245, CIKM 2026). Mechanism: a concept-provenance framework that classifies each
  concept in a simulated query as legitimate (from the user-side backstory) vs answer-side intrusion. Headline:
  across 77,004 queries from 8 LLMs, candidate-answer concepts appear in 97 of 100 topics; human initial queries
  almost never contain them. Limitation: measures intrusion, does not fully remove it; also cites a taxonomy of
  simulation-validation measures (Kruff et al., ECIR 2026) that could not be fetched here.
- Synthetic Test Collections for Retrieval Evaluation (arXiv:2405.07767, SIGIR 2024). Builds fully
  synthetic collections (queries *and* qrels) with LLMs; documents potential bias favouring LLM-based systems.
  Limitation: "reliable" is shown by agreement with real collections on leaderboards, so it inherits the
  real collection's task definition.
- Towards Understanding Bias in Synthetic Data for Evaluation (arXiv:2506.10301, CIKM 2025, Google).
  Empirical + linear mixed-effects analysis: bias in synthetic test collections is significant for absolute metrics,
  while relative system comparisons remain comparatively robust. This is the single most decision-relevant result
  for this repository: it licenses a paired relative gate on synthetic phrases and forbids reporting absolute NDCG from them.
- LLM Augmented Narrative Driven Recommendations (NDR, arXiv:2306.02250, RecSys 2023). Generates
  synthetic narrative/utterance queries from existing user–item interactions with few-shot prompting, trains a small
  retriever on them; the small retriever beats retrieval and LLM baselines. Limitation: the synthetic queries are
  derived *from* the interaction they are meant to predict, i.e. the target is visible to the generator by construction.
- Rethinking the Evaluation for Conversational Recommendation in the Era of LLMs (iEvaLM, arXiv:2305.13112,
  EMNLP 2023). Shows the standard protocol (match human-annotated ground-truth items/utterances) is
  inadequate because many items are acceptable; proposes LLM user-simulator-based evaluation. Limitation:
  LLM-simulator judgements are the same family of artefacts the protocol is trying to escape.
- Retrieval-Augmented Conversational Recommendation (arXiv:2406.00033, DOI 10.1145/3626772.3657670).
  Handles indirect preference utterances ("I'm watching my weight", "classy joint for a date") by LLM dialogue-state
  tracking + retrieval over item reviews as the bridging text. Limitation: requires rich review text per item;
  catalogue coverage becomes the binding constraint.

### Traps, each cited above
1. Target-visible generation — the generator sees the item whose held-out interaction it is generating, so rare
   titles/tokens leak into the phrase (NDR is the canonical case; CASTLE adds explicit leakage guardrails).
2. Answer-side concept intrusion — measurable, and it happens in 97/100 topics (2608.25245).
3. Absolute-metric bias / self-preference — synthetic collections inflate absolute scores and favour LLM-based
   systems; only relative comparisons survive (2506.10301, 2405.07767).
4. Generator–judge circularity — same model family writes queries, labels, and (in RecGPT-V2 style) judges (2512.14503).
5. Distributional degeneration of generated queries — length/diversity collapse vs real query distribution
   (CASTLE's InPars / Promptagator / contrastive-only table).
6. Many-valid-answers ambiguity — a phrase legitimately matches substitutes and complements; ESCI's label taxonomy
   exists for this, and the KDD-Cup ceiling (~0.90 nDCG@20 for the top 20 teams) shows ambiguous labels compress
   discrimination (2208.06264).
7. Train/test contamination from LLM priors — a generator with world knowledge of Amazon titles behaves differently
   on popular (well-known to the LLM) vs obscure catalogue items; popularity and "LLM knows it" are confounded.
8. Temporal realism — synthetic phrases carry no novelty/trend dynamics, so a temporal split can silently become
   easier than production (2607.21101's temporal protocol is the fix).
9. Ground-truth utterance matching in conversational eval penalises valid alternatives (2305.13112).

## 6. Speech input in RecSys

- Towards Building Voice-based Conversational Recommender Systems (arXiv:2306.08219, SIGIR 2023 Resource track,
  DOI 10.1145/3539618.3591876). The only dedicated RecSys-side voice-CRS resource found: two benchmarks
  (e-commerce, movies) built by turning interactions into ChatGPT-drafted conversations and then TTS-synthesising
  audio (VCRS, code released). Limitation: the "voice" data is synthesised text, so it measures nothing about
  real ASR error, disfluency, or prosody.
- An End-to-End ML System for Personalized Conversational Voice Models in Walmart E-Commerce (arXiv:2011.00866).
  Deployed personalization for voice shopping (Google Assistant, Siri, Google Home) using implicit
  feedback. Limitation: personalization is applied to the *ranking/selection layer*; the paper's contribution is
  the serving system, not a measured phrase-understanding gain.
- Verdict: there is no substantive 2024–2026 RecSys literature in which the speech channel itself changes
  recommendation quality. Adjacent IR work treats ASR as noise on retrieval (e.g. spoken-content retrieval MRR drops
  of 10–14% under ASR vs human transcripts, arXiv:2301.06056; AVATAR, arXiv:2309.01395) — i.e. engineering guidance:
  treat speech as a transcript-producing front end and keep the recommendation question text-only.

---

## 7. How large platforms wire this, on the entries above alone

Two patterns, not one:

- Pattern A — language offline, behaviour online. LLM/CLIP text becomes *features, initialisations, or
  distributions* consumed by a conventional behavioural ranker: KAR (reasoning + factual knowledge → CTR features,
  +7%/+1.7% A/B), REKI-style factorization prompting, LEARN (twin-tower item encoder), content-based initialisation with
  frozen-embedding deltas, FilterLLM (text → interaction distribution), LLMTreeRec (Huawei A/B win). Gains are single-digit
  or sub-1% online deltas (FLUID +0.55% QWD). This pattern fits a core that installs no dependencies: language never sits on the request path.
- Pattern B — language online, as an intent interface. RecGPT/RecGPT-V2 (intent mining → tag/intent-driven
  retrieval → explanation; +2.98% CTR, +3.71% IPV; Agent-as-a-Judge), OneRec-family end-to-end semantic-ID generation,
  Taobao AI Search LEAPS (expand multi-constraint phrases, then verify relevance), RA-Rec (indirect utterance → review
  text retrieval). This is where "phrase → products" lives, and it costs an LLM in the loop plus a verifier.
- The unglamorous middle that actually ships phrase search: query understanding → dual-encoder dense retrieval over
  item text, with lexical kept on purpose (SMART's adaptive rule-vs-LLM routing; BEQUE/JD for long-tail rewriting;
  domain-adaptive two-tower to fix vocabulary mismatch: Recall@10 0.26→0.66), ESCI-style labels as the training/eval
  currency, and the existing behavioural ranker (HSTU/KuaiFormer-class) still deciding the final order.
- Unseen-user utterance driving retrieval is enabled by item-side semantics (TIGER SIDs, LEARN, FLUID, content-init)
  — but SID/generative reachability has a measured ceiling (2607.21101, reproducibility study 2603.29845), so platforms
  keep content embeddings *inside* the ranker rather than trusting generated IDs alone.
- No paper above reports a phrase-conditioned ranker beating a strong behavioural hybrid on a corpus with no query
  log. Where numbers exist, they are production A/B deltas or ESCI-style nDCG. Consequence for this repository: the paired
  relative NDCG@10 gate is the correct instrument; absolute synthetic-phrase NDCG is not reportable.

## 8. Claims that could not be checked here

- "TC-RRec" — could not be located. `abs:"TC-RRec"` and `all:"text-conditioned recommendation"` in the arXiv API
  both returned 0 results; web search returned unrelated hits. Treat the name as unchecked and do not build on it
  without a DOI or URL for the work itself.
- "RecGPT" is ambiguous: at least three distinct works share the name (2507.22879 Taobao; a text-based "RecGPT-7B",
  commonly cited as 2405.12715; a chat-style sequential "RecGPT", commonly 2404.08675). Only the Taobao V1/V2 were
  fetched here; the other two IDs were not checked and are excluded.
- M5 / Walmart Shopping Queries Dataset numbers — not fetched; only the Amazon ESCI line was checked.
- Exact BM25 / lexical baseline scores on ESCI — not checked (only the dense-vs-lexical qualitative tradeoff, and
  the SMART routing result).
- Intrusion-rate and nDCG-inflation percentages for 2608.25245 beyond "77,004 queries / 8 LLMs / 97 of 100 topics".
- Kruff, Bernard & Schaer, "Validating search query simulations: a taxonomy of measures" (ECIR 2026) — seen only as
  a bibliography entry inside 2608.25245 (secondary sighting; not fetched).
- Multilingual extension of the ToT work (arXiv:2604.21096) — index-search hit only, not fetched; excluded.
- Vendor engineering-blog evidence for lexical-vs-embedding in production — none verifiable: ACM DL returned HTTP
  403 to the fetch tool, firecrawl scrape was blocked by exhausted credits, and Semantic Scholar rate-limited (429).
  The synthesis above therefore rests on papers with deployment sections, not blog posts.
- Any claim that a production recommender ran a real A/B test of preference-elicitation questions in 2024–2026 —
  not found; the surveyed elicitation work evaluates by simulation.

## 9. What this corpus can and cannot measure

Measurable now (no typed queries needed):
- Challenger-vs-baseline on real behaviour: a phrase-conditioned two-tower (or text-initialised item representation)
  scored against the existing 0.008681 hybrid and 0.008007 popularity baseline on the same temporal split, with the
  paired bootstrap gate. Text is just another side feature, so Pattern-A work is directly testable.
- Cold-item reachability and cold-user first-request NDCG@10 under the existing temporal split (2607.21101 protocol).
- Phrase→attribute parsing accuracy: does the utterance resolve to the right catalogue category/attribute? Uses the
  catalogue's own metadata as labels, no queries required.
- A *constructed* phrase test collection, used only for relative comparisons (2506.10301). Build phrases from
  training-window evidence — co-review/neighbour items or review text — rather than from the target item, then
  validate with the rank-correlation instrument: rank several internal scorers on synthetic phrases and on a
  behaviour-derived phrase set and require agreement (Kendall τ) before trusting any comparison (2502.17776 method).
- Value-of-asking curves without queries: pick k questions (items or attributes), answer them from the *future*
  window's known labels, re-rank, and plot ΔNDCG@10 per question (1703.00397 formalisation); simulate the elicitation
  policy comparison as in 2409.17436 and audit it for the selection-bias artefacts of 2405.00554.
- Guardrail diagnostics: phrase length/diversity distribution, dedup rate, attribute-coverage floor, LLM-judge label
  agreement on a sample (CASTLE's QC list, minus the KL gate).

Not measurable here (would need real query logs, real users, or audio):
- Distributional fidelity to real user phrasing — CASTLE's KL-to-real-users 1.01 and its 9.2×-better-than-baseline
  claim are only computable with a real query/seed distribution; this corpus has none, so "our synthetic phrases look
  real" is not a claim this repository can make.
- Absolute production NDCG implied by synthetic-phrase scores (2506.10301 forbids it).
- Lexical-vs-semantic routing value (SMART-style gains) — needs traffic where intent is genuinely lexical/retargeting.
- Substitute-vs-complement economics (ESCI's useful distinction) — needs human relevance labels; implicit signals
  cannot separate "substitute" from "complement".
- Speech/ASR effects, disfluency, latency-of-utterance — no audio in corpus.
- User willingness to type/speak a phrase, and preference for the phrase interface — requires a user study; nothing
  in the elicitation literature substitutes for it.

## Sources

Each identifier below was resolved against the arXiv export API on 23 September 2026, in the
order the papers appear in this note, and the title that API returned is reproduced as the link
text so a reader can check that a claim is attached to the paper it came from. The last three
entries (2606.01504, 2411.18631, 2601.08403), added the same day after the query-bank scale-up,
were resolved against their arxiv.org/abs pages instead — the export API refused all listed-search
requests that day — and their quoted sentences were checked against the abstract text.
- [Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach](https://arxiv.org/abs/2305.07001)
- [Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System](https://arxiv.org/abs/2303.14524)
- [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models](https://arxiv.org/abs/2306.10933)
- [RecGPT Technical Report](https://arxiv.org/abs/2507.22879)
- [RecGPT-V2 Technical Report](https://arxiv.org/abs/2512.14503)
- [Aligning Large Language Models for Controllable Recommendations](https://arxiv.org/abs/2403.05063)
- [Contrastive Learning for Interactive Recommendation in Fashion](https://arxiv.org/abs/2207.12033)
- [FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation](https://arxiv.org/abs/2502.16924)
- [Ask the GRU: Multi-Task Learning for Deep Text Recommendations](https://arxiv.org/abs/1609.02116)
- [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- [LEARN: Knowledge Adaptation from Large Language Model to Recommendation for Practical Industrial Application](https://arxiv.org/abs/2405.03988)
- [Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization](https://arxiv.org/abs/2507.19473)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)
- [FLUID: From Ephemeral IDs to Multimodal Semantic Codes for Industrial-Scale Livestreaming Recommendation](https://arxiv.org/abs/2605.21832)
- [Can Generative Recommendation Reach Cold Items? A Temporal Perspective on Semantic-ID Generation](https://arxiv.org/abs/2607.21101)
- [Cold-Starts in Generative Recommendation: A Reproducibility Study](https://arxiv.org/abs/2603.29845)
- [OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900)
- [LLMTreeRec: Unleashing the Power of Large Language Models for Cold-Start Recommendations](https://arxiv.org/abs/2404.00702)
- [Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://arxiv.org/abs/2206.06588)
- [A Semantic Alignment System for Multilingual Query-Product Retrieval](https://arxiv.org/abs/2208.02958)
- [A Boring-yet-effective Approach for the Product Ranking Task of the Amazon KDD Cup 2022](https://arxiv.org/abs/2208.06264)
- [Beyond Relevance: Structured Semantic Supervision for Product Search with LLM-Augmented Annotations](https://arxiv.org/abs/2609.23646)
- [Domain-Adaptive and Scalable Dense Retrieval for Content-Based Recommendation](https://arxiv.org/abs/2602.00899)
- [LEAPS: An LLM-Empowered Adaptive Plugin in Taobao AI Search](https://arxiv.org/abs/2601.05513)
- [SMART: LLM-Augmented Hybrid Retrieval for Dynamic Product Ads](https://arxiv.org/abs/2607.23121)
- [Semantic Retrieval for Product Search in E-Commerce](https://arxiv.org/abs/2606.01504)
- [Counterfactual Learning-Driven Representation Disentanglement for Search-Enhanced Recommendation](https://arxiv.org/abs/2411.18631)
- [Owen-Shapley Policy Optimization: A Principled RL Algorithm for Generative Search LLMs](https://arxiv.org/abs/2601.08403)
- [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)
- [Relevance Matters: A Multi-Task and Multi-Stage Large Language Model Approach for E-commerce Query Rewriting](https://arxiv.org/abs/2603.02555)
- [GRIT: Graph-based Recall Improvement for Task-oriented E-commerce Queries](https://arxiv.org/abs/2504.05310)
- [Combating the Cold Start User Problem in Model Based Collaborative Filtering](https://arxiv.org/abs/1703.00397)
- [Minimizing Live Experiments in Recommender Systems: User Simulation to Evaluate Preference Elicitation Policies](https://arxiv.org/abs/2409.17436)
- [A First Look at Selection Bias in Preference Elicitation for Recommendation](https://arxiv.org/abs/2405.00554)
- [Cold-start Recommendation by Personalized Embedding Region Elicitation](https://arxiv.org/abs/2406.00973)
- [Hierarchical Conversational Preference Elicitation with Bandit Feedback](https://arxiv.org/abs/2209.06129)
- [Cold-Start Personalization via Training-Free Priors from Structured World Models](https://arxiv.org/abs/2602.15012)
- [Asking Clarifying Questions for Preference Elicitation With Large Language Models](https://arxiv.org/abs/2510.12015)
- [CASTLE: Contrastive and Seed-Guided Training for Cold-Start Natural Language Search](https://arxiv.org/abs/2605.21812)
- [Tip of the Tongue Query Elicitation for Simulated Evaluation](https://arxiv.org/abs/2502.17776)
- [The "Curse of Knowledge" in LLM Query Simulation: Concept Provenance for Tracing Answer-Side Intrusion](https://arxiv.org/abs/2608.25245)
- [Synthetic Test Collections for Retrieval Evaluation](https://arxiv.org/abs/2405.07767)
- [Towards Understanding Bias in Synthetic Data for Evaluation](https://arxiv.org/abs/2506.10301)
- [Large Language Model Augmented Narrative Driven Recommendations](https://arxiv.org/abs/2306.02250)
- [Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models](https://arxiv.org/abs/2305.13112)
- [Retrieval-Augmented Conversational Recommendation with Prompt-based Semi-Structured Natural Language State Tracking](https://arxiv.org/abs/2406.00033)
- [Towards Building Voice-based Conversational Recommender Systems: Datasets, Potential Solutions, and Prospects](https://arxiv.org/abs/2306.08219)
- [An End-to-End ML System for Personalized Conversational Voice Models in Walmart E-Commerce](https://arxiv.org/abs/2011.00866)
- [Improving Noise Robustness for Spoken Content Retrieval using Semi-supervised ASR and N-best Transcripts for BERT-based Ranking Models](https://arxiv.org/abs/2301.06056)
- [AVATAR: Robust Voice Search Engine Leveraging Autoregressive Document Retrieval and Contrastive Learning](https://arxiv.org/abs/2309.01395)
- [Multilingual and Domain-Agnostic Tip-of-the-Tongue Query Generation for Simulated Evaluation](https://arxiv.org/abs/2604.21096)
