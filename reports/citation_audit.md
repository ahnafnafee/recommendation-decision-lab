# Manuscript citation audit

Audit date: 2026-09-23. Scope: the 26 references cited in `paper/main.tex` and the manuscript sentences that cite them. This is a source-and-claim audit, not an independent replication of the cited studies or of this repository's experiments. A previously reported check that 140 survey URLs resolved established link availability only; it did not establish that each citation supported its neighboring claim.

All 26 citation keys are defined once, all 26 bibliography entries are cited, and none is duplicated. The five ACM DOIs in the bibliography were checked against Crossref metadata; the added Springer and AAAI DOIs were checked there too. For the other entries, the linked owner, publisher, or author-hosted source was checked for title, author order, publication details, and the proposition used in the manuscript. The manuscript's numerical statement about production RAG routing was checked against the source abstract.

| Key | Source checked | Claim-context assessment |
| --- | --- | --- |
| `amazon5core` | [McAuley Lab 5-core documentation](https://amazon-reviews-2023.github.io/data_processing/5core.html) | Supports the supplied history field, absolute-timestamp split, cutoffs, and category files. |
| `hou2026bridging` | [Hou et al., BLaIR](https://arxiv.org/html/2403.03952v2) | Supports the Amazon Reviews 2023 dataset provenance; the specific 5-core split is documented by the owner page above. |
| `ji2023leakage` | [Ji et al., TOIS](https://doi.org/10.1145/3569930) | Supports global-timeline leakage and distorted model comparisons. |
| `gusak2025time` | [Gusak et al., RecSys](https://doi.org/10.1145/3705328.3748164) | Supports the importance of temporal split and validation construction for sequential recommendation. |
| `covington2016youtube` | [Covington et al., RecSys](https://doi.org/10.1145/2959100.2959190) | Supports learned candidate generation as an established production technique. |
| `wang2021crossbatch` | [Wang et al., SIGIR](https://doi.org/10.1145/3404835.3463032) | Supports in-batch and cross-batch negative sampling for two-tower training. |
| `rashid2008preferences` | [Rashid et al., author-hosted paper](https://kdd.org/exploration_files/WebKDD08-Al-Rashid.pdf) | Supports preference elicitation for new users without preference history. The manuscript sentence was narrowed to this direct claim. |
| `crankshaw2016clipper` | [Crankshaw et al., Clipper](https://arxiv.org/abs/1612.03079) | Supports low-latency model serving, batching, caching, and model-selection tradeoffs. |
| `hendrickx2021reject` | [Hendrickx et al., journal article](https://doi.org/10.1007/s10994-024-06534-x) | Supports abstention as a separate decision and the ambiguity/novelty distinction. |
| `hemmer2023defer` | [Hemmer et al., AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25742) | Supports learning to defer using information about the expert's capability. |
| `krauth2020offline` | [Krauth et al., arXiv](https://arxiv.org/abs/2011.07931) | Supports correlation between offline and online metrics with limits due to user-system dynamics. |
| `gilotte2018offline` | [Gilotte et al., KDD preprint](https://arxiv.org/abs/1801.07030) | Supports bias/variance compromises in counterfactual estimates for personalized recommendation. |
| `hsu2024minimizing` | [Hsu et al., arXiv](https://arxiv.org/abs/2409.17436) | Supports simulated evaluation of onboarding preference-elicitation policies. |
| `gupta2024selection` | [Gupta et al., arXiv](https://arxiv.org/abs/2405.00554) | Supports selection bias introduced during preference elicitation. The 2023 workshop date and 2024 preprint date are distinct. |
| `huang2020ebr` | [Huang et al., KDD preprint](https://arxiv.org/abs/2006.11632) | Supports embedding-based retrieval in large-scale search. |
| `reddy2022esci` | [Reddy et al., arXiv](https://arxiv.org/abs/2206.06588) | Supports the existence of manually labeled product-query relevance judgments. |
| `reimers2019sbert` | [Reimers and Gurevych, arXiv](https://arxiv.org/abs/1908.10084) | Supports separately computed sentence embeddings. Encoding catalog text before query arrival is the manuscript's engineering inference, not an experiment reported by this source. |
| `rahmani2024synthetic` | [Rahmani et al., SIGIR preprint](https://arxiv.org/abs/2405.07767) | Supports caution about synthetic retrieval test collections and absolute scores. |
| `rahmani2025bias` | [Rahmani et al., CIKM preprint](https://arxiv.org/abs/2506.10301) | Supports the distinction between bias in absolute system performance and potentially smaller effects on relative comparisons. It does not validate this manuscript's constructed phrases. |
| `hussain2026coverage` | [Hussain and Nielbo, arXiv](https://arxiv.org/abs/2605.27220) | Supports the >90% synthetic versus 27.8% production-traffic routing comparison. The manuscript now identifies the setting as RAG. |
| `meng2020splitting` | [Meng et al., paper](https://arxiv.org/pdf/2007.13237) | Supports altered rankings under different splits and explicitly recommends temporal global splitting as the default. |
| `hidasi2023flaws` | [Hidasi and Czapp, author-hosted paper](https://hidasi.eu/assets/pdf/eval_flaws_recsys23.pdf) | Supports temporal leakage, negative-item-sampling flaws, and full-catalog ranking. |
| `schnabel2022here` | [Schnabel, arXiv](https://arxiv.org/pdf/2211.01261) | Supports careful baselines, hyperparameter search reporting, and uncertainty estimates. Its 40-iteration recommendation concerns Bayesian optimization; this manuscript discloses a much smaller fixed search. |
| `pan2023sine` | [Pan et al., RecSys preprint](https://arxiv.org/abs/2308.04086) | Supports passive-negative feedback from skipped autoplay videos; not treated as equivalent to a low product rating. |
| `wang2023negative` | [Wang et al., RecSys preprint](https://arxiv.org/abs/2308.12256) | Supports the use of explicit and implicit negative feedback in sequential retrieval. |
| `ji2020popularity` | [Ji et al., SIGIR preprint](https://arxiv.org/abs/2005.13829) | Supports a difference between lighter and heavier MovieLens users in following popularity versus individual taste. |

Edits made after the audit: corrected an ungrammatical and overly broad cold-start sentence, specified that the 2026 routing comparison was from a RAG system, supplied direct URLs for three arXiv references, and completed publisher metadata for the SIGIR, *Machine Learning*, and AAAI entries. No numeric result, conclusion, or experimental claim changed. The cited literature motivates design and interpretation; it does not independently validate this study's held-out results or justify transferring effects across platforms.
