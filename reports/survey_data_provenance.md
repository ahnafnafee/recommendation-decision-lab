# Data provenance: what the published record says this corpus is

Scope: the data layer - what the Amazon Reviews'23 corpus is and how it was assembled and filtered, where the 5-core / leave-one-out / last-interaction protocol this split inherits from, why a
timestamp-ordered global split rather than a random one, the published record on cold targets and catalogue churn, what the text sidecars are and where each part comes from, and what the dataset
authors state about usage terms. The measurement-apparatus survey ([survey_evaluation_method.md](survey_evaluation_method.md)) carries the randomisation and resampling sides; repository numbers are
quoted from README.md, reports/transfer_note.md, reports/shadow_profile_design.md, reports/survey_reliable_serving.md and the tools docstrings.

## 1. The data layer of this repository

`reliability/benchmark.py` fixes two absolute cutoffs, `T1 = 1628643414042` and `T2 = 1658002729837`, over the official 5-core `timestamp_w_his` archives of the Amazon Reviews'23 benchmark for
Musical_Instruments and Video_Games, fetched as `data/{Category}.{train,valid,test}.csv.gz`; boundary crossings raise, the archives are hash-checked before the test file opens, and a challenger
activates only through a paired user-cluster bootstrap (1000 draws, seed 20260922) whose 95% lower percentile beats zero. The interaction archives carry ratings only, so every word that describes a
product comes from the dataset's own snapshot: `tools/text_corpus.py` materialises a review-prose stream (headline and body, at most 12 reviews of 240 characters each, per item and person), a
listing-text stream (title, features, description, categories, details, brand, price) and two query-pair streams from published sources into the Git-ignored `data/text/`; `tools/build_text_index.py`
encodes the catalogue's own product text so a phrase can be matched to it. The measured cohort is 35,905 eligible positive-review requests on Musical Instruments (README.md) and 35,562 on Video Games
(reports/transfer_note.md).

## 2. The corpus and its assembly

### 2.1 Where Amazon Reviews'23 comes from

The project page (amazon-reviews-2023.github.io) describes the corpus as "a large-scale Amazon Reviews dataset, collected in 2023 by McAuley Lab" and lists the release claims: "Larger Dataset: We
collected 571.54M reviews, 245.2% larger than the last version", "Newer Interactions: Current interactions range from May. 1996 to Sep. 2023", "Fine-grained Timestamp: Interaction timestamp at the
second or finer level". Its comparison table gives 34.69M reviews (2013), 82.83M (2014), 233.10M (2018) and 571.54M (2023), with 54.51M users, 48.19M items and 33 domains in 2023. The accompanying
paper (arXiv:2403.03952) introduces the corpus as "a new large-scale Amazon Reviews 2023 dataset with over 570 million reviews and 48 million items" and reports 571,544,897 reviews, 54,514,264 users
and 48,185,153 items, against 233,055,327 reviews and 15,167,257 items in the 2018 version.

### 2.2 The metadata side

The stated motivation is data quality: the earlier releases "were released several years ago (last released in 2018) and contain noisy metadata"; the new release "contains 3.18x more items and 2.58x
more text tokens in reviews and item metadata than the 2018 version", adds "new data ranging from Oct 2018 to Sep 2023", re-parses the HTML product pages "into structured JSON format, resulting in
metadata with more descriptive fields (e.g., item descriptions and features)", and moves timestamps to "millisecond precision" (arXiv:2403.03952). The metadata side, however, does not cover the whole
catalogue: the project page counts items "based on user reviews rather than item metadata files" and notes "some items lack metadata", while the paper's statistics table lists 35,393,189 items with
metadata against 48,185,153 items counted from reviews - roughly a quarter of items carry no metadata at all, so the listing-text sidecar is necessarily thin for that long tail.

### 2.3 The predecessor corpus and its filtering

The 2018 predecessor is documented at nijianmo.github.io/amazon: "an updated version of the Amazon review dataset released in 2014", "The total number of reviews is 233.1 million (142.8 million in
2014)", "reviews in the range May 1996 - Oct 2018". Its processing defines the conventions the 2023 release keeps: the 5-core subset in which "all users and items have at least 5 reviews (75.26
million reviews)", and an aggressive de-duplication pass that removes "duplicates even if they are written by different users. This accounts for users with multiple accounts or plagiarized reviews.
Such duplicates account for less than 1 percent of reviews". The page's only stated usage condition is "Please cite the following paper if you use the data in any way" - Ni, Li, McAuley (EMNLP 2019),
whose abstract frames the contribution as a recommendation-justification task. Note the version counts differ between the two pages: the 2023 comparison table lists 82.83M reviews for 2014, while the
2018 data page recalls 142.8 million.

Taken from this: the split this repository evaluates is the 5-core, de-duplicated, pure-ID subset of a scraped marketplace corpus, and both the filtering and the de-duplication are choices the authors
state explicitly rather than defaults the data arrives with.

## 3. The protocol: 5-core, leave-one-out, last interaction

### 3.1 Where each part comes from

Leave-one-out was already the standard item-recommendation protocol by 2017: NCF (arXiv:1708.05031) reports "We adopted the leave-one-out evaluation, which has been widely used in literature. For each
user, we held-out her latest interaction as the test set and utilized the remaining data for training". NCF's own filtering is a 20-core on users ("each user has at least 20 ratings") and its
evaluation ranks against a sampled candidate set ("randomly samples 100 items"), so the 5-core variant does not come from it. The 5-core for this corpus family is defined in the 2018 data page and
carried into 2023, where the project page supplies the pure-ID files "with '5-core' and 'de-duplication' processing" and states the trade-off: "Pros of 5-Core: Higher quality reviews, Reduced noise,
Balanced distribution and Computational efficiency. Cons of 5-Core: Limited diversity, Misalignment with original data distribution, Loss of context, Generalizability and Limited data size for scaling
up." The last-interaction split and the global cutoffs are defined on the same page: the leave-last-out variant keeps "the first N-2 items" for training, the "(N-1)-th" for validation and the "N-th"
for test ("This strategy is widely used in many recommendation papers"), while the absolute-timestamp variant cuts "item sequences for training and evaluation" at "t_1 = 1628643414042" and "t_2 =
1658002729837", with "The history field in the processed files represents the historical user interactions before the current timestamp" - exactly the `timestamp_w_his` layout this repository
consumes. The companion paper applies the same two cutoffs to the unfiltered data (arXiv:2403.03952): "To preserve the natural distribution, we do not filter out any users or items based on
interaction counts", and the cutoffs "split all the reviews in Amazon Reviews 2023 in a ratio of 8:1:1".

### 3.2 How sensitive results are to the choice

The data-side record says the protocol is not a neutral container. Sun (arXiv:2210.04149) re-examines the "commonly used train/test data splits" and "discuss[es] why the popularity baseline is poorly
defined under such splits": in a case study of 88 papers from the ACM RecSys conferences, 59.1% do not maintain a global timeline, so "the popularity baseline is 'forced' to use all interactions in
training set" as "a static ranking, covering the entire duration of the training data", and "recommending items that are popular in future by using the frequency counting that happened in future is
unrealistic". Krichene and Rendle (DOI 10.24963/ijcai.2021/651) show that sampled metrics "do not persist relative statements, e.g., recommender A is better than B, not even in expectation", and
"sampling should be avoided for metric calculation". Dallmann, Zoller and Hotho (arXiv:2107.13045) find that "both sampling strategies can produce inconsistent rankings compared with the full ranking
of the models", with leave-one-out "as is common in most related work". Limitation: "we only used the common leave-one-out dataset split that is often used with review datasets and did not consider
alternative splits like a split by time".

Taken from this: each knob - the core filter, the split shape, the target-set size - moves model rankings on its own, so a measurement pinned to the 5-core, leave-last-out, absolute-timestamp layout
inherits both the authors' stated trade-offs ("Limited diversity, Misalignment with original data distribution") and the published evidence that other choices would have ranked the systems
differently.

## 4. Why a timestamp-ordered global split

### 4.1 The stated rationale

The project page calls the absolute-timestamp split a strategy that "aligns with real-world scenarios but is not widely used in research", and "Researchers are encouraged to experiment with this
splitting strategy". Sun (arXiv:2210.04149) supplies the evaluation-side reason: the "two implications of neglecting a global timeline during evaluation: data leakage and oversimplification of user
preference modeling". Ji, Sun, Zhang and Li (arXiv:2010.11060) make the mechanism precise: "Data leakage is caused by not observing global timeline in evaluating recommenders ... As a result, a model
learns from the user-item interactions that are not expected to be available at prediction time", and "Recommending a future item at a past time point is not realistic and it is a sign of invalid
offline evaluation setting". A random split cannot satisfy this constraint at all, because it scatters each user's interactions across the whole timespan instead of cutting them at a time; the frozen
T1/T2, which raise on any boundary crossing, is the enforcement of that constraint.

### 4.2 The cut on the text side

The same mechanism runs through words, and `tools/text_corpus.py` is cut accordingly: "Item prose is cut at the model cutoff, so a phrase is matched only against words written before the evaluated
period", while "A person's own words are kept to the period end because each request filters them by its own timestamp". Item prose is therefore reviews before T1, and the per-person voices are that
person's own review sentences before the request's own timestamp. The published record describes the interaction-level version of the same hazard (arXiv:2010.11060: the model "learns from the
user-item interactions that are not expected to be available at prediction time"; arXiv:2210.04149: the static ranking "cover[s] the entire duration of the training data"), and it is because an uncut
review stream would violate the constraint that the prose is cut at T1 rather than T2. The listing text is the residual channel: it is the 2023 snapshot as crawled (the project page records "price ...
at time of crawling"), so a title, feature list or description is what the product page held in 2023, not what it held at T1.

### 4.3 Duplicated and near-duplicate reviews

The 2018 data page quantifies the duplication: the aggressive pass removes "duplicates even if they are written by different users ... less than 1 percent of reviews", and the 2023 5-core files apply
"de-duplication" as part of the pure-ID release. A duplicate or near-duplicate review of one item can straddle the cutoff, so identical or near-identical wording sits on both sides of the split
boundary; for the text sidecar that is a stronger overlap signal than for ratings, since the duplicated object is a whole sentence, not a number. Under a random split the same near-duplicates would
additionally mix train and test directly, one of the data-side reasons the split is time-ordered.

### 4.4 Empty histories at the cutoff

Leave-last-out combined with a global cutoff has a mechanical consequence: any user whose interactions all fall at or after T1 has no training history. Sun (arXiv:2210.04149) states the general point:
"With timeline in consideration, the cold-start issue becomes a common problem for almost every user". In this repository's own data, 1,288 of the 33,993 eligible positive validation requests on
Musical Instruments have no history at all (reports/shadow_profile_design.md), and 1,573 validation rows have an empty history (reports/survey_reliable_serving.md). For those requests nothing in the
behavioural route is personalised, and the request is scored against a catalogue the model may never have seen the target item in.

## 5. Cold targets and catalogue churn

The target of a leave-last-out request is the user's own latest interaction, so by construction it is the least seen item in that user's history; under the 5-core filter and a global cutoff, a user
active near the end of the window often holds a target absent from, or barely present in, the training catalogue. In the measured data this is the dominant regime: on Musical Instruments, 12,305 of
35,905 eligible positive test reviews (34.3%) target items absent from the train-known catalogue, and on Video Games 22,051 of 35,562 do (reports/transfer_note.md). The published record brackets this:
Dong, Qin, Shah and Wang (arXiv:2606.29947) report that "32-91% of cold-start targets are brand-new items with no training interactions", which is why "standard single retrievers place the gold item
in a 200-item pool only 4.6-22.9% of the time". Sun (arXiv:2210.04149) describes the catalogue as moving: "new items may become available in the system at any time; outdated items are removed from the
system at any time". And the 5-core page documents churn in the other direction: "28 subcategories ... (fewer than 33), due to some categories becoming empty after 5-core processing"; the Games
category runs from 115,813 items unfiltered (Table 3 of arXiv:2403.03952, cut by the same timestamps) to 25.6K items and 94.8K users in the 5-core files, and Musical_Instruments to 24.6K items and
57.4K users (project page).

Taken from this: for this split a large share of the evaluation targets are items the trained catalogue does not contain, so no ranking restricted to the train-known catalogue can recover them - "a
catalog-coverage constraint, not a ranking failure that a different ordering alone can solve" (reports/transfer_note.md) - while the 34.3% sits inside the 32-91% band reported on other domains
(arXiv:2606.29947). For the 65.7% that are present, the item's own text is the matching surface the phrase arm works on.

## 6. The text sidecars and their sources

### 6.1 Reviews as a source of language

The review stream is the only first-person language in the corpus, and the published record treats exactly this as a resource. BLaIR (arXiv:2403.03952) builds Amazon-C4 by sampling "more than 20,000
5-star reviews from the Amazon Reviews 2023 dataset with at least 100 characters per review" and rephrasing each with ChatGPT "into a first-person query expressing the intent to find a suitable item" - the same stream this repository's query sidecar reuses ("first-person queries rewritten from a review the person wrote about that item", tools/text_corpus.py). McAuley, Pandey and
Leskovec (DOI 10.1145/2783258.2783381) learn the semantics of substitutes and complements "from data associated with products", the primary source being "the text of product reviews", on a catalogue of "9
million products, 237 million links, and 144 million reviews", the
EMNLP 2019 paper (DOI 10.18653/v1/d19-1018) identifies "review segments which justify users' intentions", and Dong et al. (arXiv:2606.29947) find "pockets of semantic cold-start advantage, especially
in text-rich domains when the item is already present". So the reviews are not filler: they are the substrate from which other researchers, and this repository, derive queries, item semantics and the
per-person voices.

### 6.2 Product titles and metadata as the query side

The query-pair stream anchors the phrase arm to the published unit of e-commerce query-product relevance. The ESCI / Shopping Queries benchmark (arXiv:2206.06588) contains "around 130 thousand unique
queries and 2.6 million manually labeled (query,product) relevance judgements", and for each query "a list of up to 40 results, together with their ESCI relevance judgements (Exact, Substitute,
Complement, or Irrelevant) indicating the relevance of the product to the query". Each pair "is accompanied by additional information from the Amazon catalog, including: product title, product
description, and additional product related bullet points. This information is public, as it is displayed at the Amazon website when searching for those products". The queries are deliberately hard:
"subsets of the queries have been sampled specifically to provide a variety of challenging problems (such as negation, attribute parsing, etc.)", and the dataset "is being used in one of the KDDCup'22
challenges". BLaIR reuses it directly: "We use the ESCI dataset Reddy et al. (2022) and retain only <query, item> pairs with the label 'Exact'", in English and timestamped after 1658002729837
(arXiv:2403.03952). This repository's `QUERY_SOURCES` instead consumes the official release itself (the `shopping_queries_dataset_examples.parquet` file of the
[amazon-science/esci-data](https://github.com/amazon-science/esci-data) repository, Apache License 2.0): "shopping queries people typed, judged against catalogue items",
restricted to the US locale and to the judgements "Exact" or "Substitute" (an acceptable substitute for the item), with both splits retained.
Admitting Substitute pairs has external support: a 2026 production retrieval system trains exactly this way, with "substitute query-product pairs
 provid[ing] coarse semantic supervision in Stage 1" before graded-relevance labels refine the ranking (arXiv:2606.01504) — a substitute judgement still
binds a typed query to an item the searcher can be served, which is the property the `published` condition needs. The boundary is unchanged, though:
these are target-linked relevance judgements, not clicks, so they measure wording reach, not live-query benefit.

### 6.3 The ESCI / Shopping Queries lineage

The four-level relevance scheme is not specific to the 2022 release: the benchmark paper attributes the ESCI judgements to the KDD 2015 study of substitutable and complementary products
(arXiv:2206.06588 cites McAuley, Pandey and Leskovec, DOI 10.1145/2783258.2783381), and the release has been extended rather than replaced - SQID (arXiv:2405.15190) is "an extension of the Amazon
Shopping Queries Dataset enriched with image information associated with 190,000 products".

Taken from this: the published lineage runs from the 2015 substitute/complement semantics learned over review text, through the 2022 ESCI-labeled query-product benchmark, to image-enriched extensions;
the repository's query sidecar reuses the middle link (the full official ESCI release, US locale, pairs judged "Exact" or "Substitute", plus first-person Amazon-C4), which is why product titles and listing text, not raw review prose, carry the query arm's
matching surface. Limitation: the original 2014 "Shopping Queries" benchmark report (Anand et al., ICIR 2014) could not be opened from any reachable host, so the lineage above rests on the 2022, 2024
and 2015 papers rather than on the 2014 report itself.

## 7. Licence and redistribution

What the authors state about terms is thin and one-directional. The 2018 data page's only usage condition is "Please cite the following paper if you use the data in any way". The 2023 project page
offers the data for benchmarking ("Standard data splits to encourage RecSys benchmarking") but carries no licence or permissions section anywhere on the page (the full page was checked), and the
Hugging Face card for the snapshot (McAuley-Lab/Amazon-Reviews-2023) carries no licence field (checked through the dataset API). The companion paper states the collection method - "we exclusively
collect publicly available information that users have explicitly chosen to share. We do not include any content that users have opted to keep private or restricted" (arXiv:2403.03952) - but the arXiv
paper's CC BY 4.0 licence covers the paper, not the 571.54M review rows and metadata it points at. Nothing opened for this note grants a right to redistribute the raw review rows, the user identifiers
or the product metadata.

Keeping the raw archives and the text sidecars out of version control follows directly from that: the corpus is scraped marketplace data whose platform of origin is not the dataset authors, the
authors' own stated condition is citation rather than a redistribution licence, and this repository's MIT licence covers code and committed aggregates only. That is why DATA_LICENSE.md holds "Dataset
ownership and permissions are separate from this repository's MIT software license", why `data/`, `data/text/` and `runs/` are Git-ignored, and why the public repository redistributes "no review rows,
user identifiers, per-request histories, fitted model weights, or product metadata from that source".

## Sources

- Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders - Yupeng Hou, Jiacheng Li, Xiangjun Fu, Zhankui He, An Yan, Xiusi Chen, Julian McAuley; arXiv:2403.03952 (2024; arXiv comment: ACL 2026). [arxiv.org/abs/2403.03952](https://arxiv.org/abs/2403.03952) - "a new large-scale Amazon Reviews 2023 dataset with over 570 million reviews and 48 million items"; 571,544,897 reviews / 54,514,264 users / 48,185,153 items against 233,055,327 reviews in the 2018 version, with 35,393,189 items carrying metadata; "we do not filter out any users or items based on interaction counts", the same two cutoffs "split all the reviews in Amazon Reviews 2023 in a ratio of 8:1:1"; the Games benchmark runs 115,813 items, 2,127,563 / 189,268 / 215,809 train / val / test, "By Timestamp (08/11/2021, 07/16/2022)".
- Amazon Reviews'23 project pages (main page and data_processing/5core.html) - McAuley Lab, 2023-2024. [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) - "collected in 2023 by McAuley Lab"; "571.54M reviews, 245.2% larger than the last version"; "some items lack metadata"; the 5-core pros/cons list; the leave-last-out and absolute-timestamp definitions with t_1 = 1628643414042 and t_2 = 1658002729837; "The history field in the processed files represents the historical user interactions before the current timestamp"; 5-core rows of 94.8K users / 25.6K items for Video_Games and 57.4K / 24.6K for Musical_Instruments.
- Amazon Review Data (2018) - Jianmo Ni (data page), 2018. [nijianmo.github.io/amazon](https://nijianmo.github.io/amazon/) - "an updated version of the Amazon review dataset released in 2014"; "233.1 million (142.8 million in 2014)" reviews; "all users and items have at least 5 reviews (75.26 million reviews)"; cross-user duplicates "account for less than 1 percent of reviews"; "Please cite the following paper if you use the data in any way".
- Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects - Jianmo Ni, Jiacheng Li, Julian McAuley; EMNLP 2019, pp. 188-197, DOI 10.18653/v1/d19-1018. [aclanthology.org/D19-1018](https://aclanthology.org/D19-1018/) - the paper the 2018 data page tells users to cite; it "distantly label[s] massive review corpora and construct[s] large-scale personalized recommendation justification datasets".
- Neural Collaborative Filtering - Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua; arXiv:1708.05031 (2017). [arxiv.org/abs/1708.05031](https://arxiv.org/abs/1708.05031) - "We adopted the leave-one-out evaluation, which has been widely used in literature. For each user, we held-out her latest interaction as the test set and utilized the remaining data for training"; 20-core filtering ("each user has at least 20 ratings") and candidate-set evaluation ("randomly samples 100 items").
- Take a Fresh Look at Recommender Systems from an Evaluation Standpoint - Aixin Sun; arXiv:2210.04149 (2022; arXiv comment: SIGIR 2023 Perspectives track). [arxiv.org/abs/2210.04149](https://arxiv.org/abs/2210.04149) - "the popularity baseline is poorly defined" under random or leave-one-out splits; 59.1% of the 88 case-study papers do not maintain a global timeline; "the popularity baseline is 'forced' to use all interactions in training set"; "the cold-start issue becomes a common problem for almost every user"; "new items may become available in the system at any time; outdated items are removed from the system at any time".
- On Sampled Metrics for Item Recommendation (Extended Abstract) - Walid Krichene, Steffen Rendle; Proc. 30th IJCAI, Sister Conferences track, pp. 4784-4788, 2021, DOI 10.24963/ijcai.2021/651. [ijcai.org/proceedings/2021/0651](https://www.ijcai.org/proceedings/2021/0651) - sampled metrics "do not persist relative statements, e.g., recommender A is better than B, not even in expectation"; "sampling should be avoided for metric calculation".
- A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models - Alexander Dallmann, Daniel Zoller, Andreas Hotho; arXiv:2107.13045 (2021; arXiv comment: RecSys 2021 Datasets & Benchmarks). [arxiv.org/abs/2107.13045](https://arxiv.org/abs/2107.13045) - "both sampling strategies can produce inconsistent rankings compared with the full ranking of the models"; leave-one-out is "as is common in most related work"; "we only used the common leave-one-out dataset split ... and did not consider alternative splits like a split by time".
- A Critical Study on Data Leakage in Recommender System Offline Evaluation - Yitong Ji, Aixin Sun, Jie Zhang, Chenliang Li; arXiv:2010.11060 (2020; arXiv comment: Accepted by TOIS). [arxiv.org/abs/2010.11060](https://arxiv.org/abs/2010.11060) - "a model learns from the user-item interactions that are not expected to be available at prediction time"; "Recommending a future item at a past time point is not realistic and it is a sign of invalid offline evaluation setting".
- Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search - Chandan K. Reddy, Lluis Marquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay, Arnab Biswas, Anlu Xing, Karthik Subbian; arXiv:2206.06588 (2022). [arxiv.org/abs/2206.06588](https://arxiv.org/abs/2206.06588) - "around 130 thousand unique queries and 2.6 million manually labeled (query,product) relevance judgements"; ESCI judgements (Exact, Substitute, Complement, or Irrelevant) "indicating the relevance of the product to the query (McAuley et al., 2015)"; product metadata "public, as it is displayed at the Amazon website"; "used in one of the KDDCup'22 challenges".
- Official release of the Shopping Queries Dataset - [github.com/amazon-science/esci-data](https://github.com/amazon-science/esci-data), Apache License 2.0. The single file `shopping_queries_dataset_examples.parquet` (2,621,288 rows; SHA-256 `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a`) carries the fields `query`, `product_id`, `product_locale`, `esci_label`, `small_version`, `large_version` and `split`; the repository consumes the US-locale rows judged "E" (1,247,558) or "S" (369,313) from both splits, i.e. 1,616,871 rows.
- Semantic Retrieval for Product Search in E-Commerce - Nikhil Kothari, Saksham Samdani, Ritam Mallick; arXiv:2606.01504 (2026). [arxiv.org/abs/2606.01504](https://arxiv.org/abs/2606.01504) - "substitute query-product pairs provid[e] coarse semantic supervision in Stage 1" ahead of graded-relevance fine ranking; gains "confirmed across query-frequency strata and business verticals" and "validated through live A/B deployment at scale". Cited in 6.2 as external evidence that Substitute judgements act as target-reachable positive pairs; the abstract names no public benchmarks.
- Inferring Networks of Substitutable and Complementary Products - Julian McAuley, Rahul Pandey, Jure Leskovec; Proc. 21st ACM SIGKDD, pp. 785-794, 2015, DOI 10.1145/2783258.2783381. [dl.acm.org/doi/10.1145/2783258.2783381](https://dl.acm.org/doi/10.1145/2783258.2783381) - the KDD 2015 study the ESCI benchmark credits for the relevance scheme; learns the semantics of substitutes and complements "from data associated with products", the primary source being "the text of product reviews", on "9 million products, 237 million links, and 144 million reviews".
- Shopping Queries Image Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product Search - Marie Al Ghossein, Ching-Wei Chen, Jason Tang; arXiv:2405.15190 (2024). [arxiv.org/abs/2405.15190](https://arxiv.org/abs/2405.15190) - "an extension of the Amazon Shopping Queries Dataset enriched with image information associated with 190,000 products".
- Diagnosing and Mitigating Retrieval Bottlenecks in LLM-Based Cold-Start Recommendation - Zhe Dong, Fang Qin, Manish Shah, Yicheng Wang; arXiv:2606.29947 (2026). [arxiv.org/abs/2606.29947](https://arxiv.org/abs/2606.29947) - "32-91% of cold-start targets are brand-new items with no training interactions"; "standard single retrievers place the gold item in a 200-item pool only 4.6-22.9% of the time"; "pockets of semantic cold-start advantage, especially in text-rich domains when the item is already present".

Dropped rather than cited: arXiv:1906.03315, which resolves to "Vandermondes in superspace" rather than the predecessor corpus paper (the actual 2018 dataset paper is Ni, Li, McAuley, EMNLP 2019,
cited above); arXiv:2205.14788, which resolves to a gravitational-wave astronomy paper (the ESCI benchmark paper is arXiv:2206.06588); arXiv:2105.08141, which resolves to a video-pose paper (the
accessible Krichene-Rendle work is the IJCAI 2021 extended abstract); "Yilmaz et al." for arXiv:2210.04149, whose sole author is Aixin Sun; and the 2014 "The SQ Dataset for Shopping Queries" report
(Anand et al., ICIR 2014), whose abstract could not be opened from any reachable host, so the ESCI lineage rests on the 2022, 2024 and 2015 papers instead.
