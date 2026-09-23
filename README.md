# Recommendation Decision Lab

A runnable recommendation service built around a simple question: when does a personalized challenger earn the right to replace a reliable popularity baseline? The project combines a full-catalog temporal evaluation on two Amazon Reviews'23 product categories, a validation-only route gate, a local HTTP API, an interactive synthetic demo, and explicit fallback behavior. It is a companion engineering project to an earlier MovieLens measurement study; the two studies have separate code, data and evidence boundaries.

**Measured result:** On a fresh Musical Instruments category, the fixed hybrid scored **0.008681 NDCG@10** against **0.008007** for validation-selected recent popularity across 35,905 eligible positive-review requests. The paired difference was **+0.000674 [0.000040, 0.001338]**. This is a small offline gain with low absolute recovery, not online impact or a new algorithm. The corrected Video Games run is exploratory because an earlier replay exposed a tuple-order bug in personalization. Read the [transfer note](reports/transfer_note.md) for the protocol and limitations.

## Try the demo

Python 3.11 or newer and the standard library are sufficient. From the repository root:

```powershell
python -m unittest discover -s tests -v
python -m reliability.server --port 8899
```

Open `http://127.0.0.1:8899/`. The browser profiles and item names are invented. Compare baseline, challenger and active rankings, then click **Simulate challenger failure** to inspect the fallback. The synthetic gate stays closed because no real validation decision applies to those invented users. The service binds to loopback only. The same `web/` directory also runs as a static preview with no backend. Its GitHub Pages workflow requires manual dispatch and has not been deployed.

The API exposes `GET /api/health`, `GET /api/metrics`, and `POST /api/recommend` with a JSON body such as `{"history":["atlas","comet"],"k":5}`. Responses show the active route, shadow and baseline lists, fallback reason and local ranking time. No submitted history is logged or persisted. This is a local demonstrator rather than a hardened production service.

## Reproduce the data experiment

The source is the [McAuley Lab Amazon Reviews'23 5-core absolute-timestamp benchmark](https://amazon-reviews-2023.github.io/data_processing/5core.html). Review the owner's current page and terms before acquisition. Download the **Video_Games** and **Musical_Instruments** `timestamp_w_his` train/valid/test CSV gzip archives into `data/` with the filenames `Video_Games.train.csv.gz`, `Video_Games.valid.csv.gz`, and so on. The source paths follow this pattern:

```text
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/timestamp_w_his/<Category>.<split>.csv.gz
```

The original local run followed the committed [Video Games protocol](protocol/video_games_v1.md), then fixed a defect and committed the [Musical Instruments confirmation protocol](protocol/musical_instruments_v2.md) before opening that category. Re-running the commands below verifies the implementation; it does not create another untouched test result.

```powershell
python -m reliability.benchmark --data data --category Video_Games --output runs/video-games-replay
python -m reliability.benchmark --data data --category Musical_Instruments --output runs/musical-replay
```

Every run refuses an existing output directory. Its `validation_decision.json` is written before the test archive is opened. `aggregate.json` contains counts, metrics, paired intervals, source hashes and limitations, with no row-level predictions or user IDs. The reviewed public aggregate copies are [Video Games](reports/video_games_exploratory.json) and [Musical Instruments](reports/musical_instruments_confirmation.json).

To serve a locally trained Musical Instruments model, build an ignored, hash-checked bundle and pair it with the exact aggregate from its run:

```powershell
python -m reliability.bundle --data data --category Musical_Instruments --output runs/musical-bundle
python -m reliability.server --bundle runs/musical-bundle --aggregate runs/musical-replay/aggregate.json --port 8899
```

The real-data API accepts item IDs in `history`; the browser demo's invented profiles are unavailable in that mode. Neither the archive, model bundle nor review-derived weights are part of the public repository.

## Neural retrieval extension

The optional PyTorch path trains a two-tower retriever on chronological pairs from the training archive. A masked mean of up to 20 prior positive items passes through a small MLP to form the query vector; a separate item tower embeds train-known products. In-batch softmax supplies negatives, with duplicate targets and already reviewed products masked so they are not treated as negatives. Training uses a fixed seed and three epochs. Exact full-catalog scoring excludes prior items and keeps future items outside the train catalog as misses.

Install PyTorch for your platform using the [official selector](https://pytorch.org/get-started/locally/), then run these commands from the repository root:

```powershell
.venv\Scripts\python.exe -m reliability.neural --data data --category Musical_Instruments --output runs/musical-neural
.venv\Scripts\python.exe -m reliability.neural_calibrate --data data --weights runs/musical-neural/neural_weights.pt --output runs/musical-neural-calibrated
```

The first command saves fitted weights and a hash-checked manifest only in ignored `runs/`. The second selects a neural/popularity blend on validation and compares it with the already active hybrid before opening test rows. Each command refuses an existing output directory. To inspect the neural shadow through the local API, add `--neural-weights runs/musical-neural/neural_weights.pt --neural-train data/Musical_Instruments.train.csv.gz --neural-aggregate runs/musical-neural-calibrated/aggregate.json` to the real-data server command above. The response includes `neural_shadow`, its gate decision and fallback reason; only a validation-approved challenger can become active.

The [exploratory neural extension](reports/neural_extension.md) records the measured result. On Musical Instruments, the calibrated neural blend beat recent popularity but did not beat the existing co-review hybrid, so its stronger-challenger gate stayed closed. This category's test period was already known from the earlier study; the extension is a model-development comparison, not a new independent confirmation. The synthetic browser demo remains separate from fitted review-derived weights.

## Manuscript

The [paper](paper/main.pdf) presents the temporal comparison, neural challenger, guarded service, and limits in one standalone article. Its [LaTeX source](paper/main.tex) and [build instructions](paper/README.md) are included for inspection.

## What this demonstrates

- A validation-selected transparent control and a challenger evaluated over the entire train-known catalog, with unseen future items retained as misses.
- A gate that declines a challenger without sufficient validation evidence, plus runtime fallback when the challenger fails or a history is unsupported.
- Train-only popularity and item co-review modeling, deterministic retrieval, a hash-checked local bundle, an HTTP interface, aggregate request metrics and end-to-end service checks.
- A trained, hash-checked two-tower retrieval path that can be compared with the existing route and inspected as a shadow without displacing a stronger challenger.
- An auditable failure: the first exploratory replay silently dropped personalization because neighbor tuple fields were reversed. A training-to-ranking regression test now catches that fault.

The 5-core dataset was retrospectively filtered using the full corpus; a review is not an exposure or engagement label. A second product category is useful transfer evidence but remains on the same platform and one temporal split. Local timing excludes HTTP, initialization, concurrency and network effects. These limits are explained in the [research note](reports/transfer_note.md).

## Repository map and publication status

| Path | Purpose |
| --- | --- |
| `reliability/` | Training, full-catalog evaluation, bundle verification and local service |
| `protocol/` | Designs committed before the corresponding category outcome |
| `reports/` | Public-safe aggregates and research interpretation |
| `paper/` | Manuscript PDF, source and build instructions |
| `web/` | Dependency-free synthetic browser demo with static fallback |
| `tests/` | Exact-ranking, training-path, bundle and HTTP fallback checks |

Original source, tests and documentation use the [MIT license](LICENSE). Dataset records, fitted weights and owner metadata are outside that license; see [data permissions](DATA_LICENSE.md). The [public GitHub repository](https://github.com/ahnafnafee/recommendation-decision-lab) contains source, the manuscript and aggregate results, but no review archives or fitted weights. The paper has no claimed conference or journal acceptance. `CITATION.cff` names Ahnaf An Nafee for software citation.
