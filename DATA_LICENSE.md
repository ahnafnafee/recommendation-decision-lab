# Data and model-release boundary

The [McAuley Lab Amazon Reviews'23 benchmark](https://amazon-reviews-2023.github.io/data_processing/5core.html) supplies the Video Games and Musical Instruments CSV archives used locally. Dataset ownership and permissions are separate from this repository's MIT software license. Review the owner's current terms before downloading or reusing the data. This repository redistributes no review rows, user identifiers, per-request histories, fitted model weights, or product metadata from that source.

The `data/` and `runs/` directories are Git-ignored. The public `reports/*.json` files contain only aggregate counts, quality metrics, conditional intervals, file hashes and method decisions. The browser sample uses invented users, items and interactions. A locally built model bundle is derived from the owner data and remains under `runs/`; do not upload it with the code or use it for a different purpose without checking applicable permissions.
