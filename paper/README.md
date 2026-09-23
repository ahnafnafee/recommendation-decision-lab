# Manuscript

`main.pdf` is the standalone manuscript by Ahnaf An Nafee, and the compiled PDF is the
tracked artifact. Its tables and numerical claims correspond to the committed aggregate
reports: the corrected Video Games replay, the Musical Instruments category check, the
exploratory neural extension, the already-held-versus-elicited evidence study, and the
measured wording arm. The source is `main.tex`.

Build it in this directory with a TeX Live installation. The default engine is
pdfTeX, which is what produced the tracked PDF; three passes are needed for the
cross-references and the bibliography:

```powershell
Set-Location paper
latexmk -pdf -interaction=nonstopmode main.tex
```

Passes over `-pdf` produce the same output. LaTeX intermediates (`.aux`, `.log`,
`.fls`, `.fdb_latexmk`, `.bbl` and friends) are Git-ignored, so a local build leaves the
release boundary reported by `tools/release_check.py` unchanged; the tracked PDF is
updated only when the manuscript itself changes.

Every number in the manuscript is reproducible from a committed aggregate JSON under
`reports/` or from the commands in the top-level README; none of them is quoted from a
local file that is not in the repository. The public code and aggregate results are at
<https://github.com/ahnafnafee/recommendation-decision-lab>. Raw reviews, user-level
outputs, and model weights remain local. The manuscript is also available as a
[ResearchGate preprint](https://www.researchgate.net/publication/414634917_When_Does_Personalization_Earn_the_Route_A_Full-Catalog_Temporal_Study_of_Guarded_Recommendation).
Check the policies of any later submission venue before submitting the same work there.
