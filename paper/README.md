# Manuscript

`main.pdf` is the standalone manuscript by Ahnaf An Nafee. Its tables and numerical claims correspond to the committed aggregate reports for the corrected Video Games replay, the Musical Instruments category check, and the later exploratory neural extension. The source is `main.tex`.

Build from the repository root with TeX Live and XeLaTeX:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build/paper paper/main.tex
```

The public code and aggregate results are at <https://github.com/ahnafnafee/recommendation-decision-lab>. Raw reviews, user-level outputs, and model weights remain local. The manuscript is also available as a [ResearchGate preprint](https://doi.org/10.13140/RG.2.2.16487.61607). Check the policies of any later submission venue before submitting the same work there.
