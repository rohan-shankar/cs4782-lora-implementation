# Report

This directory contains our final 2-page report.

- `report.tex` — LaTeX source
- `report.pdf` — compiled PDF
- `figures/` — all 13 PNG figures used in the report

## Compiling

We used [tectonic](https://tectonic-typesetting.github.io/) to compile:

```bash
tectonic report/report.tex
```

This handles all dependencies and multiple passes automatically. You can also use `pdflatex` if you have a full TeX distribution installed.
