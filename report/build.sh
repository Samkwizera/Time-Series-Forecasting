#!/usr/bin/env bash
# Build report/report.pdf from the current pipeline outputs.
# Requires either tectonic (https://tectonic-typesetting.github.io) or a TeX Live with latexmk.
set -euo pipefail
cd "$(dirname "$0")"
python ../scripts/07_report_assets.py "$@"
if command -v tectonic >/dev/null; then
  tectonic report.tex
elif command -v latexmk >/dev/null; then
  latexmk -pdf -interaction=nonstopmode report.tex
else
  echo "no LaTeX engine found: install tectonic or TeX Live" >&2
  exit 1
fi
echo "built report/report.pdf"
