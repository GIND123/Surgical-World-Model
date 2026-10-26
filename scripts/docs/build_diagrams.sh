#!/usr/bin/env bash
# Compile every TikZ diagram in docs/diagrams/ to PDF (for the paper) and PNG (for the README).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../../docs/diagrams"
mkdir -p ../figures
for tex in "${@:-pipeline architecture}"; do
  for name in $tex; do
    pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" > "/tmp/diagram_$name.log" 2>&1 \
      || { grep -A4 "^!" "/tmp/diagram_$name.log" | head -20; exit 1; }
    cp "$name.pdf" "../figures/$name.pdf"
    pdftocairo -png -r 220 -singlefile "$name.pdf" "../figures/$name"
    rm -f "$name.aux" "$name.log" "$name.pdf"
    echo "built docs/figures/$name.{pdf,png}"
  done
done
