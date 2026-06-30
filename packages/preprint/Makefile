# Makefile for the single-file LaTeX manuscript (ms.tex).

MS     = ms
LATEX  = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

# Figures are generated into the project-root assets directory (by
# `uv run hp-dfr figures` in packages/experiments) and synced in here for the
# build. The local copies are git-ignored; assets/ is the source of truth.
ASSETS = ../../assets
FIGS   = $(notdir $(wildcard $(ASSETS)/*.pdf))

all: $(MS).pdf

# Copy figure PDFs from assets into the preprint directory.
.PHONY: figures
figures:
	@cp -f $(ASSETS)/*.pdf .

$(MS).pdf: $(MS).tex references.bib figures
	$(LATEX) $(MS)
	$(BIBTEX) $(MS)
	$(LATEX) $(MS)
	$(LATEX) $(MS)

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f $(FIGS)

.PHONY: distclean
distclean: clean
	rm -f $(MS).pdf

.PHONY: open
open: $(MS).pdf
	open $(MS).pdf

.PHONY: check
check:
	@grep -n "TODO\|FIXME" $(MS).tex || echo "No TODOs found"
