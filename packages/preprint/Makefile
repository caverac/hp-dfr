# Makefile for the single-file LaTeX manuscript (ms.tex).

# ms.pdf is tracked, so the build must be reproducible: pdflatex stamps
# CreationDate/ModDate into the PDF, which would make every rebuild show up as a
# modification to a binary file even when the manuscript has not changed. These
# two variables pin the timestamp (the reproducible-builds convention), so a
# rebuild of unchanged sources is byte-identical and `git status` stays honest.
export SOURCE_DATE_EPOCH = 0
export FORCE_SOURCE_DATE = 1

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

OUTDIR   = /tmp/hp-dfr
STAGEDIR = $(OUTDIR)/submission
TARFILE  = $(OUTDIR)/hp-dfr-preprint.tar

# Elsevier/CMAME (elsarticle) submission bundle. Elsevier reviewers expect a
# double-spaced, line-numbered manuscript, which is the `review` class option.
# Rather than carry it in the canonical source -- which would line-number every
# `make all` build and the docs site -- the submission target swaps `preprint`
# for `review` in a *staged* copy of $(MS).tex and compiles that, so the everyday
# build stays clean. Ships the prebuilt .bbl, so two pdflatex passes (no bibtex)
# resolve citations and line-number references. The elsarticle class and .bst are
# provided by the publisher and are not bundled.
.PHONY: submission
submission: $(MS).pdf
	@rm -rf $(STAGEDIR)
	@mkdir -p $(STAGEDIR)
	@sed -E 's/\\documentclass\[[^]]*\]\{elsarticle\}/\\documentclass[review,11pt]{elsarticle}/' $(MS).tex > $(STAGEDIR)/$(MS).tex
	@cp $(MS).bbl $(STAGEDIR)/
	@cp $(ASSETS)/*.pdf $(STAGEDIR)/
	@cd $(STAGEDIR) && $(LATEX) $(MS) && $(LATEX) $(MS)
	@COPYFILE_DISABLE=1 tar cf $(TARFILE) -C $(STAGEDIR) $(MS).tex $(MS).bbl $(MS).pdf $(FIGS)
	@echo "Created $(TARFILE) (line-numbered submission)"

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
