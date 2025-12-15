# Makefile for LaTeX preprint

# Main document
MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

# Default target
all: $(MAIN).pdf

# Build PDF
$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Quick build (no bibtex)
quick:
	$(LATEX) $(MAIN)

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f sections/*.aux

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# Watch for changes and rebuild
watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -qre modify,create,delete . 2>/dev/null || fswatch -1 . 2>/dev/null || sleep 5; \
		make quick; \
	done

# Open PDF (macOS)
open: $(MAIN).pdf
	open $(MAIN).pdf

# Check for common LaTeX issues
check:
	@echo "Checking for common issues..."
	@grep -n "TODO" sections/*.tex main.tex || echo "No TODOs found"
	@grep -n "FIXME" sections/*.tex main.tex || echo "No FIXMEs found"
	@grep -n "\\\cite{}" sections/*.tex main.tex || echo "No empty citations"

.PHONY: all quick clean distclean watch open check
