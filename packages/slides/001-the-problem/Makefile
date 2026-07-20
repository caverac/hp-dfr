MAIN     := slides
TEX      := pdflatex
TEXFLAGS := -interaction=nonstopmode -halt-on-error

# Pick up the shared style file from the parent directory.
export TEXINPUTS := .:../style:$(TEXINPUTS)

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(TEX) $(TEXFLAGS) $(MAIN)
	$(TEX) $(TEXFLAGS) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).nav $(MAIN).snm $(MAIN).toc $(MAIN).vrb

distclean: clean
	rm -f $(MAIN).pdf
