# Preprint: Extensions to the Deep Fourier Residual Method

LaTeX source for a follow-up paper extending the DFR method.

## Structure

```
preprint/
├── main.tex           # Main document
├── sections/          # Content sections
│   ├── introduction.tex
│   ├── background.tex
│   ├── methodology.tex
│   ├── theory.tex
│   ├── experiments.tex
│   ├── results.tex
│   └── conclusion.tex
├── figures/           # Figures (PDF, PNG)
├── style/             # Custom style files
├── references.bib     # Bibliography
├── Makefile           # Build commands
└── latexmkrc          # latexmk configuration
```

## Building

### Using Make

```bash
# Build PDF
make

# Quick build (skip bibtex)
make quick

# Clean auxiliary files
make clean

# Open PDF (macOS)
make open
```

### Using latexmk

```bash
# Build PDF
latexmk -pdf main.tex

# Continuous build (watch for changes)
latexmk -pdf -pvc main.tex

# Clean
latexmk -c
```

### Manual Build

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Overleaf

This project is compatible with Overleaf. To use:

1. Create a new Overleaf project
2. Upload all files from this directory
3. Set `main.tex` as the main document

## Writing Tips

### Citations

Use `\cite{key}` for citations. The bibliography file `references.bib` includes citations from the original DFR paper.

```latex
As shown by Taylor et al.~\cite{taylor2023deep}...
```

### Cross-references

Use `\label` and `\cref` for cross-references:

```latex
\begin{theorem}
\label{thm:main}
...
\end{theorem}

As shown in \cref{thm:main}...
```

### Figures

Place figures in the `figures/` directory:

```latex
\begin{figure}
\includegraphics[width=0.8\textwidth]{figures/results.pdf}
\caption{...}
\label{fig:results}
\end{figure}
```

## TODOs

- [ ] Fill in methodology section
- [ ] Add experimental results
- [ ] Generate figures from experiments
- [ ] Complete theoretical analysis
- [ ] Write abstract
- [ ] Add author information
