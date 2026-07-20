# Slides

A series of Beamer decks on the goal-oriented DFR paper for a general technical
audience: the background, the method, the theory in full, and the honest results.
Each deck lives in its own numbered subdirectory and builds independently.

## Decks

| Number | Title              | Covers                                                           |
| ------ | ------------------ | ---------------------------------------------------------------- |
| 001    | The problem        | PINNs, the residual-norm trap, DFR, quantities of interest       |
| 002    | The method         | The adjoint, the two networks, the stop-gradient, the algorithm  |
| 003    | The theory         | The setting, the error representation and the bound, with proofs |
| 004    | Results and limits | Reallocation, matched budget vs matched cost, limitations        |

## Build

A single deck:

```bash
make -C packages/slides/001-goal-oriented-dfr all
```

All decks:

```bash
make -C packages/slides all
```

The pipeline is `pdflatex` twice (no bibtex; references are inline). The output
PDF is written next to each `slides.tex`. Figures are pulled from the project
`assets` directory, so build those first with `uv run hp-dfr figures` if they
are missing.

## Theme

A minimal, sober theme in navy / slate / sky-blue accents, defined in
`style/hp-dfr-slides.sty`, with no external Beamer theme dependency.
