LATEXMK ?= latexmk
FIGURE_PDFS := \
	figures/fig1_pipeline.pdf \
	figures/fig2_lexical_controls.pdf \
	figures/fig3_writer_effect.pdf \
	figures/fig4_band_specificity.pdf \
	figures/fig5_selection_instability.pdf \
	figures/figA1_capability.pdf

.PHONY: all figures clean

all: figures paper.pdf

figures:
	python3 scripts/build_figures.py

paper.pdf: paper.tex references.bib $(FIGURE_PDFS)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error paper.tex

clean:
	$(LATEXMK) -C paper.tex
