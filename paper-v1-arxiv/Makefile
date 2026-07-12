# Makefile for LaTeX paper builds with automatic cleanup

# Paper files
PAPER_IEEE = paper_ieee
PAPER_ARTICLE = paper

# LaTeX commands
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = bibtex

# Build artifacts to clean
ARTIFACTS = *.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls *.synctex.gz

.PHONY: all clean ieee article rebuild-ieee rebuild-article figures help

# Default target
all: ieee

# Build IEEE paper with automatic cleanup
ieee: $(PAPER_IEEE).pdf
	@echo "Building IEEE paper and cleaning up..."
	@$(MAKE) clean
	@echo "Done! PDF: $(PAPER_IEEE).pdf"

# Build article paper with automatic cleanup
article: $(PAPER_ARTICLE).pdf
	@$(MAKE) clean
	@echo "Done! PDF: $(PAPER_ARTICLE).pdf"

# IEEE paper build recipe
$(PAPER_IEEE).pdf: $(PAPER_IEEE).tex references.bib
	@echo "Building $(PAPER_IEEE).pdf..."
	$(LATEX) $(PAPER_IEEE).tex
	$(BIBTEX) $(PAPER_IEEE)
	$(LATEX) $(PAPER_IEEE).tex
	$(LATEX) $(PAPER_IEEE).tex

# Article paper build recipe
$(PAPER_ARTICLE).pdf: $(PAPER_ARTICLE).tex references.bib
	@echo "Building $(PAPER_ARTICLE).pdf..."
	$(LATEX) $(PAPER_ARTICLE).tex
	$(BIBTEX) $(PAPER_ARTICLE)
	$(LATEX) $(PAPER_ARTICLE).tex
	$(LATEX) $(PAPER_ARTICLE).tex

# Clean all build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(ARTIFACTS)
	@echo "Cleaned!"

# Rebuild IEEE from scratch
rebuild-ieee: clean ieee

# Rebuild article from scratch
rebuild-article: clean article

# Generate all figures
figures:
	@echo "Generating all figures..."
	@cd .. && python3 src/generate_figures.py
	@cd .. && python3 src/visualize_steering.py
	@cd .. && python3 src/visualize_lexical_ablation.py
	@cd .. && python3 src/visualize_cross_model_layers.py
	@echo "✓ All figures generated!"

# Show help
help:
	@echo "Available targets:"
	@echo "  make ieee          - Build IEEE paper and auto-clean artifacts"
	@echo "  make article       - Build article paper and auto-clean artifacts"
	@echo "  make clean         - Remove all build artifacts"
	@echo "  make rebuild-ieee  - Clean then build IEEE paper"
	@echo "  make rebuild-article - Clean then build article paper"
	@echo "  make figures       - Generate all figures (1-6)"
	@echo "  make help          - Show this help message"
