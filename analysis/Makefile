PYTHON  := ../venv/bin/python
LATEXMK := latexmk
RESULTS := ../results
AGG     := $(RESULTS)/aggregated
# Local scratch — gitignored; preserved for ad-hoc preprocess
# invocations against fresh raw CSVs that aren't yet under
# $(AGG)/<m>/. The canonical TSV location is $(AGG)/<m>/.
DATA    := data
OUTPUT  := output

# Per-machine emission target. `make figures` defaults to the
# single machine found under $(RESULTS)/runs/ if unambiguous;
# operators with a multi-machine tree pass `make figures
# MACHINE=<id>`. Same fallback for `make preprocess`.
MACHINE ?= $(shell ls $(RESULTS)/runs 2>/dev/null | head -n 1)

# Optional corpus-size filter passed through to preprocess.py as
# `--n-passages N`. Use when the per-machine results tree carries
# runs from more than one corpus (e.g. 100k flat-scheme runs
# alongside an 8.8M IVF sweep on the same host) and the cross-IVF
# parity guard would otherwise abort the preprocess step. Empty
# = no filter, all runs feed the figures.
N_PASSAGES ?=

FIGURE_SRCS := $(filter-out %-rationale.tex,$(wildcard figures/*.tex))
FIGURE_PDFS := $(patsubst figures/%.tex,$(OUTPUT)/%.pdf,$(FIGURE_SRCS))

.PHONY: figures preprocess preprocess-all report clean

figures: preprocess $(FIGURE_PDFS)

preprocess: $(AGG)/$(MACHINE)/.preprocessed

# Per-machine canonical emission. Sentinel lives under
# $(AGG)/<machine-id>/ so multiple machines can re-preprocess
# independently without invalidating one another's TSVs.
$(AGG)/$(MACHINE)/.preprocessed: preprocess.py \
		$(shell find $(RESULTS)/runs/$(MACHINE) -name 'raw.csv' 2>/dev/null) \
		$(shell find $(RESULTS)/runs/$(MACHINE) -name 'substep-breakdown.csv' 2>/dev/null) \
		$(shell find $(RESULTS)/runs/$(MACHINE) -name 'run-metadata.toml' 2>/dev/null)
	mkdir -p $(AGG)/$(MACHINE)
	$(PYTHON) preprocess.py --results $(RESULTS) --machine $(MACHINE) \
		$(if $(N_PASSAGES),--n-passages $(N_PASSAGES),) \
		--outdir $(AGG)/$(MACHINE)
	touch $@

# Sweep every machine present under $(RESULTS)/runs/ — the form
# CI / paper-figure rebuild uses to refresh the tracked tree.
preprocess-all:
	@for m in $$(ls $(RESULTS)/runs 2>/dev/null); do \
	    echo "→ preprocess $$m"; \
	    $(MAKE) preprocess MACHINE=$$m; \
	done

$(OUTPUT)/%.pdf: figures/%.tex style.tex $(AGG)/$(MACHINE)/.preprocessed
	mkdir -p $(OUTPUT)
	$(LATEXMK) -pdf -interaction=nonstopmode \
		-pretex='\def\datadir{../$(AGG)/$(MACHINE)}' -usepretex \
		-cd -outdir=../$(OUTPUT) $<

report:
	$(PYTHON) report.py --results $(RESULTS) \
		$(if $(RUN),--run $(RUN),) \
		$(if $(MACHINE),--machine $(MACHINE),)

clean:
	find $(DATA) -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null; true
	find $(OUTPUT) -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null; true
