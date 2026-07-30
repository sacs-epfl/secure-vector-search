PYTHON  := ../../venv/bin/python
LATEXMK := latexmk
RESULTS := ../../results
DATA    := data
OUTPUT  := output

# Directory `make` was originally invoked from; the top-level Makefile
# passes this through so we can report the output path relative to the
# user's shell, not relative to this sub-dir. Defaults to CURDIR when
# this Makefile is invoked directly.
INVOCATION_DIR ?= $(CURDIR)

.PHONY: cross_machine preprocess clean

cross_machine: $(OUTPUT)/cross_machine.pdf
	@echo
	@echo "Generated:"
	@$(PYTHON) -c "import os; print(' ', os.path.relpath('$(CURDIR)/$(OUTPUT)/cross_machine.pdf', '$(INVOCATION_DIR)'))"

preprocess: $(DATA)/.preprocessed

$(DATA)/.preprocessed: preprocess.py \
		$(shell find $(RESULTS)/runs -name 'raw.csv' 2>/dev/null) \
		$(shell find $(RESULTS)/runs -name 'run-metadata.toml' 2>/dev/null)
	mkdir -p $(DATA)
	$(PYTHON) preprocess.py --results $(RESULTS) --outdir $(DATA)

$(OUTPUT)/cross_machine.pdf: cross_machine.tex ../style.tex $(DATA)/.preprocessed
	mkdir -p $(OUTPUT)
	$(LATEXMK) -pdf -interaction=nonstopmode -outdir=$(OUTPUT) cross_machine.tex

clean:
	rm -rf $(DATA) $(OUTPUT)
