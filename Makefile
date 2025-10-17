PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PACKAGE = src/casc_lite
DATA = data/gsm8k_mini.jsonl
RESULTS = results

.PHONY: setup lint test format run-fixed run-adaptive sweep plot clean

setup:
	bash scripts/install.sh

lint:
	pre-commit run --all-files

format:
	black $(PACKAGE)
	isort $(PACKAGE)

test:
	pytest -q

run-fixed:
	$(PYTHON) -m src.casc_lite.cli.run_once --config src/casc_lite/config/default.yaml --mode fixed --n_fixed 3 --data $(DATA)

run-adaptive:
	$(PYTHON) -m src.casc_lite.cli.run_once --config src/casc_lite/config/default.yaml --mode adaptive --data $(DATA)

sweep:
	$(PYTHON) -m src.casc_lite.cli.sweep_thresholds --config src/casc_lite/config/default.yaml --data $(DATA)

plot:
	$(PYTHON) -m src.casc_lite.cli.export_plots --csv $(RESULTS)/aggregate.csv

clean:
	rm -rf $(RESULTS)/*.csv $(RESULTS)/*.png $(RESULTS)/*.pdf
