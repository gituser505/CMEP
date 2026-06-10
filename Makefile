# Define default config file paths
CONFIG ?= config.json
BEST_CONFIG = best_config.json

# Tell Make these aren't real files it needs to build
.PHONY: all clean lint test tune train-best train analyze docs docs_ci build

# 'all' runs the entire pipeline sequentially
all: clean install lint test docs train analyze

# 'build' runs the pipleine to do local setup
build: clean install lint test docs

# 'run' assuming build run the core code
run: train analyze

# 'tune' for full larning pipeline with tuning
tune-full: tune train-best analyze

clean:
	@echo "Cleaning up caches and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf docs/build/*
	rm -f .latest_run.txt .latest_tune.txt

install:
	@echo "Upgrading pip and installing dependencies from requirements.txt..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	@echo "Running static analysis with pylint..."
	pylint src/ --fail-under=7.0

test:
	@echo "Running unit tests with pytest..."
	pytest -v

tune:
	@echo "Starting hyperparameter tuning using $(CONFIG)..."
	python src/tune.py --config $(CONFIG)

train-best:
	@echo "Starting training using $(BEST_CONFIG)..."
	@echo "Reading $(BEST_CONFIG) from .latest_tune.txt"
	python src/train.py --config $$(cat .latest_tune.txt)

train:
	@echo "Starting training using $(CONFIG)..."
	python src/train.py --config $(CONFIG)

analyze:
	@echo "Reading run directory from .latest_run.txt..."
	python src/analysis.py --dir $$(cat .latest_run.txt)

docs:
	@echo "Delegating to docs/Makefile to build the HTML..."
	$(MAKE) -C docs html
	@echo "Open docs/build/html/index.html in your browser."

docs_ci:
	@echo "Strictly testing Sphinx docs for CI via sub-make..."
	$(MAKE) -C docs html SPHINXOPTS="-W"