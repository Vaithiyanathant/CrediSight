.PHONY: all setup start data profile features train evaluate simulate submit app frontend test clean

PYTHON := python3
SRC := src
CONFIG := config/config.yaml

all: setup data profile features train evaluate simulate submit

setup:
	./scripts/setup.sh

start:
	./start.sh

data:
	$(PYTHON) -m lpie.pipelines.runner data

profile:
	$(PYTHON) -m lpie.pipelines.runner profile

features:
	$(PYTHON) -m lpie.pipelines.runner features

train:
	$(PYTHON) -m lpie.pipelines.runner train

evaluate:
	$(PYTHON) -m lpie.pipelines.runner evaluate

simulate:
	$(PYTHON) -m lpie.pipelines.runner simulate

submit:
	$(PYTHON) -m lpie.pipelines.runner submit

app:
	$(PYTHON) -m uvicorn lpie.api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -k "not artifacts"

compile-check:
	$(PYTHON) -m compileall src/ -q

clean:
	rm -rf artifacts/store/features artifacts/mlruns __pycache__

lsinfo:
	@echo "Model artifacts:"
	@ls -lh artifacts/models/ 2>/dev/null || echo "  (none)"
	@echo "Reports:"
	@ls reports/ 2>/dev/null || echo "  (none)"
	@echo "Submission:"
	@ls -lh artifacts/submission*.csv 2>/dev/null || echo "  (none)"

