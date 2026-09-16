# SIH26184 -- Predictive cash-out forecasting. Base prototype.
#
# PY points at the venv interpreter. On Windows (Git Bash / PowerShell) the venv
# python lives under .venv/Scripts; on Linux/mac under .venv/bin. Override if needed:
#   make train PY=python
ifeq ($(OS),Windows_NT)
  PY ?= .venv/Scripts/python.exe
else
  PY ?= .venv/bin/python
endif

.PHONY: help setup data train eval test api demo all clean

help:
	@echo "targets:"
	@echo "  setup  create venv + install dependencies"
	@echo "  data   build ATM table + training corpus + sealed corpus"
	@echo "  train  fit M1 (hazard), M2 (channel), M3 (location)"
	@echo "  eval   4-split evaluation + baselines + plots -> reports/"
	@echo "  test   run the leakage test suite"
	@echo "  api    start the FastAPI console at http://localhost:8000"
	@echo "  demo   headless end-to-end replay of one case (no browser needed)"
	@echo "  all    data + train + eval"

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install pandas numpy scikit-learn xgboost networkx fastapi \
	    uvicorn pydantic pyarrow h3 matplotlib pytest

data:
	$(PY) -m gen.atms
	$(PY) -m gen.generator
	$(PY) -m gen.generator --sealed

train:
	$(PY) -m ml.train

eval:
	$(PY) -m ml.evaluate

test:
	$(PY) -m pytest tests/ -q

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

demo:
	$(PY) demo.py

all: data train eval

clean:
	rm -rf models/*.joblib reports/*.png reports/*.json __pycache__ */__pycache__
