PY ?= python
export PYTHONPATH := src

.PHONY: all data test notebook refresh clean
all:            ## full pipeline: data -> backtest -> forecasts -> figures -> thumbnail
	$(PY) -m nowcast.pipeline
refresh:        ## same, but re-download the SAMA bulletins from KAPSARC first
	$(PY) -m nowcast.pipeline --refresh
test:
	$(PY) -m pytest -q
notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/nowcast.ipynb
