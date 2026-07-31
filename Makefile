PYTHON=python

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	ruff check .

test:
	pytest -q

demo:
	python -m sector_stat_arb.demo

all: install test demo
