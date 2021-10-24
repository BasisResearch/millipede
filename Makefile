install: FORCE
	pip install -e .[test]

lint: FORCE
	flake8
	isort --check .

format: FORCE
	isort .

test: lint FORCE
	pytest -s -vx millipede/testing.py

FORCE:
