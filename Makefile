install: FORCE
	pip install -e .[test]

lint: FORCE
	flake8
	black --check .
	isort --check .

format: FORCE
	black .
	isort .

test: lint FORCE

FORCE:
