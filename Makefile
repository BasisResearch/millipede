install: FORCE
	pip install -e .[test]

lint: FORCE
	flake8
	isort --check .

format: FORCE
	isort .

test: lint FORCE
	pytest -vx -s tests 

FORCE:
