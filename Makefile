init-pre-commit:
	pre-commit install
	pre-commit run --all-files

update-pre-commit:
	pre-commit autoupdate

init-venv:
	uv venv -p 3.12
	uv sync
	make shell

.PHONY: shell
shell:
	@echo "copy and paste the following command to activate the virtual environment"
	@echo "source .venv/bin/activate"

.PHONY: lint
lint:
	ruff check --config pyproject.toml .
	mypy .

.PHONY: lint-fix
lint-fix:
	ruff check --config pyproject.toml --fix .

.PHONY: format
format:
	ruff format --config pyproject.toml .

.PHONY: test
test:
	pytest tests/

.PHONY: test-cov
test-cov:
	pytest --cov=sequoia tests/
