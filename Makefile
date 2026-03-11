BIN_PATH=.venv/bin

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
	${BIN_PATH}/mypy --install-types --non-interactive
	${BIN_PATH}/ruff check --config pyproject.toml .
	${BIN_PATH}/mypy .

.PHONY: lint-fix
lint-fix:
	${BIN_PATH}/ruff check --config pyproject.toml --fix .

.PHONY: format
format:
	${BIN_PATH}/ruff format --config pyproject.toml .

.PHONY: test
test:
	${BIN_PATH}/pytest tests/

.PHONY: test-cov
test-cov:
	${BIN_PATH}/pytest --cov=sequoia tests/

.PHONY: dev
dev:
	open "https://agentchat.vercel.app/?apiUrl=http://localhost:2024&assistantId=agent"
	langgraph dev
