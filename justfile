sync:
    uv sync

message-passing:
    uv run python -m modules.message_passing.story

aggregation:
    uv run python -m modules.aggregation.story

run: message-passing aggregation

fmt:
    uv run ruff format .
    uv run ruff check --fix .

check:
    uv run ruff format --check .
    uv run ruff check .

mypy:
    uv run mypy common modules tests

test:
    uv run pytest

cov:
    uv run pytest --cov --cov-report=term-missing

validate: check mypy cov
