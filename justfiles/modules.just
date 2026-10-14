# The GNN chapters in modules/ and their tests.

set working-directory := ".."

paths := "modules tests/modules"

message-passing:
    uv run python -m modules.message_passing.story

aggregation:
    uv run python -m modules.aggregation.story

attention:
    uv run python -m modules.attention.story

run: message-passing aggregation attention

fmt:
    uv run ruff format {{ paths }}
    uv run ruff check --fix {{ paths }}

check:
    uv run ruff format --check {{ paths }}
    uv run ruff check {{ paths }}

mypy:
    uv run mypy {{ paths }}

test:
    uv run pytest tests/modules
