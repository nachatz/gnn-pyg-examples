# Shared code in common/, its tests, and the test helpers in tests/utils.

set working-directory := ".."

paths := "common tests/common tests/utils"

fmt:
    uv run ruff format {{ paths }}
    uv run ruff check --fix {{ paths }}

check:
    uv run ruff format --check {{ paths }}
    uv run ruff check {{ paths }}

mypy:
    uv run mypy {{ paths }}

test:
    uv run pytest tests/common
