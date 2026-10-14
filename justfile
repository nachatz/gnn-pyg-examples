mod common "justfiles/common.just"
mod modules "justfiles/modules.just"
mod repo "justfiles/repo.just"

sync:
    uv sync

message-passing: modules::message-passing

aggregation: modules::aggregation

attention: modules::attention

run: modules::run

fmt: common::fmt modules::fmt repo::fmt

check: common::check modules::check repo::check

mypy: common::mypy modules::mypy

test: common::test modules::test

# Coverage runs the whole suite at once, since fail_under = 100 applies to everything.
cov:
    uv run pytest --cov --cov-report=term-missing

validate: check mypy cov