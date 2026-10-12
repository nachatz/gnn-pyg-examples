# Repo-wide markdown and yaml (READMEs, workflows, configs).

set working-directory := ".."

fmt:
    uv run mdformat .
    yamlfmt

check:
    uv run mdformat --check .
    yamlfmt -lint
