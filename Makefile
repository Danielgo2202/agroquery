# Convenience targets for macOS/Linux. On Windows run the `uv run agroquery ...` commands directly.
.PHONY: all up data validate eval analyze test lint

all:
	uv run agroquery all

up:
	uv run agroquery up

data:
	uv run agroquery data

validate:
	uv run agroquery validate

eval:
	uv run agroquery eval $(ARGS)

analyze:
	uv run agroquery analyze

test:
	uv run pytest -q

lint:
	uv run ruff check .
