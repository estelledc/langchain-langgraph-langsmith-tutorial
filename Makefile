.PHONY: sync format lint typecheck test eval curriculum-check site-check verify site

sync:
	uv sync --frozen

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint:
	uv run ruff format --check src tests scripts
	uv run ruff check src tests scripts

typecheck:
	uv run mypy src/agent_lab

test:
	uv run pytest --cov=agent_lab --cov-report=term-missing --cov-fail-under=85

eval:
	uv run agent-lab eval --suite fast

curriculum-check:
	uv run python scripts/check_curriculum.py

site-check:
	uv run python scripts/check_site.py

verify:
	uv run agent-lab verify

site:
	bundle exec jekyll build
