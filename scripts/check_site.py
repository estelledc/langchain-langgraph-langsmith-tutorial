#!/usr/bin/env python3
"""Validate V2 source claims, local links and optional rendered Pages output."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FORBIDDEN_MARKERS = ("/Users/", "/private/tmp/", "sk-你的key", "gho_")


def public_markdown(root: Path) -> list[Path]:
    direct = [root / name for name in ("README.md", "SETUP.md", "CONTRIBUTING.md", "404.md")]
    nested = list((root / "docs").glob("**/*.md")) + list((root / "labs").glob("**/*.md"))
    return sorted(path for path in direct + nested if path.is_file())


def github_readmes(root: Path) -> list[Path]:
    excluded = {".venv", "_site", "legacy", "vendor"}
    return sorted(
        path
        for path in root.glob("**/README.md")
        if not excluded.intersection(path.relative_to(root).parts)
    )


def check_source(root: Path) -> list[str]:
    errors: list[str] = []
    readme = (root / "README.md").read_text(encoding="utf-8")
    required = (
        "# Agent Engineering Lab",
        "RunRequest",
        "PASS / FAIL / UNKNOWN / ERROR",
        "v1-legacy",
        "uv sync --frozen",
    )
    for marker in required:
        if marker not in readme:
            errors.append(f"README missing marker: {marker}")
    for stale in ("4 周、16 篇，从", "langchain-community==1.0.4        #"):
        if stale in readme:
            errors.append(f"README retained stale V1 claim: {stale}")
    for path in github_readmes(root):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            errors.append(
                f"{path.relative_to(root)} must not contain Jekyll front matter; use _config.yml"
            )

    config = yaml.safe_load((root / "_config.yml").read_text(encoding="utf-8"))
    if config.get("title") != "Agent Engineering Lab":
        errors.append("_config.yml title drift")
    if "legacy/" not in config.get("exclude", []):
        errors.append("legacy/ must be excluded from V2 Pages build")
    if "jekyll-readme-index" not in config.get("plugins", []):
        errors.append("jekyll-readme-index must render front-matter-free README files")
    readme_index = config.get("readme_index", {})
    if (
        readme_index.get("enabled") is not True
        or readme_index.get("remove_originals") is not True
        or readme_index.get("with_frontmatter") is not False
    ):
        errors.append("readme_index must render README only, without front matter or raw copies")

    lab_pages = sorted((root / "labs").glob("*/README.md"))
    if len(lab_pages) != 25:
        errors.append(f"expected 25 lab pages, found {len(lab_pages)}")

    for path in public_markdown(root):
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_MARKERS:
            if marker in text:
                errors.append(f"{path.relative_to(root)} exposes forbidden marker {marker}")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:", "/")):
                continue
            if "{{" in target or "}}" in target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{path.relative_to(root)} broken link: {target}")
    return errors


def check_built(root: Path, built: Path) -> list[str]:
    errors: list[str] = []
    index = built / "index.html"
    if not index.is_file():
        return [f"rendered index missing: {index}"]
    html = index.read_text(encoding="utf-8")
    if html.count("<h1") != 1:
        errors.append("rendered homepage must contain exactly one h1")
    if "Agent Engineering Lab" not in html:
        errors.append("rendered homepage missing V2 title")
    if "<strong>tutorial-zero</strong>" in html:
        errors.append("rendered navigation retained V1 brand")
    expected_canonical = "https://estelledc.github.io/langchain-langgraph-langsmith-tutorial/"
    if expected_canonical not in html:
        errors.append("rendered homepage canonical URL missing")
    for marker in FORBIDDEN_MARKERS:
        if marker in html:
            errors.append(f"rendered homepage exposes forbidden marker {marker}")

    labs_index = built / "labs/index.html"
    if not labs_index.is_file():
        errors.append("rendered labs index missing")
    for lab_path in sorted((root / "labs").glob("*/README.md")):
        rendered = built / "labs" / lab_path.parent.name / "index.html"
        if not rendered.is_file():
            errors.append(f"rendered lab missing: {lab_path.parent.name}")

    page_404 = built / "404.html"
    if not page_404.is_file() or "noindex,follow" not in page_404.read_text(encoding="utf-8"):
        errors.append("rendered 404 must contain noindex,follow")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--built", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors = check_source(root)
    if args.built:
        errors.extend(check_built(root, args.built.resolve()))
    if errors:
        for error in errors:
            print(f"site: {error}")
        return 1
    mode = "source + rendered" if args.built else "source"
    print(f"site: OK ({mode}; 25 lab pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
