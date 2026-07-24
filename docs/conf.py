"""Sphinx configuration for Tiresias docs."""

from __future__ import annotations

from pathlib import Path
import tomllib

project_root = Path(__file__).resolve().parents[1]

with open(project_root / "pyproject.toml", "rb") as handle:
    project_meta = tomllib.load(handle)

project = project_meta["project"]["name"]
copyright = "2026, The University of Texas Southwestern Medical Center"
author = ", ".join(author["name"] for author in project_meta["project"]["authors"])
version = project_meta["project"]["version"]
release = version

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}

exclude_patterns = ["_build", "_static", "_site", "**/Thumbs.db", "**/.DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "furo"
myst_heading_anchors = 3
