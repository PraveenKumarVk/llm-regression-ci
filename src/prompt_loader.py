"""
Loads versioned prompt artifacts from the /prompts directory.

Each file is a YAML named <version>.yaml (e.g. v1.0.0.yaml).
The loader never mutates files — it is purely a reader.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.models import FewShotExample, PromptVersion

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


def _parse_file(path: Path) -> PromptVersion:
    raw = yaml.safe_load(path.read_text())
    examples = [FewShotExample(**ex) for ex in raw.get("few_shot_examples", [])]
    return PromptVersion(
        version=raw["version"],
        timestamp=raw["timestamp"],
        system_prompt=raw["system_prompt"],
        few_shot_examples=examples,
    )


def load_version(version: str, prompts_dir: Path = _PROMPTS_DIR) -> PromptVersion:
    """Load a specific version, e.g. load_version('v1.0.0')."""
    path = prompts_dir / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version '{version}' not found at {path}")
    return _parse_file(path)


def load_latest(prompts_dir: Path = _PROMPTS_DIR) -> PromptVersion:
    """
    Return the highest semantic version found in prompts_dir.
    Versions must follow vMAJOR.MINOR.PATCH naming.
    """
    candidates = [
        p for p in prompts_dir.glob("*.yaml")
        if _VERSION_RE.match(p.stem)
    ]
    if not candidates:
        raise FileNotFoundError(f"No versioned prompt files found in {prompts_dir}")

    def _sort_key(p: Path) -> tuple[int, int, int]:
        parts = p.stem.lstrip("v").split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    latest = max(candidates, key=_sort_key)
    return _parse_file(latest)


def list_versions(prompts_dir: Path = _PROMPTS_DIR) -> list[str]:
    """Return all available version strings, sorted ascending."""
    candidates = [
        p for p in prompts_dir.glob("*.yaml")
        if _VERSION_RE.match(p.stem)
    ]

    def _sort_key(p: Path) -> tuple[int, int, int]:
        parts = p.stem.lstrip("v").split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))

    return [p.stem for p in sorted(candidates, key=_sort_key)]
