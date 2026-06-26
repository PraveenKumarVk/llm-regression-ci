"""
Versioned golden-dataset loader.

Files on disk: data/golden_dataset_v{major}.{minor}.{patch}.json
Each eval run records which dataset version it used so the eval bar itself
can be tracked for regressions independently of the prompt under test.

Public API
----------
list_versions(data_dir)       -> ["v1.0.0", "v1.1.0", ...]  (semver sorted)
load_version(version, ...)    -> GoldenDataset
load_latest(data_dir)         -> GoldenDataset  (highest semver)
load_dataset(path)            -> GoldenDataset  (explicit path; for test fixtures)
save_dataset(dataset, ...)    -> Path  (writes golden_dataset_{version}.json)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.models import GoldenDataset, GoldenTestCase

_DATA_DIR = Path(__file__).parent.parent / "data"
_FILE_RE = re.compile(r"^golden_dataset_(v\d+\.\d+\.\d+)\.json$")


def _versioned_path(version: str, data_dir: Path) -> Path:
    return data_dir / f"golden_dataset_{version}.json"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def list_versions(data_dir: Path | str | None = None) -> list[str]:
    """Return semver-sorted list of dataset version strings available on disk."""
    d = Path(data_dir) if data_dir is not None else _DATA_DIR
    hits: list[tuple[int, int, int]] = []
    for p in d.glob("golden_dataset_v*.json"):
        m = _FILE_RE.match(p.name)
        if m:
            parts = m.group(1).lstrip("v").split(".")
            hits.append((int(parts[0]), int(parts[1]), int(parts[2])))
    hits.sort()
    return [f"v{maj}.{mn}.{pat}" for maj, mn, pat in hits]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path | str) -> GoldenDataset:
    """Load from an explicit file path. Use this in test fixtures only."""
    return GoldenDataset.model_validate_json(Path(path).read_text())


def load_version(version: str, data_dir: Path | str | None = None) -> GoldenDataset:
    """Load golden_dataset_{version}.json from data_dir."""
    d = Path(data_dir) if data_dir is not None else _DATA_DIR
    return load_dataset(_versioned_path(version, d))


def load_latest(data_dir: Path | str | None = None) -> GoldenDataset:
    """Load the highest semver dataset file from data_dir."""
    versions = list_versions(data_dir)
    if not versions:
        raise FileNotFoundError(
            f"No golden_dataset_v*.json files found in {data_dir or _DATA_DIR}"
        )
    return load_version(versions[-1], data_dir)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_dataset(dataset: GoldenDataset, data_dir: Path | str | None = None) -> Path:
    """
    Serialise *dataset* to data_dir/golden_dataset_{dataset.version}.json.
    Returns the path written.
    """
    d = Path(data_dir) if data_dir is not None else _DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = _versioned_path(dataset.version, d)
    p.write_text(json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return p


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def filter_by_category(dataset: GoldenDataset, category: str) -> list[GoldenTestCase]:
    return [c for c in dataset.cases if c.failure_mode_category == category]


def filter_by_difficulty(dataset: GoldenDataset, difficulty: str) -> list[GoldenTestCase]:
    return [c for c in dataset.cases if c.difficulty == difficulty]


def filter_refusals(dataset: GoldenDataset) -> list[GoldenTestCase]:
    return [c for c in dataset.cases if c.expected_is_refusal]
