"""Tests for Phase 1 Step 1: versioned prompt artifacts."""

from pathlib import Path

import pytest
import yaml

from src.models import FewShotExample, PromptVersion
from src.prompt_loader import list_versions, load_latest, load_version

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

class TestPromptVersion:
    def test_content_hash_excludes_version_and_timestamp(self):
        base = PromptVersion(
            version="v1.0.0",
            timestamp="2026-06-24",
            system_prompt="You are an analyst.",
            few_shot_examples=[],
        )
        renamed = PromptVersion(
            version="v9.9.9",
            timestamp="2099-01-01",
            system_prompt="You are an analyst.",
            few_shot_examples=[],
        )
        assert base.content_hash == renamed.content_hash

    def test_content_hash_changes_with_system_prompt(self):
        a = PromptVersion(version="v1.0.0", timestamp="2026-06-24", system_prompt="Prompt A")
        b = PromptVersion(version="v1.0.0", timestamp="2026-06-24", system_prompt="Prompt B")
        assert a.content_hash != b.content_hash

    def test_content_hash_changes_with_few_shot_examples(self):
        base = PromptVersion(
            version="v1.0.0",
            timestamp="2026-06-24",
            system_prompt="Same prompt",
            few_shot_examples=[],
        )
        with_example = PromptVersion(
            version="v1.0.0",
            timestamp="2026-06-24",
            system_prompt="Same prompt",
            few_shot_examples=[
                FewShotExample(question="Q", context="C", answer="A")
            ],
        )
        assert base.content_hash != with_example.content_hash

    def test_content_hash_is_stable(self):
        """Same inputs always produce the same hash."""
        kwargs = dict(
            version="v1.0.0",
            timestamp="2026-06-24",
            system_prompt="You are an analyst.",
            few_shot_examples=[FewShotExample(question="Q?", context="ctx", answer="NOT_IN_DOCUMENT")],
        )
        assert PromptVersion(**kwargs).content_hash == PromptVersion(**kwargs).content_hash

    def test_content_hash_is_64_hex_chars(self):
        pv = PromptVersion(version="v1.0.0", timestamp="2026-06-24", system_prompt="x")
        assert len(pv.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in pv.content_hash)


# ---------------------------------------------------------------------------
# Prompt loader against real prompts/v1.0.0.yaml
# ---------------------------------------------------------------------------


class TestLoadVersion:
    def test_loads_v1(self):
        pv = load_version("v1.0.0", PROMPTS_DIR)
        assert pv.version == "v1.0.0"
        assert pv.timestamp == "2026-06-24"

    def test_system_prompt_contains_not_in_document_rule(self):
        pv = load_version("v1.0.0", PROMPTS_DIR)
        assert "NOT_IN_DOCUMENT" in pv.system_prompt

    def test_system_prompt_contains_citation_rule(self):
        pv = load_version("v1.0.0", PROMPTS_DIR)
        assert "CITATION" in pv.system_prompt

    def test_few_shot_examples_loaded(self):
        pv = load_version("v1.0.0", PROMPTS_DIR)
        assert len(pv.few_shot_examples) >= 2

    def test_not_in_document_example_present(self):
        """The NOT_IN_DOCUMENT failure mode must be represented in few-shots."""
        pv = load_version("v1.0.0", PROMPTS_DIR)
        refusals = [ex for ex in pv.few_shot_examples if "NOT_IN_DOCUMENT" in ex.answer]
        assert len(refusals) >= 1, "At least one NOT_IN_DOCUMENT example is required"

    def test_citation_example_present(self):
        """The citation format must be demonstrated in few-shots."""
        pv = load_version("v1.0.0", PROMPTS_DIR)
        citations = [ex for ex in pv.few_shot_examples if "[CITATION:" in ex.answer]
        assert len(citations) >= 1, "At least one citation example is required"

    def test_content_hash_is_populated(self):
        pv = load_version("v1.0.0", PROMPTS_DIR)
        assert len(pv.content_hash) == 64

    def test_missing_version_raises(self):
        with pytest.raises(FileNotFoundError, match="v9.9.9"):
            load_version("v9.9.9", PROMPTS_DIR)


class TestLoadLatest:
    def test_returns_v1_when_only_version(self):
        pv = load_latest(PROMPTS_DIR)
        assert pv.version == "v1.0.0"

    def test_picks_highest_semver(self, tmp_path):
        for ver, prompt in [("v1.0.0", "old"), ("v1.2.0", "mid"), ("v2.0.0", "new")]:
            (tmp_path / f"{ver}.yaml").write_text(
                yaml.dump({"version": ver, "timestamp": "2026-06-24", "system_prompt": prompt, "few_shot_examples": []})
            )
        pv = load_latest(tmp_path)
        assert pv.version == "v2.0.0"
        assert pv.system_prompt.strip() == "new"

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_latest(tmp_path)


class TestListVersions:
    def test_includes_v1(self):
        versions = list_versions(PROMPTS_DIR)
        assert "v1.0.0" in versions

    def test_sorted_ascending(self, tmp_path):
        for ver in ["v2.0.0", "v1.0.0", "v1.5.0"]:
            (tmp_path / f"{ver}.yaml").write_text(
                yaml.dump({"version": ver, "timestamp": "2026-06-24", "system_prompt": "x", "few_shot_examples": []})
            )
        assert list_versions(tmp_path) == ["v1.0.0", "v1.5.0", "v2.0.0"]

    def test_ignores_non_version_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text("key: value")
        (tmp_path / "v1.0.0.yaml").write_text(
            yaml.dump({"version": "v1.0.0", "timestamp": "2026-06-24", "system_prompt": "x", "few_shot_examples": []})
        )
        assert list_versions(tmp_path) == ["v1.0.0"]
