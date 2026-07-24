from __future__ import annotations

import pytest

from robots.libero.prompt_bundle import system_prompt
from rpent.utils.resources import RESOURCES_ABLATION_ENV, resource_context_disabled


def test_resource_ablation_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RESOURCES_ABLATION_ENV, raising=False)

    assert resource_context_disabled() is False
    assert "RESOURCE-ABLATION MODE" not in system_prompt()


def test_resource_ablation_overrides_resource_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RESOURCES_ABLATION_ENV, "1")

    prompt = system_prompt()

    assert resource_context_disabled() is True
    assert "RESOURCE-ABLATION MODE" in prompt[:200]
    assert "Do not call read_text_file or list_dir" in prompt
    assert "Begin immediately with `view_driver_state" in prompt
