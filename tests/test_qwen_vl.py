from __future__ import annotations

import pytest

from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.base import build_planner
from rpent.planner.qwen_vl import (
    DEFAULT_QWEN_VL_BASE_URL,
    DEFAULT_QWEN_VL_MODEL,
    build_qwen_vl_model,
    is_qwen_vl_model,
)


def test_qwen_vl_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWEN_VL_MODEL", raising=False)
    monkeypatch.delenv("QWEN_VL_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN_VL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = build_qwen_vl_model("qwen-vl:")

    assert model.model_name == DEFAULT_QWEN_VL_MODEL
    assert str(model._provider.base_url).rstrip("/") == DEFAULT_QWEN_VL_BASE_URL


def test_qwen_vl_does_not_reuse_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QWEN_VL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")

    model = build_qwen_vl_model("qwen-vl:local-qwen")

    assert model._provider.client.api_key == "EMPTY"


def test_qwen_vl_ignores_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")

    model = build_qwen_vl_model("qwen-vl:local-qwen")

    assert model._provider.client._client._trust_env is False


def test_qwen_vl_explicit_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QWEN_VL_API_KEY", "test-key")

    model = build_qwen_vl_model(
        "qwen-vl:local-qwen",
        base_url="http://qwen.internal:9000",
    )

    assert model.model_name == "local-qwen"
    assert str(model._provider.base_url).rstrip("/") == ("http://qwen.internal:9000/v1")


def test_build_planner_accepts_qwen_vl_prefix(tmp_path) -> None:
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="qwen_smoke",
        env_name="libero",
        model="qwen-vl:local-qwen",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert isinstance(planner, ApiAgentLoop)
    assert planner._model.model_name == "local-qwen"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("qwen-vl:Qwen3.5-9B", True),
        ("openai-chat:Qwen3.5-9B", False),
    ],
)
def test_is_qwen_vl_model(model: str, expected: bool) -> None:
    assert is_qwen_vl_model(model) is expected


@pytest.mark.parametrize("base_url", ["localhost:8000", "file:///tmp/model"])
def test_qwen_vl_rejects_invalid_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match=r"absolute http\(s\) URL"):
        build_qwen_vl_model(
            "qwen-vl:Qwen3.5-9B",
            base_url=base_url,
        )
