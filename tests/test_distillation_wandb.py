from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.distillation.wandb_compat import (
    apply_metric_logger_idempotent_finish,
    apply_wandb_offline_fallback,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_resolver():
    path = ROOT / "scripts/distillation/resolve_wandb_mode.py"
    spec = importlib.util.spec_from_file_location("resolve_wandb_mode", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def resolver(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())
    return _load_resolver()


def _args(mode: str) -> list[str]:
    return [
        "--mode",
        mode,
        "--username",
        "YichengDong",
        "--entity",
        "ethan-dong-nanjing-university-org",
        "--timeout",
        "1",
    ]


def test_auto_without_credentials_uses_offline(resolver, monkeypatch, capsys):
    monkeypatch.setattr(resolver, "_has_credentials", lambda: False)
    assert resolver.main(_args("auto")) == 0
    assert capsys.readouterr().out.strip() == "offline"


def test_explicit_online_without_credentials_fails(resolver, monkeypatch):
    monkeypatch.setattr(resolver, "_has_credentials", lambda: False)
    assert resolver.main(_args("online")) == 2


@pytest.mark.parametrize(
    ("mode", "expected_code", "expected_stdout"),
    [("auto", 0, "offline"), ("online", 2, "")],
)
def test_verification_failure_policy(
    resolver, monkeypatch, capsys, mode, expected_code, expected_stdout
):
    monkeypatch.setattr(resolver, "_has_credentials", lambda: True)

    def fail(**_kwargs):
        raise RuntimeError("secret-free test failure")

    monkeypatch.setattr(resolver, "_verify_online", fail)
    assert resolver.main(_args(mode)) == expected_code
    assert capsys.readouterr().out.strip() == expected_stdout


def test_online_verifies_username_and_entity(resolver, monkeypatch):
    requested = {}

    class Api:
        viewer = SimpleNamespace(username="YichengDong")

        def __init__(self, timeout):
            requested["timeout"] = timeout

        def projects(self, *, entity, per_page):
            requested.update(entity=entity, per_page=per_page)
            return iter(())

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=Api))
    resolver._verify_online(
        username="YichengDong",
        entity="ethan-dong-nanjing-university-org",
        timeout=15,
    )
    assert requested == {
        "timeout": 15,
        "entity": "ethan-dong-nanjing-university-org",
        "per_page": 1,
    }


def test_failed_online_init_retries_once_offline(monkeypatch):
    calls = []

    class FakeWandb:
        def init(self, **kwargs):
            calls.append(("init", kwargs.get("mode")))
            if kwargs.get("mode") != "offline":
                raise RuntimeError("online unavailable")
            return "offline-run"

        def teardown(self, *, exit_code):
            calls.append(("teardown", exit_code))

    fake = FakeWandb()
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("RPENT_WANDB_AUTO_FALLBACK", "1")
    apply_wandb_offline_fallback(fake)
    assert fake.init(project="harness-vla-sft") == "offline-run"
    assert calls == [("init", None), ("teardown", 1), ("init", "offline")]
    assert fake.init._rpent_offline_fallback is True


def test_metric_logger_finish_is_idempotent():
    calls = []

    class FakeMetricLogger:
        _all_loggers = []

        def finish(self):
            calls.append("finish")

    apply_metric_logger_idempotent_finish(FakeMetricLogger)
    logger = FakeMetricLogger()
    logger.finish()
    logger.finish()
    assert calls == ["finish"]
