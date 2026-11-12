"""MODEL_IMPL 환경변수로 구현체를 고르는 계약을 확인한다.

MODEL_IMPL 은 import 시점에 읽히므로 api.app 을 다시 불러온다.
테스트가 끝나면 원래 환경으로 다시 불러와 다른 테스트에 영향을 주지 않게 한다.
"""
import importlib

import pytest
from fastapi.testclient import TestClient

import api.app


@pytest.fixture
def reload_app(monkeypatch):
    """환경변수를 바꾼 뒤 api.app 을 다시 불러오는 함수를 주고, 끝나면 원상 복구한다."""
    def _reload():
        return importlib.reload(api.app)

    yield _reload
    monkeypatch.undo()
    importlib.reload(api.app)


def test_default_model_is_rule(monkeypatch, reload_app):
    """MODEL_IMPL 이 없으면 rule 구현체를 쓰고 /health 와 /eta 가 'rule' 을 알린다."""
    monkeypatch.delenv("MODEL_IMPL", raising=False)
    client = TestClient(reload_app().app)

    assert client.get("/health").json()["model"] == "rule"
    res = client.post("/eta", json={"distance_km": 1, "pending_orders": 1})
    assert res.json() == {"minutes": pytest.approx(15.0), "model": "rule"}


def test_unknown_model_impl_fails_at_import(monkeypatch, reload_app):
    """존재하지 않는 구현체 이름이면 앱이 뜨지 않고 import 단계에서 실패한다."""
    monkeypatch.setenv("MODEL_IMPL", "does_not_exist")
    with pytest.raises(ModuleNotFoundError):
        reload_app()
