import math

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


# ---------- /health ----------

def test_health_returns_ok_and_model_name():
    """GET /health 는 200 과 status=ok, 문자열 model 이름을 돌려준다."""
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert isinstance(body["model"], str) and body["model"]


# ---------- / (소개 페이지) ----------

def test_index_returns_html():
    """GET / 는 200 과 text/html 소개 페이지를 돌려준다."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")
    assert "<html" in res.text.lower()


def test_index_describes_endpoints():
    """소개 페이지는 /health, /eta 엔드포인트와 /docs 링크를 안내한다."""
    text = client.get("/").text
    assert "/health" in text
    assert "/eta" in text
    assert 'href="/docs"' in text


def test_index_shows_current_model_without_placeholder():
    """소개 페이지에 현재 구현체 이름이 들어가고 치환용 자리표시자는 남지 않는다."""
    text = client.get("/").text
    model = client.get("/health").json()["model"]
    assert model in text
    assert "__MODEL_IMPL__" not in text


def test_index_not_in_openapi_schema():
    """/ 는 사람용 페이지이므로 OpenAPI 스키마에 나오지 않는다."""
    paths = client.get("/openapi.json").json()["paths"]
    assert "/" not in paths
    assert {"/health", "/eta"} <= set(paths)


def test_index_post_method_not_allowed():
    """/ 는 GET 전용이므로 POST 는 405 를 돌려준다."""
    assert client.post("/").status_code == 405


# ---------- /eta: 정상 입력 ----------

@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (1, 1, 15.0),      # 10 + 3 + 2
        (2.5, 3, 23.5),    # 10 + 7.5 + 6
        (5, 0, 25.0),      # 10 + 15
        (0, 4, 18.0),      # 10 + 8
    ],
)
def test_eta_known_values(distance_km, pending_orders, expected):
    """알려진 입력에 대해 규칙(10 + 3*거리 + 2*주문)대로 분을 돌려준다."""
    res = client.post("/eta", json={"distance_km": distance_km, "pending_orders": pending_orders})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


def test_eta_response_shape():
    """응답은 minutes(숫자)와 model(문자열) 두 필드만 가진다."""
    res = client.post("/eta", json={"distance_km": 1, "pending_orders": 1})
    body = res.json()
    assert set(body) == {"minutes", "model"}
    assert isinstance(body["minutes"], (int, float))
    assert isinstance(body["model"], str)


def test_eta_model_matches_health():
    """/eta 의 model 은 /health 가 알려 주는 구현체 이름과 같다."""
    health_model = client.get("/health").json()["model"]
    eta_model = client.post("/eta", json={"distance_km": 1, "pending_orders": 1}).json()["model"]
    assert eta_model == health_model


# ---------- /eta: 경계값 ----------

def test_eta_zero_inputs_returns_base_time():
    """거리 0, 주문 0 이면 기본 10분을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 0, "pending_orders": 0})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10.0)


def test_eta_large_inputs():
    """큰 값(거리 1000km, 주문 10000건)도 200 과 규칙대로의 값을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 1000, "pending_orders": 10000})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10 + 3 * 1000 + 2 * 10000)


def test_eta_increases_with_distance_and_orders():
    """거리나 주문 수가 늘면 예상 시간도 늘어난다."""
    def minutes(d, p):
        return client.post("/eta", json={"distance_km": d, "pending_orders": p}).json()["minutes"]

    assert minutes(2, 1) > minutes(1, 1)
    assert minutes(1, 2) > minutes(1, 1)


# ---------- /eta: 잘못된 입력 → 422 ----------

@pytest.mark.parametrize(
    "payload",
    [
        {"distance_km": -1, "pending_orders": 0},          # 음수 거리
        {"distance_km": -0.01, "pending_orders": 0},       # 0 바로 아래
        {"distance_km": 1, "pending_orders": -1},          # 음수 주문
        {"distance_km": "abc", "pending_orders": 1},       # 숫자가 아닌 거리
        {"distance_km": 1, "pending_orders": "many"},      # 숫자가 아닌 주문
        {"distance_km": 1, "pending_orders": 1.5},         # 정수가 아닌 주문
        {"distance_km": None, "pending_orders": 1},        # null
        {"pending_orders": 1},                             # 거리 누락
        {"distance_km": 1},                                # 주문 누락
        {},                                                # 빈 본문
    ],
)
def test_eta_invalid_input_returns_422(payload):
    """음수·숫자 아님·정수 아님·누락 입력은 422 를 돌려준다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422


def test_eta_non_json_body_returns_422():
    """JSON 이 아닌 본문은 422 를 돌려준다."""
    res = client.post("/eta", content=b"not json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_eta_get_method_not_allowed():
    """/eta 는 POST 전용이므로 GET 은 405 를 돌려준다."""
    res = client.get("/eta")
    assert res.status_code == 405


def test_health_post_method_not_allowed():
    """/health 는 GET 전용이므로 POST 는 405 를 돌려준다."""
    res = client.post("/health")
    assert res.status_code == 405


@pytest.mark.parametrize(
    "raw_body",
    [
        "[1, 2]",                                          # 객체가 아닌 배열 본문
        '{"distance_km": [1], "pending_orders": 1}',       # 값이 배열
        '{"distance_km": {"v": 1}, "pending_orders": 1}',  # 값이 객체
    ],
)
def test_eta_wrong_json_shape_returns_422(raw_body):
    """JSON 이지만 모양이 틀린 본문은 422 를 돌려준다."""
    res = client.post("/eta", content=raw_body, headers={"Content-Type": "application/json"})
    assert res.status_code == 422


def test_eta_non_json_content_type_returns_422():
    """Content-Type 이 JSON 이 아니면 본문이 맞아도 422 를 돌려준다."""
    res = client.post(
        "/eta",
        content='{"distance_km": 1, "pending_orders": 1}',
        headers={"Content-Type": "text/plain"},
    )
    assert res.status_code == 422


@pytest.mark.parametrize(
    ("payload", "bad_field"),
    [
        ({"distance_km": -1, "pending_orders": 0}, "distance_km"),
        ({"distance_km": 1, "pending_orders": -1}, "pending_orders"),
        ({"distance_km": 1}, "pending_orders"),
    ],
)
def test_eta_422_detail_points_to_bad_field(payload, bad_field):
    """422 응답의 detail 은 문제가 된 필드를 loc 로 가리킨다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 422
    locs = [err["loc"] for err in res.json()["detail"]]
    assert ["body", bad_field] in [list(loc) for loc in locs]


# ---------- /eta: 경계값 (추가) ----------

@pytest.mark.parametrize(
    ("raw_distance", "expected"),
    [
        ("-0.0", 10.0),     # 음의 0 은 ge=0 을 통과한다
        ("1e-9", 10.0),     # 아주 작은 양수
    ],
)
def test_eta_tiny_distance(raw_distance, expected):
    """0 근처의 거리(-0.0, 1e-9)는 200 과 기본 시간에 가까운 값을 돌려준다."""
    body = f'{{"distance_km": {raw_distance}, "pending_orders": 0}}'
    res = client.post("/eta", content=body, headers={"Content-Type": "application/json"})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


def test_eta_huge_integer_orders():
    """64비트를 넘는 큰 정수 주문 수도 200 과 규칙대로의 값을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 1, "pending_orders": 10**23})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(10 + 3 + 2 * 10**23)


# ---------- /eta: 현재 동작 고정 (명세 미정, lax 형 변환) ----------
# Pydantic 기본(lax) 모드가 받아 주는 입력들. strict 로 바꾸기로 하면
# 이 테스트들의 기대값을 422 로 뒤집는다.

@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"distance_km": "1.5", "pending_orders": 1}, 16.5),   # 숫자 문자열 거리
        ({"distance_km": 1, "pending_orders": "3"}, 19.0),     # 숫자 문자열 주문
        ({"distance_km": 1, "pending_orders": 2.0}, 17.0),     # 소수부 0 인 float 주문
        ({"distance_km": True, "pending_orders": 1}, 15.0),    # bool → 1
        ({"distance_km": 1, "pending_orders": False}, 13.0),   # bool → 0
    ],
)
def test_eta_lax_coercion_current_behavior(payload, expected):
    """숫자 문자열·정수값 float·bool 은 현재 숫자로 변환되어 200 을 돌려준다."""
    res = client.post("/eta", json=payload)
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(expected)


def test_eta_extra_field_is_ignored():
    """정의되지 않은 추가 필드는 현재 무시되고 200 을 돌려준다."""
    res = client.post("/eta", json={"distance_km": 1, "pending_orders": 1, "extra": 5})
    assert res.status_code == 200
    assert res.json()["minutes"] == pytest.approx(15.0)


# ---------- /eta: 비유한(non-finite) 값 — 알려진 버그 ----------
# 계약: 잘못된 입력은 422, 성공 응답의 minutes 는 항상 유한한 숫자.
# 현재 앱이 이 계약을 어기므로 xfail(strict) 로 표시한다. 앱을 고치면
# XPASS 가 실패로 잡히므로 그때 xfail 표시를 지운다.

def _post_raw(body: str):
    return TestClient(app, raise_server_exceptions=False).post(
        "/eta", content=body, headers={"Content-Type": "application/json"}
    )


@pytest.mark.xfail(strict=True, reason="버그: NaN 거리 → 500 (422 여야 함)")
def test_eta_nan_distance_returns_422():
    """NaN 거리는 422 를 돌려준다."""
    res = _post_raw('{"distance_km": NaN, "pending_orders": 0}')
    assert res.status_code == 422


@pytest.mark.xfail(strict=True, reason="버그: Infinity 거리 → 200, minutes=null")
def test_eta_infinite_distance_returns_422():
    """Infinity 거리는 422 를 돌려준다."""
    res = _post_raw('{"distance_km": Infinity, "pending_orders": 0}')
    assert res.status_code == 422


@pytest.mark.xfail(strict=True, reason="버그: 1e308 거리 → 계산 결과 inf → minutes=null")
def test_eta_overflowing_distance_never_returns_null_minutes():
    """계산이 넘치는 거리(1e308)는 422 이거나, 200 이면 minutes 가 유한한 숫자다."""
    res = _post_raw('{"distance_km": 1e308, "pending_orders": 0}')
    if res.status_code == 200:
        minutes = res.json()["minutes"]
        assert isinstance(minutes, (int, float)) and math.isfinite(minutes)
    else:
        assert res.status_code == 422
