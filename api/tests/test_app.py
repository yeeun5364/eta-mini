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
