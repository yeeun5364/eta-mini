"""Python API 와 다른 언어 서비스(Go, Rust)의 예측이 같은지 비교한다.
기준은 학습한 원본 모델(MODEL_IMPL=sklearn)로 띄운 Python API 다. 서버들이 떠 있어야 한다.
하나라도 최대 오차가 TOLERANCE 분 이상이거나 응답이 없으면 exit 1 → CI 가 멈춘다.
    PY_URL=http://127.0.0.1:8080 TARGETS=go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082 \
        python -m training.check_parity
표준 라이브러리만 쓴다.
"""

import json
import os
import random
import sys
import urllib.request

TOLERANCE = 1e-4  # 분
TIMEOUT = 5       # 초

PY_URL = os.getenv("PY_URL", "http://127.0.0.1:8080").rstrip("/")
TARGETS = dict(
    item.split("=", 1)
    for item in os.getenv("TARGETS", "go=http://127.0.0.1:8081,rust=http://127.0.0.1:8082").split(",")
    if item.strip()
)

# 입력: 고정 3건 + seed 0 무작위 20건 (거리 0.5~8km, 주문 0~10건)
rng = random.Random(0)
INPUTS = [(3.0, 2), (0.0, 0), (8.0, 10)] + [
    (round(rng.uniform(0.5, 8.0), 3), rng.randint(0, 10)) for _ in range(20)
]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        return json.load(resp)


def predict(base_url: str, distance_km: float, pending_orders: int) -> float:
    body = json.dumps({"distance_km": distance_km, "pending_orders": pending_orders}).encode()
    req = urllib.request.Request(f"{base_url}/eta", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return float(json.load(resp)["minutes"])


# 기준이 정말 원본 모델인지 확인한다. rule 등으로 떠 있으면 비교 자체가 의미 없다.
try:
    base_model = get_json(f"{PY_URL}/health").get("model")
    expected = [predict(PY_URL, d, o) for d, o in INPUTS]
except Exception as e:  # noqa: BLE001 - 어떤 이유든 기준을 못 얻으면 중단
    print(f"FAIL 기준 Python API({PY_URL}) 호출 실패: {e}")
    sys.exit(1)
if base_model != "sklearn":
    print(f"FAIL 기준 Python API 의 MODEL_IMPL 이 sklearn 이 아니다: {base_model!r}")
    sys.exit(1)

print(f"기준: python ({PY_URL}, model=sklearn), 입력 {len(INPUTS)}건, 허용 오차 {TOLERANCE:g}분")
ok = True
for name, url in TARGETS.items():
    url = url.rstrip("/")
    try:
        got = [predict(url, d, o) for d, o in INPUTS]
    except Exception as e:  # noqa: BLE001 - 응답이 없으면 그 대상은 FAIL
        print(f"FAIL [{name}] {url} 호출 실패: {e}")
        ok = False
        continue
    diffs = [abs(g - w) for g, w in zip(got, expected)]
    worst = max(range(len(diffs)), key=diffs.__getitem__)
    passed = diffs[worst] < TOLERANCE
    ok &= passed
    d, o = INPUTS[worst]
    print(("PASS " if passed else "FAIL ") + f"[{name}] 최대 오차 {diffs[worst]:.2e}분 "
          f"(입력 {d}km/{o}건: python {expected[worst]:.6f}, {name} {got[worst]:.6f})")

sys.exit(0 if ok else 1)
