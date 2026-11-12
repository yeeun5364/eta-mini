import pytest

from api.predictors.rule import estimate_delivery_minutes


@pytest.mark.parametrize(
    ("distance_km", "pending_orders", "expected"),
    [
        (0, 0, 10.0),
        (1, 0, 13.0),
        (0, 1, 12.0),
        (2.5, 3, 23.5),
        (1000, 10000, 23010.0),
    ],
)
def test_estimate_known_values(distance_km, pending_orders, expected):
    """기준선 규칙 10 + 3*거리 + 2*주문 의 알려진 값을 돌려준다."""
    assert estimate_delivery_minutes(distance_km, pending_orders) == pytest.approx(expected)


def test_estimate_returns_float():
    """정수 입력이어도 float 를 돌려준다."""
    assert isinstance(estimate_delivery_minutes(1, 1), float)
