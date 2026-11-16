//! 계약 테스트: ONNX Runtime 없이 가짜 예측기로 라우터를 호출한다.

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::handler::{Predictor, router};

/// 알려진 값을 내는 가짜 예측기: 10 + 2*거리 + 3*주문.
fn app() -> Router {
    let p: Arc<Predictor> = Arc::new(|d: f32, o: f32| Ok(10.0 + 2.0 * d + 3.0 * o));
    router("fake", p)
}

async fn call(method: Method, uri: &str, ct: Option<&str>, body: &str) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(ct) = ct {
        req = req.header("content-type", ct);
    }
    let resp = app().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn post_eta(body: &str) -> (StatusCode, Value) {
    call(Method::POST, "/eta", Some("application/json"), body).await
}

fn error_types(v: &Value) -> Vec<String> {
    v["detail"].as_array().unwrap().iter().map(|e| e["type"].as_str().unwrap().to_string()).collect()
}

#[tokio::test]
async fn health_returns_status_and_model() {
    // GET /health 는 200 과 {"status": "ok", "model": 이름} 을 돌려준다.
    let (s, v) = call(Method::GET, "/health", None, "").await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v, json!({"status": "ok", "model": "fake"}));
}

#[tokio::test]
async fn eta_normal_input() {
    // 정상 입력은 200 과 예측값·모델 이름을 돌려준다.
    let (s, v) = post_eta(r#"{"distance_km": 3.0, "pending_orders": 2}"#).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v, json!({"minutes": 22.0, "model": "fake"}));
}

#[tokio::test]
async fn eta_zero_and_large_values() {
    // 경계값 0 과 큰 값도 받아들인다.
    let (s, v) = post_eta(r#"{"distance_km": 0, "pending_orders": 0}"#).await;
    assert_eq!((s, v["minutes"].as_f64()), (StatusCode::OK, Some(10.0)));
    let (s, v) = post_eta(r#"{"distance_km": 1000, "pending_orders": 1000}"#).await;
    assert_eq!((s, v["minutes"].as_f64()), (StatusCode::OK, Some(5010.0)));
}

#[tokio::test]
async fn eta_lax_coercion_like_pydantic() {
    // Pydantic lax 모드처럼 숫자 문자열, bool, 소수부 0 인 실수를 받아들이고 추가 필드는 무시한다.
    let (s, v) = post_eta(r#"{"distance_km": "1.5", "pending_orders": "2", "extra": 1}"#).await;
    assert_eq!((s, v["minutes"].as_f64()), (StatusCode::OK, Some(19.0)));
    let (s, v) = post_eta(r#"{"distance_km": true, "pending_orders": 2.0}"#).await;
    assert_eq!((s, v["minutes"].as_f64()), (StatusCode::OK, Some(18.0)));
}

#[tokio::test]
async fn eta_negative_is_422() {
    // 음수는 두 필드 모두 422 greater_than_equal.
    let (s, v) = post_eta(r#"{"distance_km": -1, "pending_orders": -1}"#).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error_types(&v), ["greater_than_equal", "greater_than_equal"]);
    assert_eq!(v["detail"][0]["loc"], json!(["body", "distance_km"]));
}

#[tokio::test]
async fn eta_not_a_number_is_422() {
    // 숫자가 아닌 값은 422.
    let (s, v) = post_eta(r#"{"distance_km": "abc", "pending_orders": 1.5}"#).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error_types(&v), ["float_parsing", "int_from_float"]);
    let (s, v) = post_eta(r#"{"distance_km": null, "pending_orders": [1]}"#).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error_types(&v), ["float_type", "int_type"]);
}

#[tokio::test]
async fn eta_missing_fields_is_422() {
    // 필드가 없으면 422 missing.
    let (s, v) = post_eta("{}").await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(error_types(&v), ["missing", "missing"]);
}

#[tokio::test]
async fn eta_malformed_body_is_422_not_400() {
    // 깨진 JSON, 빈 본문, 객체가 아닌 JSON 도 400 이 아니라 422.
    let (s, v) = post_eta(r#"{"distance_km": 1,"#).await;
    assert_eq!((s, error_types(&v)), (StatusCode::UNPROCESSABLE_ENTITY, vec!["json_invalid".to_string()]));
    let (s, v) = post_eta("").await;
    assert_eq!((s, error_types(&v)), (StatusCode::UNPROCESSABLE_ENTITY, vec!["missing".to_string()]));
    let (s, v) = post_eta("[1, 2]").await;
    assert_eq!((s, error_types(&v)), (StatusCode::UNPROCESSABLE_ENTITY, vec!["model_attributes_type".to_string()]));
}

#[tokio::test]
async fn eta_wrong_content_type_is_422_not_415() {
    // JSON 이 아닌 Content-Type 은 415 가 아니라 422, Content-Type 이 없으면 JSON 으로 읽는다.
    let body = r#"{"distance_km": 3, "pending_orders": 2}"#;
    let (s, _) = call(Method::POST, "/eta", Some("text/plain"), body).await;
    assert_eq!(s, StatusCode::UNPROCESSABLE_ENTITY);
    let (s, _) = call(Method::POST, "/eta", None, body).await;
    assert_eq!(s, StatusCode::OK);
}

#[tokio::test]
async fn eta_value_too_large_for_float32_is_422() {
    // float32 로 표현할 수 없는 값은 422 finite_number.
    let (s, v) = post_eta(r#"{"distance_km": 1e300, "pending_orders": 1}"#).await;
    assert_eq!((s, error_types(&v)), (StatusCode::UNPROCESSABLE_ENTITY, vec!["finite_number".to_string()]));
}

#[tokio::test]
async fn wrong_method_is_405() {
    // 등록하지 않은 메서드는 405.
    let (s, _) = call(Method::GET, "/eta", None, "").await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
}
