//! api/app.py 와 같은 계약(POST /eta, GET /health)을 axum 으로 구현한다.
//!
//! axum 의 `Json` 추출기는 형식 오류에 400·415 를 돌려주므로 쓰지 않는다.
//! 본문을 직접 읽고 파싱해서 모든 입력 오류를 FastAPI 처럼 422 `{"detail": [...]}` 로 돌려준다.

use std::sync::Arc;

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Serialize;
use serde_json::{Map, Value, json};

/// (거리 km, 대기 주문 수) → 예상 시간(분).
/// f32 를 쓰는 이유: ONNX 모델 입력 "X" 가 float32 이기 때문이다.
pub type Predictor = dyn Fn(f32, f32) -> Result<f32, String> + Send + Sync;

const MAX_BODY_BYTES: usize = 1 << 20;

#[derive(Clone)]
struct AppState {
    model_name: Arc<str>,
    predict: Arc<Predictor>,
}

/// FastAPI/Pydantic 의 422 오류 항목과 같은 모양: {"type", "loc", "msg", "input"}.
#[derive(Debug, Serialize, PartialEq)]
pub struct ValidationError {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub loc: Vec<Value>,
    pub msg: &'static str,
    pub input: Value,
}

pub fn router(model_name: &str, predict: Arc<Predictor>) -> Router {
    let state = AppState { model_name: model_name.into(), predict };
    // 등록하지 않은 메서드는 axum 이 405 를 돌려준다 (FastAPI 와 같음).
    Router::new()
        .route("/health", get(health))
        .route("/eta", post(eta))
        .with_state(state)
}

async fn health(State(st): State<AppState>) -> Response {
    json_response(StatusCode::OK, &json!({"status": "ok", "model": &*st.model_name}))
}

async fn eta(State(st): State<AppState>, headers: HeaderMap, body: Body) -> Response {
    let (distance, orders) = match parse_eta_request(&headers, body).await {
        Ok(v) => v,
        Err(errs) => return unprocessable(errs),
    };
    let predict = st.predict.clone();
    // ONNX 실행은 동기 호출이므로 비동기 작업자 스레드를 막지 않게 따로 돌린다.
    let result = tokio::task::spawn_blocking(move || predict(distance, orders)).await;
    let minutes = match result {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => {
            eprintln!("prediction failed: {e}");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &json!({"detail": "prediction failed"}));
        }
        Err(e) => {
            eprintln!("prediction task failed: {e}");
            return json_response(StatusCode::INTERNAL_SERVER_ERROR, &json!({"detail": "prediction failed"}));
        }
    };
    if !minutes.is_finite() {
        // 성공 응답의 minutes 는 항상 유한한 숫자여야 한다.
        return unprocessable(vec![ValidationError {
            kind: "finite_number",
            loc: vec![json!("body")],
            msg: "Input produces a non-finite prediction",
            input: Value::Null,
        }]);
    }
    json_response(StatusCode::OK, &json!({"minutes": minutes as f64, "model": &*st.model_name}))
}

fn unprocessable(errs: Vec<ValidationError>) -> Response {
    json_response(StatusCode::UNPROCESSABLE_ENTITY, &json!({"detail": errs}))
}

fn json_response(status: StatusCode, v: &Value) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], v.to_string()).into_response()
}

fn json_invalid(offset: usize) -> Vec<ValidationError> {
    vec![ValidationError {
        kind: "json_invalid",
        loc: vec![json!("body"), json!(offset)],
        msg: "JSON decode error",
        input: json!({}),
    }]
}

/// FastAPI 처럼 Content-Type 이 없거나 JSON(application/json, application/*+json)일 때만 JSON 으로 읽는다.
fn is_json_content_type(headers: &HeaderMap) -> bool {
    let Some(ct) = headers.get(header::CONTENT_TYPE) else { return true };
    let Ok(ct) = ct.to_str() else { return false };
    let mt = ct.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    mt == "application/json" || (mt.starts_with("application/") && mt.ends_with("+json"))
}

/// FastAPI(Pydantic lax 모드)의 EtaRequest 검증 규칙을 따른다.
///   - distance_km: 숫자(정수·실수·bool·숫자 문자열), >= 0
///   - pending_orders: 정수(정수·소수부 0 인 실수·bool·정수 문자열), >= 0
///   - 추가 필드는 무시
///
/// Python 과 다른 점(의도적): float32 로 표현할 수 없는 큰 값은 422.
pub async fn parse_eta_request(headers: &HeaderMap, body: Body) -> Result<(f32, f32), Vec<ValidationError>> {
    // 너무 큰 본문도 413 이 아니라 422 로 돌려준다.
    let bytes = to_bytes(body, MAX_BODY_BYTES).await.map_err(|_| json_invalid(0))?;
    parse_eta_body(is_json_content_type(headers), &bytes)
}

pub fn parse_eta_body(is_json: bool, bytes: &[u8]) -> Result<(f32, f32), Vec<ValidationError>> {
    if bytes.is_empty() {
        return Err(vec![ValidationError {
            kind: "missing",
            loc: vec![json!("body")],
            msg: "Field required",
            input: Value::Null,
        }]);
    }
    let raw: Value = if is_json {
        serde_json::from_slice(bytes).map_err(|e| json_invalid(byte_offset(bytes, e.line(), e.column())))?
    } else {
        Value::String(String::from_utf8_lossy(bytes).into_owned())
    };
    let Value::Object(obj) = raw else {
        return Err(vec![ValidationError {
            kind: "model_attributes_type",
            loc: vec![json!("body")],
            msg: "Input should be a valid dictionary or object to extract fields from",
            input: raw,
        }]);
    };

    let distance = parse_distance(&obj);
    let orders = parse_orders(&obj);
    match (distance, orders) {
        (Ok(d), Ok(o)) => Ok((d, o)),
        (d, o) => Err([d.err(), o.err()].into_iter().flatten().collect()),
    }
}

/// serde_json 의 (줄, 열) 위치를 본문 안의 바이트 위치로 바꾼다.
fn byte_offset(bytes: &[u8], line: usize, column: usize) -> usize {
    let line_start: usize = bytes
        .split(|&b| b == b'\n')
        .take(line.saturating_sub(1))
        .map(|l| l.len() + 1)
        .sum();
    (line_start + column.saturating_sub(1)).min(bytes.len())
}

fn is_float_string(s: &str) -> bool {
    // ^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$
    let s = s.strip_prefix(['+', '-']).unwrap_or(s);
    let (mantissa, exp) = match s.find(['e', 'E']) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (mantissa, None),
    };
    let digits = |p: &str| p.bytes().all(|b| b.is_ascii_digit());
    let mantissa_ok = digits(int_part)
        && frac_part.is_none_or(digits)
        && (!int_part.is_empty() || frac_part.is_some_and(|f| !f.is_empty()));
    let exp_ok = exp.is_none_or(|e| is_int_string(e));
    mantissa_ok && exp_ok
}

fn is_int_string(s: &str) -> bool {
    // ^[+-]?\d+$
    let s = s.strip_prefix(['+', '-']).unwrap_or(s);
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn field_error(field: &str, kind: &'static str, msg: &'static str, input: Value) -> ValidationError {
    ValidationError { kind, loc: vec![json!("body"), json!(field)], msg, input }
}

fn parse_distance(obj: &Map<String, Value>) -> Result<f32, ValidationError> {
    const FIELD: &str = "distance_km";
    let Some(v) = obj.get(FIELD) else {
        return Err(field_error(FIELD, "missing", "Field required", json!(obj)));
    };
    let fail = |kind, msg| Err(field_error(FIELD, kind, msg, v.clone()));
    let f = match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::INFINITY),
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::String(s) => {
            let s = s.trim();
            if !is_float_string(s) {
                return fail("float_parsing", "Input should be a valid number, unable to parse string as a number");
            }
            s.parse::<f64>().unwrap_or(f64::INFINITY)
        }
        _ => return fail("float_type", "Input should be a valid number"),
    };
    if f < 0.0 {
        return fail("greater_than_equal", "Input should be greater than or equal to 0");
    }
    let f32v = f as f32;
    if !f.is_finite() || !f32v.is_finite() {
        return fail("finite_number", "Input should be a finite number");
    }
    Ok(f32v)
}

fn parse_orders(obj: &Map<String, Value>) -> Result<f32, ValidationError> {
    const FIELD: &str = "pending_orders";
    let Some(v) = obj.get(FIELD) else {
        return Err(field_error(FIELD, "missing", "Field required", json!(obj)));
    };
    let fail = |kind, msg| Err(field_error(FIELD, kind, msg, v.clone()));
    let f = match v {
        Value::Number(n) => {
            let f = n.as_f64().unwrap_or(f64::INFINITY);
            if !f.is_finite() {
                return fail("finite_number", "Input should be a finite number");
            }
            if f != f.trunc() {
                return fail("int_from_float", "Input should be a valid integer, got a number with a fractional part");
            }
            f
        }
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::String(s) => {
            let s = s.trim();
            if !is_int_string(s) {
                return fail("int_parsing", "Input should be a valid integer, unable to parse string as an integer");
            }
            s.parse::<f64>().unwrap_or(f64::INFINITY)
        }
        _ => return fail("int_type", "Input should be a valid integer"),
    };
    if f < 0.0 {
        return fail("greater_than_equal", "Input should be greater than or equal to 0");
    }
    let f32v = f as f32;
    if !f32v.is_finite() {
        return fail("finite_number", "Input should be a finite number");
    }
    Ok(f32v)
}
