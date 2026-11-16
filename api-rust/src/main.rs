//! Rust 로 구현한 배달 예상 시간 API. api/app.py 와 같은 계약(POST /eta, GET /health)을 따른다.
//! 예측은 ONNX 모델(입력 "X" [1, 2] float32 → 출력 "variable" [1, 1])을 ONNX Runtime 으로 실행한다.
//! 정규화는 모델 안에 있으므로 원래 단위의 값을 그대로 넣는다.
//!
//! 환경 변수:
//!   PORT        듣는 포트 (기본 8080)
//!   MODEL_PATH  ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//!   ORT_LIB     libonnxruntime.so 경로 (기본 libonnxruntime.so)
//!   MODEL_IMPL  /health, /eta 응답의 model 이름 (기본 onnx)

mod handler;
#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex};

use ort::{session::Session, value::Tensor};

fn getenv(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

/// 모델을 한 번 읽어 두고, 요청마다 입력 텐서를 새로 만들어 실행한다.
fn new_onnx_predictor(lib_path: &str, model_path: &str) -> Result<Arc<handler::Predictor>, String> {
    let env = ort::init_from(lib_path)
        .map_err(|e| format!("ONNX Runtime 로드 실패({lib_path}): {e}"))?
        .build()
        .map_err(|e| format!("ONNX Runtime 초기화 실패: {e}"))?;
    let session = Session::builder(&env)
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("모델 로드 실패({model_path}): {e}"))?;
    // Session::run 이 &mut self 를 받으므로 Mutex 로 감싼다. 모델이 작아 경합은 무시할 만하다.
    let session = Mutex::new(session);
    Ok(Arc::new(move |distance_km: f32, pending_orders: f32| {
        let _env = &env; // 세션이 쓰는 환경을 프로세스 끝까지 살려 둔다.
        let input = Tensor::from_array(([1usize, 2], vec![distance_km, pending_orders])).map_err(|e| e.to_string())?;
        let mut session = session.lock().map_err(|e| e.to_string())?;
        let outputs = session.run(ort::inputs!["X" => input]).map_err(|e| e.to_string())?;
        let (_, data) = outputs["variable"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        data.first().copied().ok_or_else(|| "빈 출력".to_string())
    }))
}

#[tokio::main]
async fn main() {
    let predict = new_onnx_predictor(
        &getenv("ORT_LIB", "libonnxruntime.so"),
        &getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"),
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let app = handler::router(&getenv("MODEL_IMPL", "onnx"), predict);

    let addr = format!("0.0.0.0:{}", getenv("PORT", "8080"));
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| {
        eprintln!("{addr} 에서 듣기 실패: {e}");
        std::process::exit(1);
    });
    eprintln!("listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); })
        .await
        .unwrap();
}
