# Rust ETA API. 저장소 루트를 빌드 컨텍스트로 쓴다:
#   docker build -f api-rust/Dockerfile -t eta-api-rust .
#   docker run --rm -p 8082:8080 eta-api-rust

# ── ONNX Runtime: uv.lock 의 onnxruntime 버전과 같은 C 라이브러리를 받는다 ──
FROM --platform=$BUILDPLATFORM debian:bookworm-slim AS ort
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY uv.lock /tmp/uv.lock
RUN set -eu; \
    ORT_VERSION="$(awk '/^name = "onnxruntime"$/ { getline; gsub(/^version = "|"$/, ""); print; exit }' /tmp/uv.lock)"; \
    [ -n "$ORT_VERSION" ] || { echo "uv.lock 에서 onnxruntime 버전을 찾지 못했다" >&2; exit 1; }; \
    case "$TARGETARCH" in \
      amd64) ORT_ARCH=x64 ;; \
      arm64) ORT_ARCH=aarch64 ;; \
      *) echo "지원하지 않는 TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    echo "ONNX Runtime $ORT_VERSION ($ORT_ARCH)"; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
      | tar -xz -C /opt; \
    mv "/opt/onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}" /opt/onnxruntime

# ── Rust 빌드 (ort load-dynamic: 링크하지 않고 실행 시점에 ORT_LIB 를 연다) ──
FROM rust:1-bookworm AS build
WORKDIR /src
# 의존성만 먼저 빌드해 캐시한다.
COPY api-rust/Cargo.toml api-rust/Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release --locked && rm -rf src
COPY api-rust/src/ ./src/
RUN touch src/main.rs && cargo test --release --locked && cargo build --release --locked \
    && cp target/release/eta-api /out-eta-api

# ── 실행 ──
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=ort /opt/onnxruntime/lib/ /opt/onnxruntime/lib/
COPY --from=build /out-eta-api /app/eta-api
COPY artifacts/model_sklearn.onnx /app/model_sklearn.onnx
ENV ORT_LIB=/opt/onnxruntime/lib/libonnxruntime.so \
    MODEL_PATH=/app/model_sklearn.onnx \
    PORT=8080
USER nobody
EXPOSE 8080
CMD ["/app/eta-api"]
