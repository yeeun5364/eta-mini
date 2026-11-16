// Go 로 구현한 배달 예상 시간 API. api/app.py 와 같은 계약(POST /eta, GET /health)을 따른다.
// 예측은 ONNX 모델(입력 "X" [1, 2] float32 → 출력 "variable" [1, 1])을 ONNX Runtime 으로 실행한다.
// 정규화는 모델 안에 있으므로 원래 단위의 값을 그대로 넣는다.
//
// 환경 변수:
//
//	PORT            듣는 포트 (기본 8080)
//	MODEL_PATH      ONNX 모델 경로 (기본 artifacts/model_sklearn.onnx)
//	ONNXRUNTIME_LIB libonnxruntime.so 경로 (기본 libonnxruntime.so)
//	MODEL_IMPL      /health, /eta 응답의 model 이름 (기본 onnx)
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// newOnnxPredictor 는 모델을 한 번 읽어 두고, 요청마다 입력·출력 텐서를 새로 만들어 실행한다.
// ONNX Runtime 의 Run 은 여러 고루틴에서 동시에 불러도 안전하다.
func newOnnxPredictor(libPath, modelPath string) (Predictor, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ONNX Runtime 초기화 실패: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"X"}, []string{"variable"}, nil)
	if err != nil {
		return nil, fmt.Errorf("모델 로드 실패(%s): %w", modelPath, err)
	}
	return func(distanceKm, pendingOrders float32) (float32, error) {
		in, err := ort.NewTensor(ort.NewShape(1, 2), []float32{distanceKm, pendingOrders})
		if err != nil {
			return 0, err
		}
		defer in.Destroy()
		out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
		if err != nil {
			return 0, err
		}
		defer out.Destroy()
		if err := session.Run([]ort.Value{in}, []ort.Value{out}); err != nil {
			return 0, err
		}
		return out.GetData()[0], nil
	}, nil
}

func main() {
	predict, err := newOnnxPredictor(
		getenv("ONNXRUNTIME_LIB", "libonnxruntime.so"),
		getenv("MODEL_PATH", "artifacts/model_sklearn.onnx"),
	)
	if err != nil {
		log.Fatal(err)
	}
	addr := ":" + getenv("PORT", "8080")
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, newMux(getenv("MODEL_IMPL", "onnx"), predict)))
}
