package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// Predictor 는 (거리 km, 대기 주문 수) → 예상 시간(분)을 계산한다.
// float32 를 받는 이유: ONNX 모델 입력 "X" 가 float32 이기 때문이다.
type Predictor func(distanceKm, pendingOrders float32) (float32, error)

// api/app.py 의 EtaResponse 와 같은 모양.
type etaResponse struct {
	Minutes float64 `json:"minutes"`
	Model   string  `json:"model"`
}

// FastAPI/Pydantic 의 422 오류 항목과 같은 모양: {"type", "loc", "msg", "input"}.
type validationError struct {
	Type  string `json:"type"`
	Loc   []any  `json:"loc"`
	Msg   string `json:"msg"`
	Input any    `json:"input"`
}

const maxBodyBytes = 1 << 20

func newMux(modelName string, predict Predictor) *http.ServeMux {
	mux := http.NewServeMux()
	// Go 1.22+ 패턴: 다른 메서드는 405 가 된다 (FastAPI 와 같음).
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": modelName})
	})
	mux.HandleFunc("POST /eta", func(w http.ResponseWriter, r *http.Request) {
		handleEta(w, r, modelName, predict)
	})
	return mux
}

func handleEta(w http.ResponseWriter, r *http.Request, modelName string, predict Predictor) {
	distance, orders, errs := parseEtaRequest(r)
	if len(errs) > 0 {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": errs})
		return
	}
	minutes, err := predict(distance, orders)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "prediction failed"})
		return
	}
	if math.IsNaN(float64(minutes)) || math.IsInf(float64(minutes), 0) {
		// 성공 응답의 minutes 는 항상 유한한 숫자여야 한다.
		writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": []validationError{{
			Type: "finite_number", Loc: []any{"body"},
			Msg: "Input produces a non-finite prediction", Input: nil,
		}}})
		return
	}
	writeJSON(w, http.StatusOK, etaResponse{Minutes: float64(minutes), Model: modelName})
}

// parseEtaRequest 는 FastAPI(Pydantic lax 모드)의 EtaRequest 검증 규칙을 따른다.
//   - distance_km: 숫자(정수·실수·bool·숫자 문자열), >= 0
//   - pending_orders: 정수(정수·소수부 0 인 실수·bool·정수 문자열), >= 0
//   - 추가 필드는 무시
//
// Python 과 다른 점(의도적): NaN/Infinity 와 float32 로 표현할 수 없는 큰 값은 422.
func parseEtaRequest(r *http.Request) (float32, float32, []validationError) {
	body, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, maxBodyBytes))
	if err != nil {
		return 0, 0, []validationError{{Type: "json_invalid", Loc: []any{"body", 0}, Msg: "JSON decode error", Input: map[string]any{}}}
	}

	// FastAPI: Content-Type 이 없거나 JSON(application/json, application/*+json)일 때만 JSON 으로 읽는다.
	var raw any
	isJSON := true
	if ct := r.Header.Get("Content-Type"); ct != "" {
		mt, _, err := mime.ParseMediaType(ct)
		isJSON = err == nil && (mt == "application/json" ||
			(strings.HasPrefix(mt, "application/") && strings.HasSuffix(mt, "+json")))
	}
	switch {
	case len(body) == 0:
		return 0, 0, []validationError{{Type: "missing", Loc: []any{"body"}, Msg: "Field required", Input: nil}}
	case !isJSON:
		raw = string(body)
	default:
		dec := json.NewDecoder(bytes.NewReader(body))
		dec.UseNumber()
		if err := dec.Decode(&raw); err != nil || dec.Decode(&struct{}{}) != io.EOF {
			offset := dec.InputOffset()
			var se *json.SyntaxError
			if errors.As(err, &se) {
				offset = se.Offset
			}
			return 0, 0, []validationError{{Type: "json_invalid", Loc: []any{"body", offset}, Msg: "JSON decode error", Input: map[string]any{}}}
		}
	}

	obj, ok := raw.(map[string]any)
	if !ok {
		return 0, 0, []validationError{{Type: "model_attributes_type", Loc: []any{"body"},
			Msg: "Input should be a valid dictionary or object to extract fields from", Input: raw}}
	}

	var errs []validationError
	distance, e := parseDistance(obj)
	if e != nil {
		errs = append(errs, *e)
	}
	orders, e := parseOrders(obj)
	if e != nil {
		errs = append(errs, *e)
	}
	return distance, orders, errs
}

var (
	floatString = regexp.MustCompile(`^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$`)
	intString   = regexp.MustCompile(`^[+-]?\d+$`)
)

func parseDistance(obj map[string]any) (float32, *validationError) {
	const field = "distance_km"
	v, present := obj[field]
	fail := func(typ, msg string) (float32, *validationError) {
		return 0, &validationError{Type: typ, Loc: []any{"body", field}, Msg: msg, Input: v}
	}
	if !present {
		return fail("missing", "Field required")
	}
	var f float64
	switch x := v.(type) {
	case json.Number:
		f, _ = strconv.ParseFloat(string(x), 64) // 범위 초과는 ±Inf 가 되어 아래에서 걸러진다
	case bool:
		f = boolToFloat(x)
	case string:
		s := strings.TrimSpace(x)
		if !floatString.MatchString(s) {
			return fail("float_parsing", "Input should be a valid number, unable to parse string as a number")
		}
		f, _ = strconv.ParseFloat(s, 64)
	default:
		return fail("float_type", "Input should be a valid number")
	}
	if f < 0 {
		return fail("greater_than_equal", "Input should be greater than or equal to 0")
	}
	f32 := float32(f)
	if math.IsInf(f, 0) || math.IsInf(float64(f32), 0) {
		return fail("finite_number", "Input should be a finite number")
	}
	return f32, nil
}

func parseOrders(obj map[string]any) (float32, *validationError) {
	const field = "pending_orders"
	v, present := obj[field]
	fail := func(typ, msg string) (float32, *validationError) {
		return 0, &validationError{Type: typ, Loc: []any{"body", field}, Msg: msg, Input: v}
	}
	if !present {
		return fail("missing", "Field required")
	}
	var f float64
	switch x := v.(type) {
	case json.Number:
		f, _ = strconv.ParseFloat(string(x), 64)
		if math.IsInf(f, 0) {
			return fail("finite_number", "Input should be a finite number")
		}
		if f != math.Trunc(f) {
			return fail("int_from_float", "Input should be a valid integer, got a number with a fractional part")
		}
	case bool:
		f = boolToFloat(x)
	case string:
		s := strings.TrimSpace(x)
		if !intString.MatchString(s) {
			return fail("int_parsing", "Input should be a valid integer, unable to parse string as an integer")
		}
		f, _ = strconv.ParseFloat(s, 64)
	default:
		return fail("int_type", "Input should be a valid integer")
	}
	if f < 0 {
		return fail("greater_than_equal", "Input should be greater than or equal to 0")
	}
	f32 := float32(f)
	if math.IsInf(float64(f32), 0) {
		return fail("finite_number", "Input should be a finite number")
	}
	return f32, nil
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
