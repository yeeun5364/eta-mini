package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// 계약 테스트용 가짜 예측기: 규칙 10 + 3*거리 + 2*주문 (api/predictors/rule.py 와 같음).
func rulePredict(d, o float32) (float32, error) { return 10 + 3*d + 2*o, nil }

var testMux = newMux("onnx", rulePredict)

func do(method, path, contentType, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	testMux.ServeHTTP(rec, req)
	return rec
}

func postEta(body string) *httptest.ResponseRecorder {
	return do(http.MethodPost, "/eta", "application/json", body)
}

func decode(t *testing.T, rec *httptest.ResponseRecorder) map[string]any {
	t.Helper()
	var m map[string]any
	dec := json.NewDecoder(strings.NewReader(rec.Body.String()))
	dec.UseNumber() // 422 의 input 이 1e400 처럼 float64 범위를 넘을 수 있다
	if err := dec.Decode(&m); err != nil {
		t.Fatalf("응답이 JSON 이 아니다: %q", rec.Body.String())
	}
	return m
}

// GET /health 는 200 과 status=ok, model 이름을 돌려준다.
func TestHealth(t *testing.T) {
	rec := do(http.MethodGet, "/health", "", "")
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	m := decode(t, rec)
	if m["status"] != "ok" || m["model"] != "onnx" {
		t.Fatalf("body %v", m)
	}
}

// 알려진 입력(경계값 0, 큰 값, lax 변환 포함)에 대해 200 과 예상 분, minutes/model 두 필드만 돌려준다.
func TestEtaKnownValues(t *testing.T) {
	cases := []struct {
		body string
		want float64
	}{
		{`{"distance_km": 1, "pending_orders": 1}`, 15},
		{`{"distance_km": 2.5, "pending_orders": 3}`, 23.5},
		{`{"distance_km": 0, "pending_orders": 0}`, 10},
		{`{"distance_km": -0.0, "pending_orders": 0}`, 10},
		{`{"distance_km": 1000, "pending_orders": 10000}`, 10 + 3*1000 + 2*10000},
		{`{"distance_km": 1, "pending_orders": 100000000000000000000000}`, 10 + 3 + 2e23},
		{`{"distance_km": "1.5", "pending_orders": 1}`, 16.5},
		{`{"distance_km": 1, "pending_orders": "3"}`, 19},
		{`{"distance_km": 1, "pending_orders": 2.0}`, 17},
		{`{"distance_km": true, "pending_orders": 1}`, 15},
		{`{"distance_km": 1, "pending_orders": false}`, 13},
		{`{"distance_km": 1, "pending_orders": 1, "extra": 5}`, 15},
	}
	for _, c := range cases {
		rec := postEta(c.body)
		if rec.Code != 200 {
			t.Errorf("%s: status %d, body %s", c.body, rec.Code, rec.Body)
			continue
		}
		m := decode(t, rec)
		if len(m) != 2 || m["model"] != "onnx" {
			t.Errorf("%s: 응답 모양 %v", c.body, m)
		}
		got, _ := m["minutes"].(json.Number).Float64()
		if math.Abs(got-c.want) > 1e-6*math.Max(1, math.Abs(c.want)) {
			t.Errorf("%s: minutes %v, want %v", c.body, got, c.want)
		}
	}
}

// 음수·숫자 아님·정수 아님·null·누락·비유한·모양 틀린 JSON 은 422 를 돌려준다.
func TestEtaInvalidReturns422(t *testing.T) {
	bodies := []string{
		`{"distance_km": -1, "pending_orders": 0}`,
		`{"distance_km": -0.01, "pending_orders": 0}`,
		`{"distance_km": 1, "pending_orders": -1}`,
		`{"distance_km": "abc", "pending_orders": 1}`,
		`{"distance_km": 1, "pending_orders": "many"}`,
		`{"distance_km": 1, "pending_orders": 1.5}`,
		`{"distance_km": null, "pending_orders": 1}`,
		`{"pending_orders": 1}`,
		`{"distance_km": 1}`,
		`{}`,
		`not json`,
		``,
		`[1, 2]`,
		`{"distance_km": [1], "pending_orders": 1}`,
		`{"distance_km": {"v": 1}, "pending_orders": 1}`,
		`{"distance_km": NaN, "pending_orders": 0}`,
		`{"distance_km": Infinity, "pending_orders": 0}`,
		`{"distance_km": "inf", "pending_orders": 0}`,
		`{"distance_km": 1e308, "pending_orders": 0}`,
		`{"distance_km": 1e400, "pending_orders": 0}`,
		`{"distance_km": 1, "pending_orders": 1} trailing`,
	}
	for _, b := range bodies {
		rec := postEta(b)
		if rec.Code != 422 {
			t.Errorf("%q: status %d, want 422", b, rec.Code)
			continue
		}
		if _, ok := decode(t, rec)["detail"].([]any); !ok {
			t.Errorf("%q: detail 가 목록이 아니다: %s", b, rec.Body)
		}
	}
}

// Content-Type 이 JSON 이 아니면 본문이 맞아도 422 를 돌려준다.
func TestEtaNonJSONContentType(t *testing.T) {
	rec := do(http.MethodPost, "/eta", "text/plain", `{"distance_km": 1, "pending_orders": 1}`)
	if rec.Code != 422 {
		t.Fatalf("status %d", rec.Code)
	}
}

// 422 응답의 detail 은 문제가 된 필드를 loc ["body", 필드] 로 가리킨다.
func TestEta422DetailPointsToField(t *testing.T) {
	cases := map[string]string{
		`{"distance_km": -1, "pending_orders": 0}`: "distance_km",
		`{"distance_km": 1, "pending_orders": -1}`: "pending_orders",
		`{"distance_km": 1}`:                       "pending_orders",
	}
	for body, field := range cases {
		var resp struct {
			Detail []struct {
				Loc []any `json:"loc"`
			} `json:"detail"`
		}
		_ = json.Unmarshal(postEta(body).Body.Bytes(), &resp)
		found := false
		for _, d := range resp.Detail {
			if len(d.Loc) == 2 && d.Loc[0] == "body" && d.Loc[1] == field {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: loc [body %s] 없음: %+v", body, field, resp.Detail)
		}
	}
}

// 메서드가 틀리면 405 를 돌려준다 (GET /eta, POST /health).
func TestMethodNotAllowed(t *testing.T) {
	if c := do(http.MethodGet, "/eta", "", "").Code; c != 405 {
		t.Errorf("GET /eta: %d", c)
	}
	if c := do(http.MethodPost, "/health", "", "").Code; c != 405 {
		t.Errorf("POST /health: %d", c)
	}
}
