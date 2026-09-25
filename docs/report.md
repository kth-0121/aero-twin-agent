# 프로젝트 보고서

## Physics Reasoning Agent를 활용한 우주항공 제조 Digital Twin 지능화
제4회 경남 AI·SW경진대회 — 분야3 "제조 피지컬 AI 에이전트"

---

### 1. 배경 및 문제의식
우주항공 제조는 실측 데이터 확보가 구조적으로 제한적이다. 기존 M&S는 정해진 수치 범위 안에서만 해석 가능해 외란·소량데이터에 취약하고 현실과의 괴리(Simulation-to-Real Gap)가 발생한다. 지난 대회(2026.02 경상대 라이즈사업단대회)에서는 "AI가 정교한 가공/설계 판단을 대신할 수 있는가"라는 지적과 함께, 단순 ML 에러탐지+LLM 호출 수준의 정확도·응답성 문제를 지적받았다.

### 2. 시나리오: LPBF 용융풀 예측 (2026-09-25 팀 확정)
팀은 5가지 선정 기준(물리적 결과 도출/실측데이터 존재/중복 연구 없음/10.06까지 시연가능/우주항공 연관성)으로 7개 후보 공정(LPBF, CNC가공, 레이저용접, DED/WAAM, 복합재 경화, 금속성형, 피막)을 비교 검토했고, **LPBF(레이저 파우더베드퓨전)**를 채택했다.

> **문제 정의**: 목표 melt pool 깊이를 얻으려면 레이저 파워·스캔 속도를 어떻게 정해야 하는가? 그 조건은 keyhole·미용융 없이 안전한가?

### 3. 핵심 아이디어
- **AI↔Physics 폐루프(Looped Transformer)**: `y0=f(x)→r0=P(x,y0)→y1=f(x,y0,r0)→...` 형태로, 예측을 물리 평가기에 검증받고 잔차를 다시 입력에 반영
- **역할 분리**: 물리 계산(ROM)은 결정론적 수식이, 판단/설명은 로컬 sLLM이 담당 → 정확도·응답성 문제 대응
- **Regime gate**: 물리 모델이 신뢰 가능한 영역(conduction)과 신뢰도가 떨어지는 영역(keyhole/transition)을 구분해서 처리

### 4. 물리 모델 (팀 도출 수식)
| 항목 | 식 |
|---|---|
| 용융풀 폭 | W = √(8AP / (π·e·ρ·cp·v·(Tm−T0))) — Rosenthal 기반 근사 |
| 깊이 | D ≈ W/2 (conduction 영역 반원 단면 가정) |
| LOF 판정 | (h/W)² + (t/D)² ≤ 1 이면 안전 |
| Regime 판별 | 면적 에너지밀도 기반, 실측 D/W 분포로 보정(정확도 78%) |

### 5. 데이터
**Zenodo 16979848** (Hofmann 2025, Aconity-Midi 장비) — 316L 스테인리스, 677트랙/고유조건 **623개** (팀 조사 수치와 정확히 일치), CC BY 4.0 라이선스 확인.

### 6. 핵심 실험 결과
**실험 1 — 공정한 비교 (conduction 영역 내부 random holdout)**

| variant | test MSE |
|---|---|
| A. MLP (baseline) | 499.3 |
| B. ROM 단독 | 1315.0 |
| C. 루프(잔차 피드백 없음) | 315.9 |
| **D. 루프+물리잔차(physics-aware)** | **283.0** |

→ physics-aware가 baseline 대비 **MSE 약 43% 낮음**. 정상적인 일반화 상황에서 물리 피드백이 명확히 도움이 됨.

**실험 2 — 위험 시나리오 (conduction → keyhole/transition 영역 전환)**

| variant | test MSE |
|---|---|
| A. MLP | 10,261 |
| B. ROM 단독 | 8,402 |
| C. 루프(무피드백) | 9,995 |
| D. 루프+잔차 | 9,837 |

→ physics-aware 우위가 **약 4%로 급감**. 이는 팀이 사전에 예측한 위험("Rosenthal ROM은 conduction에서만 유효하며 keyhole에서 정확도가 떨어진다")이 실측으로 확인된 것 — ROM 신뢰도가 낮은 영역에서는 물리 잔차를 맹신하면 안 된다는 근거이자, regime gate 설계를 정당화하는 정량적 증거.

**추가 검증**: 단조성 테스트(P↑→D↑) 위반율 19.4%(정상 범위, 실측 노이즈 감안).

### 7. 시스템 구성
```
AI Agent (로컬 sLLM qwen2.5:7b) ── Tool Call ──▶ predict_melt_pool / suggest_params_for_target_depth
                                                        │
                                          Physics-aware Looped 코어 ↔ ROM(Rosenthal+LOF+regime)
```
- **정방향**: 레이저파워·스캔속도 → 깊이·폭·LOF안전여부·regime 예측
- **역방향(목표제어)**: 목표 깊이 → ROM 역산으로 안전한 conduction 영역 내 P,V 제안 (그리드서치)
- 대시보드: 실측 데이터 재생(레이저+분말 장비는 실물 재현 불가하여 채택) + What-if + Agent 권고, 전체 E2E 동작 확인

### 8. 진행 현황
| 항목 | 상태 |
|---|---|
| 시나리오/데이터/물리모델 확정 | ✅ |
| Physics-aware 코어 + 4종 ablation | ✅ |
| Regime gate | ✅ |
| 목표제어(역설계) | ✅ |
| Agent (tool-calling, 한국어 고정, 단위 정확도) | ✅ |
| 대시보드 E2E | ✅ |
| GitHub 비공개 저장소 + 협업자 초대 | ✅ |
| 발표자료 | ⬜ |

### 9. 저장소
https://github.com/kth-0121/aero-twin-agent (Private)
