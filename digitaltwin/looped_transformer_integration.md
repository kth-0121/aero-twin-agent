# Looped Transformer-Digital Twin 연동 기술 문서

## 1. 목적과 범위

이 문서는 `model/LoopedTransformer/` 모델을 Digital Twin(DT) 백엔드에서 호출할 때 필요한 데이터 계약, 추론 흐름, 출력 해석 및 연동 경계를 정리한다. 현재 구현은 LPBF 단일 트랙의 공정조건으로 용융풀 형상을 추정하는 추론 모델이며, 설비 제어기나 시간축 공정 시뮬레이터가 아니다.

이 모델에서 Transformer의 반복 축은 시간축이 아니라 **물리 피드백 추론 단계**다. 여러 입력 행은 독립된 공정조건 샘플이며, 각 행 안에서 모델이 예측과 물리 피드백을 반복한다.

## 2. 구성 요소

| 경로 | 역할 |
|---|---|
| `modelLT.py` | 입력 토큰화, 반복 Transformer 추론, 상태 예측 및 피드백 반영 |
| `dataLT.py` | 316L 데이터 로드, 7개 입력 특성 생성, 스케일러/데이터셋 정의 |
| `physicsLT.py` | Rosenthal 폭 스케일, aspect-ratio 스케일, 열-유체 무차원수 기반 피드백 계산 |
| `interfaceLT.py` | 체크포인트·학습 스케일러 로드 및 최종 추론 결과 반환 |
| `lossLT.py` | 반복 예측 감독 손실과 신뢰도 가중 물리 일관성 손실 |
| `looped_transformer_v3.pt` | 모델 state dict와 입력/출력 스케일러를 포함하는 추론 체크포인트 |

주요 추론 흐름은 다음과 같다.

```text
공정조건 4개
  -> 파생 특성 3개 생성
  -> 학습 입력 스케일러 적용
  -> 초기 상태 예측
  -> loop별 Physics Evaluator 피드백
  -> 상태 토큰 갱신 및 반복 추론
  -> 마지막 상태를 학습 출력 스케일러로 역변환
  -> 폭, 종횡비, 단면적 반환
```

## 3. 모델 입력 계약

입력 DataFrame에는 아래 7개 열이 모두 있어야 하며, 순서는 `dataLT.FEATURES` 정의를 따른다. `RUN.getPredict()`가 이 열들을 학습 시 저장된 StandardScaler로 변환하므로 호출자는 **원래 물리 단위의 값**을 전달한다.

| 특성 | 단위 | 정의/주의사항 |
|---|---:|---|
| `P_laser` | W | 레이저 출력 |
| `v_scan` | mm/s | 스캔 속도 |
| `d_laser` | mm | 레이저 스폿 직경으로 취급 |
| `t_powder` | µm | 분말층 두께 |
| `linear_energy` | J/mm | `P_laser / v_scan` |
| `peclet_number` | 무차원 | `v[m/s] * d[m] / alpha`, `alpha=5e-6 m²/s` |
| `normalized_enthalpy` | 무차원 | `eta*P / (h_s*sqrt(alpha*d³*v))`; 코드 상수 `eta=0.34`, `rho=7800 kg/m³`, `cp=460 J/(kg K)`, `Tm=1678 K`, `T0=293.15 K` |

`normalized_enthalpy` 계산에서 `v_scan`은 mm/s에서 m/s로, `d_laser`는 mm에서 m로 변환한다. 코드에서 `t_powder`는 원시 입력 특성으로 사용되며 파생식에는 들어가지 않는다. 임의의 새 조건을 만들 때도 이 특성 정의, 상수 및 단위를 일치시켜야 한다. 현재 `dataLT.load_316l_csv()`는 제공 CSV 형식 로더일 뿐, 공정조건 한 건을 받아 파생 특성을 만드는 별도 공개 함수는 없다.

## 4. 모델 출력 계약

`RUN.getPredict()`는 DataFrame을 반환하며 열은 `TARGETS` 순서인 다음 3개다. 값은 학습 출력 스케일러를 역변환한 물리 단위다.

| 출력 | 의미 | 단위 |
|---|---|---:|
| `melt_width` | 용융풀 폭 `W` | µm |
| `aspect_ratio` | 종횡비 `R = D/W` | 무차원 |
| `melt_area` | 용융풀 단면적 `A` | µm² |

DT가 깊이를 필요로 하면 아래처럼 파생한다.

```text
melt_depth_um = melt_width_um * aspect_ratio
```

`melt_area`는 모델이 별도로 예측한 값이다. Physics Evaluator는 `A ≈ π/4 * W * D`의 타원 기하 일관성을 피드백으로 사용하지만, 반환된 면적이 이 식을 항상 만족한다고 가정하면 안 된다. DT에서는 예측 단면적과 기하로부터 재구성한 면적을 구분해 보존하는 것이 좋다.

## 5. 반복 추론과 Physics Feedback

`modelLT.LoopedTransformer` 기본 구조는 7개 공정 입력, 3개 상태 출력, 7개 물리 피드백, `d_model=64`, 4개 attention head, 2개 encoder layer다. `interfaceLT.RUN`은 체크포인트를 로드하고 `loop_steps=12`로 추론한다. 실제 배포 설정은 체크포인트 생성 시 사용된 구조와 맞춰야 한다.

`FluidThermalPhysicsEvaluator`의 피드백 벡터 순서는 다음과 같다.

| 인덱스 | 신호 | 해석 |
|---:|---|---|
| 0 | 폭 잔차 | Rosenthal 폭 스케일과 예측 폭의 상대 차이 |
| 1 | 종횡비 잔차 | 물리 스케일링으로 추정한 종횡비와 예측 종횡비의 로그 차이 |
| 2 | 면적 잔차 | 예측 단면적과 타원 기하 면적의 상대 차이 |
| 3 | Ma 정규화 값 | Marangoni 수 기반 유동 지표 |
| 4 | Re 정규화 값 | Reynolds 수 기반 유동 지표 |
| 5 | We 정규화 값 | Weber 수 기반 유동 지표 |
| 6 | Ca 정규화 값 | Capillary 수 기반 유동 지표 |

현재 evaluator의 loop 단계는 다음처럼 구성된다.

| 단계 | 피드백 계산 |
|---:|---|
| 0 | 7개 피드백을 0으로 반환하는 초기 추정 단계 |
| 1 | Rosenthal 폭 일관성 |
| 2 | 유체 무차원수와 종횡비 보정 신호 |
| 3 이상 | 폭·종횡비·면적 및 유체 지표의 결합 피드백 |

피드백 신뢰도(`confidence`)는 모델 출력에 함께 보존되고 `PhysicsConsistencyLoss`에서 물리 손실 가중치로 사용된다. **현재 모델의 forward 경로는 confidence로 residual을 직접 마스킹하거나 조절하지 않는다.** 반복 갱신에는 residual 자체가 입력된다. 따라서 DT에서 confidence를 안전 점수나 보장된 불확실성으로 해석하지 말고, 별도 안전판정 기준으로 사용하지 않는다.

## 6. 기존 Python 추론 인터페이스

저장소 루트에서 실행할 때 `interfaceLT.py`의 절대 패키지 import 경로를 찾을 수 있도록 모델 폴더를 `sys.path`에 추가한다.

```python
import sys

sys.path.insert(0, "model/LoopedTransformer")
from interfaceLT import RUN

runner = RUN()
dataset = runner.setData("data/316L_meltpool.csv")
prediction = runner.getPredict(
    "model/LoopedTransformer/looped_transformer_v3.pt",
    dataset.iloc[[0]],
)

width_um = float(prediction.iloc[0]["melt_width"])
aspect_ratio = float(prediction.iloc[0]["aspect_ratio"])
depth_um = width_um * aspect_ratio
area_um2 = float(prediction.iloc[0]["melt_area"])
```

기존 CSV 추론은 위와 같이 가능하다. **새 공정조건 입력 시에는** 7개 특성을 구성한 DataFrame을 `getPredict()`에 전달해야 한다. Raw 4개 변수만 전달하거나 자체 계산한 스케일링 값을 전달하면 안 된다. 신규 입력용 특성 생성기를 DT 백엔드에 두거나, 이후 `dataLT.py`에 재사용 가능한 단일 행 feature-builder를 추가해 학습·추론 간 수식 중복을 피한다.

`interfaceLT.RUN` 호출의 출력은 마지막 loop의 예측만 포함한다. 전체 loop 궤적이 필요하면 `modelLT.LoopedTransformer.forward()`의 `LoopOutput.predictions`, `feedbacks`, `confidences`를 직접 보존하는 별도 어댑터가 필요하다. 현재 `RUN` 인터페이스는 체크포인트 로드 시 `torch.load(..., weights_only=False)`를 사용하므로 신뢰할 수 있는 체크포인트만 로드해야 한다.

### 실행 환경 의존성

이 인터페이스 경로는 PyTorch 외에도 `pandas`와 `scikit-learn`을 import한다. 현재 루트 `requirements.txt`에는 이 둘이 선언되어 있지 않으므로, 추론 환경에 별도로 설치하거나 프로젝트 의존성 목록에 추가해야 한다. 체크포인트 로드 시 이 실행 환경에는 `scikit-learn`이 없어 메타데이터 로드를 완료하지 못했다.

## 7. DT 백엔드 권장 연동 흐름

1. **입력 검증:** 출력, 속도, 스폿 직경, 분말층 두께의 누락·비수치·비양수 값을 거부하고 지원 데이터 범위를 벗어난 조건은 외삽 경고로 표시한다.
2. **특성 생성 및 추론:** 원시 조건으로 위의 3개 파생 특성을 계산하고, `RUN.getPredict()`에 한 행 또는 배치 DataFrame으로 전달한다.
3. **출력 변환:** `depth_um = width_um * aspect_ratio`를 계산하되 원본 모델 출력도 유지한다.
4. **별도 상태/위험 평가:** LOF 기준, conduction/keyhole regime, balling 위험은 이 모델의 반환값이 아니다. 공정 기하와 공정조건을 사용하는 명시적 evaluator에서 산출하고, 사용할 수 없는 입력은 `unknown` 또는 `not_evaluated`로 구분한다.
5. **가상 검증:** What-if 또는 행동 후보의 새 조건마다 동일한 feature-builder와 추론/평가 경로를 실행한다. 후보의 예측값과 위험상태, 제약 위반을 함께 비교한다.
6. **결과 기록:** 입력 원시값, 파생 특성, 체크포인트 버전, loop 수, 예측 형상, evaluator 버전, 위험 판정 및 후보별 결과를 함께 기록한다.

권장 서비스 경계 예시는 다음과 같다.

```text
ProcessCondition
  -> FeatureBuilder
  -> LoopedTransformerAdapter
  -> GeometryDeriver (depth = width * aspect_ratio)
  -> ProcessRiskEvaluator (LOF / regime / balling 등)
  -> DigitalTwinScenarioRunner
  -> DecisionResult + provenance
```

각 구성 요소는 독립적으로 교체할 수 있도록 분리한다. 특히 모델 예측, 형상 변환, 안전 판정, 시뮬레이션 결과를 하나의 숫자나 단일 `safe` 플래그로 뭉치지 않는다.

### 7.1 L/T와 Action/ROM 사이의 현재 계약

두 모델의 결과는 같은 물리량을 추정하더라도 동일한 평가기로 간주하지 않는다. L/T Physics Evaluator는 반복 추론을 위한 형상 일관성 피드백과 유동 지표를 제공한다. `sim.lpbf`는 현재 LOF/regime 판정과 Action 후보의 가상 재평가를 담당한다. 두 evaluator의 상수와 깊이 추정식이 다르므로 결과를 덮어쓰거나 평균내지 않고, 각 결과에 출처를 보존한다.

| 계약 항목 | 값/소유자 | 의미 |
|---|---|---|
| 입력 공정조건 | `P_laser[W]`, `v_scan[mm/s]`, `d_laser[mm]`, `t_powder[um]` | L/T가 요구하는 원시 입력. Action/ROM에는 `spot_diameter_um`, `hatch_um`, `layer_thickness_um`도 전달한다. `d_laser[mm]`는 spot diameter로 변환하며, `t_powder`와 layer thickness는 물리적 정의가 일치하는지 확인한 뒤 매핑한다. |
| L/T Physical State | `melt_width[um]`, `aspect_ratio[-]`, `melt_area[um^2]`; `depth = width * aspect_ratio` | 데이터 기반 추정과 loop refinement 결과. 원본 예측값과 파생 깊이를 구분한다. |
| L/T Physics Evidence | loop별 feedback residual 및 Ma/Re/We/Ca 지표, evaluator/checkpoint 식별자 | L/T 내부 refinement 근거. confidence는 보정된 불확실성이나 안전 확률이 아니다. |
| ROM Risk/Regime | `sim.decision.assess_state()` 결과와 `sim.lpbf` 기준 | 현재 Action 정책에서 사용하는 LOF criterion 및 conduction/keyhole-or-transition 판정. 확률이 아닌 현재 ROM의 rule-based 결과다. |
| Action 후보 검증 | `sim.decision.recommend_action()` | 후보 공정조건마다 ROM 상태를 다시 계산한다. 후보 평가가 L/T로 전환되기 전까지 ROM이 Action 선택의 기준이다. |

Agent의 예측 및 Action 도구는 레이저 출력과 속도뿐 아니라 spot diameter, hatch spacing, layer thickness를 입력받는다. 생략 시 각각 100 um, 100 um, 30 um 기본값을 쓴다. 실제 L/T 결과와 비교하거나 후보를 평가할 때는 이 기본값에 의존하지 말고 해당 공정의 값을 명시적으로 전달한다.

현재 L/T와 Action 경로는 아직 실행 시 연결되어 있지 않다. 통합 시에는 같은 `ProcessCondition`을 두 경로에 전달하고, 다음을 별도 필드로 기록한다.

```text
physical_state: L/T의 W, R, A 및 파생 D
physics_evidence: L/T evaluator의 loop별 신호와 출처
process_risk: sim ROM의 LOF/regime 결과와 evaluator 버전
counterfactuals: 기존 Action 후보별 ROM 재평가 결과
```

L/T evidence는 Action Generator의 설명 근거로 전달할 수 있지만, 검증 전에는 ROM risk를 대체하거나 위험 확률로 해석하지 않는다. 두 경로의 판정이 다르면 불일치 자체를 남기고 검토 대상으로 표시한다. 물성 상수나 regime 기준을 통합하려면 데이터/체크포인트 기준으로 별도 교차 검증하고, L/T 물리 피드백 변경 시에는 재학습 및 checkpoint 버전 변경을 요구한다.

## 8. 현재 범위와 미지원 사항

- 시간 이력, 센서 스트림, 이전 상태는 모델 입력에 포함되지 않는다. 각 호출은 현재 입력 행을 독립 추론한다.
- 모델 출력만으로 LOF, keyhole, balling의 최종 판정을 내릴 수 없다. `balling`은 학습 CSV의 별도 라벨이며 현재 `TARGETS` 및 `RUN.getPredict()` 출력에 포함되지 않는다.
- 신뢰구간, 예측 불확실성, OOD 점수, 수렴 보장은 제공되지 않는다. 보고서의 loop 성능 실험을 임의 운전조건의 안전 보장으로 해석하지 않는다.
- PyBullet 및 제조 형상 생성/운영 시뮬레이터는 이 폴더에서 구현되지 않는다. 별도의 DT 모듈이 이 추론 adapter의 출력을 소비하는 형태로 연동한다.
- loop 출력 궤적은 `modelLT` 내부에서 얻을 수 있지만, 기존 `RUN`은 마지막 예측만 반환한다.
- 입력 범위, 물리 evaluator 상수, 학습 데이터/체크포인트 버전이 달라지면 별도 검증이 필요하다.

## 9. 검증 체크리스트

- [ ] 학습 시 사용한 체크포인트, 7개 feature 정의, 스케일러와 추론 환경이 일치한다.
- [ ] 기존 CSV 행을 통해 체크포인트 로드 및 결과의 행 수·열 이름·유한값을 확인했다.
- [ ] 새 조건 입력의 원시 단위와 feature 계산 결과가 `dataLT.py` 정의와 일치한다.
- [ ] `depth = width * aspect_ratio` 변환의 단위와 물리적 범위를 확인했다.
- [ ] LOF/regime/balling evaluator가 모델 출력과 독립된 검증 코드로 테스트됐다.
- [ ] 행동 후보별로 변경 조건부터 위험 재평가까지 동일한 추론 경로를 사용한다.
- [ ] 정상·경계·위험·학습범위 밖 조건에서 DT 결과와 경고 정책을 확인했다.
- [ ] 로그에 체크포인트 및 evaluator 버전, 조건, 예측 및 판정 근거가 남는다.

## 10. 참고 코드

- 모델 입출력과 loop: `model/LoopedTransformer/modelLT.py`
- feature 및 데이터 계약: `model/LoopedTransformer/dataLT.py`
- 물리 피드백: `model/LoopedTransformer/physicsLT.py`
- 추론 adapter: `model/LoopedTransformer/interfaceLT.py`
- 손실 정의: `model/LoopedTransformer/lossLT.py`
- 실험 관찰 기록: `model/LoopedTransformer/train/testingReport.md`