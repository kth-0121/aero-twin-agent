# aero-twin-agent

Physics Reasoning Agent를 활용한 우주항공 제조 Digital Twin 지능화 — 제4회 경남 AI·SW경진대회.
팀 601Solutions (전혁건·문기준·권태훈).

**시나리오: LPBF(레이저 파우더베드퓨전) 용융풀 예측.** 레이저 파워·스캔속도를 입력하면
용융풀 깊이/폭을 예측하고, LOF(미융착) 없이 안전한 공정 조건인지, conduction/keyhole 중
어느 영역인지 판정한다. 실측 데이터(316L 스테인리스, 623개 조건)로 학습하되, 물리 평가기(P)를
쓰는 AI↔Physics 폐루프(Looped Transformer)로 제한된 데이터·미지 구간에서도 물리적으로
일관된 예측을 유지한다. 현재 공정 상태를 판단해 위험시 행동(파워/속도 조정)을 추천하고,
ROM으로 가상 검증한다. 목표 깊이를 주면 안전한 조건을 역으로 제안하는 목표제어(inverse
design)도 지원한다. 그 위에서 로컬 sLLM 에이전트가 자연어로 What-if·의사결정을 지원한다.

## 역할 분담
- **전혁건** — PM, 개발 일정 관리, 데이터 전처리 및 baseline 생성·성능 측정
- **문기준** — Technical Lead, Looped Transformer + Physics Evaluator 구현 (`model/LoopedTransformer/`)
- **권태훈** — Physics & Digital Twin Lead, 공정 상태별 판단·행동결정·Digital Twin 가상검증·E2E Platform 구성 (`sim/`, `agent/`, `app.py`)

## 핵심 실험 결과 (`model/train.py`, 2개 실험)
- conduction 영역 내부 일반화(공정한 비교): physics-aware가 baseline MLP 대비 MSE 약 43% 낮음
- conduction→keyhole 영역 전환(위험 시나리오): 물리 ROM 자체의 정확도가 떨어지는 구간이라
  physics-aware의 우위가 크게 줄어듦(약 4%) — ROM 신뢰도가 낮은 영역에서는 물리 잔차를
  맹신하면 안 된다는 걸 실측으로 확인, regime gate(`classify_regime`) 설계의 근거가 됨

## 준비

```powershell
pip install -r requirements.txt
```

로컬 sLLM(에이전트용)도 필요합니다:
1. https://ollama.com 에서 Ollama 설치 (또는 `winget install Ollama.Ollama`)
2. `ollama pull qwen2.5:7b`
3. 설치하면 서버가 자동 실행됩니다. 안 떠 있으면 `ollama serve`

**경상국립대 AI 데이터센터 GPU 서버**(H200, 팀 계정 공유)도 사용 가능 — `ssh -p 26917 team21@203.230.214.11`,
Conda `edu` 환경(`/opt/anaconda3/envs/edu`) 제공. VS Code Remote-SSH로 접속.

## 실행 전 환경변수 (매 터미널 세션마다)

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # Anaconda numpy/torch OpenMP 충돌 방지
$env:PYTHONIOENCODING = "utf-8"       # 콘솔 한글 깨짐 방지
```

## 실행

패키지끼리 서로 import하므로 (`sim`, `model`, `agent`), 저장소 루트에서
`python -m 패키지.파일` 형태로 실행해야 합니다. (`sim/lpbf.py`만 예외적으로 직접 실행 가능)

```powershell
python sim\lpbf.py              # 물리 ROM(Rosenthal/LOF/regime) self-check
python -m sim.decision          # 공정상태 판단 + 행동결정 self-check
python -m model.physics_core    # Looped 코어 구조 self-check
python -m model.baseline        # baseline 모델 self-check
python -m model.train           # 실측 데이터 학습 + physics-aware vs baseline 비교 (핵심 실험)
python -m agent.tools           # 용융풀 예측/행동추천 도구 self-check
python -m agent.loop            # 에이전트 tool-calling 테스트 (Ollama 필요)
streamlit run app.py            # 데모 대시보드
```

## 구조

```
data/               316L 실측 melt pool 데이터 (CSV)
sim/                LPBF 물리 ROM(lpbf.py) + 공정상태 판단·행동결정(decision.py)
model/              physics-aware looped 코어(초기 버전) + baseline + 학습 스크립트
model/LoopedTransformer/   문기준님 공식 Looped Transformer 모델 (인터페이스: interfaceLT.py)
digitaltwin/        STL/CLI 툴패스 + FiPy/PyBullet 기반 LT 이전 DT 모듈
agent/              로컬 sLLM 도구호출 루프
app.py              Streamlit 데모 대시보드
DOCS/               개발계획서 등 공식 제출 문서
```

## 현재 상태 / 알려진 이슈
- `model/LoopedTransformer/`의 공식 LT 모델(v3)이 최신 물리 평가기(FluidThermalPhysicsEvaluator,
  7차원 잔차)까지 반영됨. 단 `interfaceLT.py`가 추론 시 물리평가기를 실제로 안 넘기는 버그가 있어
  로컬에서 임시 수정 후 v2·v3 정상 동작 확인함 (아직 리포지토리에는 미반영)
- `sim/decision.py`의 행동결정은 현재 `sim/lpbf.py`의 자체 ROM을 쓰고 있음 — 위 LT 모델 연동은
  인터페이스 확정 후 진행 예정
- STL -> ASCII CLI -> 공정 이벤트 -> FiPy 열장/PyBullet 경로 재생의 LT 이전 모듈 구현 완료 (`digitaltwin/README.md`)
- CLI/분말/공정 설정은 현재 316L 데모 프로파일이며, 분말 분포와 preheat는 장비 인증값이 아닌 명시적 데모 가정
- FiPy 결과는 정성적 열 추세이고 PyBullet은 시각화용이며, LT 예측·결함 보증·장비 제어와 아직 연결되지 않음

## 데이터 출처
- 316L 실측 melt pool 데이터: [Zenodo 16979848](https://zenodo.org/records/16979848) (Hofmann 2025, Aconity-Midi 장비, 677트랙/고유조건 623개, CC BY 4.0)
