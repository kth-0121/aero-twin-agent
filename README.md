# Physics-aware Physical AI 기반 우주항공 제조 Digital Twin Agent 개발

Physics Reasoning Agent를 활용한 우주항공 제조 Digital Twin 지능화 — 제4회 경남 AI·SW경진대회.
팀 601Solutions (전혁건·문기준·권태훈).

**시나리오: LPBF(레이저 파우더베드퓨전) 용융풀 예측.** 레이저 파워·스캔속도를 입력하면
용융풀 깊이/폭을 예측하고, LOF(미융착) 없이 안전한 공정 조건인지, conduction/keyhole 중
어느 영역인지 판정한다. 실측 데이터(316L 스테인리스, 623개 조건)로 학습하되, 물리 평가기(P)를
쓰는 AI↔Physics 폐루프(Looped Transformer)로 제한된 데이터·미지 구간에서도 물리적으로
일관된 예측을 유지한다. 현재 공정 상태를 판단해 위험시 행동(파워/속도 조정)을 추천하고,
ROM으로 가상 검증한다. 목표 깊이를 주면 안전한 조건을 역으로 제안하는 목표제어(inverse
design)도 지원한다. 그 위에서 로컬 sLLM 에이전트가 자연어로 What-if·의사결정을 지원한다.

## 연결형 LPBF 콘솔 (FastAPI)

저장소 루트에서 `python -m uvicorn digitaltwin.backend.main:app --host 0.0.0.0 --port 8000`으로 실행한다.

## 역할 분담
- **전혁건** — PM, 개발 일정 관리, 데이터 전처리 및 baseline 생성·성능 측정
- **문기준** — Technical Lead, Looped Transformer + Physics Evaluator 구현
- **권태훈** — Physics & Digital Twin Lead, 공정 상태별 판단·행동결정·Digital Twin 가상검증·E2E Platform 구성

## 준비

```powershell
pip install -r requirements.txt
```

로컬 sLLM(에이전트용)도 필요합니다:
1. https://ollama.com 에서 Ollama 설치 (또는 `winget install Ollama.Ollama`)
2. `ollama pull qwen2.5:7b`
3. 설치하면 서버가 자동 실행됩니다. 안 떠 있으면 `ollama serve`

경상국립대 AI 데이터센터 GPU 서버(H200)도 사용할 수 있다. 접속 정보는 팀 내부에서만 공유한다.

## 실행 전 환경변수 (매 터미널 세션마다)

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # Anaconda numpy/torch OpenMP 충돌 방지
$env:PYTHONIOENCODING = "utf-8"       # 콘솔 한글 깨짐 방지
```

## 실행 (TEST)

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

## 실행 (APP)
root에서
```powershell
python -m uvicorn digitaltwin.backend.main:app --host 0.0.0.0 --port 8000

## 구조

```
data/               실측 melt pool 데이터 (CSV)
sim/                LPBF 물리 ROM(lpbf.py) + 공정상태 판단·행동결정
model/              baseline + 학습 스크립트
model/LoopedTransformer/    Looped Transformer 모델
digitaltwin/        STL/CLI 툴패스 + FiPy/PyBullet 기반 DT 모듈
agent/              로컬 sLLM 도구호출 루프
DOCS/               개발계획서 등 공식 제출 문서
```

## 데이터 출처
- 316L 실측 melt pool 데이터: [Zenodo 16979848](https://zenodo.org/records/16979848) (Hofmann 2025, Aconity-Midi 장비, 677트랙/고유조건 623개, CC BY 4.0)
