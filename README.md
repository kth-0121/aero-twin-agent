# Physics-aware Physical AI 기반 우주항공 제조 Digital Twin Agent

> **제4회 경남 AI·SW경진대회**
> **팀명:** 601Solutions (전혁건 · 문기준 · 권태훈)

Physics Reasoning Agent를 활용하여 우주항공 제조 분야의 **LPBF(레이저 파우더 베드 퓨전)** 공정을 지능화하고 가상 검증 및 제어를 수행하는 Digital Twin 솔루션입니다.

---

## 주요 특징 (Project Overview)

* **AI ↔ Physics 폐루프 (Looped Transformer):** 제한된 실측 데이터(316L 스테인리스, 623개 조건) 환경에서도 물리 평가기($P$)를 결합하여 미지 구간에서 물리적으로 일관된 용융풀(Melt Pool) 깊이/폭을 예측합니다.
* **공정 상태 및 영역 판정:** LOF(미융착) 발생 여부, Conduction / Keyhole 영역을 실시간으로 분석합니다.
* **역설계(Inverse Design) 제어:** 목표 용융풀 깊이를 입력받아 안전한 레이저 파워 및 스캔 속도 조건 역제안합니다.
* **자연어 기반 What-if 의사결정:** 로컬 sLLM(Ollama / Qwen2.5) 기반 에이전트가 공정 상태 판단, 위험 시 행동(파워/속도 조정) 추천, 가상 검증(ROM)을 자연어로 지원합니다.

---

## 팀원 및 역할 분담

| 이름 | 담당 역할 | 주요 작업 내용 |
| --- | --- | --- |
| **전혁건** | **PM** | 개발 일정 관리, 데이터 전처리 및 Baseline 생성·성능 측정 |
| **문기준** | **Technical Lead** | Looped Transformer + Physics Evaluator 모델 구현 |
| **권태훈** | **Physics & DT Lead** | 공정 상태 판단·행동 결정, Digital Twin 가상 검증 및 E2E Platform 구성 |

---

## 프로젝트 구조

```text
.
├── data/                    # 316L 실측 Melt Pool 데이터 (CSV)
├── sim/                     # LPBF 물리 ROM (lpbf.py) + 공정 상태 판단 및 행동 결정
├── model/                   # Baseline 및 학습 스크립트
│   └── LoopedTransformer/   # Physics-aware Looped Transformer 모델
├── digitaltwin/             # STL/CLI 툴패스 + FiPy/PyBullet 기반 Digital Twin 모듈
│   └── backend/             # FastAPI 백엔드
├── agent/                   # 로컬 sLLM 도구 호출(Tool-calling) 루프
├── DOCS/                    # 개발계획서 및 제출 문서 목록
├── app.py                   # Streamlit 데모 대시보드
└── requirements.txt         # 의존성 패키지 목록

```

---

## 사전 준비

### 1. Python 패키지 설치

```powershell
pip install -r requirements.txt

```

### 2. 로컬 sLLM 에이전트 설정 (Ollama)

1. [Ollama 공식 홈페이지](https://ollama.com)에서 설치 (또는 `winget install Ollama.Ollama` 실행)
2. 모델 다운로드 및 실행:
```powershell
ollama pull qwen2.5:7b

```


*설치 후 백그라운드 서버가 자동으로 시작됩니다. 미실행 시 `ollama serve`를 직접 입력하세요.*

> **참고:** 경상국립대 AI 데이터센터 GPU 서버(H200) 접속 정보는 팀 내부 공유용 문서 및 환경 설정을 참고하세요.

---

## 실행 전 환경 변수 설정

터미널 세션마다 실행 전 아래 명령어로 환경 변수를 설정합니다.

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # Anaconda numpy/torch OpenMP 충돌 방지
$env:PYTHONIOENCODING = "utf-8"       # 콘솔 한글 출력 깨짐 방지

```

---

## 검증 및 테스트 실행 (TEST)

각 모듈 간 의존성(`sim`, `model`, `agent`)이 있으므로 저장소 루트에서 `python -m 패키지.파일` 형식으로 실행합니다.

```powershell
# 1. 물리 ROM (Rosenthal/LOF/Regime) 검증
python sim\lpbf.py

# 2. 공정 상태 판단 및 행동 결정 기능 검증
python -m sim.decision

# 3. Looped Transformer 코어 구조 검증
python -m model.physics_core

# 4. Baseline 모델 검증
python -m model.baseline

# 5. 실측 데이터 학습 및 성능 비교 (Physics-aware vs Baseline 핵심 실험)
python -m model.train

# 6. 에이전트 Tool 테스트
python -m agent.tools

# 7. LLM 에이전트 Loop 테스트 (Ollama 실행 필수)
python -m agent.loop

# 8. Streamlit 데모 대시보드 실행
streamlit run app.py

```

---

## 서비스 실행 (APP)

FastAPI 기반의 연결형 LPBF 콘솔 서버를 실행합니다.

```powershell
python -m uvicorn digitaltwin.backend.main:app --host 0.0.0.0 --port 8000

```

> 서버 실행 후 브라우저에서 `http://localhost:8000/docs`로 접속하면 API 명세를 확인할 수 있습니다.

---

## Dataset Citation

* **316L Stainless Steel Melt Pool Dataset:** [Zenodo 16979848](https://zenodo.org/records/16979848)
* Reference: Hofmann (2025), Aconity-Midi 장비 수집 데이터
* Total Tracks: 677개 트랙 (고유 공정 조건 623개)
* License: CC BY 4.0
