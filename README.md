# aero-twin-agent

Physics Reasoning Agent를 활용한 우주항공 제조 Digital Twin 지능화 — 제4회 경남 AI·SW경진대회.

물리 시뮬레이터로 학습데이터를 만들고, AI↔Physics 폐루프(Looped Transformer 컨셉)로
제한된 데이터·미지 환경에서도 물리적으로 일관된 상태를 추론하는 Physics-aware 모델과,
그 위에서 What-if 시뮬레이션/의사결정을 지원하는 로컬 sLLM 에이전트로 구성됩니다.

## 준비

```powershell
pip install -r requirements.txt
```

로컬 sLLM(에이전트용)도 필요합니다:
1. https://ollama.com 에서 Ollama 설치 (또는 `winget install Ollama.Ollama`)
2. `ollama pull qwen2.5:7b`
3. 설치하면 서버가 자동 실행됩니다. 안 떠 있으면 `ollama serve`

## 실행 전 환경변수 (매 터미널 세션마다)

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"   # Anaconda numpy/torch OpenMP 충돌 방지
$env:PYTHONIOENCODING = "utf-8"       # 콘솔 한글 깨짐 방지
```

## 실행

패키지끼리 서로 import하므로 (`sim`, `model`, `agent`), 저장소 루트에서
`python -m 패키지.파일` 형태로 실행해야 합니다. (`sim/cure.py`만 예외적으로 직접 실행 가능)

```powershell
python sim\cure.py              # 물리 시뮬레이터 self-check
python -m model.physics_core    # Looped 코어 구조 self-check
python -m model.baseline        # baseline 모델 self-check
python -m model.train           # 학습 + physics-aware vs baseline 비교 (핵심 실험)
python -m agent.tools           # what-if 도구 self-check
python -m agent.loop            # 에이전트 tool-calling 테스트 (Ollama 필요)
streamlit run app.py            # 데모 대시보드
```

## 구조

```
sim/      물리 시뮬레이터 (학습데이터 생성 + 물리 평가기 P)
model/    physics-aware looped 코어 + baseline 비교모델 + 학습 스크립트
agent/    로컬 sLLM 도구호출 루프
rig/      Arduino 펌웨어 + 시리얼 리더 (실물 리그용)
app.py    Streamlit 데모 대시보드
```
