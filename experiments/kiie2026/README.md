# KIIE 2026 추계학술대회 논문 실험 (Conduction 학습 → Keyhole 평가)

논문 "물리 피드백 Looped Transformer 기반 LPBF 디지털 트윈 에이전트"의 표·수치를 만든 스크립트와 결과 요약입니다.
레포 루트의 `model/Baseline`, `model/LoopedTransformer`(legacy 포함)와 `data/316L_meltpool.csv`를 그대로 import하므로 레포 안에서 실행해야 합니다.
학습·평가 데이터 분리와 사전 등록 기록은 `prereg20.txt`에 있습니다.

## 실행 환경
```
pip install torch numpy pandas scikit-learn scipy
set KMP_DUPLICATE_LIB_OK=TRUE      # Windows + conda MKL 중복 로드 회피
python exp_ens3.py train 4         # 30시드 × 4모델 × 2분할 학습 (CPU 코어 4개 기준 수 시간)
python exp_ens3.py judge           # results/ens3_summary.csv
```
다른 레포 위치에서 돌릴 때는 환경변수 `REPO`에 레포 루트를 지정합니다.

## 파일과 논문 표 대응
| 파일 | 내용 | 논문 |
|---|---|---|
| `exp_lt.py` | 공통 데이터 분할(D/W, 에너지), 합성 Keyhole 생성(2,000행), MLP 5설정·LT 학습 루프, 라운드별 탐색 잡 정의 | 3.1, 3.3, 4.1절 |
| `exp_ens3.py` | 시드 800~829를 10개씩 앙상블 3개로 묶어 오차·단조성·범위·물리 잔차·위험 판정·추천 평가 | 표 1, 4, 5 |
| `exp_risk.py` | 위험 판정(전도/전이/키홀)과 31×31 격자 조건 추천 규칙 | 3.4, 4.5절 |
| `exp_scen.py` | 에너지 부족/정상/과다 60개 시나리오 보조 실험 (격자 15/31/61) | 4.5절 보조 실험 |
| `exp_h2b.py` | 단일 모델 물리 피드백 유무 비교 (시드 500~509) | 4.4절, 표 A1 |
| `prereg20.txt` | 라운드 20 이후 사전 등록(판정 기준을 실행 전에 기록)과 결과 로그 | 4.1절, 5절 |
| `results/ens3_summary.csv` | 앙상블 3개 × 모델 4 × 분할 2 지표 | 표 1, 4, 5 |
| `results/lt_jobs9/10.jsonl.gz` | 합성 데이터 규모 탐색 (시드 300~309) | 표 3 |
| `results/lt_jobs11.jsonl.gz` | 단일 모델 확인 (시드 500~509) | 표 A1 |
| `results/lt_jobs12/13.jsonl.gz` | 앙상블 탐색·1차 확인 (시드 600~629, 700~729) | 표 2 |
| `results/scen_*.csv` | 60개 시나리오 결과·안정성 | 4.5절 보조 실험 |

표준화 MSE는 실측 Conduction 학습 행으로 맞춘 스케일러 기준이며, W·R·A 세 출력의 평균입니다.
실측 Keyhole 행은 어떤 단계에서도 학습·검증·설정 선택에 쓰지 않습니다.

## 데이터
Hofmann 등의 316L 단일 트랙 데이터(CC BY 4.0), `data/316L_meltpool.csv`. 677행.
