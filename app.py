"""E2E demo dashboard: real 316L melt pool data -> physics-aware core -> agent recommendation.

No physical rig here -- LPBF needs an actual laser + metal powder bed, which
isn't reproducible on a desk in two weeks. Instead we replay real measured
rows (data/316L_meltpool.csv) as if they were arriving live, so the "real,
not synthetic" story still holds: the model reacts to genuine measurements,
not numbers we made up.
"""
import torch
import streamlit as st

from agent.tools import predict_melt_pool
from agent.loop import ask_agent
from model.train import fit_core, load_dataset, physics_fn

st.set_page_config(page_title="Aero Twin Agent", layout="wide")
st.title("우주항공 제조 Digital Twin — LPBF Physics Reasoning Agent")


@st.cache_resource
def get_core():
    with st.spinner("Physics-aware 코어 학습 중 (최초 1회, 수십 초 소요)..."):
        x_train, y_train = load_dataset(power_range=(0, 400))
        return fit_core(x_train, y_train, hidden_dim=32, n_loops=3)


@st.cache_resource
def get_replay_rows():
    x, y = load_dataset(power_range=(0, 10_000))
    return x.tolist(), y.tolist()


core = get_core()
replay_x, replay_y = get_replay_rows()

st.header("1. 실측 데이터 재생 → 코어 예측 vs 실제값")
st.caption("data/316L_meltpool.csv의 실측 행을 하나씩 흘려보내며, 학습된 physics-aware 코어의 예측과 실제 측정값을 비교합니다.")

if "replay_i" not in st.session_state:
    st.session_state.replay_i = 0
    st.session_state.pred_depths, st.session_state.true_depths = [], []
    st.session_state.pred_widths, st.session_state.true_widths = [], []

if st.button("다음 실측 행 재생"):
    i = st.session_state.replay_i % len(replay_x)
    x_row = torch.tensor([[replay_x[i]]], dtype=torch.float32)
    pred, _ = core(x_row, physics_fn)
    pred = pred.squeeze().tolist()
    st.session_state.pred_depths.append(pred[0])
    st.session_state.true_depths.append(replay_y[i][0])
    st.session_state.pred_widths.append(pred[1])
    st.session_state.true_widths.append(replay_y[i][1])
    st.session_state.replay_i += 1

col1, col2 = st.columns(2)
with col1:
    st.line_chart({"예측 depth": st.session_state.pred_depths, "실제 depth": st.session_state.true_depths}, height=250)
with col2:
    st.line_chart({"예측 width": st.session_state.pred_widths, "실제 width": st.session_state.true_widths}, height=250)

st.header("2. What-if 시뮬레이션")
power = st.slider("레이저 파워 (W)", 50, 500, 200)
speed = st.slider("스캔속도 (mm/s)", 100, 2000, 800)
if st.button("시뮬레이션 실행"):
    st.session_state.whatif_result = predict_melt_pool(power, speed)
if "whatif_result" in st.session_state:
    st.json(st.session_state.whatif_result)

st.header("3. Agent 권고")
question = st.text_input(
    "질문", f"레이저 파워 {power}W, 스캔속도 {speed}mm/s로 찍으면 LOF 없이 안전하게 붙어?"
)
if st.button("Agent에게 물어보기"):
    try:
        with st.spinner("Agent 응답 대기 중..."):
            st.write(ask_agent(question))
    except Exception as e:
        st.error(f"Agent 호출 실패 (Ollama 서버가 켜져 있는지 확인하세요): {e}")
