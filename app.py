"""E2E demo dashboard: live rig temperature -> physics-aware core -> agent recommendation."""
import torch
import streamlit as st

from rig.serial_reader import read_mock
from agent.tools import run_whatif
from agent.loop import ask_agent
from model.train import fit_core, physics_fn

st.set_page_config(page_title="Aero Twin Agent", layout="wide")
st.title("우주항공 제조 Digital Twin — Physics Reasoning Agent")

READING_DT_S = 1.0  # 센서 한 스텝 사이 가정 시간(초), read_mock 주기와 맞춤


@st.cache_resource
def get_core():
    with st.spinner("Physics-aware 코어 학습 중 (최초 1회, 수십 초 소요)..."):
        return fit_core(hidden_dim=32, n_loops=3)


core = get_core()


def infer_next_alpha(temp_c, prev_alpha):
    x = torch.tensor([[[temp_c + 273.15, prev_alpha, READING_DT_S]]], dtype=torch.float32)
    with torch.no_grad():
        pred, _ = core(x, physics_fn)
    return float(pred.clamp(0.0, 1.0).item())


st.header("1. 실시간 경화 리그 센서 → 내부 경화도 추론")
st.caption("표면 온도(관측 가능)만 보고, physics-aware 코어가 내부 경화도(직접 관측 불가)를 추론합니다.")

if "rig" not in st.session_state:
    st.session_state.rig = read_mock(sleep_s=READING_DT_S)
    st.session_state.temps = []
    st.session_state.alphas = []

if st.button("다음 센서값 읽기"):
    temp = next(st.session_state.rig)
    prev_alpha = st.session_state.alphas[-1] if st.session_state.alphas else 0.0
    st.session_state.temps.append(temp)
    st.session_state.alphas.append(infer_next_alpha(temp, prev_alpha))

col1, col2 = st.columns(2)
col1.line_chart(st.session_state.temps, height=200)
col1.caption("표면 온도 (도C)")
col2.line_chart(st.session_state.alphas, height=200)
col2.caption("추론된 경화도 (0~1)")

st.header("2. What-if 시뮬레이션")
ramp = st.slider("승온속도 (도/분)", 1.0, 10.0, 3.0)
hold = st.slider("유지온도 (도C)", 120, 200, 180)
if st.button("시뮬레이션 실행"):
    st.session_state.whatif_result = run_whatif(ramp, hold)
if "whatif_result" in st.session_state:
    st.json(st.session_state.whatif_result)

st.header("3. Agent 권고")
question = st.text_input(
    "질문", f"승온속도 {ramp}도/분, 유지온도 {hold}도로 경화하면 완전히 경화돼?"
)
if st.button("Agent에게 물어보기"):
    try:
        with st.spinner("Agent 응답 대기 중..."):
            st.write(ask_agent(question))
    except Exception as e:
        st.error(f"Agent 호출 실패 (Ollama 서버가 켜져 있는지 확인하세요): {e}")
