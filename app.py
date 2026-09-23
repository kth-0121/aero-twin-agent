"""E2E demo dashboard: live rig temperature -> physics-aware core -> agent recommendation."""
import streamlit as st

from rig.serial_reader import read_mock
from agent.tools import run_whatif

st.set_page_config(page_title="Aero Twin Agent", layout="wide")
st.title("우주항공 제조 Digital Twin — Physics Reasoning Agent")

st.header("1. 실시간 경화 리그 센서")
if "rig" not in st.session_state:
    st.session_state.rig = read_mock()
    st.session_state.readings = []

if st.button("다음 센서값 읽기"):
    st.session_state.readings.append(next(st.session_state.rig))

st.line_chart(st.session_state.readings)

st.header("2. What-if 시뮬레이션")
ramp = st.slider("승온속도 (도/분)", 1.0, 10.0, 3.0)
hold = st.slider("유지온도 (도C)", 120, 200, 180)
if st.button("시뮬레이션 실행"):
    result = run_whatif(ramp, hold)
    st.json(result)

st.header("3. Agent 권고")
st.caption("agent.loop.ask_agent() 연결 예정 — 로컬 Ollama 서버 필요")
