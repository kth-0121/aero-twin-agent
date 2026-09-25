"""Minimal tool-calling loop against a local Ollama sLLM.

No agent framework: a couple of tools doesn't need one. The model decides
which tool to call, we execute it locally, and feed the result back for a
final natural-language recommendation. Requires `ollama serve` running with
MODEL pulled (e.g. `ollama pull qwen2.5:7b`).
"""
import json
import re
import requests

from agent.tools import TOOLS

# ponytail: qwen's native tool-call format is a literal <tool_call>{json}</tool_call>
# block; Ollama usually lifts this into the structured tool_calls field but not
# always -- fall back to parsing it out of raw content when that happens.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(msg):
    if msg.get("tool_calls"):
        return msg["tool_calls"]
    match = TOOL_CALL_RE.search(msg.get("content") or "")
    if not match:
        return []
    try:
        call = json.loads(match.group(1))
        return [{"function": {"name": call["name"], "arguments": call["arguments"]}}]
    except (json.JSONDecodeError, KeyError):
        return []


def _looks_like_broken_tool_call(content):
    """Sometimes the model emits a tool-call attempt that's truncated/malformed
    enough that even the regex fallback can't parse it -- that's still a clear
    signal the sampled response is broken and worth retrying, not returning."""
    return bool(content) and "tool_call" in content.lower()

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

# ponytail: qwen models sometimes drift into Chinese mid-answer; a blunt
# system prompt is the cheap fix -- swap models if this stops being enough
SYSTEM_PROMPT = (
    "당신은 LPBF(금속 3D프린팅) 제조 공정 Digital Twin 에이전트입니다. "
    "LOF는 Lack of Fusion(미융착, 층/트랙이 제대로 붙지 않는 결함)의 약자입니다. "
    "반드시 한국어로만 답변하세요. 중국어나 다른 언어를 절대 섞지 마세요."
)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "predict_melt_pool",
            "description": "Predict LPBF melt pool depth/width for given laser power and scan speed, and check Lack-of-Fusion safety.",
            "parameters": {
                "type": "object",
                "properties": {
                    "laser_power_w": {"type": "number"},
                    "scan_speed_mm_s": {"type": "number"},
                    "hatch_um": {"type": "number"},
                    "layer_thickness_um": {"type": "number"},
                },
                "required": ["laser_power_w", "scan_speed_mm_s"],
            },
        },
    }
]


def ask_agent(user_message, history=None, max_attempts=3):
    messages = list(history) if history else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_message})

    # small local models occasionally sample a malformed tool-call attempt;
    # retrying (fresh sample each time) is cheap and usually fixes it
    for attempt in range(max_attempts):
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "tools": TOOL_SPECS, "stream": False,
            "options": {"temperature": 0.3},
        })
        resp.raise_for_status()
        msg = resp.json()["message"]
        tool_calls = _extract_tool_calls(msg)
        if tool_calls or not _looks_like_broken_tool_call(msg.get("content")):
            break

    if tool_calls:
        messages.append(msg)
    for call in tool_calls:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        result = TOOLS[name](**args)
        messages.append({"role": "tool", "content": json.dumps(result)})

    if tool_calls:
        final = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "stream": False,
        })
        final.raise_for_status()
        return final.json()["message"]["content"]

    return msg["content"]


if __name__ == "__main__":
    # manual check -- needs a running Ollama server, so not an assert-based demo()
    print(ask_agent("레이저 파워 200W, 스캔속도 800mm/s로 찍으면 LOF 없이 안전하게 붙어?"))
