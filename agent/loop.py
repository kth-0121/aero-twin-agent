"""Minimal tool-calling loop against a local Ollama sLLM.

No agent framework: a couple of tools doesn't need one. The model decides
which tool to call, we execute it locally, and feed the result back for a
final natural-language recommendation. Requires `ollama serve` running with
MODEL pulled (e.g. `ollama pull qwen2.5:7b`).
"""
import json
import requests

from agent.tools import TOOLS

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

# ponytail: qwen models sometimes drift into Chinese mid-answer; a blunt
# system prompt is the cheap fix -- swap models if this stops being enough
SYSTEM_PROMPT = "당신은 우주항공 제조 공정 Digital Twin 에이전트입니다. 반드시 한국어로만 답변하세요. 중국어나 다른 언어를 절대 섞지 마세요."

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "run_whatif",
            "description": "Simulate a composite cure cycle and report the resulting degree of cure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ramp_rate_c_per_min": {"type": "number"},
                    "hold_temp_c": {"type": "number"},
                    "hold_seconds": {"type": "number"},
                },
                "required": ["ramp_rate_c_per_min", "hold_temp_c"],
            },
        },
    }
]


def ask_agent(user_message, history=None):
    messages = list(history) if history else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_message})
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL, "messages": messages, "tools": TOOL_SPECS, "stream": False,
    })
    resp.raise_for_status()
    msg = resp.json()["message"]

    if msg.get("tool_calls"):
        messages.append(msg)
    for call in msg.get("tool_calls", []):
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        result = TOOLS[name](**args)
        messages.append({"role": "tool", "content": json.dumps(result)})

    if msg.get("tool_calls"):
        final = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "stream": False,
        })
        final.raise_for_status()
        return final.json()["message"]["content"]

    return msg["content"]


if __name__ == "__main__":
    # manual check -- needs a running Ollama server, so not an assert-based demo()
    print(ask_agent("승온속도 3도/분, 유지온도 180도로 경화하면 완전히 경화돼?"))
