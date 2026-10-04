/* 의사결정 지원 패널: 조건 → 상태 판단 → 행동 후보 → 재검증, 그리고 로컬 sLLM Agent 질의.
   백엔드: POST /api/recommend, GET /api/scenarios, POST /api/agent (digitaltwin/backend/decision_api.py) */
(() => {
  const mount = document.getElementById('decision-panel');
  if (!mount) return;

  const STATUS = { normal: ['정상', 'normal'], lof_risk: ['LOF 위험', 'risk'], keyhole_risk: ['Keyhole 위험', 'risk'] };
  const FIELDS = [
    { key: 'power_w', label: '레이저 출력 (W)', value: 250, min: 50, max: 600, step: 5 },
    { key: 'speed_mm_s', label: '스캔 속도 (mm/s)', value: 800, min: 100, max: 2000, step: 10 },
    { key: 'spot_diameter_um', label: '스폿 직경 (µm)', value: 100, min: 50, max: 200, step: 5 },
    { key: 'hatch_um', label: '해치 간격 (µm)', value: 100, min: 50, max: 200, step: 5 },
    { key: 'layer_thickness_um', label: '층 두께 (µm)', value: 30, min: 20, max: 100, step: 5, wide: true },
  ];

  const el = (tag, props = {}, ...kids) => {
    const node = Object.assign(document.createElement(tag), props);
    kids.flat().forEach(k => node.append(k instanceof Node ? k : document.createTextNode(String(k))));
    return node;
  };
  const api = async (path, body) => {
    const res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
    return data;
  };

  // ── 입력 영역 ──
  const inputs = {};
  const presets = el('div', { className: 'dp-presets' });
  const fields = el('div', { className: 'dp-fields' });
  FIELDS.forEach(f => {
    const input = el('input', { type: 'number', min: f.min, max: f.max, step: f.step, value: f.value, id: `dp-${f.key}` });
    inputs[f.key] = input;
    fields.append(el('div', { className: 'dp-field' + (f.wide ? ' wide' : '') }, el('label', { htmlFor: input.id, textContent: f.label }), input));
  });
  const runButton = el('button', { className: 'primary', type: 'button', textContent: '판단 · 재검증 실행' });
  const useJobButton = el('button', { type: 'button', textContent: '작업 권고 조건으로 검토', title: '위 "장기 공정 변수 권고"의 출력·속도를 불러옵니다' });
  const setConditions = c => Object.entries(c).forEach(([k, v]) => { if (inputs[k]) inputs[k].value = v; });
  const readConditions = () => Object.fromEntries(Object.entries(inputs).map(([k, n]) => [k, Number(n.value)]));

  // ── 결과 영역 ──
  const summary = el('div', { className: 'dp-summary' }, el('strong', { textContent: '조건을 선택하고 실행하세요' }));
  const facts = el('div', { className: 'result-list' });
  const notes = el('div');
  let table = el('div', { className: 'dp-empty', textContent: '위험 상태일 때 행동 후보와 재검증 결과가 표시됩니다.' });
  const results = el('div', { className: 'dp-results' }, summary, facts, notes, table);

  mount.append(
    el('section', { className: 'card' },
      el('div', { className: 'card-head' },
        el('span', { className: 'card-title', textContent: '의사결정 지원 · 조건 검토' }),
        el('span', { className: 'card-subtitle', textContent: '조건 → 상태 판단 → 행동 후보 → 재검증 (가상 검증, 장비 명령 없음)' })),
      el('div', { className: 'dp-body' },
        el('div', { className: 'dp-controls' }, presets, fields, runButton, useJobButton),
        results)),
  );

  const row = (label, value) => el('div', { className: 'result-row' }, el('span', { textContent: label }), el('strong', { textContent: value }));

  function render(r) {
    const s = r.current_state;
    const [label, kind] = STATUS[s.status] || [s.status, 'risk'];
    summary.replaceChildren(
      el('strong', { textContent: r.reason ? `판단: ${r.reason}` : `판단: ${r.recommended_action}` }),
      el('span', { className: `status-pill ${kind}`, textContent: label }));
    facts.replaceChildren(
      row('예측 폭 / 깊이', `${s.width_um} / ${s.depth_um} µm`),
      row('LOF 지표 (≤ 1 안전)', String(s.lof_criterion)),
      row('Regime', s.regime === 'conduction' ? 'conduction' : 'keyhole / transition'),
      row('예측 모델', r.prediction_model),
      row('불확실성', r.uncertainty));
    notes.replaceChildren(...(r.warnings || []).map(w => el('p', { className: 'dp-warn', textContent: '⚠ ' + w })));

    const list = r.candidates_evaluated || [];
    if (!list.length) {
      table.replaceWith(table = el('div', { className: 'dp-empty', textContent: '현재 상태가 정상이라 행동 후보가 없습니다 (공정 유지).' }));
      return;
    }
    const head = el('tr', {}, ...['행동', '새 출력 (W)', '새 속도 (mm/s)', '재검증 결과', 'LOF', '해소'].map(h => el('th', { textContent: h })));
    const body = list.map(c => {
      const st = c.resulting_state;
      return el('tr', { className: c.action === r.recommended_action ? 'best' : '' },
        el('td', { textContent: c.action }), el('td', { textContent: c.new_power_w }), el('td', { textContent: c.new_speed_mm_s }),
        el('td', { textContent: (STATUS[st.status] || [st.status])[0] }), el('td', { textContent: st.lof_criterion }),
        el('td', { className: c.resolved ? 'ok' : 'no', textContent: c.resolved ? '✔ 해소' : '✖ 미해소' }));
    });
    const next = el('div', {}, el('table', { className: 'dp-table' }, el('thead', {}, head), el('tbody', {}, body)),
      el('p', { className: 'dp-note', textContent: `검증 방식: ${r.verification}. 강조된 행이 추천 행동입니다.` }));
    table.replaceWith(next);
    table = next;
  }

  async function run() {
    runButton.disabled = true;
    try { render(await api('/api/recommend', readConditions())); }
    catch (e) { summary.replaceChildren(el('strong', { textContent: '판단 실패' })); notes.replaceChildren(el('p', { className: 'dp-warn', textContent: e.message })); }
    finally { runButton.disabled = false; }
  }
  runButton.onclick = run;
  useJobButton.onclick = () => {
    const num = id => parseFloat((document.getElementById(id)?.textContent || '').replace(/[^0-9.]/g, ''));
    const power = num('recommended-power'), speed = num('recommended-speed');
    if (!power || !speed) { notes.replaceChildren(el('p', { className: 'dp-warn', textContent: '아직 작업 권고가 없습니다. AI 분석을 먼저 완료하세요.' })); return; }
    setConditions({ power_w: power, speed_mm_s: speed });
    run();
  };

  api('/api/scenarios').then(list => list.forEach(s => presets.append(
    el('button', { type: 'button', textContent: s.label, onclick: () => { setConditions({ power_w: s.power_w, speed_mm_s: s.speed_mm_s }); run(); } })
  ))).catch(() => {});

  // ── Agent 질의 ──
  const question = el('input', { type: 'text', value: '레이저 출력 80W, 스캔속도 1500mm/s로 찍으면 LOF 없이 안전해? 위험하면 어떻게 바꿔야 해?', 'aria-label': 'Agent 질문' });
  const askButton = el('button', { type: 'button', textContent: '질문' });
  const answer = el('div');
  askButton.onclick = async () => {
    askButton.disabled = true;
    answer.replaceChildren(el('div', { className: 'dp-answer', textContent: 'Agent 응답 대기 중… (로컬 모델이라 수십 초 걸릴 수 있습니다)' }));
    try { answer.replaceChildren(el('div', { className: 'dp-answer', textContent: (await api('/api/agent', { question: question.value })).answer })); }
    catch (e) { answer.replaceChildren(el('div', { className: 'dp-answer err', textContent: e.message })); }
    finally { askButton.disabled = false; }
  };
  mount.append(el('section', { className: 'card' },
    el('div', { className: 'card-head' },
      el('span', { className: 'card-title', textContent: 'Agent 질의' }),
      el('span', { className: 'card-subtitle', textContent: '로컬 sLLM(Ollama)이 Tool을 호출해 판단 결과를 설명 · 물리 계산은 Tool이 수행' })),
    el('div', { className: 'dp-ask' }, question, askButton), answer));
})();
