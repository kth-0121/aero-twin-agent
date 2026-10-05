/* 공정 조건 검토 패널. */
(() => {
  const mount = document.getElementById('decision-panel');
  if (!mount) return;

  const STATUS = { normal: ['정상', 'normal'], lof_risk: ['미융착 위험', 'risk'], keyhole_risk: ['과열 위험', 'risk'] };
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
  const runButton = el('button', { className: 'primary', type: 'button', textContent: '조건 분석' });
  const useJobButton = el('button', { type: 'button', textContent: '권고 조건 불러오기', title: '작업 분석에서 제시한 출력과 속도를 입력합니다' });
  const setConditions = c => Object.entries(c).forEach(([k, v]) => { if (inputs[k]) inputs[k].value = v; });
  const readConditions = () => Object.fromEntries(Object.entries(inputs).map(([k, n]) => [k, Number(n.value)]));

  // ── 결과 영역 ──
  const summary = el('div', { className: 'dp-summary' }, el('strong', { textContent: '조건을 입력하고 분석하세요' }));
  const facts = el('div', { className: 'result-list' });
  const notes = el('div');
  let table = el('div', { className: 'dp-empty', textContent: '위험 조건이 확인되면 조정안과 재평가 결과가 표시됩니다.' });
  const results = el('div', { className: 'dp-results' }, summary, facts, notes, table);

  mount.append(
    el('section', { className: 'card' },
      el('div', { className: 'card-head' },
        el('span', { className: 'card-title', textContent: '공정 조건 분석' }),
        el('span', { className: 'card-subtitle', textContent: '입력 조건의 용융 상태를 확인하고 조정안을 비교합니다.' })),
      el('div', { className: 'dp-body' },
        el('div', { className: 'dp-controls' }, presets, fields, runButton, useJobButton),
        results)),
  );

  const row = (label, value) => el('div', { className: 'result-row' }, el('span', { textContent: label }), el('strong', { textContent: value }));

  function render(r) {
    const s = r.current_state;
    const [label, kind] = STATUS[s.status] || [s.status, 'risk'];
    summary.replaceChildren(
      el('strong', { textContent: `현재 상태: ${label} · 권고: ${r.recommended_action}` }),
      el('span', { className: `status-pill ${kind}`, textContent: label }));
    facts.replaceChildren(
      row('용융풀 폭 / 깊이 (추정)', `${s.width_um} / ${s.depth_um} µm`),
      row('미융착 위험 지표 (1 이하)', String(s.lof_criterion)),
      row('용융 모드', s.regime === 'conduction' ? '전도 용융' : '천이·과열 구간'),
      row('예측 한계', '단일 추정값이며 품질을 보증하지 않습니다.'));
    notes.replaceChildren(...(r.warnings || []).map(w => el('p', { className: 'dp-warn', textContent: '⚠ ' + w })));

    const list = r.candidates_evaluated || [];
    if (!list.length) {
      table.replaceWith(table = el('div', { className: 'dp-empty', textContent: '현재 입력 조건에서 조정 권고가 없습니다. 입력한 조건을 유지하세요.' }));
      return;
    }
    const head = el('tr', {}, ...['조정 항목', '출력 (W)', '속도 (mm/s)', '재평가 상태', '미융착 지표', '위험 개선'].map(h => el('th', { textContent: h })));
    const body = list.map(c => {
      const st = c.resulting_state;
      return el('tr', { className: c.action === r.recommended_action ? 'best' : '' },
        el('td', { textContent: c.action }), el('td', { textContent: c.new_power_w }), el('td', { textContent: c.new_speed_mm_s }),
        el('td', { textContent: (STATUS[st.status] || [st.status])[0] }), el('td', { textContent: st.lof_criterion }),
        el('td', { className: c.resolved ? 'ok' : 'no', textContent: c.resolved ? '개선' : '미개선' }));
    });
    const next = el('div', {}, el('table', { className: 'dp-table' }, el('thead', {}, head), el('tbody', {}, body)),
      el('p', { className: 'dp-note', textContent: '입력 조건을 기준으로 조정안을 다시 계산했습니다. 강조된 항목이 권고 조건입니다. 결과는 참고용 추정치입니다.' }));
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
    if (!power || !speed) {
      notes.replaceChildren(el('p', {
        className: 'dp-warn',
        textContent: '불러올 권고 조건이 없습니다. 먼저 작업 파일을 등록하고 위험 분석을 완료하세요.',
      }));
      return;
    }
    setConditions({ power_w: power, speed_mm_s: speed });
    run();
  };

  api('/api/scenarios').then(list => list.forEach(s => presets.append(
    el('button', {
      type: 'button',
      textContent: ({ normal: '기준 조건', lof: '미융착 위험 조건', keyhole: '과열 위험 조건' })[s.id] || s.label,
      onclick: () => { setConditions({ power_w: s.power_w, speed_mm_s: s.speed_mm_s }); run(); },
    })
  ))).catch(() => {});

  const question = el('input', {
    type: 'text',
    value: '출력 80 W, 속도 1,500 mm/s 조건의 미융착 위험과 조정 권고를 알려주세요.',
    'aria-label': '공정 조건 질문',
  });
  const askButton = el('button', { type: 'button', textContent: '문의' });
  const answer = el('div');
  askButton.onclick = async () => {
    askButton.disabled = true;
    answer.replaceChildren(el('div', { className: 'dp-answer', textContent: '조건을 검토하고 있습니다…' }));
    try { answer.replaceChildren(el('div', { className: 'dp-answer', textContent: (await api('/api/agent', { question: question.value })).answer })); }
    catch (e) { answer.replaceChildren(el('div', { className: 'dp-answer err', textContent: e.message })); }
    finally { askButton.disabled = false; }
  };
  mount.append(el('section', { className: 'card' },
    el('div', { className: 'card-head' },
      el('span', { className: 'card-title', textContent: '공정 조건 문의' }),
      el('span', { className: 'card-subtitle', textContent: '입력 조건과 분석 결과에 대해 질문할 수 있습니다.' })),
    el('div', { className: 'dp-ask' }, question, askButton), answer));
})();
