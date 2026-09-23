const $ = id => document.getElementById(id);
let selected = 'cosmic';
let profiles = {};
let staticBundle = null;

function renderList(id, rows) {
  $(id).replaceChildren(...rows.map(row => {
    const li = document.createElement('li');
    li.textContent = row.title;
    return li;
  }));
}

function staticRank(history, alpha, k) {
  const base = staticBundle.recent;
  const seen = new Set(history);
  const latest = [];
  const distinct = new Set();
  for (const item of [...history].reverse()) {
    if (!distinct.has(item)) {
      latest.push(item);
      distinct.add(item);
      if (latest.length === 20) break;
    }
  }
  const personal = {};
  for (const source of latest) {
    for (const [item, similarity] of staticBundle.neighbors[source] || []) {
      if (!seen.has(item)) personal[item] = (personal[item] || 0) + similarity;
    }
  }
  return Object.keys(base).filter(item => !seen.has(item)).sort((a, b) => {
    const left = (1 - alpha) * base[a] + alpha * (personal[a] || 0);
    const right = (1 - alpha) * base[b] + alpha * (personal[b] || 0);
    return right - left || base[b] - base[a] || a.localeCompare(b);
  }).slice(0, k).map(id => ({ id, title: staticBundle.titles[id] || id }));
}

function staticRecommend(history, simulateFailure) {
  const baseline = staticRank(history, 0, 5);
  const shadow = simulateFailure ? [] : staticRank(history, staticBundle.alpha, 5);
  return {
    baseline, shadow, active: baseline, method: 'recent',
    fallback: simulateFailure ? 'challenger_unavailable' : 'validation_gate_closed',
    data_scope: 'invented items', local_rank_ms: null
  };
}

async function refreshMetrics() {
  if (staticBundle) {
    $('metrics').textContent = 'Static preview. Run the local service for live request counters and timing.';
    return;
  }
  const response = await fetch('/api/metrics');
  const data = await response.json();
  $('metrics').textContent = `Local requests: ${data.counts.requests || 0} · fallbacks: ${data.counts.fallbacks || 0} · in-process p95: ${data.local_rank_ms.p95 ?? '—'} ms`;
}

async function compare(simulateFailure = false) {
  const history = $('history').value.split(',').map(value => value.trim()).filter(Boolean);
  $('status').textContent = 'Ranking…';
  try {
    let data;
    if (staticBundle) {
      data = staticRecommend(history, simulateFailure);
    } else {
      const response = await fetch('/api/recommend', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history, k: 5, simulate_failure: simulateFailure })
      });
      data = await response.json();
      if (!response.ok) throw new Error(data.error);
    }
    renderList('baseline', data.baseline);
    renderList('shadow', data.shadow);
    renderList('active', data.active);
    $('status').replaceChildren();
    const strong = document.createElement('strong');
    strong.textContent = data.method === 'hybrid' ? 'Challenger active' : 'Baseline active';
    const timing = data.local_rank_ms == null ? 'static preview' : `${data.local_rank_ms} ms local ranking`;
    $('status').append(strong, document.createTextNode(` · ${data.fallback?.replaceAll('_', ' ') || 'validation gate passed'} · ${data.data_scope} · ${timing}`));
    $('route-explanation').textContent = data.fallback
      ? `Fallback reason: ${data.fallback.replaceAll('_', ' ')}.`
      : 'Validation-approved challenger for this local model.';
    await refreshMetrics();
  } catch (error) {
    $('status').textContent = `Request failed: ${error.message}`;
  }
}

function addProfiles() {
  for (const name of ['cosmic', 'story', 'mixed', 'new'].filter(key => key in profiles)) {
    const items = profiles[name];
    const button = document.createElement('button');
    button.className = 'choice' + (name === selected ? ' active' : '');
    button.textContent = name[0].toUpperCase() + name.slice(1);
    button.onclick = () => {
      selected = name;
      document.querySelectorAll('.choice').forEach(other => other.classList.remove('active'));
      button.classList.add('active');
      $('history').value = items.join(', ');
      compare();
    };
    $('profiles').append(button);
  }
  $('history').value = profiles[selected].join(', ');
}

async function start() {
  try {
    const health = await fetch('/api/health');
    if (!health.ok) throw new Error('local API unavailable');
    const response = await fetch('/api/profiles');
    profiles = response.ok ? (await response.json()).profiles : {};
  } catch {
    try {
      const response = await fetch('demo_bundle.json');
      if (!response.ok) throw new Error('static demo bundle unavailable');
      staticBundle = await response.json();
      profiles = staticBundle.profiles;
    } catch (error) {
      $('status').textContent = `Demo unavailable: ${error.message}`;
      return;
    }
  }
  if (Object.keys(profiles).length) addProfiles();
  else $('status').textContent = 'Real local bundle loaded. Enter item IDs to compare routes.';
  await compare();
}

let held = 0;

function renderEvidence(data) {
  const parts = [];
  if (data.heard?.length) parts.push(`used: ${data.heard.join(', ')}`);
  if (data.ruled_out?.length) parts.push(`ruled out: ${data.ruled_out.join(', ')}`);
  if (data.budget != null) parts.push(`budget up to $${data.budget}`);
  $('phrase-evidence').textContent = data.said
    ? `Because you said “${data.said}”${parts.length ? ` · ${parts.join(' · ')}` : ''}`
    : 'Nothing asked yet.';
}

function heldNote(count) {
  $('standing-note').textContent = count
    ? `${count} sentence${count > 1 ? 's' : ''} held from this session. “Offer without asking” answers from the last one, with nothing typed in.`
    : 'No sentence held yet, so there is nothing to offer unprompted.';
}

function sampleAnswer(phrase, history) {
  // Product text stays on the machine it was fetched on. A page served without
  // the local service can therefore show the words it understood, but not the
  // catalogue they point at — so the behavioural route answers here.
  return { said: phrase, heard: (phrase.toLowerCase().match(/[a-z0-9]+/g) || []).slice(0, 12),
           ruled_out: [], budget: null, spoken: [], active: staticRecommend(history, false).active,
           combined: staticRecommend(history, false).active, answering: 'behavioural',
           phrase_fallback: 'service_unavailable', held: 0 };
}

async function speak(standing = false) {
  const typed = $('phrase').value.trim();
  if (!typed && !standing) { $('phrase-status').textContent = 'Type a sentence first.'; return; }
  $('phrase-status').textContent = standing ? 'Offering from a sentence you already left…' : 'Matching your words…';
  const history = $('history').value.split(',').map(value => value.trim()).filter(Boolean);
  try {
    let data;
    if (staticBundle) {
      data = sampleAnswer(standing ? '' : typed, history);
    } else {
      if (standing) {
        const response = await fetch('/api/ambient');
        data = await response.json();
        if (!response.ok) throw new Error(data.error);
        if (!data.offered) throw new Error(data.fallback.replaceAll('_', ' '));
      } else {
        const response = await fetch('/api/utterance', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ phrase: typed, history, k: 5, remember: true })
        });
        data = await response.json();
        if (!response.ok) throw new Error(data.error);
      }
    }
    renderList('phrase-list', data.spoken);
    renderList('phrase-answer', data.combined);
    renderList('phrase-behavioural', data.active || []);
    renderEvidence(data);
    held = Math.max(held, data.held || 0);
    heldNote(held);
    $('phrase-route').textContent = data.answering === 'spoken'
      ? 'Your wording answered this. The behavioural route is beside it, not hidden inside it.'
      : 'Your wording matched no product, so the behavioural route answered instead.';
    const note = data.phrase_fallback ? ` · ${data.phrase_fallback.replaceAll('_', ' ')}` : '';
    const ms = data.phrase_ms == null ? 'static preview' : `${data.phrase_ms} ms in-process`;
    $('phrase-status').textContent = `${data.offered ? 'Offered without being asked' : data.answering === 'spoken' ? 'Answered from your words' : 'Answered from history'} · ${ms}${note}`;
    await refreshMetrics();
  } catch (error) {
    $('phrase-status').textContent = `Could not answer: ${error.message}`;
  }
}

$('run').onclick = () => compare();
$('failure').onclick = () => compare(true);
$('ask').onclick = () => speak();
$('offer').onclick = () => speak(true);
$('phrase').addEventListener('keydown', event => { if (event.key === 'Enter') speak(); });
start();
