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

$('run').onclick = () => compare();
$('failure').onclick = () => compare(true);
start();
