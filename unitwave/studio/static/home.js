// Homepage: choose a session. The page sends filters and draws what the server returns;
// filtering, counts, sorting, the region tree and probe positions are all server-side.
import { createOverview } from '/static/home3d.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const state = { region: '', sort: 'date', desc: false, options: null, seq: 0, selected: new Set(), probes: null };
const COLUMNS = [
  ['lab', 'Lab'], ['subject', 'Subject'], ['date', 'Date'], ['n_probes', 'Probes', 'num'],
  ['n_good_units', 'Good units*', 'num'], ['n_trials', 'Trials', 'num'],
  ['n_included_trials', 'Included', 'num'], ['n_regions', 'Beryl regions', 'num'],
];

// ---------- theme (same behaviour as the session view) ----------
function setTheme(v) {
  if (v === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = v;
  for (const b of $('theme').querySelectorAll('button')) b.setAttribute('aria-pressed', b.dataset.v === v);
  try { localStorage.setItem('studio-theme', v); } catch { /* storage may be unavailable */ }
  if (state.probes) drawOverview(state.probes);
  if ($('summaryRun') && $('summaryRun').value) drawSummary();
}
function theme() {
  const t = document.documentElement.dataset.theme;
  return t || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
$('theme').addEventListener('click', (e) => { if (e.target.dataset.v) setTheme(e.target.dataset.v); });
try { const t = localStorage.getItem('studio-theme'); if (t) setTheme(t); } catch { /* ignore */ }

// ---------- filters -> request ----------
function sessionFilter() {
  const f = {};
  if ($('lab').value) f.labs = [$('lab').value];
  if ($('subject').value) f.subjects = [$('subject').value];
  if ($('dateFrom').value) f.date_from = $('dateFrom').value;
  if ($('dateTo').value) f.date_to = $('dateTo').value;
  if (+$('minUnits').value) f.min_good_units = +$('minUnits').value;
  if (+$('minTrials').value) f.min_included_trials = +$('minTrials').value;
  if ($('nProbes').value) f.n_probes = [+$('nProbes').value];
  const mods = [...$('modalities').querySelectorAll('input:checked')].map((i) => i.value);
  if (mods.length) f.modalities = mods;
  if (state.region) {
    f.region = state.region;
    f.min_region_units = Math.max(1, +$('minRegionUnits').value || 1);
  }
  return f;
}
function checked(id) {
  const boxes = [...$(id).querySelectorAll('input')];
  const on = boxes.filter((i) => i.checked).map((i) => +i.value);
  return on.length === boxes.length ? [] : on;  // all ticked means no filter
}
function trialFilter() {
  return {
    bwm_include: $('tfInclude').checked, exclude_nogo: $('tfNogo').checked,
    contrasts: checked('tfContrasts'), blocks: checked('tfBlocks'), outcomes: checked('tfOutcomes'),
  };
}

// ---------- load and draw ----------
async function refresh() {
  const seq = ++state.seq;
  const p = new URLSearchParams({ f: JSON.stringify(sessionFilter()), sort: state.sort, desc: state.desc ? 1 : 0 });
  const r = await fetch('/api/home?' + p);
  if (seq !== state.seq) return;  // a newer request replaced this one
  if (!r.ok) { $('counts').textContent = await r.text(); return; }
  const d = await r.json();
  if (!state.options) setup(d);
  const s = d.summary;
  $('counts').textContent = `${s.n_sessions} sessions, ${s.n_probes} probes, ${s.n_units.toLocaleString()} units match` +
    (s.n_region_units == null ? '' : ` · ${s.n_region_units.toLocaleString()} of those units in ${state.region}`);
  $('unitsNote').textContent = `* ${d.notes.units} Manifest v${d.notes.manifest.manifest_version}.`;
  $('backToSession').hidden = !d.open;
  drawTree(d.tree);
  drawTable(d.sessions);
  state.probes = d.probes;
  drawOverview(d.probes);
}

// ---------- 3D overview ----------
const overview = createOverview($('overview'), {
  onHover(row, e) {
    const tip = $('overviewTip');
    tip.hidden = !row;
    if (!row) return;
    tip.textContent = `${row.lab} · ${row.subject} · ${row.date} · ${row.probe_name} (click to open)`;
    const rect = $('overview').getBoundingClientRect();
    tip.style.left = `${e.clientX - rect.left}px`;
    tip.style.top = `${e.clientY - rect.top}px`;
  },
  onPick(row) { openData({ kind: 'ibl', eid: row.eid, trials: trialFilter() }, `session ${row.eid.slice(0, 8)}`); },
});
let brainLoaded = false;
function drawOverview(p) {
  const colours = p.colours[theme()];
  if (!brainLoaded) {
    brainLoaded = true;
    // A failed download is tried again on the next redraw; the probes draw without it.
    overview.loadBrain(p.brain_id, css('--muted')).catch(() => { brainLoaded = false; });
  } else overview.setMuted(css('--muted'));
  overview.update(p, colours);
  const present = new Set(p.lines.map((l) => l.lab));
  const shown = p.labs.filter((l) => present.has(l.lab) && !l.other);
  const others = p.labs.filter((l) => present.has(l.lab) && l.other);
  $('labLegend').innerHTML = shown.map((l) => `<span class="item"><span class="sw" style="background:${colours[l.lab]}"></span>${esc(l.lab)}</span>`).join('') +
    (others.length ? `<span class="item" title="${esc(others.map((l) => l.lab).join(', '))}"><span class="sw" style="background:${css('--muted')}"></span>${others.length} other labs</span>` : '');
  $('overviewMeta').textContent = `${p.lines.length} probes · tip to top, from the manifest · drag to rotate`;
}

function setup(d) {
  const o = (state.options = d.options);
  for (const v of o.labs) $('lab').add(new Option(v, v));
  for (const v of o.subjects) $('subject').add(new Option(v, v));
  for (const v of o.n_probes) $('nProbes').add(new Option(String(v), v));
  $('dateFrom').min = $('dateTo').min = o.dates[0];
  $('dateFrom').max = $('dateTo').max = o.dates[1];
  $('minRegionUnits').value = o.min_region_units;
  $('modalities').innerHTML = o.modalities.map((m) => `<label><input type="checkbox" value="${esc(m)}"> ${esc(m)}</label>`).join('');
  $('phySyncNote').textContent += ` A fit with any pulse more than ${o.sync_tolerance_ms} ms off the line is refused, and so are pulse files of different lengths.`;
  for (const t of o.tasks) $('phyTask').add(new Option(`${t.label} (${t.name})`, t.name, t.name === o.default_task, t.name === o.default_task));
  // NWB: a layout brings its own task, which can still be changed.
  $('nwbLayout').add(new Option('Generic NWB (no layout)', ''));
  for (const l of o.layouts) $('nwbLayout').add(new Option(l.label, l.name, false, false));
  for (const t of o.tasks) $('nwbTask').add(new Option(`${t.label} (${t.name})`, t.name));
  const layoutTask = () => {
    const l = o.layouts.find((x) => x.name === $('nwbLayout').value);
    $('nwbTask').value = (l && l.task) || o.default_task;
  };
  $('nwbLayout').addEventListener('change', layoutTask);
  if (o.layouts.length) $('nwbLayout').value = o.layouts[0].name;
  layoutTask();
  const t = o.default_trial_filter, lv = o.trial_levels;
  $('tfInclude').checked = t.bwm_include;
  $('tfNogo').checked = t.exclude_nogo;
  const boxes = (id, values, chosen, name) => {
    $(id).innerHTML = values.map((v) => `<label><input type="checkbox" value="${v}" ${!chosen.length || chosen.includes(v) ? 'checked' : ''}> ${esc(name(v))}</label>`).join('');
  };
  boxes('tfContrasts', lv.contrasts, t.contrasts, (v) => `${v * 100}%`);
  boxes('tfBlocks', lv.blocks, t.blocks, (v) => `p(left) ${v}`);
  boxes('tfOutcomes', lv.outcomes, t.outcomes, (v) => (v < 0 ? 'error' : 'reward'));
  $('heads').innerHTML = '<th class="pick" title="Select for a session set"></th>' +
    COLUMNS.map(([k, label, cls]) => `<th class="sortable ${cls || ''}" data-k="${k}">${label}</th>`).join('') + '<th></th><th></th>';
}

function drawTree(tree) {
  const top = `<div class="node" data-node="" aria-selected="${!state.region}"><span>Any region</span></div>`;
  $('tree').innerHTML = top + tree.map((n) =>
    `<div class="node" data-node="${esc(n.acronym)}" aria-selected="${state.region === n.acronym}" style="padding-left:${6 + n.depth * 10}px" title="${esc(n.name)}">` +
    `<span class="sw" style="background:${n.colour}"></span><span>${esc(n.acronym)}</span><span class="n">${n.n_total.toLocaleString()}</span></div>`).join('');
  $('regionNote').textContent = state.region
    ? `Sessions with at least ${$('minRegionUnits').value} good units in ${state.region} or below it.`
    : 'No region chosen. Pick one to require units there, descendants included.';
}

function drawTable(rows) {
  for (const th of $('heads').querySelectorAll('th[data-k]')) {
    if (th.dataset.k === state.sort) th.setAttribute('aria-sort', state.desc ? 'descending' : 'ascending');
    else th.removeAttribute('aria-sort');
  }
  $('rows').innerHTML = rows.map((r) => `<tr data-eid="${esc(r.eid)}">` +
    `<td class="pick"><input type="checkbox" data-pick="${esc(r.eid)}" ${state.selected.has(r.eid) ? 'checked' : ''}></td>` +
    COLUMNS.map(([k, , cls]) => `<td class="${cls || ''}" ${k === 'n_regions' ? `title="${esc(r.regions.join(', '))}"` : ''}>${esc(r[k])}</td>`).join('') +
    `<td title="${r.cached ? 'in the local cache' : 'loads from the release'}">${r.cached ? '<span class="dot">●</span>' : ''}</td>` +
    `<td class="open-cell"><button class="btn" data-open="${esc(r.eid)}">Open</button></td></tr>`).join('');
}

// ---------- opening data ----------
async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function openData(body, label) {
  $('loadingText').textContent = `Loading ${label}…`;
  $('loading').hidden = false;
  try {
    window.location = (await post('/api/open', body)).url;
  } catch (e) {
    $('loadingText').textContent = e.message;
    setTimeout(() => { $('loading').hidden = true; }, 5000);
  }
}
const openSession = (eid) => openData({ kind: 'ibl', eid, trials: trialFilter() }, `session ${eid.slice(0, 8)}… from the cache if it's there, else from the release`);

// ---------- other ways in: Phy folders, recent projects, session sets ----------
async function completePhy() {
  const d = await (await fetch('/api/phy/complete?' + new URLSearchParams({ prefix: $('phyPath').value }))).json();
  $('phyRoot').textContent = `Folders under ${d.root}. Each opens with the events.csv beside its params.py.`;
  $('phyChoices').innerHTML = d.choices.map((c) => `<option value="${esc(c.path)}${c.phy ? '' : '/'}">${c.phy ? (c.events ? 'Phy folder' : 'Phy folder, no events.csv') : 'folder'}</option>`).join('');
}
async function completeNwb() {
  const d = await (await fetch('/api/nwb/complete?' + new URLSearchParams({ prefix: $('nwbPath').value }))).json();
  $('nwbRoot').textContent = `Files under ${d.root}.`;
  $('nwbChoices').innerHTML = d.choices.map((c) => `<option value="${esc(c.path)}${c.nwb ? '' : '/'}">${c.nwb ? `NWB file, ${c.size_mb} MB` : 'folder'}</option>`).join('');
}
async function loadProjects() {
  const d = await (await fetch('/api/projects')).json();
  $('projects').innerHTML = d.projects.length
    ? d.projects.map((p) => `<div class="row"><span title="${esc(p.file)} · ${esc(JSON.stringify(p.source))}">${esc(p.name)} · ${esc(p.source.kind === 'phy' ? 'Phy' : p.source.kind === 'nwb' ? `NWB · ${(p.source.file || '').split('/').pop()}` : (p.source.eid || '').slice(0, 8))}${p.file.endsWith('.unitwave.json') ? '' : ' <span class="note">(old file ending)</span>'}</span><button class="btn" data-project="${esc(p.file)}">Open</button></div>`).join('')
    : '<p class="note">No saved projects yet.</p>';
}
async function loadSets() {
  const d = await (await fetch('/api/sets')).json();
  state.setsFolder = d.folder;
  summaryHow();
  $('setList').length = 1;
  for (const s of d.sets) {
    const old = s.file.endsWith('.unitwave-set.json') ? '' : ', old file ending';
    $('setList').add(new Option(`${s.name} (${s.n_sessions} sessions${old})`, s.file));
  }
}
function applySet(set) {
  const f = set.session_filter, t = set.trial_filter;
  $('lab').value = (f.labs || [''])[0];
  $('subject').value = (f.subjects || [''])[0];
  $('dateFrom').value = f.date_from || '';
  $('dateTo').value = f.date_to || '';
  $('minUnits').value = f.min_good_units || 0;
  $('minTrials').value = f.min_included_trials || 0;
  $('nProbes').value = (f.n_probes || [''])[0];
  for (const i of $('modalities').querySelectorAll('input')) i.checked = (f.modalities || []).includes(i.value);
  state.region = f.region || '';
  if (f.min_region_units) $('minRegionUnits').value = f.min_region_units;
  $('tfInclude').checked = Boolean(t.bwm_include);
  $('tfNogo').checked = Boolean(t.exclude_nogo);
  for (const [id, key] of [['tfContrasts', 'contrasts'], ['tfBlocks', 'blocks'], ['tfOutcomes', 'outcomes']]) {
    for (const i of $(id).querySelectorAll('input')) i.checked = !(t[key] || []).length || t[key].includes(+i.value);
  }
  state.selected = new Set(set.eids);
}

// ---------- wiring ----------
for (const id of ['lab', 'subject', 'dateFrom', 'dateTo', 'minUnits', 'minTrials', 'nProbes', 'minRegionUnits']) {
  $(id).addEventListener('change', refresh);
}
$('modalities').addEventListener('change', refresh);
$('tree').addEventListener('click', (e) => {
  const n = e.target.closest('.node');
  if (!n) return;
  state.region = n.dataset.node;
  refresh();
});
$('heads').addEventListener('click', (e) => {
  const th = e.target.closest('th[data-k]');
  if (!th) return;
  state.desc = state.sort === th.dataset.k ? !state.desc : false;
  state.sort = th.dataset.k;
  refresh();
});
$('rows').addEventListener('click', (e) => { const b = e.target.closest('[data-open]'); if (b) openSession(b.dataset.open); });
$('rows').addEventListener('change', (e) => {
  const box = e.target.closest('[data-pick]');
  if (!box) return;
  if (box.checked) state.selected.add(box.dataset.pick); else state.selected.delete(box.dataset.pick);
  $('setMsg').textContent = `${state.selected.size} sessions selected.`;
});
$('phyPath').addEventListener('input', completePhy);
$('nwbPath').addEventListener('input', completeNwb);
$('nwbOpen').addEventListener('click', () => openData(
  { kind: 'nwb', path: $('nwbPath').value, layout: $('nwbLayout').value, task: $('nwbTask').value },
  `NWB file ${$('nwbPath').value}`));
$('phyOpen').addEventListener('click', () => {
  const body = { kind: 'phy', path: $('phyPath').value, task: $('phyTaskFile').value.trim() || $('phyTask').value };
  // Sync pulses (step 13a): both or neither; the server says which is missing.
  const probe = $('phySyncProbe').value.trim(), events = $('phySyncEvents').value.trim();
  if (probe || events) Object.assign(body, { sync_probe: probe, sync_events: events });
  openData(body, `Phy folder ${$('phyPath').value}`);
});
$('projects').addEventListener('click', (e) => { const b = e.target.closest('[data-project]'); if (b) openData({ kind: 'project', name: b.dataset.project }, `project ${b.dataset.project}`); });
$('setSave').addEventListener('click', async () => {
  try {
    const d = await post('/api/sets/save', { name: $('setName').value, eids: [...state.selected], session_filter: sessionFilter(), trial_filter: trialFilter() });
    $('setMsg').textContent = `Saved ${d.saved}: ${state.selected.size} sessions, with the filters that chose them.`;
    loadSets();
  } catch (e) { $('setMsg').textContent = e.message; }
});
$('setOpen').addEventListener('click', async () => {
  if (!$('setList').value) return;
  try {
    const d = await post('/api/sets/open', { name: $('setList').value });
    applySet(d.set);
    $('setMsg').textContent = `Loaded ${d.set.name}: ${d.set.eids.length} sessions selected, filters restored.` + (d.warnings.length ? ` ⚠ ${d.warnings.join(' ')}` : '');
    refresh();
  } catch (e) { $('setMsg').textContent = e.message; }
});
// ---------- region summaries (S5) ----------
// Runs are written by `python -m unitwave.cli.summarise` (slow: every session of a set
// is tested); the server reads them and draws the figures. The page shows them.
async function loadSummaries() {
  const runs = await (await fetch('/api/summaries')).json();
  $('summaryRun').innerHTML = runs.length
    ? runs.map((r) => `<option value="${esc(r.run)}">${esc(r.set)} · ${esc(r.label)} · ${r.n_sessions} sessions` +
      ` · ${r.n_claims} of ${r.n_tested} regions with a claim · ${esc(r.created.slice(0, 16).replace('T', ' '))}</option>`).join('')
    : '<option value="">None yet</option>';
  $('summaryMeta').textContent = runs.length ? `${runs.length} in runs/` : '';
  showSummary();
}
function summaryHow() {
  const file = $('setList').value || 'NAME.unitwave-set.json';
  $('summaryHow').textContent = 'Made from a saved set on the command line, because every session is tested (minutes per session): ' +
    `python -m unitwave.cli.summarise "${state.setsFolder}/${file}" --label responsive --event stim_on ` +
    '(or --label selective --event stim_on --split choice, or --label locked). ' +
    'NWB files with brain regions: python -m unitwave.cli.summarise --nwb FILE … --layout LAYOUT --name NAME --label …';
}
function drawSummary() {
  const run = encodeURIComponent($('summaryRun').value);
  $('summaryMap').src = `/api/summary.png?run=${run}&kind=flatmap&theme=${theme()}`;
  $('summarySpread').src = `/api/summary.png?run=${run}&kind=spread&theme=${theme()}`;
}
async function showSummary() {
  const run = $('summaryRun').value;
  for (const id of ['summaryMap', 'summarySpread']) $(id).hidden = !run;
  for (const id of ['summaryCaption', 'summaryClaims', 'summaryErr']) $(id).textContent = '';
  if (!run) return;
  try {
    const r = await fetch(`/api/summary?run=${encodeURIComponent(run)}`);
    if (!r.ok) throw new Error((await r.text()).replace(/^Cannot show this: /, ''));
    const d = await r.json();
    $('summaryCaption').textContent = d.caption;
    $('summaryClaims').textContent = d.claims.length
      ? 'Regions with a claim: ' + d.claims.map((c) => `${c.region} ${c.direction} (q = ${Number(c.q).toPrecision(2)}, ` +
        `${c.n_sessions} sessions, ${c.n_units} units)`).join(' · ')
      : d.n_tested ? 'No region differs from its sessions after correction.'
        : `No region has units in at least ${d.min_sessions} sessions of this set, so none gets a verdict.`;
    drawSummary();
  } catch (e) {
    $('summaryErr').textContent = e.message;
  }
}
$('summaryRun').addEventListener('change', showSummary);
$('setList').addEventListener('change', summaryHow);

refresh();
completePhy();
completeNwb();
loadProjects();
loadSets();
loadSummaries();
