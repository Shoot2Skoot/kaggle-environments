import type { RendererOptions } from '@kaggle-environments/core';
import type { SHFrame, Target } from './transformer';

function esc(s: any): string {
  return String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c] as string);
}

// Fascist-board power labels by player count (mirrors the engine's power ladder).
function powerLabels(n: number): Record<number, string> {
  if (n <= 6) return { 3: 'PEEK', 4: 'KILL', 5: 'KILL' };
  if (n <= 8) return { 2: 'INV', 3: 'ELECT', 4: 'KILL', 5: 'KILL' };
  return { 1: 'INV', 2: 'INV', 3: 'ELECT', 4: 'KILL', 5: 'KILL' };
}

const REL_COLOR: Record<Target['rel'], string> = {
  nominate: '#f0b323',
  ja: '#3fb46b',
  nein: '#e0524a',
  investigate: '#39c5e0',
  execute: '#e0524a',
  special: '#b07ff0',
  mention: '#6b7280',
  gov: '#f0b323',
};

function roleShort(role: string): string {
  if (role === 'Hitler') return 'HITLER';
  if (role === 'Fascist') return 'FASCIST';
  return 'LIBERAL';
}

// Ring coordinates: player 0 at top, clockwise.
function ringPos(i: number, n: number, cx: number, cy: number, r: number): { x: number; y: number } {
  const angle = (i / n) * Math.PI * 2;
  return { x: cx + r * Math.sin(angle), y: cy - r * Math.cos(angle) };
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options as any;
  try {
    renderFrame(options);
  } catch (err: any) {
    // eslint-disable-next-line no-console
    console.error('[secret_hitler renderer] threw:', err);
    const f0 = (replay?.steps as any)?.[step];
    (parent as HTMLElement).innerHTML =
      `<div class="sh-root"><div class="sh-panel"><b>Renderer error:</b> ${esc(err?.message)}<br/><br/>frame keys: ${esc(f0 ? Object.keys(f0).join(', ') : '(no frame)')}<br/>steps len: ${esc((replay?.steps as any)?.length)}</div></div>`;
  }
}

function renderFrame(options: RendererOptions) {
  const { step, replay, parent } = options as any;
  const frames = (replay.steps as SHFrame[]) || [];
  const f = frames[Math.max(0, Math.min(step, frames.length - 1))];
  if (!f) {
    parent.innerHTML = '<div class="sh-root">No replay data.</div>';
    return;
  }

  const { playerIds, roles, alive, numPlayers } = f;
  const models = f.models || {};
  const shortModel = (pid: string): string => {
    const m = models[pid];
    if (!m) return '';
    // strip provider prefix and common date/version suffixes for a compact chip
    return m
      .split('/')
      .pop()!
      .replace(/-(latest|preview|\d{8})$/i, '');
  };
  const ended = f.ended;

  // ---- Board (policy tracks + tracker) ----
  const labels = powerLabels(numPlayers);
  const libSlots = Array.from({ length: 5 }, (_, i) => {
    const filled = i < f.board.liberal;
    const fresh = f.kind === 'policy' && filled && i === f.board.liberal - 1;
    return `<div class="sh-slot lib ${filled ? 'filled' : ''} ${fresh ? 'flash' : ''}"></div>`;
  }).join('');
  const fasSlots = Array.from({ length: 6 }, (_, i) => {
    const filled = i < f.board.fascist;
    const fresh = f.kind === 'policy' && filled && i === f.board.fascist - 1;
    const lbl = labels[i + 1] ? `<span class="sh-power">${labels[i + 1]}</span>` : '';
    return `<div class="sh-slot fas ${filled ? 'filled' : ''} ${fresh ? 'flash' : ''}">${lbl}</div>`;
  }).join('');
  const pips = Array.from(
    { length: 3 },
    (_, i) => `<div class="sh-pip ${i < f.board.tracker ? 'on' : ''}"></div>`
  ).join('');

  // ---- Player ring (SVG lines underneath, HTML cards on top) ----
  const W = 560;
  const H = 460;
  const cx = W / 2;
  const cy = H / 2;
  const r = Math.min(W, H) / 2 - 64;
  const pos: Record<string, { x: number; y: number }> = {};
  playerIds.forEach((pid, i) => (pos[pid] = ringPos(i, numPlayers, cx, cy, r)));

  // SVG connectors: actor -> target, or (for vote_result) each voter -> government.
  const lines: string[] = [];
  const sourceFor = (t: Target): string | null => {
    if (f.kind === 'vote_result') return t.id; // voter is the source; aim at government
    return f.actor;
  };
  const govPoint = (() => {
    // For vote_result, draw toward the president (representative government node).
    const pid = f.president;
    return pid && pos[pid] ? pos[pid] : { x: cx, y: cy };
  })();
  f.targets.forEach((t) => {
    const from = sourceFor(t);
    let a = from && pos[from] ? pos[from] : null;
    let b = pos[t.id] || null;
    if (f.kind === 'vote_result') {
      a = pos[t.id];
      b = govPoint;
    }
    if (!a || !b) return;
    const color = REL_COLOR[t.rel] || '#888';
    const dash = t.rel === 'mention' ? 'stroke-dasharray="4 4"' : '';
    lines.push(
      `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${color}" stroke-width="2.5" ${dash} marker-end="url(#arrow-${t.rel})" opacity="0.9"/>`
    );
  });
  const markers = Object.entries(REL_COLOR)
    .map(
      ([rel, color]) =>
        `<marker id="arrow-${rel}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${color}"/></marker>`
    )
    .join('');

  const cards = playerIds
    .map((pid) => {
      const p = pos[pid];
      const role = roles[pid]?.role || 'Liberal';
      const isDead = !alive.includes(pid);
      const isPres = pid === f.president;
      const isChan = pid === f.chancellor;
      const isNominee = pid === f.nominee && !isChan;
      const isActor = pid === f.actor;
      const markerTxt = isPres ? 'PRES' : isChan ? 'CHAN' : isNominee ? 'NOM' : '';
      const cls = [
        'sh-pcard',
        `team-${roles[pid]?.team?.toLowerCase() || 'liberal'}`,
        `role-${role.toLowerCase()}`,
        isDead ? 'dead' : '',
        isActor ? 'actor' : '',
        isPres ? 'pres' : '',
        isChan ? 'chan' : '',
      ].join(' ');
      return `
        <div class="${cls}" style="left:${p.x}px; top:${p.y}px;">
          ${markerTxt ? `<div class="sh-pmark">${markerTxt}</div>` : ''}
          <div class="sh-pid">${esc(pid)}</div>
          <div class="sh-prole">${roleShort(role)}</div>
          ${shortModel(pid) ? `<div class="sh-pmodel" title="${esc(models[pid])}">${esc(shortModel(pid))}</div>` : ''}
        </div>`;
    })
    .join('');

  // ---- Reasoning / action panel ----
  const actorRole = f.actor ? roleShort(roles[f.actor]?.role || '') : '';
  const panel = `
    <div class="sh-panel">
      <div class="sh-phasechip">${esc(f.phaseLabel)}</div>
      <div class="sh-headline">${esc(f.headline)}</div>
      ${f.message ? `<div class="sh-speech">“${esc(f.message)}”</div>` : ''}
      ${f.detail ? `<div class="sh-detail">${esc(f.detail)}</div>` : ''}
      ${
        f.reasoning
          ? `<div class="sh-reasonbox">
               <div class="sh-reasonhdr">${esc(f.actor)} ${actorRole ? `· <span class="sh-rtag role-${(roles[f.actor!]?.role || '').toLowerCase()}">${actorRole}</span>` : ''} — private reasoning</div>
               <div class="sh-reasontext">${esc(f.reasoning)}</div>
             </div>`
          : ''
      }
      ${
        ended
          ? `<div class="sh-endbox"><b class="team-${String(ended.winner_team).toLowerCase()}">${esc(ended.winner_team)} win.</b> ${esc(ended.reason)}</div>`
          : ''
      }
    </div>`;

  // ---- Running log up to this frame ----
  const logHtml = frames
    .slice(0, step + 1)
    .slice(-40)
    .map((g, idx, arr) => {
      const globalIdx = step + 1 - arr.length + idx;
      const cur = globalIdx === step ? 'cur' : '';
      return `<div class="sh-logline ${g.kind} ${cur}"><span class="sh-logr">R${g.round}</span> ${esc(g.headline)}</div>`;
    })
    .join('');

  parent.innerHTML = `
    <div class="sh-root renderer-container">
      <div class="sh-top">
        <span class="sh-title">SECRET HITLER</span>
        <span class="sh-sub">Round ${esc(f.round)} · Step ${step + 1}/${frames.length}</span>
      </div>

      <div class="sh-board">
        <div class="sh-trackrow"><span class="sh-tl">Liberal ${f.board.liberal}/5</span><div class="sh-track">${libSlots}</div></div>
        <div class="sh-trackrow"><span class="sh-tl">Fascist ${f.board.fascist}/6</span><div class="sh-track">${fasSlots}</div></div>
        <div class="sh-trackrow">
          <span class="sh-tl">Election tracker</span><div class="sh-tracker">${pips}</div>
          ${f.board.veto ? '<span class="sh-veto">VETO UNLOCKED</span>' : ''}
          <span class="sh-deck">draw ${f.board.draw} · discard ${f.board.discard}</span>
        </div>
      </div>

      <div class="sh-main">
        <div class="sh-ringwrap" style="width:${W}px; height:${H}px;">
          <svg class="sh-svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs>${markers}</defs>${lines.join('')}</svg>
          ${cards}
        </div>
        ${panel}
      </div>

      <div class="sh-log">${logHtml}</div>
    </div>
  `;

  // keep the current log line in view
  const curEl = parent.querySelector('.sh-logline.cur') as HTMLElement | null;
  curEl?.scrollIntoView({ block: 'nearest' });
}
