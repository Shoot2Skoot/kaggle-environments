import type { RendererOptions } from '@kaggle-environments/core';

// Events that are safe to show to spectators (allowlist = default-deny). Private events
// (role reveals, investigation results, policy peeks) are deliberately excluded so the
// visualizer never leaks hidden information mid-game. Roles are revealed only at game end.
const LOG_ALLOWLIST = new Set<string>([
  'GameStartDataEntry',
  'NominationDataEntry',
  'VoteResultDataEntry',
  'ChatDataEntry',
  'BidResultDataEntry',
  'DiscardActionDataEntry',
  'PolicyEnactedDataEntry',
  'ChaosDataEntry',
  'ElectionTrackerDataEntry',
  'PowerUsedDataEntry',
  'ExecutionResultDataEntry',
  'GameEndResultsDataEntry',
]);

interface ParsedEvent {
  data_type: string;
  event: any;
}

function rawObs(step: any): any | null {
  if (!Array.isArray(step)) return null;
  for (const p of step) {
    const ro = p?.observation?.raw_observation;
    if (ro) return ro;
  }
  return null;
}

function moderatorEvents(replay: any): any[][] {
  return replay?.info?.MODERATOR_OBSERVATION ?? [];
}

function parseStepEvents(stepEntries: any[]): ParsedEvent[] {
  const out: ParsedEvent[] = [];
  for (const entry of stepEntries || []) {
    try {
      out.push({ data_type: entry.data_type, event: JSON.parse(entry.json_str) });
    } catch {
      /* ignore */
    }
  }
  return out;
}

function powerLabels(numPlayers: number): Record<number, string> {
  if (numPlayers <= 6) return { 3: 'Peek', 4: 'Kill', 5: 'Kill' };
  if (numPlayers <= 8) return { 2: 'Investigate', 3: 'Sp. Election', 4: 'Kill', 5: 'Kill' };
  return { 1: 'Investigate', 2: 'Investigate', 3: 'Sp. Election', 4: 'Kill', 5: 'Kill' };
}

function esc(s: any): string {
  return String(s ?? '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c] as string);
}

function describe(pe: ParsedEvent): string | null {
  const e = pe.event;
  const d = e.data || {};
  switch (pe.data_type) {
    case 'GameStartDataEntry':
      return `Game starts with ${d.number_of_players} players.`;
    case 'NominationDataEntry':
      return `${d.actor_id} nominates ${d.target_id} as Chancellor.`;
    case 'VoteResultDataEntry':
      return `Election ${d.passed ? 'PASSED' : 'FAILED'} — Ja: [${(d.ja_voters || []).join(', ')}] / Nein: [${(d.nein_voters || []).join(', ')}].`;
    case 'ChatDataEntry':
      return `${d.actor_id}: ${e.description?.includes(':') ? e.description.split(/:(.+)/)[1].trim() : d.message}`;
    case 'DiscardActionDataEntry':
      return d.veto_proposed ? `${d.actor_id} proposes a veto.` : `${d.actor_id} passes policies along.`;
    case 'PolicyEnactedDataEntry':
      return `A ${d.policy_color} policy is enacted${d.by_chaos ? ' (chaos)' : ''}. [L ${d.liberal_policies} | F ${d.fascist_policies}]`;
    case 'ChaosDataEntry':
      return `Chaos! Top policy (${d.policy_color}) auto-enacted.`;
    case 'ElectionTrackerDataEntry':
      return `${d.reason} Tracker: ${d.election_tracker}/3.`;
    case 'PowerUsedDataEntry':
      return `${d.actor_id} uses ${d.power}${d.target_id ? ` on ${d.target_id}` : ''}.`;
    case 'ExecutionResultDataEntry':
      return `${d.executed_player_id} is executed.`;
    case 'GameEndResultsDataEntry':
      return `GAME OVER — ${d.winner_team} win. ${d.reason}`;
    default:
      return null;
  }
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options as any;
  const steps = (replay.steps as any[]) || [];
  const mo = moderatorEvents(replay);

  const ro = rawObs(steps[step]) || {};
  const prevRo = step > 0 ? rawObs(steps[step - 1]) || {} : {};
  const players: string[] = ro.all_player_ids || [];
  const alive: string[] = ro.alive_players || players;
  const numPlayers = players.length;

  // Cumulative spectator events up to and including this step.
  const log: ParsedEvent[] = [];
  let finalRoles: Record<string, string> | null = null;
  let endInfo: any = null;
  for (let s = 0; s <= step && s < mo.length; s++) {
    for (const pe of parseStepEvents(mo[s])) {
      if (pe.data_type === 'GameEndResultsDataEntry') {
        finalRoles = pe.event.data?.all_players_and_role || null;
        endInfo = pe.event.data;
      }
      if (LOG_ALLOWLIST.has(pe.data_type)) log.push(pe);
    }
  }

  const libDelta = (ro.liberal_policies || 0) - (prevRo.liberal_policies || 0);
  const fasDelta = (ro.fascist_policies || 0) - (prevRo.fascist_policies || 0);
  const labels = powerLabels(numPlayers);

  const libSlots = Array.from({ length: 5 }, (_, i) => {
    const filled = i < (ro.liberal_policies || 0);
    const isNew = libDelta > 0 && i === (ro.liberal_policies || 0) - 1;
    return `<div class="sh-slot lib ${filled ? 'filled' : ''} ${isNew ? 'flash' : ''}"></div>`;
  }).join('');

  const fasSlots = Array.from({ length: 6 }, (_, i) => {
    const filled = i < (ro.fascist_policies || 0);
    const isNew = fasDelta > 0 && i === (ro.fascist_policies || 0) - 1;
    const label = labels[i + 1] ? `<span class="sh-power">${labels[i + 1]}</span>` : '';
    return `<div class="sh-slot fas ${filled ? 'filled' : ''} ${isNew ? 'flash' : ''}">${label}</div>`;
  }).join('');

  const trackerPips = Array.from(
    { length: 3 },
    (_, i) => `<div class="sh-pip ${i < (ro.election_tracker || 0) ? 'on' : ''}"></div>`
  ).join('');

  const playerCards = players
    .map((pid) => {
      const isPres = pid === ro.president_id;
      const isChan = pid === ro.chancellor_id;
      const isDead = !alive.includes(pid);
      const role = finalRoles ? finalRoles[pid] : null;
      const marker = isPres ? 'P' : isChan ? 'C' : '';
      return `
        <div class="sh-player ${isDead ? 'dead' : ''} ${isPres ? 'pres' : ''} ${isChan ? 'chan' : ''}">
          <div class="sh-marker">${marker}</div>
          <div class="sh-pid">${esc(pid)}</div>
          ${role ? `<div class="sh-role ${role.toLowerCase()}">${esc(role)}</div>` : ''}
        </div>`;
    })
    .join('');

  const logHtml = log
    .slice(-14)
    .map((pe) => {
      const text = describe(pe);
      if (!text) return '';
      const cls =
        pe.data_type === 'GameEndResultsDataEntry' ? 'end' : pe.data_type === 'PolicyEnactedDataEntry' ? 'policy' : '';
      return `<div class="sh-logline ${cls}">${esc(text)}</div>`;
    })
    .join('');

  const status = endInfo
    ? `<span class="sh-winner ${String(endInfo.winner_team).toLowerCase()}">${esc(endInfo.winner_team)} win</span> — ${esc(endInfo.reason)}`
    : `President: <b>${esc(ro.president_id || '—')}</b> &nbsp; Chancellor: <b>${esc(ro.chancellor_id || ro.nominee_chancellor_id || '—')}</b>`;

  parent.innerHTML = `
    <div class="sh-root renderer-container">
      <div class="sh-header">
        <span class="sh-title">Secret Hitler</span>
        <span class="sh-sub">Round ${esc(ro.round_number ?? '?')} — ${esc(ro.detailed_phase ?? '')}</span>
      </div>
      <div class="sh-board">
        <div class="sh-trackrow">
          <span class="sh-tracklabel">Liberal ${ro.liberal_policies || 0}/5</span>
          <div class="sh-track">${libSlots}</div>
        </div>
        <div class="sh-trackrow">
          <span class="sh-tracklabel">Fascist ${ro.fascist_policies || 0}/6</span>
          <div class="sh-track">${fasSlots}</div>
        </div>
        <div class="sh-trackrow">
          <span class="sh-tracklabel">Election tracker</span>
          <div class="sh-tracker">${trackerPips}</div>
          <span class="sh-veto">${ro.veto_unlocked ? 'VETO UNLOCKED' : ''}</span>
        </div>
      </div>
      <div class="sh-players">${playerCards}</div>
      <div class="sh-status">${status}</div>
      <div class="sh-log">${logHtml}</div>
    </div>
  `;
}
