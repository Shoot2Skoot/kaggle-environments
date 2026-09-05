// Transforms a Secret Hitler replay into a fine-grained, god-mode spectator narrative.
//
// The kaggle engine batches simultaneous decisions (e.g. all five votes) into a single
// engine step, but every individual decision is recorded as its own event in
// info.MODERATOR_OBSERVATION with its actor_id + reasoning. This transformer flattens that
// moderator log into one DISPLAY FRAME per decision / state-change, so a spectator can scrub
// through the game one action at a time and always see who did what, to whom, and why.
//
// Roles are revealed at all times (this is a spectator view, not a player view) — we pre-scan
// the GameStartRoleDataEntry events and attach the full role map to every frame.

interface RawEvent {
  data_type: string;
  json_str: string;
}

export interface RoleInfo {
  role: string; // Liberal | Fascist | Hitler
  team: string; // Liberal | Fascist
  allies: Record<string, string>; // known_allies: playerId -> role
}

export interface Board {
  liberal: number;
  fascist: number;
  tracker: number;
  veto: boolean;
  draw: number;
  discard: number;
}

export interface Target {
  id: string;
  rel: 'nominate' | 'ja' | 'nein' | 'investigate' | 'execute' | 'special' | 'mention' | 'gov';
}

export interface SHFrame {
  step: number;
  players: {
    id: number;
    name: string;
    thumbnail: string;
    isTurn: boolean;
    actionDisplayText?: string;
    thoughts?: string;
  }[];
  // --- spectator fields consumed by the renderer ---
  kind: string;
  actor: string | null;
  targets: Target[];
  headline: string;
  message: string | null;
  reasoning: string | null;
  detail: string | null;
  board: Board;
  alive: string[];
  roles: Record<string, RoleInfo>;
  models: Record<string, string>;
  playerIds: string[];
  numPlayers: number;
  phaseLabel: string;
  round: number;
  president: string | null;
  chancellor: string | null;
  nominee: string | null;
  ended: any | null;
}

function parse(e: RawEvent): { data_type: string; description: string; data: any } | null {
  try {
    const ev = JSON.parse(e.json_str);
    return { data_type: e.data_type, description: ev.description ?? '', data: ev.data ?? {} };
  } catch {
    return null;
  }
}

function moEvents(replay: any): { data_type: string; description: string; data: any }[] {
  const mo: RawEvent[][] = replay?.info?.MODERATOR_OBSERVATION ?? [];
  const flat: { data_type: string; description: string; data: any }[] = [];
  for (const stepEvents of mo) {
    for (const raw of stepEvents || []) {
      const p = parse(raw);
      if (p) flat.push(p);
    }
  }
  return flat;
}

const POWER_LABEL: Record<string, string> = {
  INVESTIGATE_LOYALTY: 'Investigate Loyalty',
  CALL_SPECIAL_ELECTION: 'Special Election',
  POLICY_PEEK: 'Policy Peek',
  EXECUTION: 'Execution',
};

function powerRel(power: string): Target['rel'] {
  if (power?.includes('INVESTIGATE')) return 'investigate';
  if (power?.includes('EXECUTION')) return 'execute';
  if (power?.includes('SPECIAL')) return 'special';
  return 'mention';
}

export function secretHitlerTransformer(replay: any): any {
  const events = moEvents(replay);

  // ---- Pass 1: roles (god-mode), player order, end results ----
  const roles: Record<string, RoleInfo> = {};
  let playerIds: string[] = [];
  let ended: any = null;
  for (const ev of events) {
    if (ev.data_type === 'GameStartDataEntry' && Array.isArray(ev.data.player_ids)) {
      playerIds = ev.data.player_ids.slice();
    }
    if (ev.data_type === 'GameStartRoleDataEntry') {
      roles[ev.data.player_id] = {
        role: ev.data.role,
        team: ev.data.team,
        allies: ev.data.known_allies || {},
      };
    }
    if (ev.data_type === 'GameEndResultsDataEntry') ended = ev.data;
  }
  if (playerIds.length === 0) playerIds = Object.keys(roles);
  const numPlayers = playerIds.length;
  const models: Record<string, string> = replay?.info?.PlayerModels || {};

  // ---- Pass 2: walk events in order, emit one frame per decision / state change ----
  const board: Board = { liberal: 0, fascist: 0, tracker: 0, veto: false, draw: 0, discard: 0 };
  let alive = playerIds.slice();
  let president: string | null = null;
  let chancellor: string | null = null;
  let nominee: string | null = null;
  let round = 1;
  // Speeches after a policy is enacted (and before the next nomination) are the post-policy
  // debrief; speeches after a nomination are the pre-vote discussion.
  let inDebrief = false;
  const frames: SHFrame[] = [];

  const phaseFor = (kind: string): string =>
    ({
      setup: 'Setup',
      nomination: 'Nomination',
      bid: 'Discussion — Bidding',
      chat: 'Discussion',
      debrief: 'Debrief',
      vote: 'Election — Voting',
      vote_result: 'Election Result',
      discard: 'Legislative Session',
      policy: 'Legislative Session',
      power: 'Executive Power',
      execution: 'Executive Power',
      tracker: 'Election Tracker',
      chaos: 'Chaos',
      game_end: 'Game Over',
    })[kind] ?? '';

  const emit = (f: Partial<SHFrame> & { kind: string; headline: string }) => {
    const actor = f.actor ?? null;
    const reasoning = f.reasoning ?? null;
    const players = playerIds.map((pid, i) => ({
      id: i,
      name: pid,
      thumbnail: '',
      isTurn: pid === actor,
      actionDisplayText: pid === actor ? f.headline : undefined,
      thoughts: pid === actor && reasoning ? reasoning : undefined,
    }));
    frames.push({
      step: frames.length,
      players,
      kind: f.kind,
      actor,
      targets: f.targets ?? [],
      headline: f.headline,
      message: f.message ?? null,
      reasoning,
      detail: f.detail ?? null,
      board: { ...board },
      alive: alive.slice(),
      roles,
      models,
      playerIds,
      numPlayers,
      phaseLabel: phaseFor(f.kind),
      round,
      president,
      chancellor,
      nominee,
      ended: f.kind === 'game_end' ? ended : null,
    });
  };

  let setupEmitted = false;
  for (const ev of events) {
    const d = ev.data || {};
    switch (ev.data_type) {
      case 'GameStartDataEntry': {
        const rc = d.role_counts || {};
        emit({
          kind: 'setup',
          actor: null,
          headline: `Game begins — ${d.number_of_players} players`,
          detail:
            `Roles dealt: ${Object.entries(rc)
              .map(([k, v]) => `${v} ${k}`)
              .join(', ')}. Fascists know each other; ` +
            `${numPlayers <= 6 ? 'Hitler knows the Fascists' : 'Hitler is in the dark'}.`,
        });
        setupEmitted = true;
        break;
      }
      case 'BoardSnapshotDataEntry': {
        board.liberal = d.liberal_policies ?? board.liberal;
        board.fascist = d.fascist_policies ?? board.fascist;
        board.tracker = d.election_tracker ?? board.tracker;
        board.veto = !!d.veto_unlocked;
        board.draw = d.draw_pile_count ?? board.draw;
        board.discard = d.discard_pile_count ?? board.discard;
        if (Array.isArray(d.alive_player_ids)) alive = d.alive_player_ids.slice();
        president = d.president_id ?? president;
        chancellor = d.chancellor_id ?? chancellor;
        break;
      }
      case 'NominationDataEntry': {
        inDebrief = false;
        president = d.actor_id;
        nominee = d.target_id;
        chancellor = null;
        emit({
          kind: 'nomination',
          actor: d.actor_id,
          targets: [{ id: d.target_id, rel: 'nominate' }],
          headline: `${d.actor_id} nominates ${d.target_id} as Chancellor`,
          reasoning: d.reasoning ?? null,
        });
        break;
      }
      case 'BidDataEntry': {
        emit({
          kind: 'bid',
          actor: d.actor_id,
          headline: `${d.actor_id} bids ${d.bid_amount} for the floor`,
          reasoning: d.reasoning ?? null,
        });
        break;
      }
      case 'ChatDataEntry': {
        const mentioned: Target[] = (d.mentioned_player_ids || [])
          .filter((m: string) => m !== d.actor_id)
          .map((m: string) => ({ id: m, rel: 'mention' as const }));
        emit({
          kind: inDebrief ? 'debrief' : 'chat',
          actor: d.actor_id,
          targets: mentioned,
          headline: inDebrief ? `${d.actor_id} debriefs` : `${d.actor_id} speaks`,
          message: d.message ?? null,
          reasoning: d.reasoning ?? null,
        });
        break;
      }
      case 'VoteDataEntry': {
        const isJa = String(d.vote).toLowerCase().includes('ja');
        const rel = isJa ? 'ja' : 'nein';
        const govTargets: Target[] = [];
        if (president) govTargets.push({ id: president, rel });
        if (nominee) govTargets.push({ id: nominee, rel });
        emit({
          kind: 'vote',
          actor: d.actor_id,
          targets: govTargets,
          headline: `${d.actor_id} votes ${isJa ? 'JA' : 'NEIN'}`,
          reasoning: d.reasoning ?? null,
        });
        break;
      }
      case 'VoteResultDataEntry': {
        const targets: Target[] = [];
        for (const v of d.ja_voters || []) targets.push({ id: v, rel: 'ja' });
        for (const v of d.nein_voters || []) targets.push({ id: v, rel: 'nein' });
        if (d.passed) chancellor = d.chancellor_id ?? nominee;
        emit({
          kind: 'vote_result',
          actor: null,
          targets,
          headline: `Election ${d.passed ? 'PASSED' : 'FAILED'} (${(d.ja_voters || []).length} Ja / ${(d.nein_voters || []).length} Nein)`,
          detail: d.passed
            ? `Government formed: President ${d.president_id}, Chancellor ${d.chancellor_id}.`
            : 'No government — the election tracker advances.',
        });
        break;
      }
      case 'DiscardActionDataEntry': {
        const color = d.discard ?? d.discarded_policy ?? null;
        const veto = d.veto_proposed || d.veto;
        emit({
          kind: 'discard',
          actor: d.actor_id,
          headline: veto
            ? `${d.actor_id} proposes a VETO`
            : `${d.actor_id} discards${color ? ` a ${color} policy` : ' a policy'}`,
          reasoning: d.reasoning ?? null,
        });
        break;
      }
      case 'PolicyEnactedDataEntry': {
        inDebrief = true;
        board.liberal = d.liberal_policies ?? board.liberal;
        board.fascist = d.fascist_policies ?? board.fascist;
        const govTargets: Target[] = [];
        if (d.president_id) govTargets.push({ id: d.president_id, rel: 'gov' });
        if (d.chancellor_id) govTargets.push({ id: d.chancellor_id, rel: 'gov' });
        emit({
          kind: 'policy',
          actor: null,
          targets: govTargets,
          headline: `A ${d.policy_color} policy is enacted${d.by_chaos ? ' (chaos)' : ''}`,
          detail: `Board: ${board.liberal}/5 Liberal · ${board.fascist}/6 Fascist.`,
        });
        break;
      }
      case 'PowerUsedDataEntry': {
        const targets: Target[] = d.target_id ? [{ id: d.target_id, rel: powerRel(d.power) }] : [];
        emit({
          kind: 'power',
          actor: d.actor_id,
          targets,
          headline: `${d.actor_id} uses ${POWER_LABEL[d.power] || d.power}${d.target_id ? ` on ${d.target_id}` : ''}`,
          reasoning: d.reasoning ?? null,
        });
        break;
      }
      case 'ExecutionResultDataEntry': {
        const victim = d.executed_player_id;
        alive = alive.filter((p) => p !== victim);
        emit({
          kind: 'execution',
          actor: null,
          targets: victim ? [{ id: victim, rel: 'execute' }] : [],
          headline: `${victim} is executed`,
        });
        break;
      }
      case 'ElectionTrackerDataEntry': {
        board.tracker = d.election_tracker ?? board.tracker;
        emit({
          kind: 'tracker',
          actor: null,
          headline: `Election tracker → ${d.election_tracker}/3`,
          detail: d.reason ?? null,
        });
        break;
      }
      case 'ChaosDataEntry': {
        emit({
          kind: 'chaos',
          actor: null,
          headline: `Chaos! Top policy (${d.policy_color}) auto-enacted`,
        });
        round += 1;
        break;
      }
      case 'GameEndResultsDataEntry': {
        emit({
          kind: 'game_end',
          actor: null,
          headline: `GAME OVER — ${d.winner_team} win`,
          detail: d.reason ?? null,
        });
        break;
      }
      default:
        break;
    }
    // advance round counter at each new election result so headers feel right
    if (ev.data_type === 'VoteResultDataEntry') round += 1;
  }

  if (!setupEmitted && frames.length === 0) {
    // Degenerate replay; emit a single placeholder so the player has something to show.
    emit({ kind: 'setup', actor: null, headline: 'No events found in replay.' });
  }

  return { ...replay, steps: frames, isTransformed: true };
}

export function getStepLabel(step: any): string {
  const f = step as SHFrame;
  const map: Record<string, string> = {
    setup: 'Setup',
    nomination: 'Nominate',
    bid: 'Bid',
    chat: 'Speak',
    debrief: 'Debrief',
    vote: 'Vote',
    vote_result: 'Result',
    discard: 'Legislate',
    policy: 'Policy',
    power: 'Power',
    execution: 'Execute',
    tracker: 'Tracker',
    chaos: 'Chaos',
    game_end: 'End',
  };
  return map[f?.kind] ?? f?.kind ?? '';
}

export function getStepDescription(step: any): string {
  const f = step as SHFrame;
  if (!f) return '';
  if (f.reasoning) return `${f.headline}\n\n${f.reasoning}`;
  return f.detail ? `${f.headline} — ${f.detail}` : f.headline;
}
