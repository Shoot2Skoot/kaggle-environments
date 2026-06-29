"""LLM harness for Secret Hitler.

Mirrors werewolf's litellm action-registry paradigm (the right fit for a free-form
chat + structured-action social game), with two deliberate upgrades from the
create-harness/review-harness guidance:

* JSON extraction uses ``core_harness.extract_last_json_object`` (last-block-wins) instead
  of werewolf's first-block ``find("```json")``. Because that helper returns ``None`` on
  failure (it does not raise), we raise ``LLMActionException`` ourselves so the tenacity
  repair loop still fires. We also pre-strip trailing commas to recover the most common
  malformed-JSON case that standard ``json`` (unlike pyjson5) rejects.
* The discard action renders the hand *with multiplicity* and validates the chosen color is
  actually in hand.

litellm is imported lazily/guarded so this module (and its parse/prompt tests) load even
when litellm is not installed.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from kaggle_environments.core_harness import extract_last_json_object

from ..game.actions import (
    BidAction,
    ChatAction,
    DiscardPolicyAction,
    ExecutionAction,
    InvestigateAction,
    NominateChancellorAction,
    NoOpAction,
    PolicyPeekAckAction,
    SpecialElectionAction,
    VetoConsentAction,
    VoteAction,
)
from ..game.consts import DetailedPhase, Power, Vote
from ..game.records import get_raw_observation

logger = logging.getLogger(__name__)

try:  # litellm is optional at import time.
    import litellm
    from litellm import completion

    litellm.drop_params = True
except Exception:  # noqa: BLE001
    litellm = None
    completion = None


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class LLMActionException(Exception):
    def __init__(self, message, raw_out=None, prompt=None):
        super().__init__(message)
        self.raw_out = raw_out
        self.prompt = prompt


SYSTEM_PROMPT = (
    "You are a strategic player in a game of Secret Hitler, a hidden-role social deduction "
    "game. Liberals win by enacting 5 Liberal policies or assassinating Hitler. Fascists win "
    "by enacting 6 Fascist policies or by electing Hitler as Chancellor once 3 Fascist "
    "policies are enacted. Play to win for your team. Refer to other players by their exact "
    "string ids."
)

# Per-role rule text, pinned into every prompt's role block so it never scrolls out of the
# truncated event log (mirrors werewolf's always-present rule_of_role).
ROLE_RULES = {
    "Liberal": ("You do not know anyone else's role. Win by enacting 5 Liberal policies or by having Hitler executed."),
    "Fascist": (
        "You secretly know your fellow Fascist(s) and Hitler (see allies below). Win by enacting 6 "
        "Fascist policies, or by getting Hitler elected Chancellor after 3 Fascist policies are "
        "enacted. Protect Hitler's identity and sow distrust among Liberals."
    ),
    "Hitler": (
        "You are on the Fascist team but play like a Liberal to stay hidden. In 5-6 player games you "
        "know the Fascist; in 7+ player games you do not. Your team wins if 6 Fascist policies are "
        "enacted, or if you are elected Chancellor after 3 Fascist policies are enacted. You appear "
        "as a Fascist if investigated, and if you are executed the Liberals win immediately."
    ),
}


PROMPT_TEMPLATE = """{system_prompt}

### Your secret role
You are {player_id}. Role: {role} (team: {team}).
Rule of your role: {role_rule}
{allies_line}

### Current game state
{current_state}

### Recent events (chronological)
{event_log}

### Your task
{instruction}{error_instruction}

Respond with a single JSON object in a ```json fenced block, conforming to:
{schema}
For example:
{exemplar}
Put your private reasoning in a "reasoning" field first; it is invisible to other players.
"""


def parse_json(out: str, required_key: str) -> dict:
    """Extract the last JSON object (last-mention-wins), tolerating trailing commas.

    Raises ``LLMActionException`` when nothing parseable is found so the repair loop runs.
    """
    cleaned = _TRAILING_COMMA_RE.sub(r"\1", out)
    data = extract_last_json_object(cleaned, required_keys=(required_key,))
    if data is None:
        raise LLMActionException(f"No JSON object with key '{required_key}' found.", raw_out=out)
    return data


class LLMSecretHitlerAgent:
    """Stateful per-player LLM agent. One instance per player (it accumulates an event log)."""

    def __init__(
        self,
        model_name: str,
        agent_config: Optional[dict] = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_retries: int = 3,
        litellm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._max_retries = max_retries
        self._decoding_kwargs = dict(litellm_kwargs or {})
        agent_config = agent_config or {}
        if agent_config.get("llms"):
            self._decoding_kwargs.update(agent_config["llms"][0].get("parameters", {}) or {})
        self._event_log: List[str] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self._handlers = {
            DetailedPhase.ELECTION_NOMINATION_AWAIT: self._h_nominate,
            DetailedPhase.ELECTION_BIDDING_AWAIT: self._h_bid,
            DetailedPhase.ELECTION_CHAT_AWAIT: self._h_chat,
            DetailedPhase.ELECTION_VOTE_AWAIT: self._h_vote,
            DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT: self._h_president_discard,
            DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT: self._h_chancellor_discard,
            DetailedPhase.LEGISLATIVE_VETO_CONSENT_AWAIT: self._h_veto_consent,
            DetailedPhase.EXECUTIVE_POWER_AWAIT: self._h_power,
        }

    # ------------------------------------------------------------------ #
    # LLM call
    # ------------------------------------------------------------------ #
    def query(self, prompt: str) -> str:
        if completion is None:
            raise RuntimeError("litellm is not installed; cannot query a live model.")
        response = completion(
            model=self._model_name, messages=[{"role": "user", "content": prompt}], **self._decoding_kwargs
        )
        msg = "".join(c.get("message", {}).get("content", "") or "" for c in response.get("choices", []))
        try:
            usage = response.get("usage", {})
            self.total_prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.total_completion_tokens += usage.get("completion_tokens", 0) or 0
        except Exception:  # noqa: BLE001
            pass
        if not msg:
            raise LLMActionException("Empty model response.", raw_out=str(response), prompt=prompt)
        return msg

    def query_parse(self, prompt: str, required_key: str) -> dict:
        last_exc = None
        error_suffix = ""
        for attempt in range(self._max_retries):
            full_prompt = prompt + error_suffix
            try:
                raw = self.query(full_prompt)
                return parse_json(raw, required_key)
            except LLMActionException as exc:
                last_exc = exc
                tail = (exc.raw_out or "")[-500:]
                error_suffix = (
                    f"\n\n### Your previous response was not parseable\n{tail}\n"
                    f'Reply with ONLY a ```json block containing the "{required_key}" field.'
                )
                logger.warning(f"{self._model_name} parse retry {attempt + 1}: {exc}")
        raise last_exc

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #
    @staticmethod
    def current_state(raw) -> str:
        board = {
            "liberal_policies": f"{raw.liberal_policies}/5",
            "fascist_policies": f"{raw.fascist_policies}/6",
            "election_tracker": f"{raw.election_tracker}/3",
            "veto_unlocked": raw.veto_unlocked,
            "president": raw.president_id,
            "chancellor": raw.chancellor_id,
            "nominee_chancellor": raw.nominee_chancellor_id,
            "alive_players": raw.alive_players,
            "draw_pile_count": raw.draw_pile_count,
            "discard_pile_count": raw.discard_pile_count,
            "your_investigations": raw.investigation_results,
        }
        return json.dumps(board, default=str)

    def _allies_line(self, raw) -> str:
        if raw.known_allies:
            return f"Known allies (secret): {raw.known_allies}."
        return "You have no known allies."

    def _build_prompt(self, raw, instruction, schema, exemplar, error_instruction="") -> str:
        return PROMPT_TEMPLATE.format(
            system_prompt=self._system_prompt,
            player_id=raw.player_id,
            role=raw.role,
            team=raw.team,
            role_rule=ROLE_RULES.get(str(raw.role), ""),
            allies_line=self._allies_line(raw),
            current_state=self.current_state(raw),
            event_log="\n".join(self._event_log[-60:]) or "None yet.",
            instruction=instruction,
            error_instruction=error_instruction,
            schema=json.dumps(schema),
            exemplar=exemplar,
        )

    # ------------------------------------------------------------------ #
    # Per-phase handlers -> serialized action dict
    # ------------------------------------------------------------------ #
    def _args(self, raw):
        return {"round_number": raw.round_number, "phase": raw.game_state_phase, "actor_id": raw.player_id}

    def _match_target(self, value: Any, valid: Sequence[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        for v in valid:
            if text == v or text.lower() == v.lower():
                return v
        return None

    def _h_nominate(self, raw):
        valid = raw.eligible_chancellor_ids or [p for p in raw.alive_players if p != raw.player_id]
        prompt = self._build_prompt(
            raw,
            f"As President, nominate a Chancellor from: {valid}. The table then votes on the two of "
            f"you as a government; if it passes, you and this Chancellor jointly decide the next "
            f"policy. Note: after 3 Fascist policies are enacted, electing Hitler as Chancellor "
            f"instantly wins the game for the Fascists.",
            NominateChancellorAction.schema_for_player(),
            json.dumps({"reasoning": "...", "target_id": valid[0] if valid else "player_x"}),
        )
        data = self.query_parse(prompt, "target_id")
        target = self._match_target(data.get("target_id"), valid) or valid[0]
        return NominateChancellorAction(**self._args(raw), target_id=target, reasoning=data.get("reasoning"))

    def _h_bid(self, raw):
        prompt = self._build_prompt(
            raw,
            "Bid 0-3 to win the floor and speak before the vote. Higher bids win and only the winner "
            "speaks this round, but bidding high draws the table's attention to you.",
            BidAction.schema_for_player(),
            json.dumps({"reasoning": "...", "amount": 2}),
        )
        data = self.query_parse(prompt, "amount")
        try:
            amount = max(0, min(int(data.get("amount", 0)), 3))
        except (TypeError, ValueError):
            amount = 0
        return BidAction(**self._args(raw), amount=amount, reasoning=data.get("reasoning"))

    def _h_chat(self, raw):
        prompt = self._build_prompt(
            raw,
            "You won the floor — address the table before the vote. Everything you say is public and "
            "will be remembered and used by others to infer your role; you may speak truthfully or deceive.",
            ChatAction.schema_for_player(),
            json.dumps({"reasoning": "...", "message": "I trust this government because..."}),
        )
        data = self.query_parse(prompt, "message")
        return ChatAction(**self._args(raw), message=str(data.get("message", "")), reasoning=data.get("reasoning"))

    def _h_vote(self, raw):
        prompt = self._build_prompt(
            raw,
            f"Vote Ja or Nein on this government (President {raw.president_id}, Chancellor "
            f"{raw.nominee_chancellor_id}). If it passes, the President draws 3 policies and the "
            f"Chancellor enacts 1. If it fails, the election tracker advances — after 3 failed "
            f"governments in a row the top policy is force-enacted with no powers (chaos). Note: once "
            f"3 Fascist policies are enacted, electing Hitler as Chancellor instantly loses the game "
            f"for the Liberals.",
            VoteAction.schema_for_player(),
            json.dumps({"reasoning": "...", "vote": "Ja"}),
        )
        data = self.query_parse(prompt, "vote")
        vote = Vote.JA if str(data.get("vote", "")).strip().lower() in ("ja", "yes", "true") else Vote.NEIN
        return VoteAction(**self._args(raw), vote=vote, reasoning=data.get("reasoning"))

    def _render_hand(self, hand) -> str:
        from collections import Counter

        counts = Counter(str(c) for c in (hand or []))
        return ", ".join(f"{color} x{n}" for color, n in counts.items()) or "empty"

    def _h_president_discard(self, raw):
        hand = list(raw.policy_hand or [])
        prompt = self._build_prompt(
            raw,
            f"You drew 3 policies ({self._render_hand(hand)}). Discard exactly ONE; the other two go "
            f"to the Chancellor, who will enact one of them (they cannot see which you discarded). The "
            f"policy enacted this round will be one of the two you pass — so if you pass two of the "
            f"same color, that color is enacted and you cannot prevent it. Others cannot see your draw, "
            f"so consider what you will claim. Discard one by color ('Liberal' or 'Fascist').",
            DiscardPolicyAction.schema_for_player(),
            json.dumps({"reasoning": "...", "discard": str(hand[0]) if hand else "Fascist"}),
        )
        data = self.query_parse(prompt, "discard")
        discard = self._match_discard(data.get("discard"), hand)
        return DiscardPolicyAction(**self._args(raw), discard=discard, reasoning=data.get("reasoning"))

    def _h_chancellor_discard(self, raw):
        hand = list(raw.policy_hand or [])
        veto_note = ' You may instead set "veto": true to propose vetoing this agenda.' if raw.veto_available else ""
        prompt = self._build_prompt(
            raw,
            f"You received 2 policies ({self._render_hand(hand)}). You will enact ONE and discard the "
            f"other. If both are the same color, that color is enacted — you have no choice. Others "
            f"cannot see the discarded policy or what the President drew.{veto_note} Discard one by color.",
            DiscardPolicyAction.schema_for_chancellor(),
            json.dumps({"reasoning": "...", "discard": str(hand[0]) if hand else "Fascist", "veto": False}),
        )
        data = self.query_parse(prompt, "discard")
        if raw.veto_available and bool(data.get("veto")):
            return DiscardPolicyAction(**self._args(raw), veto=True, reasoning=data.get("reasoning"))
        discard = self._match_discard(data.get("discard"), hand)
        return DiscardPolicyAction(**self._args(raw), discard=discard, veto=False, reasoning=data.get("reasoning"))

    def _match_discard(self, value, hand):
        text = str(value).strip().lower() if value is not None else ""
        for color in hand:
            if str(color).lower() == text:
                return color
        return hand[0] if hand else None

    def _h_veto_consent(self, raw):
        prompt = self._build_prompt(
            raw,
            f"Chancellor {raw.chancellor_id} proposes to veto this agenda (possible now that 5 Fascist "
            f"policies are enacted). If you consent, both policies are discarded and no policy is "
            f"enacted — but this counts as a failed government and advances the election tracker toward "
            f"chaos. If you refuse, the Chancellor must enact one of the two policies.",
            VetoConsentAction.schema_for_player(),
            json.dumps({"reasoning": "...", "consent": True}),
        )
        data = self.query_parse(prompt, "consent")
        consent = (
            bool(data.get("consent"))
            if not isinstance(data.get("consent"), str)
            else data["consent"].strip().lower() in ("true", "yes", "ja")
        )
        return VetoConsentAction(**self._args(raw), consent=consent, reasoning=data.get("reasoning"))

    def _h_power(self, raw):
        power = raw.pending_power
        valid = raw.power_valid_targets or [p for p in raw.alive_players if p != raw.player_id]
        if power == Power.POLICY_PEEK:
            prompt = self._build_prompt(
                raw,
                "You used Policy Peek and privately saw the top three policies that will be drawn next "
                "(in your events). No one else knows what you saw — you may use this knowledge later or "
                "misrepresent it. Acknowledge to continue.",
                PolicyPeekAckAction.schema_for_player(),
                json.dumps({"reasoning": "..."}),
            )
            data = self.query_parse(prompt, "reasoning")
            return PolicyPeekAckAction(**self._args(raw), reasoning=data.get("reasoning"))

        names = {
            Power.INVESTIGATE: (
                "Use your power to investigate a player's party membership. You will privately learn "
                "whether the target is a Liberal or a Fascist (Hitler appears as Fascist). Only you see "
                "the result — you may report it truthfully to the table or lie about it.",
                InvestigateAction,
            ),
            Power.SPECIAL_ELECTION: (
                "Use your power to choose the next Presidential candidate, out of the normal rotation. "
                "You are handing that player presidential power for the next round; the usual rotation "
                "resumes afterward.",
                SpecialElectionAction,
            ),
            Power.EXECUTION: (
                "Use your power to execute a player, removing them permanently. If you execute Hitler, "
                "the Liberals win the game instantly; if you execute a Liberal, you weaken your own "
                "side. No one learns the executed player's role.",
                ExecutionAction,
            ),
        }
        desc, action_cls = names[power]
        prompt = self._build_prompt(
            raw,
            f"{desc} Valid targets: {valid}.",
            action_cls.schema_for_player(),
            json.dumps({"reasoning": "...", "target_id": valid[0] if valid else "player_x"}),
        )
        data = self.query_parse(prompt, "target_id")
        target = self._match_target(data.get("target_id"), valid) or valid[0]
        return action_cls(**self._args(raw), target_id=target, reasoning=data.get("reasoning"))

    # ------------------------------------------------------------------ #
    # Self-memory
    # ------------------------------------------------------------------ #
    def _self_note(self, action) -> str:
        """A compact record of this agent's own action + reasoning, re-injected into the event
        log so later turns can stay consistent with earlier claims (mirrors werewolf)."""
        try:
            d = action.serialize()
        except Exception:  # noqa: BLE001
            d = {}
        name = type(action).__name__
        if name == "DiscardPolicyAction":
            label = "You proposed a veto." if d.get("veto") else f"You discarded a {d.get('discard', '?')} policy."
        else:
            label = {
                "NominateChancellorAction": f"You nominated {d.get('target_id', '?')} as Chancellor.",
                "VoteAction": f"You voted {str(d.get('vote', '?')).split('.')[-1]}.",
                "ChatAction": f'You told the table: "{str(d.get("message", ""))[:200]}"',
                "BidAction": f"You bid {d.get('amount', '?')} for the floor.",
                "VetoConsentAction": f"You {'consented to' if d.get('consent') else 'refused'} the veto.",
                "InvestigateAction": f"You investigated {d.get('target_id', '?')}.",
                "SpecialElectionAction": f"You called a special election for {d.get('target_id', '?')}.",
                "ExecutionAction": f"You executed {d.get('target_id', '?')}.",
                "PolicyPeekAckAction": "You reviewed the top three policies (Policy Peek).",
                "NoOpAction": "You took no action.",
            }.get(name, f"You acted ({name}).")
        reasoning = (getattr(action, "reasoning", None) or d.get("reasoning") or "").strip()
        note = f"[YOUR ACTION] {label}"
        if reasoning:
            note += f" (your reasoning: {reasoning[:300]})"
        return note

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def __call__(self, obs, config=None) -> dict:
        raw = get_raw_observation(obs)
        for view in raw.new_player_event_views:
            self._event_log.append(f"[R{view.day}] {view.description}")

        phase = DetailedPhase(raw.detailed_phase)
        handler = self._handlers.get(phase)
        if handler is None:
            return NoOpAction(**self._args(raw), reasoning="No action required.").serialize()

        try:
            action = handler(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{self._model_name} failed to act in {phase}: {exc}")
            action = NoOpAction(**self._args(raw), reasoning="Fell back to NoOp after repeated failures.")
        # Re-inject our own decision + reasoning so future turns can stay self-consistent.
        self._event_log.append(f"[R{raw.round_number}] {self._self_note(action)}")
        return action.serialize()


def _litellm_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "litellm_models.yaml")
