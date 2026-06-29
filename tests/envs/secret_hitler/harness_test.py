"""Tests for the Secret Hitler LLM harness."""

import json

import pytest

from kaggle_environments import make
from kaggle_environments.envs.secret_hitler.game.consts import DetailedPhase, ObsKeys, Power, Vote
from kaggle_environments.envs.secret_hitler.game.records import SecretHitlerObservationModel
from kaggle_environments.envs.secret_hitler.game.roles import assign_role_counts
from kaggle_environments.envs.secret_hitler.harness import base as harness
from kaggle_environments.envs.secret_hitler.harness.base import (
    LLMActionException,
    LLMSecretHitlerAgent,
    parse_json,
)


def _obs(**overrides):
    base_fields = dict(
        player_id="p0",
        role="Liberal",
        team="Liberal",
        is_alive=True,
        round_number=1,
        detailed_phase=DetailedPhase.ELECTION_NOMINATION_AWAIT,
        game_state_phase="Election",
        all_player_ids=["p0", "p1", "p2", "p3", "p4"],
        alive_players=["p0", "p1", "p2", "p3", "p4"],
        liberal_policies=0,
        fascist_policies=0,
        election_tracker=0,
        veto_unlocked=False,
        draw_pile_count=17,
        discard_pile_count=0,
        president_id="p0",
    )
    base_fields.update(overrides)
    model = SecretHitlerObservationModel(**base_fields)
    return {ObsKeys.RAW_OBSERVATION: model.model_dump()}


def _agent(monkeypatch, response: str):
    a = LLMSecretHitlerAgent(model_name="test/model")
    monkeypatch.setattr(a, "query", lambda prompt: response)
    return a


def fenced(d):
    return f"```json\n{json.dumps(d)}\n```"


# ------------------------------------------------------------------ #
# parse_json
# ------------------------------------------------------------------ #
class TestParse:
    def test_fenced(self):
        assert parse_json(fenced({"vote": "Ja"}), "vote") == {"vote": "Ja"}

    def test_bare(self):
        assert parse_json('reasoning here {"vote": "Nein"}', "vote")["vote"] == "Nein"

    def test_last_block_wins(self):
        out = fenced({"target_id": "p1"}) + "\nactually no\n" + fenced({"target_id": "p2"})
        assert parse_json(out, "target_id")["target_id"] == "p2"

    def test_trailing_comma_tolerated(self):
        assert parse_json('```json\n{"amount": 3,}\n```', "amount")["amount"] == 3

    def test_missing_raises(self):
        with pytest.raises(LLMActionException):
            parse_json("no json at all", "vote")

    def test_wrong_key_raises(self):
        with pytest.raises(LLMActionException):
            parse_json(fenced({"something_else": 1}), "vote")


# ------------------------------------------------------------------ #
# Prompt building
# ------------------------------------------------------------------ #
class TestPrompt:
    def test_prompt_has_rules_role_and_state(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"target_id": "p1"}))
        captured = {}
        monkeypatch.setattr(a, "query", lambda prompt: captured.setdefault("p", prompt) or fenced({"target_id": "p1"}))
        a(_obs())
        p = captured["p"]
        assert "Secret Hitler" in p and "Liberal" in p and "liberal_policies" in p

    def test_discard_prompt_shows_hand_multiplicity(self, monkeypatch):
        captured = {}
        a = LLMSecretHitlerAgent(model_name="test/model")
        monkeypatch.setattr(
            a, "query", lambda prompt: captured.setdefault("p", prompt) or fenced({"discard": "Fascist"})
        )
        a(
            _obs(
                player_id="p0",
                detailed_phase=DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT,
                game_state_phase="Legislative",
                policy_hand=["Fascist", "Fascist", "Liberal"],
            )
        )
        assert "Fascist x2" in captured["p"] and "Liberal x1" in captured["p"]

    def test_fascist_allies_in_prompt_but_not_for_liberal(self, monkeypatch):
        captured = {}
        a = LLMSecretHitlerAgent(model_name="test/model")
        monkeypatch.setattr(a, "query", lambda prompt: captured.setdefault("p", prompt) or fenced({"vote": "Ja"}))
        a(
            _obs(
                role="Fascist",
                team="Fascist",
                detailed_phase=DetailedPhase.ELECTION_VOTE_AWAIT,
                nominee_chancellor_id="p1",
                known_allies={"p3": "Hitler"},
            )
        )
        assert "p3" in captured["p"]


# ------------------------------------------------------------------ #
# Per-phase handlers
# ------------------------------------------------------------------ #
class TestHandlers:
    def test_nominate_matches_valid(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"target_id": "p2"}))
        out = a(_obs(eligible_chancellor_ids=["p1", "p2", "p3"]))
        assert out["action_type"] == "NominateChancellorAction"
        assert out["kwargs"]["target_id"] == "p2"

    def test_nominate_illegal_falls_back(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"target_id": "ghost"}))
        out = a(_obs(eligible_chancellor_ids=["p1", "p2"]))
        assert out["kwargs"]["target_id"] == "p1"

    def test_vote(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"vote": "Nein"}))
        out = a(
            _obs(
                detailed_phase=DetailedPhase.ELECTION_VOTE_AWAIT,
                game_state_phase="Election",
                nominee_chancellor_id="p1",
            )
        )
        assert out["action_type"] == "VoteAction" and out["kwargs"]["vote"] == Vote.NEIN.value

    def test_discard_matches_hand(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"discard": "Liberal"}))
        out = a(
            _obs(
                detailed_phase=DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT,
                game_state_phase="Legislative",
                policy_hand=["Fascist", "Fascist", "Liberal"],
            )
        )
        assert out["action_type"] == "DiscardPolicyAction" and out["kwargs"]["discard"] == "Liberal"

    def test_discard_illegal_color_falls_back_to_hand(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"discard": "Liberal"}))
        out = a(
            _obs(
                detailed_phase=DetailedPhase.LEGISLATIVE_PRESIDENT_AWAIT,
                game_state_phase="Legislative",
                policy_hand=["Fascist", "Fascist", "Fascist"],  # no Liberal in hand
            )
        )
        assert out["kwargs"]["discard"] == "Fascist"

    def test_chancellor_veto(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"discard": "Fascist", "veto": True}))
        out = a(
            _obs(
                detailed_phase=DetailedPhase.LEGISLATIVE_CHANCELLOR_AWAIT,
                game_state_phase="Legislative",
                policy_hand=["Fascist", "Liberal"],
                veto_available=True,
                chancellor_id="p0",
            )
        )
        assert out["kwargs"]["veto"] is True

    def test_power_execution(self, monkeypatch):
        a = _agent(monkeypatch, fenced({"target_id": "p3"}))
        out = a(
            _obs(
                detailed_phase=DetailedPhase.EXECUTIVE_POWER_AWAIT,
                game_state_phase="Executive",
                pending_power=Power.EXECUTION.value,
                power_valid_targets=["p1", "p3"],
            )
        )
        assert out["action_type"] == "ExecutionAction" and out["kwargs"]["target_id"] == "p3"

    def test_unparseable_falls_back_to_noop(self, monkeypatch):
        a = _agent(monkeypatch, "no json here, just prose")
        out = a(_obs(eligible_chancellor_ids=["p1"]))
        assert out["action_type"] == "NoOpAction"


# ------------------------------------------------------------------ #
# Integration: a full game driven by a mocked LLM
# ------------------------------------------------------------------ #
def test_full_game_with_mocked_llm(monkeypatch):
    canned = json.dumps(
        {
            "reasoning": "auto",
            "target_id": "p1",
            "amount": 1,
            "vote": "Ja",
            "discard": "Fascist",
            "message": "hello",
            "consent": False,
        }
    )

    def fake_completion(**kwargs):
        return {
            "choices": [{"message": {"content": f"```json\n{canned}\n```"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(harness, "completion", fake_completion)

    from kaggle_environments.envs.secret_hitler.game.consts import RoleConst

    n = 5
    counts = assign_role_counts(n)
    roles = (
        [RoleConst.LIBERAL.value] * counts[RoleConst.LIBERAL]
        + [RoleConst.FASCIST.value] * counts[RoleConst.FASCIST]
        + [RoleConst.HITLER.value] * counts[RoleConst.HITLER]
    )
    cfg = [{"id": f"p{i}", "role": roles[i], "agent_id": "llm"} for i in range(n)]

    env = make("secret_hitler", debug=True, configuration={"agents": cfg})
    llm_agents = [LLMSecretHitlerAgent(model_name="test/model") for _ in range(n)]
    env.run(llm_agents)
    assert set(env.toJSON()["statuses"]) == {"DONE"}
