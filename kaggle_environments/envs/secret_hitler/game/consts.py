from enum import Enum

MODERATOR_ID = "MODERATOR"

EVENT_HANDLER_FOR_ATTR_NAME = "_event_handler_for"


class StrEnum(str, Enum):
    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


class Phase(StrEnum):
    ELECTION = "Election"
    LEGISLATIVE = "Legislative"
    EXECUTIVE = "Executive"
    GAME_OVER = "Game Over"


ELECTION, LEGISLATIVE, EXECUTIVE, GAME_OVER = Phase


class Team(StrEnum):
    LIBERAL = "Liberal"
    FASCIST = "Fascist"


class RoleConst(StrEnum):
    LIBERAL = "Liberal"
    FASCIST = "Fascist"
    HITLER = "Hitler"


class PolicyColor(StrEnum):
    LIBERAL = "Liberal"
    FASCIST = "Fascist"


class Power(StrEnum):
    POLICY_PEEK = "PolicyPeek"
    INVESTIGATE = "InvestigateLoyalty"
    SPECIAL_ELECTION = "SpecialElection"
    EXECUTION = "Execution"


class Vote(StrEnum):
    JA = "Ja"
    NEIN = "Nein"


class PerceivedThreatLevel(StrEnum):
    SAFE = "SAFE"
    UNEASY = "UNEASY"
    DANGER = "DANGER"


class PhaseDivider(StrEnum):
    ROUND_START = "ROUND START"
    ELECTION_START = "ELECTION START"
    ELECTION_END = "ELECTION END"
    LEGISLATIVE_START = "LEGISLATIVE START"
    LEGISLATIVE_END = "LEGISLATIVE END"
    EXECUTIVE_START = "EXECUTIVE START"
    EXECUTIVE_END = "EXECUTIVE END"


class EnvInfoKeys:
    MODERATOR_OBS = "MODERATOR_OBSERVATION"
    GAME_END = "GAME_END"


class ObsKeys:
    RAW_OBSERVATION = "raw_observation"


class DetailedPhase(StrEnum):
    def __new__(cls, value, category: Phase):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.category = category
        return obj

    GAME_START = "GAME_START", ELECTION

    # Election
    ELECTION_NOMINATION_AWAIT = "ELECTION_NOMINATION_AWAIT", ELECTION
    ELECTION_BIDDING_AWAIT = "ELECTION_BIDDING_AWAIT", ELECTION
    ELECTION_CHAT_AWAIT = "ELECTION_CHAT_AWAIT", ELECTION
    ELECTION_VOTE_AWAIT = "ELECTION_VOTE_AWAIT", ELECTION
    ELECTION_CONCLUDE = "ELECTION_CONCLUDE", ELECTION

    # Legislative
    LEGISLATIVE_PRESIDENT_AWAIT = "LEGISLATIVE_PRESIDENT_AWAIT", LEGISLATIVE
    LEGISLATIVE_CHANCELLOR_AWAIT = "LEGISLATIVE_CHANCELLOR_AWAIT", LEGISLATIVE
    LEGISLATIVE_VETO_CONSENT_AWAIT = "LEGISLATIVE_VETO_CONSENT_AWAIT", LEGISLATIVE
    LEGISLATIVE_CONCLUDE = "LEGISLATIVE_CONCLUDE", LEGISLATIVE
    # Post-policy debrief: round-robin discussion after a policy is enacted, before the next round.
    LEGISLATIVE_DEBRIEF_AWAIT = "LEGISLATIVE_DEBRIEF_AWAIT", LEGISLATIVE

    # Executive
    EXECUTIVE_POWER_AWAIT = "EXECUTIVE_POWER_AWAIT", EXECUTIVE
    EXECUTIVE_CONCLUDE = "EXECUTIVE_CONCLUDE", EXECUTIVE

    GAME_OVER = "GAME_OVER", GAME_OVER


class EventName(StrEnum):
    GAME_START = "game_start"
    GAME_START_ROLE = "game_start_role"
    PHASE_DIVIDER = "phase_divider"
    MODERATOR_ANNOUNCEMENT = "moderator_announcement"

    NOMINATION_REQUEST = "nomination_request"
    NOMINATION = "nomination"

    BID_REQUEST = "bid_request"
    BID_ACTION = "bid_action"
    BID_RESULT = "bid_result"

    CHAT_REQUEST = "chat_request"
    DISCUSSION = "discussion"
    DISCUSSION_ORDER = "discussion_order"

    VOTE_REQUEST = "vote_request"
    VOTE_ACTION = "vote_action"
    ELECTION_RESULT = "election_result"
    GOVERNMENT_ELECTED = "government_elected"

    DISCARD_REQUEST = "discard_request"
    DISCARD_ACTION = "discard_action"
    VETO_PROPOSED = "veto_proposed"
    VETO_CONSENT_REQUEST = "veto_consent_request"
    VETO_RESULT = "veto_result"
    POLICY_ENACTED = "policy_enacted"
    CHAOS_ENACTED = "chaos_enacted"
    ELECTION_TRACKER = "election_tracker"

    POWER_REQUEST = "power_request"
    POWER_USED = "power_used"
    INVESTIGATE_RESULT = "investigate_result"
    POLICY_PEEK = "policy_peek"
    SPECIAL_ELECTION = "special_election"
    EXECUTION = "execution"

    BOARD_SNAPSHOT = "board_snapshot"
    GAME_END = "game_end"
    ERROR = "error"
    NOOP_ACTION = "no_op_action"
