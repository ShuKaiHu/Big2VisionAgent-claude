from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentCard(BaseModel):
    code: str
    display: str
    rank: str
    suit: str


class AgentActionOption(BaseModel):
    action: Literal["play", "pass"]
    cards: list[AgentCard] = Field(default_factory=list)
    combo_type: str | None = None


class AgentDecision(BaseModel):
    action: Literal["play", "pass"]
    card_codes: list[str] = Field(default_factory=list)
    combo_type: str | None = None
    note: str | None = None


class TurnConstraint(BaseModel):
    lead_actor: Literal["self", "left", "top", "right"] | None = None
    required_combo_type: str | None = None
    last_played_cards: list[AgentCard] = Field(default_factory=list)
    last_played_by: Literal["self", "left", "top", "right"] | None = None
    passes_since_last_play: int = 0


class OpponentState(BaseModel):
    seat: Literal["left", "top", "right"]
    remaining_count: int | None = None


class PlayEvent(BaseModel):
    """A single play or pass in the CURRENT game, in chronological order.

    This is the authoritative, complete record of who played/passed what — built
    from the full WS network timeline, not the lossy single-snapshot constraint.
    The decision agent rebuilds each player's played-cards set from this so the
    belief model (determinization) and policy/value model see every opponent play.
    """

    actor: Literal["self", "left", "top", "right"]
    action: Literal["play", "pass"]
    card_codes: list[str] = Field(default_factory=list)
    combo_type: str | None = None  # 牌型（single/pair/straight/full_house/...），pass 為 None
    ts: int | None = None          # 事件時間（epoch 毫秒，來自 WS 封包）


class AgentObservation(BaseModel):
    game_index: int
    trick_index: int | None = None
    self_hand: list[AgentCard] = Field(default_factory=list)
    hand_count: int = 0
    turn: Literal["self", "left", "top", "right", "unknown"] = "unknown"
    constraint: TurnConstraint = Field(default_factory=TurnConstraint)
    opponents: list[OpponentState] = Field(default_factory=list)
    legal_actions: list[AgentActionOption] = Field(default_factory=list)
    # Complete ordered play/pass history of the current game (authoritative).
    play_history: list[PlayEvent] = Field(default_factory=list)
    source_seq: int | None = None
