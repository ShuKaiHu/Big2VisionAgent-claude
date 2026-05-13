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


class AgentObservation(BaseModel):
    game_index: int
    trick_index: int | None = None
    self_hand: list[AgentCard] = Field(default_factory=list)
    hand_count: int = 0
    turn: Literal["self", "left", "top", "right", "unknown"] = "unknown"
    constraint: TurnConstraint = Field(default_factory=TurnConstraint)
    opponents: list[OpponentState] = Field(default_factory=list)
    legal_actions: list[AgentActionOption] = Field(default_factory=list)
    source_seq: int | None = None
