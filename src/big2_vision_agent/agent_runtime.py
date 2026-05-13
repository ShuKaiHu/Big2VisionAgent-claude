from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Protocol

from big2_vision_agent.action_executor import ActionExecutor
from big2_vision_agent.agent_schema import AgentDecision, AgentObservation


class DecisionAgent(Protocol):
    def decide(self, observation: AgentObservation) -> AgentDecision | None: ...


class FallbackDecisionAgent:
    def __init__(self) -> None:
        self.executor = ActionExecutor()

    def decide(self, observation: AgentObservation) -> AgentDecision | None:
        play_actions = [action for action in observation.legal_actions if action.action == "play"]
        if play_actions:
            # "能不 pass 就不 pass": prefer the first legal play after packet/runtime filtering.
            action = play_actions[0]
            return AgentDecision(
                action="play",
                card_codes=[card.code for card in action.cards],
                combo_type=action.combo_type,
                note="fallback:no_pass",
            )
        return self.executor.choose_fallback_action(observation.legal_actions)


class ExternalCommandAgent:
    def __init__(self, command: str) -> None:
        self.command = command

    def decide(self, observation: AgentObservation) -> AgentDecision | None:
        completed = subprocess.run(
            self.command,
            input=json.dumps(observation.model_dump(), ensure_ascii=False),
            text=True,
            shell=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"External agent command failed: {completed.returncode}: {completed.stderr.strip()}"
            )
        payload = completed.stdout.strip()
        if not payload:
            return None
        return AgentDecision.model_validate_json(payload)


def build_decision_agent() -> DecisionAgent:
    command = os.getenv("BIG2_AGENT_COMMAND")
    if command:
        return ExternalCommandAgent(command)
    return FallbackDecisionAgent()


def sample_random_decision(observation: AgentObservation) -> AgentDecision | None:
    play_actions = [action for action in observation.legal_actions if action.action == "play"]
    if play_actions:
        chosen = random.choice(play_actions)
        return AgentDecision(
            action="play",
            card_codes=[card.code for card in chosen.cards],
            combo_type=chosen.combo_type,
            note="random_sample",
        )
    return AgentDecision(action="pass", note="random_sample")


def save_observation(observation: AgentObservation, path: Path) -> None:
    path.write_text(json.dumps(observation.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
