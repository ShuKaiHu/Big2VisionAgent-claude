from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import threading
from pathlib import Path
from typing import Protocol

from big2_vision_agent.action_executor import ActionExecutor
from big2_vision_agent.agent_schema import AgentDecision, AgentObservation

log = logging.getLogger(__name__)


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
    """長駐 subprocess wrapper agent。

    使用 subprocess.Popen 保持 wrapper process 存活，透過 stdin/stdout pipe
    逐輪傳送 observation JSON 並讀回 decision JSON。

    好處：MockGame 在 wrapper 內部累積完整出牌歷史，MCTS 的局面重建更準確，
    大幅減少 no_env_overlap 的發生頻率。
    舊架構（subprocess.run）每輪重新啟動 process，MockGame 從零開始，
    歷史全部遺失，是 no_env_overlap 的根本原因。
    """

    def __init__(self, command: str) -> None:
        self.command = command
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None

    def _ensure_process(self) -> None:
        """若 process 未啟動或已死亡，重新啟動。"""
        if self._process is not None and self._process.poll() is None:
            return  # 仍在執行中

        if self._process is not None:
            log.warning("[wrapper] process died (returncode=%s), restarting", self._process.poll())

        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,          # line-buffered
            shell=True,
        )
        # 在背景執行緒讀 stderr 並轉發到 log，避免 stderr buffer 滿造成 deadlock
        self._stderr_thread = threading.Thread(
            target=self._forward_stderr,
            args=(self._process,),
            daemon=True,
        )
        self._stderr_thread.start()
        log.info("[wrapper] process started (pid=%s)", self._process.pid)

    def _forward_stderr(self, proc: subprocess.Popen) -> None:
        """背景執行緒：持續讀 wrapper 的 stderr 並寫到 log。"""
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    log.debug("[wrapper] %s", line)
        except Exception:
            pass

    def decide(self, observation: AgentObservation) -> AgentDecision | None:
        line = json.dumps(observation.model_dump(), ensure_ascii=False) + "\n"
        # The wrapper subprocess can die intermittently (observed: an OS SIGKILL
        # -9 that is NOT this wrapper's own OOM — RSS stays ~190MB with the system
        # mostly free). Rather than crash the whole autoplay session, RESTART the
        # wrapper and retry the move ONCE. A fresh wrapper loses its in-memory
        # MockGame history for this single move (slightly less accurate env
        # reconstruction), but the session keeps playing instead of aborting; the
        # following moves rebuild state normally.
        for attempt in (1, 2):
            self._ensure_process()
            proc = self._process
            assert proc is not None
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
                payload = proc.stdout.readline()
            except (BrokenPipeError, OSError, ValueError) as exc:
                log.error("[wrapper] write/read failed (%s) — restarting (attempt %d/2)",
                          exc, attempt)
                self._process = None
                continue
            if proc.poll() is not None:
                log.error("[wrapper] exited unexpectedly (returncode=%s) — restarting (attempt %d/2)",
                          proc.poll(), attempt)
                self._process = None
                continue
            payload = payload.strip()
            if not payload:
                return None
            return AgentDecision.model_validate_json(payload)

        log.error("[wrapper] still unavailable after restart — skipping this decision")
        return None

    def close(self) -> None:
        """關閉 wrapper process（agent 結束時呼叫）。"""
        if self._process is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None


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
