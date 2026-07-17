#!/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/.venv/bin/python3
"""heuristic_wrapper.py — belief/dominance heuristic bot (Big2-Belief-Policy v1)

A drop-in replacement for alpha_big2_wrapper.py that swaps the AlphaZero
value+MCTS decision core for the simple, offline-champion heuristic
`bot.choose_action` (aggressive shedding, never voluntarily pass, dump when
cornered). The whole exploration converged on this being the strongest offline
heuristic; the AlphaZero stack added no win and the go-out planner / holding
variants all scored lower (see AlphaBig2-claude memory). Per the project's
methodology the only remaining judge is ONLINE play vs real humans — this wrapper
is what puts v1 in front of them.

It REUSES all the robust plumbing from alpha_big2_wrapper:
  * MockGame            — drift-free game-state reconstruction from play_history
  * _build_mask         — server-authoritative legal-action mask (AlphaBig2 enum)
  * _apply_one_card_rule— house rule: don't feed a low single to a 1-card player
  * _to_decision        — action index → AgentDecision JSON
  * dashboard + opp-hand belief display, error handling, stdin/stdout protocol
Importing that module no longer loads torch (lazy _ensure_model), so this stays a
pure-CPU heuristic with no model in memory.

Usage (see play_heuristic.sh):
    BIG2_AGENT_COMMAND=/path/to/heuristic_wrapper.py \\
        uv run big2-agent autoplay-agent --executor packet
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

# alpha_big2_wrapper lives at this repo's root (same dir as this file) and, on
# import, sets up sys.path + chdir to AlphaBig2-claude and exposes all helpers.
import alpha_big2_wrapper as W

import numpy as np
import enumerateOptions

# Big2-Belief-Policy holds the decision core (`bot`) and its own top-level
# `dominance` (AlphaBig2 only has engine/dominance, so no name clash). Append so
# AlphaBig2's already-loaded enumerateOptions wins for shared names.
_BOT_DIR = os.environ.get(
    "BIG2_BOT_DIR", "/Users/shukaihu/Code_Project_Local/Big2-Belief-Policy"
)
sys.path.append(_BOT_DIR)
import bot  # noqa: E402

log = W.log
_HEUR_LOG_PATH = os.path.join(W._BV_DIR, "artifacts", "heuristic_moves.jsonl")


class _BotGame:
    """Minimal big2Game-shaped view over MockGame + the authoritative legal mask,
    exposing exactly what bot.choose_action reads."""

    def __init__(self, mock: "W.MockGame", obs_mask: np.ndarray) -> None:
        self.currentHands = mock.currentHands   # {1..4: card_id array}; opp = zero-arrays sized by remaining
        self.cardsPlayed = mock.cardsPlayed     # (4,52) [player-1][card_id-1]
        self.control = mock.control             # 1=lead, 0=follow
        self._mask = obs_mask

    def returnAvailableActions(self) -> np.ndarray:
        return self._mask


def _log_move(mock_game, obs, action, note) -> None:
    """Append per-move context to heuristic_moves.jsonl for online-data analysis
    (mirrors mcts_moves.jsonl). Never breaks play."""
    try:
        hand = [int(c) for c in mock_game.currentHands[1]]
        played = bot.played_ids(_BotGame(mock_game, np.array([])))
        c = obs.get("constraint", {})
        to_beat = [W._cid_label(W._bv_to_id(card["code"]))
                   for card in c.get("last_played_cards", [])]
        chosen = W._action_cards(int(action))
        if action != enumerateOptions.passInd:
            cards, _n = enumerateOptions.getOptionNC(int(action))
            safety = round(float(bot.play_safety(list(cards), hand, played)), 3)
        else:
            safety = None
        rec = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "agent": "heuristic_v1",
            "game_index": obs.get("game_index"),
            "control": "lead" if mock_game.control == 1 else "follow",
            "my_hand": [W._cid_label(x) for x in sorted(hand)],
            "to_beat": to_beat or None,
            "opp_counts": {"right": int(len(mock_game.currentHands[2])),
                           "top": int(len(mock_game.currentHands[3])),
                           "left": int(len(mock_game.currentHands[4]))},
            "chosen": chosen,
            "chosen_safety": safety,
            "note": note,
        }
        os.makedirs(os.path.dirname(_HEUR_LOG_PATH), exist_ok=True)
        with open(_HEUR_LOG_PATH, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("heuristic-move log failed: %s", e)


def _infer(mock_game: "W.MockGame", obs: dict) -> tuple[int, str, dict]:
    """Decide with bot.choose_action on the server-authoritative legal mask.
    Returns (action_idx, note, extra) — same contract as W._infer."""
    extra: dict = {}
    PASS = enumerateOptions.passInd

    obs_mask = W._build_mask(obs)
    if not obs_mask.any():
        log.warning("No legal actions — defaulting to pass")
        return PASS, "fallback:no_legal_actions", extra

    if mock_game.mustPlayClub3:
        obs_mask[PASS] = 0.0
        if not obs_mask.any():
            obs_mask[PASS] = 1.0   # safety net: never leave the agent with no action

    obs_mask = W._apply_one_card_rule(obs, obs_mask, mock_game)
    extra["obs_mask"] = obs_mask

    # opponent-hand belief is display-only; compute for the dashboard.
    try:
        extra["opp_hands"] = W._sample_opponent_hands(mock_game)
    except Exception:
        extra["opp_hands"] = {}

    non_pass = [int(a) for a in np.flatnonzero(obs_mask) if int(a) != PASS]
    if not non_pass:
        return PASS, "only_legal:pass", extra

    action = int(bot.choose_action(_BotGame(mock_game, obs_mask), 1))
    if obs_mask[action] == 0:   # safety: keep the choice legal
        log.warning("bot chose illegal action %d — restricting to legal", action)
        action = non_pass[0]

    note = ("heuristic:lead" if mock_game.control == 1 else "heuristic:follow")
    _log_move(mock_game, obs, action, note)
    return action, note, extra


def main() -> None:
    try:
        os.makedirs(os.path.dirname(W._DASH_STATE_PATH), exist_ok=True)
        with open(W._DASH_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"__status": "waiting", "updated_at": None}, f)
    except Exception:
        pass

    log.info("heuristic_wrapper: belief/dominance v1 (bot.choose_action), no model")
    game = W.MockGame()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obs = json.loads(raw)
            game.update(obs)
            action_idx, note, extra = _infer(game, obs)
            decision = W._to_decision(action_idx)

            if action_idx == enumerateOptions.passInd:
                game.record_our_pass()
            else:
                card_ids, _ = enumerateOptions.getOptionNC(action_idx)
                game.record_our_play([int(c) for c in card_ids])

            decision["note"] = note
            log.info("→ %s %s (%s) [%s]", decision["action"],
                     decision["card_codes"], decision["combo_type"], note)
            print(json.dumps(decision), flush=True)
            W._write_dashboard_state(obs, decision, note, extra, game)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            short_err = " | ".join(tb[-2:]) if len(tb) >= 2 else str(exc)
            log.exception("Error during inference — falling back to pass")
            print(json.dumps({
                "action": "pass", "card_codes": [], "combo_type": None,
                "note": f"fallback:exception {short_err}",
            }), flush=True)


if __name__ == "__main__":
    main()
