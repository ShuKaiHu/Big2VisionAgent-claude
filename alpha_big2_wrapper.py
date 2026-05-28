#!/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/.venv/bin/python3
"""alpha_big2_wrapper.py  (AlphaBig2 新版 engine — Big2Net + MCTS)

Bridges Big2VisionAgent ↔ AlphaBig2 ML model (engine/ 框架).

Usage:
    BIG2_AGENT_COMMAND=/path/to/alpha_big2_wrapper.py \\
        uv run big2-agent autoplay-agent --executor packet

讀一行 AgentObservation JSON from stdin.
寫一行 AgentDecision JSON to stdout.
所有 debug 輸出走 stderr。

決策流程：
  每輪收到 observation → 更新 MockGame（局面重建）
  → 對對手手牌做 determinization（隨機採樣可能的牌）
  → 建構完整的 big2Game → 包入 Big2Env → MCTS 跑 3 秒
  → 選 visit count 最高的動作 → 回傳 AgentDecision
"""

from __future__ import annotations

import json
import logging
import os
import sys
import copy

# ── Path setup ──────────────────────────────────────────────────────────────
_AB2_DIR = os.environ.get(
    "ALPHA_BIG2_DIR",
    "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/.claude/worktrees/lucid-liskov-29de5d",
)
sys.path.insert(0, _AB2_DIR)
# enumerateOptions.py uses relative path for actionIndices.pkl
os.chdir(_AB2_DIR)

import numpy as np
import torch

import gameLogic
import enumerateOptions
from big2Game import big2Game, handPlayed
from engine.model import Big2Net
from engine.env import Big2Env, PASS_IDX
from engine.mcts import MCTS

# ── Logging (stderr only) ────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[alpha_big2] %(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Model loading ────────────────────────────────────────────────────────────
_CKPT_DIR = os.path.join(_AB2_DIR, "engine", "checkpoints")
_BEST_PT   = os.path.join(_CKPT_DIR, "best.pt")
_LATEST_PT = os.path.join(_CKPT_DIR, "latest.pt")

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

def _load_model() -> Big2Net:
    model = Big2Net()
    path = _BEST_PT if os.path.exists(_BEST_PT) else _LATEST_PT
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys (new heads): %s", missing)
    model.eval()
    log.info("Model loaded from %s", path)
    return model

_model = _load_model()
_mcts  = MCTS(_model, n_simulations=200)   # fallback if time_limit not used

# ── Card-code conversion ─────────────────────────────────────────────────────
# Big2Vision code: "<suit_char><rank_char>"
#   suit_char: '1'=Spade  '2'=Heart  '3'=Diamond  '4'=Club
#   rank_char: '3'..'9', 'T'(10), 'J', 'Q', 'K', '1'(Ace), '2'
#
# AlphaBig2 card_id = (rank_value-1)*4 + suit_idx  (1-indexed, range 1..52)
#   rank_value: 1=Three … 13=Two
#   suit_idx:   1=Club  2=Diamond  3=Heart  4=Spade

_BV_SUIT_TO_AB: dict[str, int] = {"1": 4, "2": 3, "3": 2, "4": 1}
_BV_RANK_TO_AB: dict[str, int] = {
    "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
    "T": 8, "J": 9, "Q": 10, "K": 11, "1": 12, "2": 13,
}
_AB_SUIT_TO_BV: dict[int, str] = {4: "1", 3: "2", 2: "3", 1: "4"}
_AB_RANK_TO_BV: dict[int, str] = {
    1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9",
    8: "T", 9: "J", 10: "Q", 11: "K", 12: "1", 13: "2",
}


def _bv_to_id(code: str) -> int:
    return (_BV_RANK_TO_AB[code[1]] - 1) * 4 + _BV_SUIT_TO_AB[code[0]]


def _id_to_bv(card_id: int) -> str:
    rank_val = (card_id - 1) // 4 + 1
    suit_idx = (card_id - 1) % 4 + 1
    return _AB_SUIT_TO_BV[suit_idx] + _AB_RANK_TO_BV[rank_val]


def _five_combo_type(card_ids) -> str:
    arr = np.array(list(card_ids), dtype=np.int64)
    if gameLogic.isStraightFlush(arr):
        return "straight_flush"
    if gameLogic.isFourOfAKind(arr):
        return "four_of_kind"
    if gameLogic.isFullHouse(arr)[0]:
        return "full_house"
    return "straight"


# ── MockGame ──────────────────────────────────────────────────────────────────

class MockGame:
    """
    Reconstructs a big2Game-compatible state from the AgentObservation stream.
    We are always perspective player 1.  Seat mapping:
        self=1  left=2  top=3  right=4

    Note on goIndex semantics:
      MockGame.goIndex starts at 0, incremented BEFORE storing in handsPlayed.
      big2Game.goIndex starts at 1, incremented AFTER storing in handsPlayed.
      When building a big2Game for MCTS: g.goIndex = mock_game.goIndex + 1
    """

    _SEAT = {"self": 1, "left": 2, "top": 3, "right": 4}

    def __init__(self) -> None:
        self._game_index: int | None = None
        self._prev_constraint: dict = {}
        self._reset()

    def _reset(self) -> None:
        self.currentHands: dict[int, np.ndarray] = {
            p: np.array([], dtype=np.int64) for p in range(1, 5)
        }
        self.cardsPlayed: np.ndarray = np.zeros((4, 52), dtype=np.int32)
        self.playersGo: int = 1
        self.control: int = 1
        self.mustPlayClub3: bool = False
        self.passedThisRound: dict[int, bool] = {p: False for p in range(1, 5)}
        self.lastPlayedPlayer: int = 1
        self.goIndex: int = 0          # 0 = no hands played yet
        self.handsPlayed: dict = {}
        self.actionHistory: list[dict] = []
        self.gameOver: int = 0
        self.rewards: np.ndarray = np.zeros(4)
        self._prev_constraint = {}

    def update(self, obs: dict) -> None:
        gidx = obs.get("game_index", 0)
        if gidx != self._game_index:
            self._reset()
            self._game_index = gidx
            log.info("New game (game_index=%d)", gidx)

        # Our exact hand
        self.currentHands[1] = np.array(
            sorted(_bv_to_id(c["code"]) for c in obs["self_hand"]),
            dtype=np.int64,
        )

        # Opponent hand sizes (actual cards unknown)
        for opp in obs.get("opponents", []):
            p = self._SEAT.get(opp["seat"])
            rc = opp.get("remaining_count")
            if p is not None and rc is not None and len(self.currentHands[p]) != rc:
                self.currentHands[p] = np.zeros(rc, dtype=np.int64)

        # Flags from constraint
        c = obs.get("constraint", {})
        last_played = c.get("last_played_cards", [])
        lead_actor  = c.get("lead_actor")
        # control=1（自由出牌）的條件：
        #   - lead_actor 明確設定（有人主導這一墩），或
        #   - 牌桌上沒有牌（全新一墩，包含開局 mustPlayClub3 的情況）
        # control=0（必須跟牌）只在 lead_actor=None 且桌上有牌時成立。
        self.control = 1 if (lead_actor is not None or not last_played) else 0
        self.playersGo = self._SEAT.get(obs.get("turn", "self"), 1)

        # mustPlayClub3: 3♣ = card_id 1 must be in legal play
        self.mustPlayClub3 = any(
            _bv_to_id(card["code"]) == 1
            for action in obs.get("legal_actions", [])
            if action.get("action") == "play"
            for card in action.get("cards", [])
        )

        # Detect opponent plays between our turns
        prev_cards = self._prev_constraint.get("last_played_cards", [])
        prev_by    = self._prev_constraint.get("last_played_by")
        cur_cards  = c.get("last_played_cards", [])
        cur_by     = c.get("last_played_by")
        passes_since = c.get("passes_since_last_play", 0)

        if cur_cards and (cur_cards != prev_cards or cur_by != prev_by):
            player = self._SEAT.get(cur_by) if cur_by else None
            if player and player != 1:
                card_ids = [_bv_to_id(card["code"]) for card in cur_cards]
                self._record_play(player, card_ids)
                self.lastPlayedPlayer = player
                self.passedThisRound = {p: False for p in range(1, 5)}

        for i in range(1, passes_since + 1):
            p = ((self.lastPlayedPlayer - 1 + i) % 4) + 1
            if p != 1:
                self.passedThisRound[p] = True

        self._prev_constraint = c

    class _H:
        """Minimal handPlayed-compatible object for MockGame entries."""
        def __init__(self, cards: list[int]) -> None:
            self.hand = np.array(cards, dtype=np.int64)

    def _record_play(self, player: int, card_ids: list[int]) -> None:
        self.goIndex += 1
        self.actionHistory.append({
            "player": player,
            "hand": np.array(card_ids, dtype=np.int64),
            "pass": False,
            "forced_skip": False,
            "control_break": self.control == 0,
            "passed_snapshot": np.array(
                [1 if self.passedThisRound[p] else 0 for p in range(1, 5)],
                dtype=np.float32,
            ),
        })
        self.handsPlayed[self.goIndex] = MockGame._H(card_ids)
        for cid in card_ids:
            self.cardsPlayed[player - 1][int(cid) - 1] = 1

    def record_our_play(self, card_ids: list[int]) -> None:
        self._record_play(1, card_ids)
        self.lastPlayedPlayer = 1
        self.passedThisRound = {p: False for p in range(1, 5)}
        played = set(card_ids)
        self.currentHands[1] = np.array(
            [c for c in self.currentHands[1] if c not in played], dtype=np.int64
        )

    def record_our_pass(self) -> None:
        self.goIndex += 1
        self.actionHistory.append({
            "player": 1,
            "hand": None,
            "pass": True,
            "forced_skip": False,
            "control_break": False,
            "passed_snapshot": np.array(
                [1 if self.passedThisRound[p] else 0 for p in range(1, 5)],
                dtype=np.float32,
            ),
        })
        self.passedThisRound[1] = True


# ── Determinization helpers ───────────────────────────────────────────────────

def _sample_opponent_hands(mock_game: MockGame) -> dict[int, np.ndarray]:
    """
    Sample plausible opponent hands given public info.

    Known:  our hand (player 1) + all played cards
    Unknown: remaining 52 - known cards, distributed to 3 opponents
             according to their remaining hand counts.

    若對手 remaining_count 尚未從觀察中取得（遊戲剛開始時常見），
    以「剩餘牌均分」估計，確保 Big2Env 不會看到「對手 0 張牌」的假局面。
    """
    my_hand = set(int(c) for c in mock_game.currentHands[1])
    played  = set()
    for p in range(4):
        for c in range(52):
            if mock_game.cardsPlayed[p][c] == 1:
                played.add(c + 1)

    remaining = list(set(range(1, 53)) - my_hand - played)
    np.random.shuffle(remaining)

    # 若三位對手都是 0 張（remaining_count 尚未讀到），
    # 用剩餘牌均分作為估計（遊戲開局每人 13 張時最常見）。
    total_known = sum(len(mock_game.currentHands[p]) for p in [2, 3, 4])
    if total_known == 0 and len(remaining) > 0:
        per_opp = len(remaining) // 3
        sizes = {2: per_opp, 3: per_opp, 4: len(remaining) - 2 * per_opp}
        log.info("對手手牌數未知，估計每人 %d / %d / %d 張", sizes[2], sizes[3], sizes[4])
    else:
        sizes = {p: len(mock_game.currentHands[p]) for p in [2, 3, 4]}

    # ── 安全上限：每人最多 13 張 ──────────────────────────────────────────────
    # enumerateOptions 的 twoCardIndices / fiveCardIndices 預先計算時
    # 以 indexInHand（0-12）為索引，若手牌超過 13 張就會 IndexError。
    # 根本原因：cardsPlayed 只追蹤最近一手出牌，played 集合偏小 →
    # remaining 被高估 → per_opp 可能 ≥ 14。
    # 直接 cap 到 13 即可保持安全（Big2 任何玩家最多持 13 張）。
    sizes = {p: min(13, sizes[p]) for p in [2, 3, 4]}

    opp_hands = {}
    offset = 0
    for p in [2, 3, 4]:
        n = sizes[p]
        opp_hands[p] = np.array(
            sorted(remaining[offset : offset + n]), dtype=np.int64
        )
        offset += n

    total_needed = sum(sizes[p] for p in [2, 3, 4])
    if total_needed > len(remaining):
        log.warning(
            "剩餘牌 (%d) 不足以分給對手 (%d) — 局面可能已過時",
            len(remaining), total_needed,
        )

    return opp_hands


def _build_game_for_mcts(mock_game: MockGame, opp_hands: dict[int, np.ndarray]) -> big2Game:
    """
    Build a complete big2Game from MockGame state + sampled opponent hands.

    big2Game semantics: goIndex is the NEXT empty slot (starts at 1).
    MockGame semantics: goIndex is the number of plays so far (starts at 0).
    Mapping: g.goIndex = mock_game.goIndex + 1
    """
    g = big2Game.__new__(big2Game)

    # Hands
    g.currentHands = {
        1: mock_game.currentHands[1].copy(),
        2: opp_hands[2].copy(),
        3: opp_hands[3].copy(),
        4: opp_hands[4].copy(),
    }

    # Public info
    g.cardsPlayed       = mock_game.cardsPlayed.copy()
    g.playersGo         = mock_game.playersGo
    g.control           = mock_game.control
    g.mustPlayClub3     = mock_game.mustPlayClub3
    g.passedThisRound   = dict(mock_game.passedThisRound)
    g.lastPlayedPlayer  = mock_game.lastPlayedPlayer

    # goIndex: big2Game convention (points to next empty slot)
    g.goIndex       = mock_game.goIndex + 1
    g.handsPlayed   = dict(mock_game.handsPlayed)   # _H objects, .hand is compatible
    g.actionHistory = list(mock_game.actionHistory)

    # Game lifecycle
    g.gameOver  = 0
    g.rewards   = np.zeros(4)
    g.goCounter = 0

    # passCount: number of unique players who passed since last real play
    g.passCount = sum(1 for p in range(1, 5) if mock_game.passedThisRound.get(p, False))

    # club3Player: only matters when mustPlayClub3=True (first move of game)
    if mock_game.mustPlayClub3:
        g.club3Player = mock_game.playersGo   # the player who has 3♣
    else:
        g.club3Player = 1   # irrelevant once mustPlayClub3=False

    # Legacy neural network inputs (not used by new model)
    g.neuralNetworkInputs = {p: np.zeros(412, dtype=int) for p in range(1, 5)}

    return g


# ── Action mask from legal_actions ────────────────────────────────────────────

def _build_mask(obs: dict) -> np.ndarray:
    mask = np.zeros(enumerateOptions.passInd + 1, dtype=np.float32)
    for action in obs.get("legal_actions", []):
        if action.get("action") == "pass":
            mask[enumerateOptions.passInd] = 1.0
            continue
        card_ids = tuple(
            sorted(_bv_to_id(c["code"]) for c in action.get("cards", []))
        )
        n = len(card_ids)
        try:
            if n == 1:
                mask[enumerateOptions.SINGLE_INDEX[card_ids[0]]] = 1.0
            elif n == 2:
                mask[enumerateOptions.PAIR_OFFSET + enumerateOptions.PAIR_INDEX[card_ids]] = 1.0
            elif n == 5:
                mask[enumerateOptions.FIVE_OFFSET + enumerateOptions.FIVE_INDEX[card_ids]] = 1.0
        except KeyError:
            log.warning("Legal action not in AlphaBig2 enum: %s", card_ids)
    return mask


# ── Action index → AgentDecision ──────────────────────────────────────────────

def _to_decision(action_idx: int) -> dict:
    if action_idx == enumerateOptions.passInd:
        return {"action": "pass", "card_codes": [], "combo_type": None}
    card_ids, n = enumerateOptions.getOptionNC(action_idx)
    codes = [_id_to_bv(int(cid)) for cid in card_ids]
    if n == 1:
        combo = "single"
    elif n == 2:
        combo = "pair"
    else:
        combo = _five_combo_type(card_ids)
    return {"action": "play", "card_codes": codes, "combo_type": combo}


# ── One-card rule: restrict singles when any opponent has 1 card ──────────────

def _apply_one_card_rule(obs: dict, obs_mask: np.ndarray) -> np.ndarray:
    """Filter obs_mask to enforce the house rule:

    When ANY opponent has exactly 1 card remaining, playing a single is
    restricted to the HIGHEST single in the legal actions only.
    Non-single plays (pair, 5-card combo, pass) are completely unrestricted.

    This applies whether we are the lead actor OR following someone else's
    single — the old _forced_highest_single only handled the follower case.

    Returns a (possibly modified) copy of obs_mask.
    """
    opponents = obs.get("opponents", [])
    if not any(opp.get("remaining_count") == 1 for opp in opponents):
        return obs_mask   # Rule inactive

    # Collect all single-card action indices that are currently legal
    legal_singles: list[tuple[int, int]] = []   # (card_id, action_idx)
    for action in obs.get("legal_actions", []):
        if action.get("action") == "play" and len(action.get("cards", [])) == 1:
            card_id = _bv_to_id(action["cards"][0]["code"])
            try:
                idx = enumerateOptions.SINGLE_INDEX[card_id]
                if obs_mask[idx] > 0:
                    legal_singles.append((card_id, idx))
            except KeyError:
                pass

    if len(legal_singles) <= 1:
        return obs_mask   # 0 or 1 singles — nothing to restrict

    # Keep only the highest single; zero-out all others
    highest_id, _ = max(legal_singles, key=lambda x: x[0])
    new_mask = obs_mask.copy()
    for card_id, idx in legal_singles:
        if card_id != highest_id:
            new_mask[idx] = 0.0
            log.debug("One-card rule: removed single card_id=%d from mask", card_id)

    highest_code = _id_to_bv(highest_id)
    log.info("One-card rule active: only highest single %s (+ multi-card combos) allowed",
             highest_code)
    return new_mask


# ── Inference (MCTS + determinization) ───────────────────────────────────────

def _infer(mock_game: MockGame, obs: dict) -> tuple[int, str]:
    """回傳 (action_idx, ml_note)。ml_note 會寫入 AgentDecision.note，
    在 run.log 裡可見，用來確認 ML 運算是否正常。"""
    import time

    # Build valid action mask from legal_actions (ground truth from server)
    obs_mask = _build_mask(obs)
    legal = np.flatnonzero(obs_mask)
    if len(legal) == 0:
        log.warning("No legal actions — defaulting to pass")
        return enumerateOptions.passInd, "fallback:no_legal_actions"

    # ── mustPlayClub3：開局必須出含梅花三的牌，不得 PASS ─────────────────
    # packet parser 有時把 pass 放進 legal_actions，但遊戲伺服器會拒絕。
    # 直接把 pass 從 mask 移除，讓 MCTS 只能選含梅花三的出法。
    if mock_game.mustPlayClub3:
        obs_mask[enumerateOptions.passInd] = 0.0
        legal = np.flatnonzero(obs_mask)
        if len(legal) == 0:
            log.warning("mustPlayClub3 但 mask 裡沒有合法出牌 — 保留 pass 作為 fallback")
            obs_mask[enumerateOptions.passInd] = 1.0
            legal = np.array([enumerateOptions.passInd])

    # ── One-card rule: when any opponent has 1 card, only highest single legal ─
    # Applies to both lead and follower situations. Must run BEFORE early-exit
    # so the pass-only check sees the filtered mask.
    obs_mask = _apply_one_card_rule(obs, obs_mask)
    legal = np.flatnonzero(obs_mask)   # recompute after filter

    # ── Early exit: if the only legal action is pass, no need for MCTS ────
    # This is common when we can't beat the current table hand.
    non_pass_legal = [a for a in legal if a != enumerateOptions.passInd]
    if len(non_pass_legal) == 0:
        return enumerateOptions.passInd, "only_legal:pass"

    # Sample opponent hands (determinization)
    opp_hands = _sample_opponent_hands(mock_game)
    g = _build_game_for_mcts(mock_game, opp_hands)

    # Wrap in Big2Env
    env = Big2Env.__new__(Big2Env)
    env._game = g
    env._done = False

    # ── ALWAYS sync critical game state from obs (authoritative ground truth) ─
    # MockGame can drift from reality (missed opponent plays, wrong control
    # flag).  obs.constraint is the server's ground truth — always apply it
    # before querying get_valid_actions() so the env stays consistent.
    c = obs.get("constraint", {})
    last_played = c.get("last_played_cards", [])
    lead_actor  = c.get("lead_actor")

    # control=0 only when there are cards on the table AND we are not the
    # lead actor (we need to beat the last hand).
    if lead_actor is None and last_played:
        g.control = 0
        # Replace handsPlayed[goIndex-1] with the actual table hand so
        # returnAvailableActions reads the right cards to beat.
        last_card_ids = sorted(_bv_to_id(card["code"]) for card in last_played)
        if g.goIndex == 0:
            g.goIndex = 1
        g.handsPlayed[g.goIndex - 1] = MockGame._H(last_card_ids)
    else:
        g.control = 1

    # It is definitely our turn; make sure the game knows.
    g.playersGo = 1
    # We have not passed this round (we are about to act).
    g.passedThisRound[1] = False

    # Verify that the env's valid actions overlap with obs_mask
    env_mask = env.get_valid_actions()
    overlap = (env_mask * obs_mask).sum()
    if overlap == 0:
        # Debug: log what obs_mask and env_mask contain to understand the mismatch
        obs_legal_indices = np.flatnonzero(obs_mask).tolist()
        env_legal_indices = np.flatnonzero(env_mask).tolist()
        obs_n = len(obs_legal_indices)
        env_n = len(env_legal_indices)
        log.warning(
            "No env/obs overlap after sync | control=%d goIndex=%d | "
            "obs_mask has %d actions (indices: %s) | env_mask has %d actions (sample: %s)",
            g.control, g.goIndex, obs_n, obs_legal_indices[:5], env_n, env_legal_indices[:5],
        )
        log.warning("Falling back to greedy policy")
        from engine.features import encode_static, encode_history_steps
        static  = encode_static(mock_game, 1)
        history = encode_history_steps(mock_game)
        probs, value = _model.predict(static, history, obs_mask)
        masked = probs * obs_mask
        action = int(np.argmax(masked)) if masked.sum() > 0 else enumerateOptions.passInd
        note = f"greedy:no_env_overlap v={float(value):.3f}"
        log.info("Greedy policy (no env overlap): action=%d value=%.3f", action, float(value))
        return action, note

    # Run MCTS for 1 second (online real-time limit)
    t0 = time.time()
    action, visits = _mcts.run(env, temperature=0.0, time_limit=1.0)
    elapsed = time.time() - t0
    n_sims = int(visits.sum())
    log.info("MCTS: %d sims in %.2fs → action=%d", n_sims, elapsed, action)

    # Restrict to actions that are ACTUALLY legal (obs_mask is ground truth)
    if obs_mask[action] == 0:
        log.warning("MCTS chose action %d not in obs_mask — restricting to legal", action)
        masked_visits = visits * obs_mask
        action = int(np.argmax(masked_visits)) if masked_visits.sum() > 0 else int(legal[0])

    note = f"mcts:{n_sims}sims_{elapsed:.2f}s"
    return action, note


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    game = MockGame()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obs = json.loads(raw)
            game.update(obs)
            action_idx, ml_note = _infer(game, obs)
            decision   = _to_decision(action_idx)

            # Keep MockGame in sync with our own decision
            if action_idx == enumerateOptions.passInd:
                game.record_our_pass()
            else:
                card_ids, _ = enumerateOptions.getOptionNC(action_idx)
                game.record_our_play([int(c) for c in card_ids])

            # 把 ML 運算資訊寫入 note 欄位，讓 run.log 可見
            decision["note"] = ml_note
            log.info("→ %s %s (%s) [%s]", decision["action"], decision["card_codes"], decision["combo_type"], ml_note)
            print(json.dumps(decision), flush=True)

        except Exception as exc:
            import traceback
            tb = traceback.format_exc().strip().splitlines()
            # 取最後兩行作為簡短摘要，避免 note 過長
            short_err = " | ".join(tb[-2:]) if len(tb) >= 2 else str(exc)
            log.exception("Error during inference — falling back to pass")
            print(json.dumps({
                "action": "pass",
                "card_codes": [],
                "combo_type": None,
                "note": f"fallback:exception {short_err}",
            }), flush=True)


if __name__ == "__main__":
    main()
