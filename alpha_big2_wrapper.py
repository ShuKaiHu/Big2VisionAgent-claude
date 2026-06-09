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
    "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude",
)
# Resolve Big2VisionAgent root before chdir changes CWD
_BV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AB2_DIR)
# 看板狀態的單一真相模組（純標準庫，main.py 與本 wrapper 共用，確保結構一致）
sys.path.insert(0, os.path.join(_BV_DIR, "src"))
from big2_vision_agent import dashboard_writer
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

# ── Online void-constrained determinization (opt-in, default OFF) ────────────
# When an opponent passes while a SINGLE leads the trick, a rational player
# reveals it holds no single higher than that card. We accumulate this as a hard
# cap on the opponent's max card_id and exclude higher cards when sampling
# opponent hands for MCTS determinization. This is a RATIONAL-PLAY assumption
# (humans sometimes pass while saving a high card), so it is gated behind
# BIG2_VOID=1 and MUST be A/B-validated online — if it hurts (as the soft belief
# experiment did), leave it off. Default OFF keeps V6 deployment byte-identical.
_VOID_ON = os.environ.get("BIG2_VOID") == "1"
if _VOID_ON:
    log.info("BIG2_VOID=1 → void-constrained determinization ENABLED")

# ── Model loading ────────────────────────────────────────────────────────────
_CKPT_DIR = os.path.join(_AB2_DIR, "engine", "checkpoints")
_BEST_PT   = os.path.join(_CKPT_DIR, "best.pt")
_LATEST_PT = os.path.join(_CKPT_DIR, "latest.pt")

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

def _load_model() -> Big2Net:
    model = Big2Net()
    # ALPHA_BIG2_CKPT lets you A/B a non-default checkpoint online without
    # touching best.pt. Accepts an absolute path OR a name relative to _CKPT_DIR
    # (e.g. "saved/v8_td_deploy.pt"). Unset → default best.pt (V6). Both V6 and
    # V8 are 306-dim, so launch with BIG2_DOMINANCE=1 either way.
    _override = os.environ.get("ALPHA_BIG2_CKPT")
    if _override:
        path = _override if os.path.isabs(_override) else os.path.join(_CKPT_DIR, _override)
        if not os.path.exists(path):
            raise RuntimeError(f"ALPHA_BIG2_CKPT not found: {path}")
    else:
        path = _BEST_PT if os.path.exists(_BEST_PT) else _LATEST_PT
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    # ── Feature-dim safety guard ───────────────────────────────────────────
    # A model trained with dominance features (BIG2_DOMINANCE=1, STATIC_DIM 306)
    # has a 434-wide input_proj; a 302-dim build is 430-wide. load_state_dict
    # with strict=False would SILENTLY DROP the mismatched input layer, leaving
    # it randomly initialized → the model plays garbage with NO error. Detect
    # this and fail LOUDLY with the fix instead.
    ckpt_in = None
    for k in ("input_proj.0.weight", "input_proj.weight"):
        if k in state:
            ckpt_in = state[k].shape[1]
            break
    model_in = model.input_proj[0].weight.shape[1]
    if ckpt_in is not None and ckpt_in != model_in:
        import engine.features as _f
        raise RuntimeError(
            f"FEATURE-DIM MISMATCH: checkpoint expects input {ckpt_in} but this "
            f"process builds {model_in} (STATIC_DIM={_f.STATIC_DIM}, "
            f"DOMINANCE_ON={_f.DOMINANCE_ON}). "
            f"This v6 model needs dominance features — set BIG2_DOMINANCE=1 "
            f"before launching. (ckpt {ckpt_in - 128}-dim static vs "
            f"{model_in - 128}-dim here.)"
        )
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        log.warning("Missing keys (new heads): %s", missing)
    model.eval()
    log.info("Model loaded from %s (input_dim=%d)", path, model_in)
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

    # 出牌順序：self → right → top → left（伺服器 actor_index 遞增方向，已由
    # 對局封包驗證）。AlphaBig2 的 player 1→2→3→4 即出牌順序，故：
    #   self=1, 下家(right)=2, 對家(top)=3, 上家(left)=4。
    # 先前誤把 left=2/right=4，導致下家/上家顛倒，餵給模型的相鄰關係是反的。
    _SEAT = {"self": 1, "right": 2, "top": 3, "left": 4}

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
        # BIG2_VOID: per-opponent max card_id they can hold, inferred from passes
        # on a single lead. 52 = no constraint (no informative pass yet).
        self.opp_void_single_cap: dict[int, int] = {p: 52 for p in range(2, 5)}
        # Authoritative-rebuild bookkeeping (play_history path):
        self._logged_events: int = 0      # play_history entries already turned into dashboard events
        self._last_lead_single: int | None = None  # card_id of the current trick's leading single (void)

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

        # ── Authoritative play-tracking ────────────────────────────────────────
        # The observation now carries play_history: the COMPLETE ordered log of
        # every play/pass this game (built from the full WS timeline). Rebuilding
        # from it each turn is drift-free — every opponent play is recorded, so
        # cardsPlayed / actionHistory are exact for the belief + policy models.
        play_history = obs.get("play_history")
        if play_history:
            self.rebuild_from_history(play_history)
        else:
            # Backward-compat: legacy snapshot-based incremental detection
            # (lossy — misses opponents that aren't the immediately-prior player).
            self._update_from_snapshot(c)

        self._prev_constraint = c

    class _H:
        """Minimal handPlayed-compatible object for MockGame entries."""
        def __init__(self, cards: list[int]) -> None:
            self.hand = np.array(cards, dtype=np.int64)

    def rebuild_from_history(self, play_history: list[dict]) -> None:
        """Deterministically rebuild ALL play-derived state from the complete,
        ordered play/pass log of the current game. Replaces the lossy snapshot
        path: every opponent play is captured, so cardsPlayed (the model's
        "what each player played" feature) and actionHistory are exact.

        Also drives the dashboard event log via a diff against already-logged
        entries, so opponents' plays AND passes appear completely."""
        self.cardsPlayed = np.zeros((4, 52), dtype=np.int32)
        self.handsPlayed = {}
        self.actionHistory = []
        self.goIndex = 0
        self.passedThisRound = {p: False for p in range(1, 5)}
        self.lastPlayedPlayer = 1
        self.opp_void_single_cap = {p: 52 for p in range(2, 5)}
        self._last_lead_single = None

        for ev in play_history:
            actor = ev.get("actor")
            p = self._SEAT.get(actor)
            if p is None:
                continue
            snap = np.array(
                [1 if self.passedThisRound[q] else 0 for q in range(1, 5)],
                dtype=np.float32,
            )
            if ev.get("action") == "play":
                card_ids = sorted(_bv_to_id(code) for code in ev.get("card_codes", []))
                self.goIndex += 1
                self.actionHistory.append({
                    "player": p,
                    "hand": np.array(card_ids, dtype=np.int64),
                    "pass": False,
                    "forced_skip": False,
                    "control_break": False,
                    "passed_snapshot": snap,
                })
                self.handsPlayed[self.goIndex] = MockGame._H(card_ids)
                for cid in card_ids:
                    self.cardsPlayed[p - 1][int(cid) - 1] = 1
                self.lastPlayedPlayer = p
                # New hand on the table → everyone's pass-flag resets.
                self.passedThisRound = {q: False for q in range(1, 5)}
                self._last_lead_single = card_ids[0] if len(card_ids) == 1 else None
            else:  # pass
                self.goIndex += 1
                self.actionHistory.append({
                    "player": p,
                    "hand": None,
                    "pass": True,
                    "forced_skip": False,
                    "control_break": False,
                    "passed_snapshot": snap,
                })
                self.passedThisRound[p] = True
                # Void: passing on a single reveals no single stronger than it.
                # cap 一律計算（供看板 belief 估計用），是否套用到 MCTS 抽樣才看
                # _VOID_ON（_sample_opponent_hands）。兩者解耦：看板的對手手牌信心
                # 總是用得到 void 資訊，MCTS 行為維持原樣。
                if (p != 1 and self._last_lead_single is not None
                        and self._last_lead_single < self.opp_void_single_cap[p]):
                    self.opp_void_single_cap[p] = self._last_lead_single
        # 看板事件紀錄改由 dashboard_writer.build_events 直接從 play_history 算
        # （main.py 與 wrapper 共用，完整且一致），此處不再另行累積。

    def _update_from_snapshot(self, c: dict) -> None:
        """Legacy fallback when play_history is absent. Lossy by design."""
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

        lead_single = None
        if _VOID_ON and passes_since > 0:
            for e in reversed(self.actionHistory):
                if not e.get("pass") and e.get("hand") is not None:
                    if len(e["hand"]) == 1:
                        lead_single = int(e["hand"][0])
                    break

        for i in range(1, passes_since + 1):
            p = ((self.lastPlayedPlayer - 1 + i) % 4) + 1
            if p != 1:
                self.passedThisRound[p] = True
                if lead_single is not None and lead_single < self.opp_void_single_cap[p]:
                    self.opp_void_single_cap[p] = lead_single

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
        # Advisory only on the play_history path: the next observation's
        # play_history is authoritative and a full rebuild overwrites this.
        # We still shrink our own hand so any same-turn followup reads are sane.
        played = set(card_ids)
        self.currentHands[1] = np.array(
            [c for c in self.currentHands[1] if c not in played], dtype=np.int64
        )
        self.lastPlayedPlayer = 1
        self.passedThisRound = {p: False for p in range(1, 5)}

    def record_our_pass(self) -> None:
        # Advisory only on the play_history path (see record_our_play).
        self.passedThisRound[1] = True


# ── Determinization helpers ───────────────────────────────────────────────────

def _assign_with_void(mock_game: "MockGame", remaining: list, sizes: dict) -> dict[int, np.ndarray]:
    """Partition `remaining` (already shuffled) to opponents 2/3/4 respecting
    per-seat void caps (max card_id from passes on a single). Tightest cap first;
    if a seat cannot be filled from eligible cards, take all eligible and fill
    the shortfall from the rest (relaxing the cap) so determinization never fails.
    With all caps = 52 this reduces to a uniform random partition (== no-void)."""
    caps = mock_game.opp_void_single_cap
    pool = [int(c) for c in remaining]
    opp_hands: dict[int, np.ndarray] = {}
    for p in sorted([2, 3, 4], key=lambda q: caps.get(q, 52)):  # tightest first
        n = int(sizes[p])
        cap = caps.get(p, 52)
        if n <= 0:
            opp_hands[p] = np.array([], dtype=np.int64)
            continue
        eligible = [c for c in pool if c <= cap]
        if len(eligible) >= n:
            chosen = [int(c) for c in np.random.choice(eligible, size=n, replace=False)]
        else:
            chosen = list(eligible)
            rest = [c for c in pool if c not in set(chosen)]
            short = n - len(chosen)
            if short > 0 and rest:
                chosen += [int(c) for c in
                           np.random.choice(rest, size=min(short, len(rest)), replace=False)]
            log.warning("void: P%d 約束過嚴 (需%d, 合格%d) → 放寬", p, n, len(eligible))
        chosen_set = set(chosen)
        opp_hands[p] = np.array(sorted(chosen_set), dtype=np.int64)
        pool = [c for c in pool if c not in chosen_set]
    return opp_hands


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

    if _VOID_ON:
        opp_hands = _assign_with_void(mock_game, remaining, sizes)
    else:
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


# ── 對手手牌信心估計（看板用） ────────────────────────────────────────────────

def _unknown_and_sizes(mock_game: MockGame):
    """未知牌（不在我手、不在已出）與各對手張數。與 _sample_opponent_hands 同邏輯。"""
    my_hand = set(int(c) for c in mock_game.currentHands[1])
    played = set()
    for p in range(4):
        for c in range(52):
            if mock_game.cardsPlayed[p][c] == 1:
                played.add(c + 1)
    unknown = sorted(set(range(1, 53)) - my_hand - played)
    total_known = sum(len(mock_game.currentHands[p]) for p in (2, 3, 4))
    if total_known == 0 and unknown:
        per = len(unknown) // 3
        sizes = {2: per, 3: per, 4: len(unknown) - 2 * per}
    else:
        sizes = {p: len(mock_game.currentHands[p]) for p in (2, 3, 4)}
    sizes = {p: min(13, sizes[p]) for p in (2, 3, 4)}
    return unknown, sizes


def _estimate_belief(mock_game: MockGame, n_samples: int = 120):
    """蒙地卡羅估計每張未知牌落在各對手(2,3,4)的機率。

    重複抽樣 N 次（一律套用 void 約束，因為只是顯示、不影響決策），統計每張
    未知牌被分到各家的頻率。無 void 資訊時退化為「正比各家張數」的均勻分布。
    回傳 ({card_id: {2:p, 3:p, 4:p}}, sizes)。"""
    unknown, sizes = _unknown_and_sizes(mock_game)
    if not unknown or sum(sizes.values()) == 0:
        return {}, sizes
    counts = {cid: {2: 0, 3: 0, 4: 0} for cid in unknown}
    runs = 0
    for _ in range(n_samples):
        remaining = list(unknown)
        np.random.shuffle(remaining)
        try:
            hands = _assign_with_void(mock_game, remaining, sizes)
        except Exception:
            continue
        for p in (2, 3, 4):
            for cid in hands.get(p, []):
                ci = int(cid)
                if ci in counts:
                    counts[ci][p] += 1
        runs += 1
    if runs == 0:
        return {}, sizes
    belief = {cid: {p: counts[cid][p] / runs for p in (2, 3, 4)} for cid in unknown}
    return belief, sizes


def _assign_belief(belief: dict, sizes: dict) -> dict:
    """依 belief 貪婪指派每張未知牌給「機率最高且仍有空位」的對手，張數自洽且穩定
    （belief 是多次平均，不隨單次抽樣跳動）。
    回傳 {seat: [(card_id, confidence_prob), ...]}。"""
    pairs = [(dist[p], cid, p) for cid, dist in belief.items() for p in (2, 3, 4)]
    pairs.sort(key=lambda t: t[0], reverse=True)   # 機率高者優先指派
    slots = {p: sizes[p] for p in (2, 3, 4)}
    assigned: dict[int, int] = {}
    result: dict[int, list] = {2: [], 3: [], 4: []}
    for prob, cid, p in pairs:
        if cid in assigned or slots[p] <= 0:
            continue
        assigned[cid] = p
        slots[p] -= 1
        result[p].append((cid, prob))
    return result


def _conf_level(prob: float) -> str:
    """信心等級：機率越集中在這家越有把握。3 家隨機基線約 33%。"""
    if prob >= 0.55:
        return "high"   # 綠：明顯集中在這家
    if prob >= 0.40:
        return "mid"    # 黃：略高於隨機
    return "low"        # 紅：接近隨機，沒把握


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

    When the NEXT player (下家 = the "right" seat, who plays immediately after
    us; turn order is self → right → top → left) has exactly 1 card remaining,
    playing a single is restricted to the HIGHEST single in the legal actions
    only.  Non-single plays (pair, 5-card combo, pass) are completely
    unrestricted.

    Only the right seat matters — if "left" or "top" has 1 card but "right"
    does not, the rule is inactive and any single may be played.

    Returns a (possibly modified) copy of obs_mask.
    """
    opponents = obs.get("opponents", [])
    right_remaining = None
    for opp in opponents:
        if opp.get("seat") == "right":
            right_remaining = opp.get("remaining_count")
            break
    if right_remaining != 1:
        return obs_mask   # Rule inactive (next player does not have exactly 1 card)

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

def _infer(mock_game: MockGame, obs: dict) -> tuple[int, str, dict]:
    """回傳 (action_idx, ml_note, extra)。
    extra 包含 policy_probs、visits、opp_hands，供 dashboard 顯示。"""
    import time
    extra: dict = {}

    obs_mask = _build_mask(obs)
    legal = np.flatnonzero(obs_mask)
    if len(legal) == 0:
        log.warning("No legal actions — defaulting to pass")
        return enumerateOptions.passInd, "fallback:no_legal_actions", extra

    if mock_game.mustPlayClub3:
        obs_mask[enumerateOptions.passInd] = 0.0
        legal = np.flatnonzero(obs_mask)
        if len(legal) == 0:
            log.warning("mustPlayClub3 但 mask 裡沒有合法出牌 — 保留 pass 作為 fallback")
            obs_mask[enumerateOptions.passInd] = 1.0
            legal = np.array([enumerateOptions.passInd])

    obs_mask = _apply_one_card_rule(obs, obs_mask)
    legal = np.flatnonzero(obs_mask)
    extra["obs_mask"] = obs_mask

    # determinization：先算好供看板顯示對手推測手牌（即使只能 pass 也要有）。
    opp_hands = _sample_opponent_hands(mock_game)
    extra["opp_hands"] = opp_hands

    non_pass_legal = [a for a in legal if a != enumerateOptions.passInd]
    if len(non_pass_legal) == 0:
        return enumerateOptions.passInd, "only_legal:pass", extra

    g = _build_game_for_mcts(mock_game, opp_hands)

    env = Big2Env.__new__(Big2Env)
    env._game = g
    env._done = False

    c = obs.get("constraint", {})
    last_played = c.get("last_played_cards", [])
    lead_actor  = c.get("lead_actor")

    if lead_actor is None and last_played:
        g.control = 0
        last_card_ids = sorted(_bv_to_id(card["code"]) for card in last_played)
        if g.goIndex == 0:
            g.goIndex = 1
        g.handsPlayed[g.goIndex - 1] = MockGame._H(last_card_ids)
    else:
        g.control = 1

    g.playersGo = 1
    g.passedThisRound[1] = False

    env_mask = env.get_valid_actions()
    overlap = (env_mask * obs_mask).sum()

    # Policy 評分（一次 forward pass，成本遠小於 MCTS）
    from engine.features import encode_static, encode_history_steps
    static  = encode_static(mock_game, 1)
    history = encode_history_steps(mock_game)
    policy_probs, value = _model.predict(static, history, obs_mask)
    extra["policy_probs"] = policy_probs

    if overlap == 0:
        obs_legal_indices = np.flatnonzero(obs_mask).tolist()
        env_legal_indices = np.flatnonzero(env_mask).tolist()
        log.warning(
            "No env/obs overlap after sync | control=%d goIndex=%d | "
            "obs_mask has %d actions (indices: %s) | env_mask has %d actions (sample: %s)",
            g.control, g.goIndex, len(obs_legal_indices), obs_legal_indices[:5],
            len(env_legal_indices), env_legal_indices[:5],
        )
        log.warning("Falling back to greedy policy")
        masked = policy_probs * obs_mask
        action = int(np.argmax(masked)) if masked.sum() > 0 else enumerateOptions.passInd
        self_v = float(np.asarray(value).reshape(-1)[0])
        note = f"greedy:no_env_overlap v={self_v:.3f}"
        log.info("Greedy policy (no env overlap): action=%d value=%.3f", action, self_v)
        return action, note, extra

    t0 = time.time()
    action, visits = _mcts.run(env, temperature=0.0, time_limit=1.0)
    elapsed = time.time() - t0
    n_sims = int(visits.sum())
    extra["visits"] = visits
    log.info("MCTS: %d sims in %.2fs → action=%d", n_sims, elapsed, action)

    if obs_mask[action] == 0:
        log.warning("MCTS chose action %d not in obs_mask — restricting to legal", action)
        masked_visits = visits * obs_mask
        action = int(np.argmax(masked_visits)) if masked_visits.sum() > 0 else int(legal[0])

    note = f"mcts:{n_sims}sims_{elapsed:.2f}s"
    return action, note, extra


# ── Dashboard state writer ────────────────────────────────────────────────────

_DASH_STATE_PATH = os.path.join(_BV_DIR, "state", "dashboard_state.json")

_SUIT_SYM = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}
_RANK_CHAR_DISP = {
    "1": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "10", "J": "J", "Q": "Q", "K": "K",
}

_dash_stats: dict = {
    "decisions": 0, "plays": 0, "passes": 0, "fallbacks": 0,
    "mcts_total_sims": 0, "mcts_total_time_s": 0.0,
}
_dash_events: list[dict] = []
_MAX_EVENTS = 60

# player index ↔ 座位字串 ↔ 中文。出牌順序 self(1)→right(2)→top(3)→left(4)。
_SEAT_ID  = {1: "self", 2: "right", 3: "top", 4: "left"}
_SEAT_ZH  = {"self": "自己", "right": "下家", "top": "對家", "left": "上家"}
_COMBO_ZH_MAP = {
    "single": "單張", "pair": "對子", "straight": "順子",
    "full_house": "葫蘆", "four_of_kind": "四條",
    "four_of_a_kind": "四條", "straight_flush": "同花順",
}


def _combo_from_ids(card_ids) -> str | None:
    n = len(card_ids)
    if n == 1: return "single"
    if n == 2: return "pair"
    if n == 5: return _five_combo_type(list(card_ids))
    return None


def _log_event(seat: str, ev_type: str, cards: list[str], combo: str | None) -> None:
    from datetime import datetime as _dt
    ts      = _dt.now().strftime("%H:%M:%S")
    zh      = _SEAT_ZH.get(seat, seat)
    czh     = _COMBO_ZH_MAP.get(combo or "", "")
    if ev_type == "play":
        cs  = " ".join(cards)
        msg = f"{zh} 出牌：{cs}（{czh}）" if czh else f"{zh} 出牌：{cs}"
    elif ev_type == "pass":
        msg = f"{zh} PASS"
    else:
        msg = f"{zh} {ev_type}"
    _dash_events.append({"ts": ts, "type": ev_type, "actor": seat, "cards": cards, "combo": combo, "msg": msg})
    if len(_dash_events) > _MAX_EVENTS:
        _dash_events.pop(0)


def _card_sym(card: dict) -> str:
    d = card.get("display", "")
    if not d:
        return card.get("code", "?")
    return _SUIT_SYM.get(d[0], d[0]) + d[1:]


def _code_sym(code: str) -> str:
    if len(code) < 2:
        return code
    suit = {"1": "♠", "2": "♥", "3": "♦", "4": "♣"}.get(code[0], code[0])
    rank = _RANK_CHAR_DISP.get(code[1], code[1:])
    return suit + rank


def _parse_ml_note(note: str) -> dict:
    import re as _re
    info: dict = {"mode": "unknown", "mcts_sims": None, "mcts_time_s": None}
    if not note:
        return info
    m = _re.match(r"mcts:(\d+)sims_([\d.]+)s", note)
    if m:
        info.update(mode="mcts", mcts_sims=int(m.group(1)), mcts_time_s=float(m.group(2)))
    elif note.startswith("greedy:"):
        info["mode"] = "greedy"
    elif note.startswith("only_legal:"):
        info["mode"] = "forced"
    elif note.startswith("fallback:"):
        info["mode"] = "fallback"
    return info


def _action_idx_from_legal(la: dict) -> int | None:
    """把 obs legal_action dict 轉成 enumerateOptions action index。"""
    try:
        if la.get("action") == "pass":
            return enumerateOptions.passInd
        cards = la.get("cards", [])
        n = len(cards)
        if n == 0:
            return None
        ids = tuple(sorted(_bv_to_id(c["code"]) for c in cards))
        if n == 1:
            return enumerateOptions.SINGLE_INDEX[ids[0]]
        if n == 2:
            return enumerateOptions.PAIR_OFFSET + enumerateOptions.PAIR_INDEX[ids]
        if n == 5:
            return enumerateOptions.FIVE_OFFSET + enumerateOptions.FIVE_INDEX[ids]
    except Exception:
        pass
    return None


def _scored_legal_actions(obs: dict, infer_extra: dict, chosen_idx: int) -> list[dict]:
    """每個合法動作加上 policy% 和 MCTS visit%。"""
    policy_probs = infer_extra.get("policy_probs")
    visits       = infer_extra.get("visits")
    total_visits = int(visits.sum()) if visits is not None else 0

    result = []
    for la in obs.get("legal_actions", []):
        idx = _action_idx_from_legal(la)
        cards_sym = [_card_sym(c) for c in la.get("cards", [])]
        combo     = la.get("combo_type") or ("pass" if la.get("action") == "pass" else None)
        combo_zh  = _COMBO_ZH_MAP.get(combo or "", "PASS" if la.get("action") == "pass" else "")

        policy_pct  = None
        visit_count = None
        visit_pct   = None
        if idx is not None and policy_probs is not None and idx < len(policy_probs):
            policy_pct = float(policy_probs[idx]) * 100
        if idx is not None and visits is not None and idx < len(visits):
            visit_count = int(visits[idx])
            visit_pct   = visit_count / total_visits * 100 if total_visits > 0 else 0.0

        result.append({
            "action":      la.get("action"),
            "cards":       cards_sym,
            "combo_type":  combo,
            "combo_zh":    combo_zh,
            "action_idx":  idx,
            "policy_pct":  round(policy_pct, 1) if policy_pct is not None else None,
            "visits":      visit_count,
            "visit_pct":   round(visit_pct, 1) if visit_pct is not None else None,
            "chosen":      (idx is not None and idx == chosen_idx),
        })

    # 按 policy_pct 降序排（MCTS 沒跑時也有合理排序），PASS 放最後
    def _sort_key(x):
        is_pass = x["action"] == "pass"
        pct = x["policy_pct"] if x["policy_pct"] is not None else -1.0
        return (is_pass, -pct)
    result.sort(key=_sort_key)
    return result


def _build_players_data(obs: dict, mock_game: MockGame, infer_extra: dict) -> list[dict]:
    """回傳 4 個 seat 的資料列表（self=1, left=2, top=3, right=4）。"""
    opp_hands = infer_extra.get("opp_hands", {})
    # opponents 的 seat 是字串（left/top/right）→ 轉成整數 seat 對應，
    # 否則 int 查 str-key 永遠 miss，remaining 會悄悄退回 len(currentHands)。
    _seat_str2int = {"right": 2, "top": 3, "left": 4}
    opp_count: dict[int, int | None] = {}
    for opp in obs.get("opponents", []):
        s = _seat_str2int.get(opp.get("seat"))
        if s is not None:
            opp_count[s] = opp.get("remaining_count")

    players = []
    for seat in [1, 2, 3, 4]:
        seat_id = _SEAT_ID[seat]
        seat_zh = _SEAT_ZH[seat_id]

        # 已出牌（從 cardsPlayed 矩陣，0-indexed player，0-indexed card_id）
        played_ids = [c + 1 for c in range(52) if mock_game.cardsPlayed[seat - 1][c] == 1]
        played_syms = [_code_sym(_id_to_bv(cid)) for cid in sorted(played_ids, key=lambda cid: _bv_sort_key(_id_to_bv(cid)))]

        # 已出張數（完整追蹤，cardsPlayed 由 play_history 重建，無遺漏）
        known_played = len(played_ids)

        if seat == 1:
            remaining = obs.get("hand_count", len(obs.get("self_hand", [])))
            played_total = 13 - remaining  # 精確
            # 自己手牌：已知，依牌力由小到大排序
            self_hand_sorted = sorted(obs.get("self_hand", []), key=lambda c: _bv_sort_key(c.get("code", "")))
            hand_syms = [_card_sym(c) for c in self_hand_sorted]
            is_estimated = False
        else:
            # ── 對手剩餘張數：精確計算，永不為 None ──────────────────────────
            # 每家開局 13 張，cardsPlayed 已完整追蹤 → 剩餘 = 13 - 已出。
            # 這比依賴 UI remaining_count（可能讀不到）更可靠，也保證有值。
            remaining = 13 - known_played
            played_total = known_played
            # 與 UI 讀數交叉驗證：不一致代表追蹤可能有漏，記警告以便偵測。
            ui_remaining = opp_count.get(seat)
            if ui_remaining is not None and ui_remaining != remaining:
                log.warning(
                    "對手 %s 剩餘張數不一致：追蹤=%d UI=%d（已出%d張）— 可能有漏追蹤",
                    seat_id, remaining, ui_remaining, known_played,
                )
                # UI 是遊戲真實顯示，視為 ground truth；以它為準。
                remaining = ui_remaining
                played_total = 13 - ui_remaining
            if seat in opp_hands and len(opp_hands[seat]) > 0:
                # 對手估計牌：依牌力由小到大排序
                sorted_ids = sorted(opp_hands[seat], key=lambda cid: _bv_sort_key(_id_to_bv(int(cid))))
                hand_syms = [_code_sym(_id_to_bv(int(cid))) for cid in sorted_ids]
                is_estimated = True
            else:
                hand_syms = []
                is_estimated = False

        players.append({
            "seat":          seat,
            "seat_id":       seat_id,
            "seat_zh":       seat_zh,
            "remaining":     remaining,
            "played":        played_syms,
            "played_total":  played_total,   # 實際出牌總數（伺服器推算）
            "hand":          hand_syms,
            "is_estimated":  is_estimated,
        })
    return players


def _bv_sort_key(bv_code: str) -> tuple:
    _rank_order = {"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"T":10,"J":11,"Q":12,"K":13,"1":14,"2":15}
    _suit_order = {"4":1,"3":2,"2":3,"1":4}  # C<D<H<S
    if len(bv_code) < 2:
        return (99, 99)
    return (_rank_order.get(bv_code[1], 99), _suit_order.get(bv_code[0], 99))


def _write_dashboard_state(
    obs: dict, decision: dict, ml_note: str,
    infer_extra: dict | None = None,
    mock_game: "MockGame | None" = None,
) -> None:
    from datetime import datetime as _dt
    if infer_extra is None:
        infer_extra = {}
    try:
        note_info = _parse_ml_note(ml_note)

        _dash_stats["decisions"] += 1
        if decision["action"] == "pass":
            _dash_stats["passes"] += 1
        else:
            _dash_stats["plays"] += 1
        if note_info["mode"] in ("fallback", "greedy"):
            _dash_stats["fallbacks"] += 1
        if note_info["mcts_sims"] is not None:
            _dash_stats["mcts_total_sims"] += note_info["mcts_sims"]
        if note_info["mcts_time_s"] is not None:
            _dash_stats["mcts_total_time_s"] += note_info["mcts_time_s"]

        # 對手推測手牌 + 信心：用 belief 蒙地卡羅估計每張未知牌的家別機率，
        # 再貪婪指派（穩定、張數自洽）。→ {player_index: [(bv_code, conf_level)]}
        opp_hands_codes: dict[int, list[tuple[str, str]]] = {}
        if mock_game is not None:
            belief, sizes = _estimate_belief(mock_game)
            if belief:
                for seat, items in _assign_belief(belief, sizes).items():
                    opp_hands_codes[seat] = [
                        (_id_to_bv(cid), _conf_level(prob)) for cid, prob in items
                    ]

        # 決策牌符號 + 選中的 action index
        decision_cards = [dashboard_writer.code_to_symbol(code) for code in decision.get("card_codes", [])]
        chosen_idx = _action_idx_from_legal({
            "action": decision["action"],
            "cards":  [{"code": code} for code in decision.get("card_codes", [])],
            "combo_type": decision.get("combo_type"),
        }) if decision["action"] != "pass" else enumerateOptions.passInd

        # 牌況核心（players/events/constraint/turn）由共用模組建構 → 與 main.py 完全一致。
        # wrapper 額外帶 AI 決策選項、對手推測手牌、本場統計。
        state = dashboard_writer.base_state(obs, opp_hands=opp_hands_codes)
        state["legal_actions"] = _scored_legal_actions(obs, infer_extra, chosen_idx)
        state["last_decision"] = {
            "action":     decision["action"],
            "cards":      decision_cards,
            "combo_type": decision.get("combo_type"),
            "combo_zh":   _COMBO_ZH_MAP.get(decision.get("combo_type") or "", ""),
            "note":       ml_note,
            **note_info,
        }
        state["session"] = dict(_dash_stats)

        dashboard_writer.atomic_write(_DASH_STATE_PATH, state)
    except Exception as _e:
        log.debug("dashboard write failed: %s", _e)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    # 啟動時清空舊狀態，讓 dashboard 從空白開始
    try:
        os.makedirs(os.path.dirname(_DASH_STATE_PATH), exist_ok=True)
        with open(_DASH_STATE_PATH, "w", encoding="utf-8") as _f:
            json.dump({"__status": "waiting", "updated_at": None}, _f)
    except Exception:
        pass

    game = MockGame()

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obs = json.loads(raw)
            game.update(obs)
            action_idx, ml_note, infer_extra = _infer(game, obs)
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
            _write_dashboard_state(obs, decision, ml_note, infer_extra, game)

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
