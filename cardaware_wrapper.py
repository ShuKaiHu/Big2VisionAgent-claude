#!/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo/.venv/bin/python
"""cardaware_wrapper.py — Big2VisionAgent decision agent backed by the cardaware PPO model.

Drop-in replacement for alpha_big2_wrapper.py. Reads an AgentObservation JSON
line from stdin, writes an AgentDecision JSON line to stdout (all debug on stderr).

It REUSES alpha_big2_wrapper's proven vision-bridge logic — MockGame (state
reconstruction), _build_mask, the one-card house rule, _build_game_for_mcts,
_to_decision — and only swaps the brain. The cardaware actor-critic scores the
legal actions by dot-product and needs ONLY public info (our hand, played cards,
opponent counts, trick, pass count), so there is NO belief sampling / MCTS — much
simpler and faster than the AlphaZero path.

Env vars:
  CARDAWARE_CKPT  cardaware checkpoint (default: AlphaBig2-ppo/ppo/checkpoints/ppo_cardaware_best.pt)
  CARDAWARE_DIR   AlphaBig2-ppo worktree path
"""
from __future__ import annotations

import json
import os
import gc
import re
import sys
import time

import numpy as np
import torch

# Importing alpha_big2_wrapper sets up the AlphaBig2-claude path + chdir (for
# actionIndices.pkl) and gives us MockGame + all the vision-bridge helpers.
import alpha_big2_wrapper as abw

# Add the AlphaBig2-ppo worktree for the cardaware modules. AlphaBig2-claude is
# already at sys.path[0] (inserted by abw), so `engine`/`enumerateOptions`/
# `gameLogic` keep resolving to claude's copies — which are byte-identical to
# ppo's — while `ppo.*` resolves here.
_AB2PPO = os.environ.get("CARDAWARE_DIR", "/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo")
if _AB2PPO not in sys.path:
    sys.path.append(_AB2PPO)

from ppo.network_cardaware import CardAwareActorCritic, obs_from_env as ca_obs_from_env, _to_batch
from ppo.network_cardaware_history import (
    CardAwareActorCriticHistory, steps_from_action_history, encode_history_steps_v2,
)
from ppo.action_features import ACTION_FEATURES
from ppo.belief_model import BeliefNet, belief_from_obs, unseen_mask_from_obs
from ppo.belief_model_history import BeliefNetHistory
from engine.env import Big2Env

enumerateOptions = abw.enumerateOptions
PASS_IDX = abw.PASS_IDX
log = abw.log

# Free the AlphaZero MCTS model + value net that abw loads at import — cardaware
# never uses them, and that footprint (plus the browser's RAM) was OOM-killing
# the wrapper online (returncode -9).
for _attr in ("_mcts", "_model", "_value_model"):
    if hasattr(abw, _attr):
        setattr(abw, _attr, None)
gc.collect()

# ── PPO_V2: inference-time PIMC search (V1 weights + determinized rollouts) ────
# CARDAWARE_SEARCH=1 turns it on. For each of the top-M policy candidates, sample
# W random opponent-hand worlds consistent with public info, roll each out to
# terminal with cardaware-greedy for all seats, and pick the action with the best
# mean self-score. Bounded by a wall-clock budget so it fits the online turn timer.
_SEARCH = os.environ.get("CARDAWARE_SEARCH") == "1"
_SEARCH_WORLDS = int(os.environ.get("CARDAWARE_WORLDS", "24"))   # hard cap (bounds memory); 1s budget also applies
_SEARCH_TOPM = int(os.environ.get("CARDAWARE_TOPM", "4"))
_SEARCH_BUDGET = float(os.environ.get("CARDAWARE_BUDGET", "1.0"))  # ~1s/move to fit the online turn timer

# ── Future main line: PPO_V4 (policy) + BELIEF.pt (belief) + VALUE.pt (value) ──
# All three are HUMAN-DATA-grounded models. The search:
#   BELIEF_SEARCH=1 → importance-sample each determinized world from the dedicated
#     belief model BELIEF.pt (BeliefNet, 82.8% P@count on human data).
#   VALUE_LEAF=1    → evaluate each candidate with the public-info value model
#     VALUE.pt (truncated rollout to our next turn, then value) instead of rolling
#     all the way to terminal with greedy.
# Policy = whatever CARDAWARE_CKPT points to (set it to PPO_V4.pt at launch).
_BELIEF_SEARCH = os.environ.get("BELIEF_SEARCH") == "1"
_BELIEF_CKPT = os.environ.get(
    "BELIEF_CKPT", os.path.join(_AB2PPO, "ppo", "checkpoints", "saved", "BELIEF.pt"))
_VALUE_LEAF = os.environ.get("VALUE_LEAF") == "1"
# Deploy value = VALUE_minplays.pt (public obs + min-plays/13). Ablation (200k):
# min-plays earns the whole +8.5% MSE; dominance was ~redundant for position value.
# extra_dim is auto-detected from the checkpoint, so plain VALUE.pt still loads.
_VALUE_CKPT = os.environ.get(
    "VALUE_CKPT", "/Users/shukaihu/Code_Project_Local/AlphaBig2-Value/checkpoints/VALUE_minplays.pt")

# ISMCTS=1 → true single-tree information-set MCTS (vs the flat PIMC _search_action).
# One tree keyed by OUR action-path; each simulation re-determinizes a world from
# belief; opponents play forward via the policy (environment); our nodes use PUCT
# with the policy as prior; the leaf is the value net (our perspective); one shared
# scorecard. Needs belief (determinize) + value (leaf).
_ISMCTS = os.environ.get("ISMCTS") == "1"
_ISMCTS_CPUCT = float(os.environ.get("ISMCTS_CPUCT", "1.5"))
_ISMCTS_SIMS = int(os.environ.get("ISMCTS_SIMS", "200"))   # cap; _SEARCH_BUDGET also applies
# De-anchor knobs (default OFF -> deployed behavior UNCHANGED unless enabled).
# Mirror eval_alpha1.py's root Dirichlet noise + min-visit floor, ROOT-only.
_ISMCTS_ROOT_NOISE = os.environ.get("ISMCTS_ROOT_NOISE") == "1"
_ISMCTS_NOISE_EPS = float(os.environ.get("ISMCTS_NOISE_EPS", "0.25"))
_ISMCTS_NOISE_ALPHA = float(os.environ.get("ISMCTS_NOISE_ALPHA", "0.3"))
_ISMCTS_MIN_VISITS = int(os.environ.get("ISMCTS_MIN_VISITS", "0"))

# POSTACTION_VALUE=1 → the ISMCTS value leaf evaluates the POST-action state (apply
# my greedy action first, then read value from my perspective: my hand is reduced,
# the trick is my own just-played combo) instead of the pre-action state. Requires
# VALUE_CKPT to point at a post-action-trained value net (VALUE_postaction.pt).
# Single-variable swap vs V4PBV_Iter9 (see post-action-value-decision-design):
# policy/belief/ISMCTS/sims all unchanged, ONLY the value component (train timing +
# where it's queried) changes. Default OFF -> existing V4PBV behavior untouched.
_POSTACTION_VALUE = os.environ.get("POSTACTION_VALUE") == "1"

assert ACTION_FEATURES.shape[0] == enumerateOptions.passInd + 1, "action-space size mismatch"

_DEV = "cpu"  # tiny net; single inferences — CPU is plenty and avoids subprocess GPU issues
_CKPT = os.environ.get(
    "CARDAWARE_CKPT",
    os.path.join(_AB2PPO, "ppo", "checkpoints", "ppo_cardaware_best.pt"),
)
# Single-threaded torch: the net is tiny, and repeated multi-threaded (OpenMP)
# bursts during search were the likely cause of the wrapper getting SIGKILL'd
# (-9) online after a few search moves. 1 thread is also often faster for tiny ops.
torch.set_num_threads(1)

_ck = torch.load(_CKPT, map_location=_DEV, weights_only=False)
# policy_578/policy_1266 (and any future policy_*_bomb fine-tune) were trained
# with a GRU play-history head (same per-step schema as belief_model_history's
# BeliefNetHistory) -- detect via the "arch" tag saved at train time and switch
# the net class + inference path accordingly. Search modes (ISMCTS/PIMC) are
# NOT wired up for history checkpoints yet -- only the plain greedy path below.
_HISTORY_POLICY = _ck.get("arch") == "cardaware_history"
if _HISTORY_POLICY:
    _net = CardAwareActorCriticHistory().to(_DEV)
else:
    _net = CardAwareActorCritic().to(_DEV)
_net.load_state_dict(_ck["model"])
_net.eval()
log.info("cardaware loaded: %s (arch=%s upd=%s history=%s)", os.path.basename(_CKPT),
         _ck.get("arch"), _ck.get("update"), _HISTORY_POLICY)
if _HISTORY_POLICY and (os.environ.get("ISMCTS") == "1" or os.environ.get("CARDAWARE_SEARCH") == "1"):
    raise RuntimeError(
        "history-augmented policy checkpoints (arch=cardaware_history) only "
        "support the plain greedy inference path so far -- ISMCTS/PIMC search "
        "is not wired up for them yet. Unset ISMCTS/SEARCH to test this checkpoint."
    )

_belief_net = None
_BELIEF_HISTORY = False
if _BELIEF_SEARCH:
    _bk = torch.load(_BELIEF_CKPT, map_location=_DEV, weights_only=False)
    _BELIEF_HISTORY = _bk.get("arch") == "belief_history_v2"
    if _BELIEF_HISTORY:
        # gru_hidden auto-detected from the checkpoint (gru.weight_ih_l0 = 3*gru_hidden)
        _gru_h = _bk["model"]["gru.weight_ih_l0"].shape[0] // 3
        _belief_net = BeliefNetHistory(gru_hidden=_gru_h).to(_DEV)
    else:
        _belief_net = BeliefNet().to(_DEV)
    _belief_net.load_state_dict(_bk["model"])
    _belief_net.eval()
    log.info("belief net loaded: %s (arch=%s val_pcount=%s) — belief-guided determinization ON",
             os.path.basename(_BELIEF_CKPT), "history" if _BELIEF_HISTORY else "plain",
             _bk.get("val_pcount"))


# Public-info value model — mirror of AlphaBig2-Value/value_model.py ValueNet
# (kept inline to avoid that workspace's import-time chdir). Leaf evaluator when
# VALUE_LEAF=1: V(public state) -> expected final score for the to-move player.
_VALUE_SCALE, _VD, _VE = 13.0, 64, 256


class ValueNet(torch.nn.Module):
    def __init__(self, extra_dim=0):
        super().__init__()
        self.extra_dim = extra_dim            # +obs["feat"] (deploy: 1 = min-plays/13)
        self.card_emb = torch.nn.Embedding(53, _VD, padding_idx=0)
        self.hand_attn = torch.nn.MultiheadAttention(_VD, num_heads=4, batch_first=True)
        _si = _VD + _VD + 3 * _VD + _VD + 3 + 1 + extra_dim
        self.enc = torch.nn.Sequential(
            torch.nn.Linear(_si, _VE), torch.nn.ReLU(),
            torch.nn.Linear(_VE, _VE), torch.nn.ReLU(),
            torch.nn.Linear(_VE, _VE), torch.nn.ReLU(), torch.nn.LayerNorm(_VE))
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(_VE, _VE // 2), torch.nn.ReLU(),
            torch.nn.Linear(_VE // 2, 1), torch.nn.Tanh())

    def forward(self, obs):
        hand_ids = obs["hand_ids"]; pad = hand_ids == 0
        h = self.card_emb(hand_ids)
        attn, _ = self.hand_attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        hand_vec = (attn * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        cards = self.card_emb.weight[1:]
        seen_vec = obs["seen"] @ cards
        opp_vec = (obs["opp"] @ cards).reshape(obs["opp"].shape[0], -1)
        trick_vec = obs["trick"] @ cards
        si = torch.cat([hand_vec, seen_vec, opp_vec, trick_vec, obs["counts"], obs["passc"]], dim=-1)
        if self.extra_dim:
            si = torch.cat([si, obs["feat"]], dim=-1)
        return self.value_head(self.enc(si)).squeeze(-1)


# min-plays-to-empty (greedy fewest legal plays to go out) — the deploy value's
# extra feature. Inlined (like ValueNet) to keep the wrapper self-contained; mirror
# of AlphaBig2-Value/hand_features.py.
from collections import Counter as _Counter
_MP_WIN = [tuple(range(s, s + 5)) for s in range(1, 9)] + [(12, 13, 1, 2, 3), (13, 1, 2, 3, 4)]


def _mp_rank(c):
    return (c - 1) // 4 + 1


def _mp_extract_five(rem):
    rc = _Counter(_mp_rank(c) for c in rem)
    for r, n in rc.items():
        if n >= 4:
            quad = [c for c in rem if _mp_rank(c) == r][:4]
            kicker = [c for c in rem if c not in quad][:1]
            if kicker:
                return quad + kicker
    trips = [r for r, n in rc.items() if n >= 3]
    for t in trips:
        for r, n in rc.items():
            if r != t and n >= 2:
                return [c for c in rem if _mp_rank(c) == t][:3] + [c for c in rem if _mp_rank(c) == r][:2]
    have = {}
    for c in rem:
        have.setdefault(_mp_rank(c), []).append(c)
    for w in _MP_WIN:
        if all(r in have for r in w):
            return [have[r][0] for r in w]
    return None


def min_plays_to_empty(cards):
    rem = [int(c) for c in cards]; plays = 0
    while True:
        five = _mp_extract_five(rem)
        if not five:
            break
        for c in five:
            rem.remove(c)
        plays += 1
    rc = _Counter(_mp_rank(c) for c in rem)
    for r in rc:
        while sum(1 for c in rem if _mp_rank(c) == r) >= 2:
            for c in [c for c in rem if _mp_rank(c) == r][:2]:
                rem.remove(c)
            plays += 1
    return plays + len(rem)


_value_net = None
_value_extra_dim = 0
if _VALUE_LEAF:
    _vk = torch.load(_VALUE_CKPT, map_location=_DEV, weights_only=False)
    _value_extra_dim = int(_vk.get("extra_dim", 0))   # 1 = min-plays; 0 = plain VALUE.pt
    _value_net = ValueNet(extra_dim=_value_extra_dim).to(_DEV)
    _value_net.load_state_dict(_vk["model"])
    _value_net.eval()
    log.info("value net loaded: %s (extra_dim=%d, val_mse=%s) — value-leaf eval ON",
             os.path.basename(_VALUE_CKPT), _value_extra_dim, _vk.get("val_mse"))


@torch.no_grad()
def _value_eval(env, me, real_units=False) -> float:
    """Value net at an OUR-turn leaf, me's perspective. Adds the min-plays feature
    (over me's KNOWN remaining hand) when the deploy value expects it. real_units
    multiplies by the score scale (PIMC wants real score; ISMCTS wants tanh)."""
    b = _to_batch([ca_obs_from_env(env)], _DEV)
    if _value_extra_dim:
        mp = min_plays_to_empty([int(c) for c in env.game.currentHands[me]]) / 13.0
        b["feat"] = torch.tensor([[mp]], dtype=torch.float32, device=_DEV)
    v = float(_value_net.forward(b)[0].item())
    return v * _VALUE_SCALE if real_units else v


class _MePerspectiveShim:
    """obs_from_env view forced to a FIXED perspective seat, even when it is no
    longer that seat's turn — needed to read a POST-action obs from my own
    perspective right after I've played (playersGo has already advanced to the
    next seat, but I want hand=my reduced hand, trick=my own just-played combo,
    opponents clockwise from me)."""
    __slots__ = ("game", "_me")

    def __init__(self, g, me):
        self.game = g
        self._me = me

    @property
    def current_player(self):
        return self._me


@torch.no_grad()
def _postaction_value_eval(env, me, real_units=False) -> float:
    """POST-action leaf: apply my greedy action, then evaluate the value of MY
    resulting position (reduced hand, trick = my own play) from my perspective —
    matching how VALUE_postaction.pt was trained (in-distribution). Called only
    when env.current_player == me (an OUR-turn leaf)."""
    a = _ca_greedy(env)
    e2 = env.clone()
    e2.step(a)
    if e2.done:                                    # my play emptied my hand / ended the game
        r = float(e2.game.rewards[me - 1])
        return r if real_units else float(np.tanh(r / _VALUE_SCALE))
    shim = _MePerspectiveShim(e2.game, me)
    b = _to_batch([ca_obs_from_env(shim)], _DEV)
    if _value_extra_dim:                           # min-plays over my POST-play hand
        mp = min_plays_to_empty([int(c) for c in e2.game.currentHands[me]]) / 13.0
        b["feat"] = torch.tensor([[mp]], dtype=torch.float32, device=_DEV)
    v = float(_value_net.forward(b)[0].item())
    return v * _VALUE_SCALE if real_units else v


class _EnvShim:
    """Minimal env view so we can reuse network_cardaware.obs_from_env(env)."""
    __slots__ = ("game",)

    def __init__(self, g):
        self.game = g

    @property
    def current_player(self):
        return self.game.playersGo


def _dummy_opp_hands(game) -> dict:
    """Valid filler hands for opponents (correct sizes, no overlap with our hand
    or played cards). cardaware never reads opponents' card identities — only
    their counts and the public played-cards — so the contents are irrelevant;
    this just lets _build_game_for_mcts construct a well-formed big2Game."""
    played = {int(c) + 1 for c in np.flatnonzero(game.cardsPlayed.any(axis=0))}
    ours = {int(c) for c in game.currentHands[1]}
    unknown = [c for c in range(1, 53) if c not in played and c not in ours]
    out, i = {}, 0
    for p in (2, 3, 4):
        n = int(game.currentHands[p].size)
        out[p] = np.array(unknown[i:i + n], dtype=np.int64)
        i += n
    return out


def _apply_trick(bg, obs):
    """obs_from_env reads the active trick from handsPlayed[goIndex-1], which the
    online-reconstructed game doesn't always populate. The vision obs is
    authoritative — set it explicitly when following."""
    if bg.control == 0:
        lp = obs.get("constraint", {}).get("last_played_cards", [])
        trick_ids = sorted(abw._bv_to_id(c["code"]) for c in lp)
        if trick_ids:
            bg.handsPlayed = dict(bg.handsPlayed)
            bg.handsPlayed[bg.goIndex - 1] = abw.handPlayed(
                np.array(trick_ids, dtype=np.int64), 0)
    return bg


def _wrap(bg) -> Big2Env:
    e = Big2Env.__new__(Big2Env)
    e._game = bg
    e._done = bool(bg.gameOver)
    return e


_DEBUG = os.environ.get("CARDAWARE_DEBUG") == "1"
_PROBE = os.environ.get("CARDAWARE_PROBE", "/tmp/cw_probe.txt")


def _probe(stage, env=None, extra=""):
    """Overwrite a single-line file with the op about to run, so when the process
    is SIGKILL'd the file shows exactly what was executing."""
    if not _DEBUG:
        return
    try:
        s = stage
        if env is not None:
            g = env.game
            p = g.playersGo
            s += (f" p={p} ctrl={g.control} "
                  f"hand={sorted(int(c) for c in g.currentHands[p])}")
        s += " " + extra
        with open(_PROBE, "w") as f:
            f.write(s)
            f.flush()
    except Exception:
        pass


@torch.no_grad()
def _ca_greedy(env) -> int:
    """cardaware greedy action for whichever seat is to move (rollout policy)."""
    _probe("GETVALID", env)
    mask = env.get_valid_actions()
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return PASS_IDX
    st = ca_obs_from_env(env)
    obs_t = _to_batch([st], _DEV)
    feats = torch.from_numpy(ACTION_FEATURES[legal]).unsqueeze(0).to(_DEV)
    am = torch.ones(1, legal.size, dtype=torch.bool, device=_DEV)
    logits, _ = _net.forward(obs_t, feats, am)
    return int(legal[int(torch.argmax(logits[0]).item())])


@torch.no_grad()
def _rollout(env) -> np.ndarray:
    steps = 0
    while not env.done and steps < 160:
        a = _ca_greedy(env)
        _probe("STEP", env, f"action={a} step={steps}")
        env.step(a)
        steps += 1
    return env.game.rewards


@torch.no_grad()
def _value_leaf(env, me) -> float:
    """Leaf evaluation with the public-info VALUE.pt instead of a full rollout:
    greedy-play opponents until it is `me`'s turn again (or terminal), then read the
    value net from `me`'s perspective. Returns `me`'s expected final score (real
    units), directly comparable to a terminal reward."""
    steps = 0
    while not env.done and env.current_player != me and steps < 60:
        env.step(_ca_greedy(env))
        steps += 1
    if env.done:
        return float(env.game.rewards[me - 1])
    return _value_eval(env, me, real_units=True)


def _random_opp_hands(game) -> dict:
    """A random opponent-hand world consistent with public info (correct sizes,
    drawn from the unseen cards)."""
    played = {int(c) + 1 for c in np.flatnonzero(game.cardsPlayed.any(axis=0))}
    ours = {int(c) for c in game.currentHands[1]}
    unknown = [c for c in range(1, 53) if c not in played and c not in ours]
    np.random.shuffle(unknown)
    out, i = {}, 0
    for p in (2, 3, 4):
        n = int(game.currentHands[p].size)
        out[p] = np.array(unknown[i:i + n], dtype=np.int64)
        i += n
    return out


@torch.no_grad()
def _belief_from_state(st, game=None) -> np.ndarray:
    """(3,52) P(opp i holds card c), i -> players 2,3,4, masked to unseen cards.
    History belief additionally reads the play/pass sequence from game.actionHistory
    (same encoding the history policy already uses)."""
    if not _BELIEF_HISTORY:
        return belief_from_obs(_belief_net, st, _DEV)
    hist_np = encode_history_steps_v2(steps_from_action_history(game.actionHistory))
    ht = torch.from_numpy(hist_np).unsqueeze(0).to(_DEV)
    with torch.no_grad():
        logits = _belief_net(_to_batch([st], _DEV), ht)
    return torch.sigmoid(logits)[0].cpu().numpy() * unseen_mask_from_obs(st)


def _belief_opp_hands(game, belief) -> dict:
    """A determinized world drawn from the belief: assign each unknown card to an
    opponent ~ P(opp holds it), respecting exact counts (importance sampling vs the
    uniform _random_opp_hands). belief[i] corresponds to player i+2."""
    played = {int(c) + 1 for c in np.flatnonzero(game.cardsPlayed.any(axis=0))}
    ours = {int(c) for c in game.currentHands[1]}
    unknown = [c for c in range(1, 53) if c not in played and c not in ours]
    np.random.shuffle(unknown)  # randomize tie-breaks; weights do the biasing
    remaining = {p: int(game.currentHands[p].size) for p in (2, 3, 4)}
    out = {2: [], 3: [], 4: []}
    for c in unknown:
        ps = [p for p in (2, 3, 4) if remaining[p] > 0]
        if not ps:
            break
        w = np.array([belief[p - 2][c - 1] for p in ps], dtype=np.float64) + 1e-6
        w /= w.sum()
        p = ps[int(np.random.choice(len(ps), p=w))]
        out[p].append(c); remaining[p] -= 1
    return {p: np.array(out[p], dtype=np.int64) for p in (2, 3, 4)}


def _search_action(game, obs, legal, probs, belief=None):
    """PIMC: for top-M policy candidates, average self-score over W determinized
    rollouts; return (best_action, extra). belief != None -> belief-guided worlds."""
    me = game.playersGo
    order = sorted(range(legal.size), key=lambda i: -float(probs[i]))
    cand = [int(legal[i]) for i in order[:_SEARCH_TOPM]]
    if bool(legal.size) and PASS_IDX in legal.tolist() and PASS_IDX not in cand:
        cand.append(PASS_IDX)
    if len(cand) == 1:
        return cand[0], {"worlds": 0, "q": {}}

    Q = {a: 0.0 for a in cand}
    N = {a: 0 for a in cand}
    t0 = time.time()
    worlds = 0
    while worlds < _SEARCH_WORLDS and (time.time() - t0) < _SEARCH_BUDGET:
        world = _belief_opp_hands(game, belief) if belief is not None else _random_opp_hands(game)
        base = _apply_trick(abw._build_game_for_mcts(game, world), obs)
        # Engine bug guard: a reconstructed world can start with passCount==3
        # (vision saw 3 passes). big2Game.updateGame only resets on passCount==3,
        # so a PASS from there pushes it to 4 → the "skip passed players" while-loop
        # spins forever (the SIGKILL we hit). Normalize the pass bookkeeping so
        # rollouts never reach passCount==4. Scoped to the rollout; engine untouched.
        if base.passCount >= 3:
            base.passCount = 0
            base.passedThisRound = {1: False, 2: False, 3: False, 4: False}
        for a in cand:
            try:
                env = _wrap(base.clone())
                env.step(a)
                Q[a] += _value_leaf(env, me) if _VALUE_LEAF else float(_rollout(env)[me - 1])
                N[a] += 1
            except Exception:
                pass
        del base
        worlds += 1
        if worlds % 8 == 0:
            gc.collect()
    gc.collect()

    scored = {a: (Q[a] / N[a] if N[a] else -1e9) for a in cand}
    best = max(scored, key=scored.get)
    return best, {"worlds": worlds, "q": {int(a): round(scored[a], 2) for a in cand}}


# ── True ISMCTS (single information-set tree, re-determinized per simulation) ──
@torch.no_grad()
def _policy_eval(env):
    """Policy net: (legal_idx, softmax probs) for the to-move player's legal actions."""
    mask = env.get_valid_actions()
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return np.array([PASS_IDX]), np.array([1.0], dtype=np.float32)
    obs_t = _to_batch([ca_obs_from_env(env)], _DEV)
    feats = torch.from_numpy(ACTION_FEATURES[legal]).unsqueeze(0).to(_DEV)
    am = torch.ones(1, legal.size, dtype=torch.bool, device=_DEV)
    logits, _ = _net.forward(obs_t, feats, am)
    return legal, torch.softmax(logits[0], dim=-1).cpu().numpy()


@torch.no_grad()
def _leaf_value(env, me) -> float:
    """Normalized (tanh, [-1,1]) expected final score for `me` at this OUR-turn leaf."""
    if _value_net is not None:
        if _POSTACTION_VALUE:
            return _postaction_value_eval(env, me, real_units=False)
        return _value_eval(env, me, real_units=False)
    return float(np.tanh(np.asarray(_rollout(env))[me - 1] / _VALUE_SCALE))  # fallback: rollout


def _ismcts_simulate(env, tree, me, allowed=None, root_noise=None):
    """One simulation on a determinized world. Tree nodes keyed by OUR action-path;
    opponents play forward via the policy (environment); our nodes PUCT-select with
    the policy prior; the leaf is value-net evaluated; value backed up to OUR edges.
    `allowed` (a set of action indices) restricts OUR ROOT move to the house-rule-legal
    set (e.g. the one-card rule) — the engine's valid moves don't know that rule."""
    key = (); path = []; value = 0.0
    while True:
        if env.done:
            value = float(np.tanh(env.game.rewards[me - 1] / _VALUE_SCALE)); break
        if env.current_player != me:
            env.step(_ca_greedy(env)); continue            # opponent = environment
        legal_now, probs = _policy_eval(env)               # our legal + priors in THIS world
        if not key and allowed is not None:                # ROOT: enforce house-rule-legal set
            km = np.array([int(a) in allowed for a in legal_now], dtype=bool)
            if km.any():
                legal_now, probs = legal_now[km], probs[km]
        if key not in tree:                                # new info-set node = leaf → expand+eval
            tree[key] = {"N": {}, "W": {}, "avail": {}}
            value = _leaf_value(env, me); break
        node = tree[key]
        if not key and root_noise is not None:             # ROOT: blend Dirichlet into the prior
            probs = np.array([(1 - _ISMCTS_NOISE_EPS) * float(probs[j])
                              + _ISMCTS_NOISE_EPS * root_noise.get(int(a), 0.0)
                              for j, a in enumerate(legal_now)], dtype=np.float32)
        if not key and _ISMCTS_MIN_VISITS > 0:             # ROOT: min-visit floor before PUCT
            starved = [int(a) for a in legal_now if node["N"].get(int(a), 0) < _ISMCTS_MIN_VISITS]
            if starved:
                best_a = min(starved, key=lambda a: node["N"].get(a, 0))
                for a in legal_now:
                    a = int(a); node["avail"][a] = node["avail"].get(a, 0) + 1
                path.append((key, best_a)); env.step(best_a); key = key + (best_a,)
                continue
        sqrt_av = max(sum(node["avail"].get(int(a), 0) for a in legal_now), 1) ** 0.5
        best_a, best_s = int(legal_now[0]), -1e18
        for j, a in enumerate(legal_now):
            a = int(a); N = node["N"].get(a, 0)
            Q = (node["W"][a] / N) if N > 0 else 0.0
            s = Q + _ISMCTS_CPUCT * float(probs[j]) * sqrt_av / (1 + N)
            if s > best_s:
                best_s, best_a = s, a
        for a in legal_now:                                # availability counts (Cowling ISMCTS)
            a = int(a); node["avail"][a] = node["avail"].get(a, 0) + 1
        path.append((key, best_a))
        env.step(best_a)
        key = key + (best_a,)
    for k, a in path:                                      # backup OUR value along OUR edges
        nd = tree[k]
        nd["N"][a] = nd["N"].get(a, 0) + 1
        nd["W"][a] = nd["W"].get(a, 0.0) + value


def _ismcts_action(game, obs, belief, allowed=None):
    """ISMCTS root: re-determinize from belief each simulation, share one tree,
    return (best_action, info). best = most-visited OUR root action. `allowed`
    restricts the root move to the house-rule-legal set (one-card rule etc.)."""
    me = game.playersGo
    tree = {}
    # Root Dirichlet noise (de-anchor): sample ONCE per decision over the root legal set.
    # Root legality depends only on OUR hand + the table trick (fixed across determinizations),
    # so a single dummy determinization gives the correct root action set.
    root_noise = None
    if _ISMCTS_ROOT_NOISE:
        w0 = _belief_opp_hands(game, belief) if belief is not None else _random_opp_hands(game)
        b0 = _apply_trick(abw._build_game_for_mcts(game, w0), obs)
        legal0, _ = _policy_eval(_wrap(b0.clone()))
        legal0 = [int(a) for a in legal0 if allowed is None or int(a) in allowed]
        if legal0:
            noise = np.random.dirichlet([_ISMCTS_NOISE_ALPHA] * len(legal0))
            root_noise = {a: float(n) for a, n in zip(legal0, noise)}
    t0 = time.time(); sims = 0
    while sims < _ISMCTS_SIMS and (time.time() - t0) < _SEARCH_BUDGET:
        world = _belief_opp_hands(game, belief) if belief is not None else _random_opp_hands(game)
        base = _apply_trick(abw._build_game_for_mcts(game, world), obs)
        if base.passCount >= 3:                            # same engine-loop guard as PIMC
            base.passCount = 0
            base.passedThisRound = {1: False, 2: False, 3: False, 4: False}
        try:
            _ismcts_simulate(_wrap(base.clone()), tree, me, allowed, root_noise)
        except Exception:
            pass
        sims += 1
        if sims % 8 == 0:
            gc.collect()
    gc.collect()
    root = tree.get((), {"N": {}, "W": {}})
    visits = {a: n for a, n in root["N"].items() if allowed is None or int(a) in allowed}
    if not visits:
        return None, {"sims": sims, "visits": {}}
    best = max(visits, key=visits.get)
    q = {int(a): round(root["W"][a] / visits[a], 3) for a in visits}
    return best, {"sims": sims, "visits": {int(a): visits[a] for a in visits}, "q": q}


def _infer_cardaware(game, obs):
    """Return (action_idx, note, extra)."""
    mask = abw._build_mask(obs)
    mask = abw._apply_must_play_club3(mask, game)
    mask = abw._apply_one_card_rule(obs, mask, game)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return PASS_IDX, "cardaware:no-legal->pass", {}

    bg = _apply_trick(abw._build_game_for_mcts(game, _dummy_opp_hands(game)), obs)
    st = ca_obs_from_env(_EnvShim(bg))
    obs_t = _to_batch([st], _DEV)
    feats = torch.from_numpy(ACTION_FEATURES[legal]).unsqueeze(0).to(_DEV)
    am = torch.ones(1, legal.size, dtype=torch.bool, device=_DEV)
    with torch.no_grad():
        if _HISTORY_POLICY:
            # game.actionHistory (MockGame) uses the exact same dict schema as
            # the real engine's big2Game.actionHistory -- steps_from_action_history
            # works unmodified on either source.
            steps = steps_from_action_history(game.actionHistory)
            hist_np = encode_history_steps_v2(steps)
            hist_t = torch.from_numpy(hist_np).unsqueeze(0).to(_DEV)
            logits, value = _net.forward(obs_t, hist_t, feats, am)
        else:
            logits, value = _net.forward(obs_t, feats, am)
        probs = torch.softmax(logits[0], dim=-1)
        pos = int(torch.argmax(logits[0]).item())
    action_idx = int(legal[pos])
    # For the dashboard: policy% per legal action, + a belief-based opponent-hand
    # estimate. belief is ONE BeliefNet forward (cheap) — not the 120x Monte Carlo
    # resampling in alpha_big2_wrapper._estimate_belief that was the suspected
    # SIGKILL cause; computed once here and reused by whichever branch runs below.
    policy_map = {int(legal[i]): float(probs[i]) for i in range(legal.size)}
    belief = _belief_from_state(st, game) if _belief_net is not None else None

    _legal_set = set(int(a) for a in legal)   # house-rule-legal (incl. one-card rule)
    if _ISMCTS and legal.size > 1:
        t0 = time.time()
        best, sx = _ismcts_action(game, obs, belief, _legal_set)
        if best is not None and int(best) in _legal_set:   # invariant: never play outside legal
            action_idx = best
        note = (f"cardaware+ismcts{'+belief' if belief is not None else ''}"
                f"{'+value' if _value_net is not None else ''} sims={sx['sims']} "
                f"({time.time()-t0:.1f}s) policy_p={float(probs[pos]):.2f}")
        extra = {"ismcts": sx, "value": float(value.item()), "n_legal": int(legal.size),
                 "policy_map": policy_map, "belief": belief}
        return action_idx, note, extra

    if _SEARCH and legal.size > 1:
        t0 = time.time()
        best, sx = _search_action(game, obs, legal, probs.cpu().numpy(), belief=belief)
        action_idx = best
        note = (f"cardaware+search{'+belief' if belief is not None else ''}"
                f"{'+value' if _VALUE_LEAF else ''} "
                f"worlds={sx['worlds']} q={sx['q']} "
                f"({time.time()-t0:.1f}s) policy_p={float(probs[pos]):.2f}")
        extra = {"search": sx, "value": float(value.item()), "n_legal": int(legal.size),
                 "policy_map": policy_map, "belief": belief}
        return action_idx, note, extra

    note = f"cardaware p={float(probs[pos]):.2f} v={float(value.item()):.2f} L={legal.size}"
    extra = {"value": float(value.item()), "prob": float(probs[pos]), "n_legal": int(legal.size),
             "policy_map": policy_map, "belief": belief}
    return action_idx, note, extra


# ── Dashboard (localhost:7373) ────────────────────────────────────────────────
# Reuses the shared dashboard_writer module (players/events/constraint — identical
# to what main.py writes) but builds the "AI decision" block from data THIS wrapper
# already computed for its own decision — no extra model calls, no resampling.
# (alpha_big2_wrapper's version re-estimates belief via 120x Monte Carlo every
# decision and was the suspected cause of an earlier online SIGKILL; we avoid that
# entirely by reusing the single BeliefNet forward pass computed in _infer_cardaware.)
_dash_stats = {"decisions": 0, "plays": 0, "passes": 0, "fallbacks": 0,
               "mcts_total_sims": 0, "mcts_total_time_s": 0.0}


def _parse_note(note: str) -> dict:
    """note string -> {mode, mcts_sims, mcts_time_s} for the dashboard's mode chip."""
    info = {"mode": "unknown", "mcts_sims": None, "mcts_time_s": None}
    if not note:
        return info
    if note.startswith("cardaware+ismcts"):
        info["mode"] = "ismcts"
    elif note.startswith("cardaware+search"):
        info["mode"] = "search"
    elif note.startswith("cardaware:no-legal"):
        info["mode"] = "forced"
    elif note.startswith("fallback:"):
        info["mode"] = "fallback"
    elif note.startswith("cardaware "):
        info["mode"] = "greedy"
    m = re.search(r"(?:sims|worlds)=(\d+).*?\(([\d.]+)s\)", note)
    if m:
        info["mcts_sims"] = int(m.group(1))
        info["mcts_time_s"] = float(m.group(2))
    return info


def _opp_hands_from_belief(game, belief) -> dict:
    """Greedy opponent-hand estimate straight from the (3,52) belief matrix already
    computed for this decision — one pass over ~52 unseen cards, no sampling loop.
    belief[i] corresponds to seat i+2 (right/top/left), matching _belief_opp_hands.
    Returns {seat: [(card_id, prob), ...]}."""
    played = {int(c) + 1 for c in np.flatnonzero(game.cardsPlayed.any(axis=0))}
    ours = {int(c) for c in game.currentHands[1]}
    unseen = [c for c in range(1, 53) if c not in played and c not in ours]
    slots = {p: int(game.currentHands[p].size) for p in (2, 3, 4)}
    pairs = [(float(belief[i][c - 1]), c, p) for c in unseen for i, p in enumerate((2, 3, 4))]
    pairs.sort(key=lambda t: t[0], reverse=True)   # assign highest-probability pairs first
    assigned: dict = {}
    out = {p: [] for p in (2, 3, 4)}
    for prob, c, p in pairs:
        if c in assigned or slots[p] <= 0:
            continue
        assigned[c] = p; slots[p] -= 1
        out[p].append((c, prob))
    return out


def _scored_legal_actions(obs: dict, extra: dict, chosen_idx: int) -> list:
    """Each legal action + policy% (from this decision's policy_map) and MCTS
    visit% (from the ISMCTS/search root, when a search ran this decision)."""
    policy_map = extra.get("policy_map") or {}
    visits = ((extra.get("ismcts") or extra.get("search") or {}).get("visits")) or {}
    total_visits = sum(visits.values())
    result = []
    for la in obs.get("legal_actions", []):
        idx = abw._action_idx_from_legal(la)
        cards_sym = [abw._card_sym(c) for c in la.get("cards", [])]
        combo = la.get("combo_type") or ("pass" if la.get("action") == "pass" else None)
        combo_zh = abw._COMBO_ZH_MAP.get(combo or "", "PASS" if la.get("action") == "pass" else "")
        policy_pct = round(policy_map[idx] * 100, 1) if idx in policy_map else None
        visit_count = visits.get(idx)
        visit_pct = (round(visit_count / total_visits * 100, 1) if total_visits > 0 else 0.0) \
            if visit_count is not None else None
        result.append({
            "action": la.get("action"), "cards": cards_sym, "combo_type": combo,
            "combo_zh": combo_zh, "action_idx": idx,
            "policy_pct": policy_pct, "visits": visit_count, "visit_pct": visit_pct,
            "chosen": (idx is not None and idx == chosen_idx),
        })

    def _sort_key(x):
        is_pass = x["action"] == "pass"
        pct = x["policy_pct"] if x["policy_pct"] is not None else -1.0
        return (is_pass, -pct)
    result.sort(key=_sort_key)
    return result


def _write_dashboard(game, obs: dict, decision: dict, note: str, extra: dict) -> None:
    try:
        extra = extra or {}
        note_info = _parse_note(note)
        _dash_stats["decisions"] += 1
        _dash_stats["passes" if decision["action"] == "pass" else "plays"] += 1
        if note_info["mode"] in ("fallback", "greedy"):
            _dash_stats["fallbacks"] += 1
        if note_info["mcts_sims"] is not None:
            _dash_stats["mcts_total_sims"] += note_info["mcts_sims"]
        if note_info["mcts_time_s"] is not None:
            _dash_stats["mcts_total_time_s"] += note_info["mcts_time_s"]

        opp_hands_codes = {}
        belief = extra.get("belief")
        if belief is not None:
            for seat, items in _opp_hands_from_belief(game, belief).items():
                opp_hands_codes[seat] = [(abw._id_to_bv(cid), abw._conf_level(prob))
                                          for cid, prob in items]

        decision_cards = [abw.dashboard_writer.code_to_symbol(c) for c in decision.get("card_codes", [])]
        chosen_idx = abw._action_idx_from_legal({
            "action": decision["action"],
            "cards": [{"code": c} for c in decision.get("card_codes", [])],
            "combo_type": decision.get("combo_type"),
        }) if decision["action"] != "pass" else enumerateOptions.passInd

        state = abw.dashboard_writer.base_state(obs, opp_hands=opp_hands_codes)
        state["legal_actions"] = _scored_legal_actions(obs, extra, chosen_idx)
        state["last_decision"] = {
            "action": decision["action"], "cards": decision_cards,
            "combo_type": decision.get("combo_type"),
            "combo_zh": abw._COMBO_ZH_MAP.get(decision.get("combo_type") or "", ""),
            "note": note, **note_info,
        }
        state["session"] = dict(_dash_stats)
        abw.dashboard_writer.atomic_write(abw._DASH_STATE_PATH, state)
    except Exception as exc:
        log.debug("dashboard write failed: %s", exc)


def main() -> None:
    # Start from a blank dashboard so stale state from a previous agent isn't shown.
    try:
        os.makedirs(os.path.dirname(abw._DASH_STATE_PATH), exist_ok=True)
        with open(abw._DASH_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"__status": "waiting", "updated_at": None}, f)
    except Exception:
        pass

    game = abw.MockGame()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obs = json.loads(raw)
            game.update(obs)
            action_idx, note, extra = _infer_cardaware(game, obs)
            decision = abw._to_decision(action_idx)
            if action_idx == enumerateOptions.passInd:
                game.record_our_pass()
            else:
                cids, _ = enumerateOptions.getOptionNC(action_idx)
                game.record_our_play([int(c) for c in cids])
            decision["note"] = note
            log.info("→ %s %s (%s) [%s]", decision["action"], decision["card_codes"],
                     decision["combo_type"], note)
            print(json.dumps(decision), flush=True)
            # Our own dashboard write (NOT abw._write_dashboard_state — that one
            # re-estimates belief via 120x Monte Carlo every decision and was the
            # suspected cause of an earlier online SIGKILL). This reuses data
            # _infer_cardaware already computed, so it adds ~no cost.
            _write_dashboard(game, obs, decision, note, extra)
        except Exception as exc:
            import traceback
            short = " | ".join(traceback.format_exc().strip().splitlines()[-2:])
            log.exception("cardaware inference error — falling back to pass")
            fallback = {"action": "pass", "card_codes": [], "combo_type": None,
                        "note": f"fallback:{short}"}
            print(json.dumps(fallback), flush=True)
            try:
                _write_dashboard(game, obs, fallback, fallback["note"], {})
            except Exception:
                pass


if __name__ == "__main__":
    main()
