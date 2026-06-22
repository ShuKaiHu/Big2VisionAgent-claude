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
from ppo.action_features import ACTION_FEATURES
from ppo.network_v6 import CardAwareV6, unseen_mask_from_obs
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

# Belief-guided search (the V6 idea): instead of dealing opponents' unknown cards
# UNIFORMLY at random for each PIMC world, importance-sample them from the V6
# belief head P(opp holds card) — ~85% accurate on the human/online distribution.
# The search then evaluates LIKELIER worlds. Policy/value/rollouts stay = _net.
_BELIEF_SEARCH = os.environ.get("BELIEF_SEARCH") == "1"
_BELIEF_CKPT = os.environ.get(
    "BELIEF_CKPT", os.path.join(_AB2PPO, "ppo", "checkpoints", "saved", "PPO_V6.pt"))

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
_net = CardAwareActorCritic().to(_DEV)
_net.load_state_dict(_ck["model"])
_net.eval()
log.info("cardaware loaded: %s (arch=%s upd=%s)", os.path.basename(_CKPT),
         _ck.get("arch"), _ck.get("update"))

_belief_net = None
if _BELIEF_SEARCH:
    _bk = torch.load(_BELIEF_CKPT, map_location=_DEV, weights_only=False)
    _belief_net = CardAwareV6().to(_DEV)
    _belief_net.load_state_dict(_bk["model"])
    _belief_net.eval()
    log.info("belief net loaded: %s (belief_prec=%s) — belief-guided search ON",
             os.path.basename(_BELIEF_CKPT), _bk.get("belief_prec"))


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
def _belief_from_state(st) -> np.ndarray:
    """(3,52) P(opp i holds card c), i -> players 2,3,4, masked to unseen cards."""
    S = _belief_net.state_embedding(_to_batch([st], _DEV))
    bel = torch.sigmoid(_belief_net.belief_head(S)).reshape(3, 52).cpu().numpy()
    return bel * unseen_mask_from_obs(st)


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
                r = _rollout(env)
                Q[a] += float(r[me - 1])
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


def _infer_cardaware(game, obs):
    """Return (action_idx, note, extra)."""
    mask = abw._build_mask(obs)
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
        logits, value = _net.forward(obs_t, feats, am)
        probs = torch.softmax(logits[0], dim=-1)
        pos = int(torch.argmax(logits[0]).item())
    action_idx = int(legal[pos])

    if _SEARCH and legal.size > 1:
        t0 = time.time()
        belief = _belief_from_state(st) if _belief_net is not None else None
        best, sx = _search_action(game, obs, legal, probs.cpu().numpy(), belief=belief)
        action_idx = best
        note = (f"cardaware+search{'+belief' if belief is not None else ''} "
                f"worlds={sx['worlds']} q={sx['q']} "
                f"({time.time()-t0:.1f}s) policy_p={float(probs[pos]):.2f}")
        extra = {"search": sx, "value": float(value.item()), "n_legal": int(legal.size)}
        return action_idx, note, extra

    note = f"cardaware p={float(probs[pos]):.2f} v={float(value.item()):.2f} L={legal.size}"
    extra = {"value": float(value.item()), "prob": float(probs[pos]), "n_legal": int(legal.size)}
    return action_idx, note, extra


def main() -> None:
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
            # NOTE: abw._write_dashboard_state intentionally NOT called — it is
            # MCTS-shaped, unused by cardaware, and was the prime suspect for the
            # wrapper getting SIGKILL'd (-9) after a few search moves online.
        except Exception as exc:
            import traceback
            short = " | ".join(traceback.format_exc().strip().splitlines()[-2:])
            log.exception("cardaware inference error — falling back to pass")
            print(json.dumps({"action": "pass", "card_codes": [], "combo_type": None,
                              "note": f"fallback:{short}"}), flush=True)


if __name__ == "__main__":
    main()
