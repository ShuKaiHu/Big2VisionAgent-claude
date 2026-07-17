#!/usr/bin/env bash
# Wait for the in-flight online test (PID $1) to finish, then run two ISMCTS legs
# sequentially (one 神來也 account):
#   1. V4PBV_postaction_v1 — the post-action-value experiment (active priority)
#   2. V4PBV_base_v2        — the un-iterated base baseline (secondary)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-Value"
cd "$HERE"
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
    echo "=== waiting for in-flight online test (pid $WAIT_PID) ==="
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "=== in-flight test done ==="
fi

# ── Leg 1: post-action value ISMCTS (single-variable swap vs V4PBV_Iter9_v2) ──
# policy=V4PBV_Iter9_policy, belief=BELIEF.pt, ISMCTS sims=200 — ALL identical to
# Iter9_v2; ONLY value changes: V4PBV_Iter9_value(pre) -> VALUE_postaction(post)
# + POSTACTION_VALUE=1 (leaf evaluates post-action state).
echo "=== LEG 1: V4PBV_postaction_v1 (post-action value ISMCTS) ==="
env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS \
    ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 POSTACTION_VALUE=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
    VALUE_CKPT="$VAL/checkpoints/VALUE_postaction.pt" \
    ./.venv/bin/python3 run_online_leg.py \
        --tag V4PBV_postaction_v1 \
        --ckpt "$VAL/checkpoints/V4PBV_Iter9_policy.pt" \
        --goal 100
echo "=== LEG 1 DONE: V4PBV_postaction_v1 ==="

# ── Leg 2: V4PBV base = PPO_V4 + BELIEF + VALUE_minplays + ISMCTS (secondary) ──
echo "=== LEG 2: V4PBV_base_v2 (base 3-model ISMCTS) ==="
env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
    ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
    VALUE_CKPT="$VAL/checkpoints/VALUE_minplays.pt" \
    ./.venv/bin/python3 run_online_leg.py \
        --tag V4PBV_base_v2 \
        --ckpt "$PPO/ppo/checkpoints/saved/PPO_V4.pt" \
        --goal 100
echo "=== LEG 2 DONE: V4PBV_base_v2 ==="
echo "=== BOTH DONE ==="
