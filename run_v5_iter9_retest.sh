#!/usr/bin/env bash
# Retest the two historical value-guided bests under the FIXED one-card rule
# (2026-07-08 engine + vision-layer fix), same logic as policy_all -> policy_all_v2.
# One 神來也 account => sequential legs. Each leg = 100 real games via run_online_leg.py.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-Value"
cd "$HERE"

# ── Leg 1: V5 = PPO_V4 policy + flat PIMC search (worlds=24, budget 1s) ──
echo "=== LEG 1: V5_v2 (PPO_V4 + CARDAWARE_SEARCH flat PIMC) ==="
env -u ISMCTS -u BELIEF_SEARCH -u VALUE_LEAF -u VALUE_CKPT -u ISMCTS_SIMS \
    CARDAWARE_SEARCH=1 CARDAWARE_WORLDS=24 CARDAWARE_BUDGET=1.0 \
    ./.venv/bin/python3 run_online_leg.py \
        --tag V5_v2 \
        --ckpt "$PPO/ppo/checkpoints/saved/PPO_V4.pt" \
        --goal 100
echo "=== LEG 1 DONE: V5_v2 ==="

# ── Leg 2: V4PBV_Iter9 = policy + value + belief + ISMCTS (sims=200, budget 1s) ──
echo "=== LEG 2: V4PBV_Iter9_v2 (3-model ISMCTS) ==="
env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS \
    ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
    VALUE_CKPT="$VAL/checkpoints/V4PBV_Iter9_value.pt" \
    ./.venv/bin/python3 run_online_leg.py \
        --tag V4PBV_Iter9_v2 \
        --ckpt "$VAL/checkpoints/V4PBV_Iter9_policy.pt" \
        --goal 100
echo "=== LEG 2 DONE: V4PBV_Iter9_v2 ==="
echo "=== BOTH RETESTS DONE ==="
