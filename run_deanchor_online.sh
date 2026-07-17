#!/usr/bin/env bash
# DE-ANCHOR online test (300 games).
# Same strongest stack as best_p4500_bhist_vmp:
#   P = policy_4500   B = belief_hist_sc4500   V = VALUE_minplays   + ISMCTS (200 sims)
# DIFFERENCE: root Dirichlet noise (eps=0.25, alpha=0.3) + min-visit floor M=2 ENABLED.
#
# Baseline to beat (plain ISMCTS, same stack):  avg_score = -4.31  (299 games)
# Local fixed-deal test said de-anchor is WORSE vs greedy opponents; this run is the
# real-human transfer check the user asked for. Attribution via tag=deanchor_dir_floor2.
set -uo pipefail
cd /Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude
echo "=== DE-ANCHOR (Dirichlet eps0.25 + floor M=2) 300 games START ==="
env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
    ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
    ISMCTS_ROOT_NOISE=1 ISMCTS_NOISE_EPS=0.25 ISMCTS_NOISE_ALPHA=0.3 ISMCTS_MIN_VISITS=2 \
    BELIEF_CKPT="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo/ppo/checkpoints/belief_hist_sc4500_best.pt" \
    VALUE_CKPT="/Users/shukaihu/Code_Project_Local/AlphaBig2-Value/checkpoints/VALUE_minplays.pt" \
    ./.venv/bin/python3 run_online_leg.py \
        --tag deanchor_dir_floor2 \
        --ckpt "/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo/ppo/checkpoints/policy_4500_best.pt" \
        --goal 300 --already 0
echo "=== DE-ANCHOR 300 DONE ==="
