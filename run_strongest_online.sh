#!/usr/bin/env bash
# Strongest assembled model, 300 online games:
#   P = policy_4500 (clean human BC)   B = belief_hist_sc4500 (clean history belief)
#   V = VALUE_minplays (proven RL-generator value)   + ISMCTS
# Waits for the grid finish-chain to end first (one browser at a time).
set -uo pipefail
cd /Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude
echo "=== 等 grid finish chain(pid 92275)結束... ==="
while kill -0 92275 2>/dev/null; do sleep 120; done
echo "=== grid 結束,開始最強組合 300 場 (Wed Jul 15 23:47:16 CST 2026) ==="
env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE     ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0     BELIEF_CKPT="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/ppo/checkpoints/belief_hist_sc4500_best.pt"     VALUE_CKPT="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/value/checkpoints/VALUE_minplays.pt"     ./.venv/bin/python3 run_online_leg.py         --tag best_p4500_bhist_vmp         --ckpt "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/ppo/checkpoints/policy_4500_best.pt"         --goal 300 --already 0
echo "=== STRONGEST 300 DONE (Wed Jul 15 23:47:16 CST 2026) ==="
