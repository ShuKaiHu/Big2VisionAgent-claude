#!/usr/bin/env bash
# Online-test the 3 diagonal data-scaling cells (500/1500/2500), each =
# policy_N + belief_N + VALUE_scale_N + ISMCTS. Sequential (one account).
# Waits for each cell's value checkpoint before its leg (1500/2500 train in bg).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/value"
cd "$HERE"

run_cell () {
  local N=$1
  local VCK="$VAL/checkpoints/VALUE_scale_${N}.pt"
  echo "=== CELL $N: waiting for value checkpoint $VCK ==="
  while [ ! -f "$VCK" ]; do sleep 60; done
  # give it a moment to finish writing
  sleep 5
  echo "=== CELL $N: online 100 (policy_$N + belief_$N + VALUE_scale_$N + ISMCTS) ==="
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
      BELIEF_CKPT="$PPO/ppo/checkpoints/belief_${N}_best.pt" \
      VALUE_CKPT="$VCK" \
      ./.venv/bin/python3 run_online_leg.py \
          --tag scale_${N}_v1 \
          --ckpt "$PPO/ppo/checkpoints/policy_${N}_best.pt" \
          --goal 100
  echo "=== CELL $N DONE ==="
}

run_cell 500
run_cell 1500
run_cell 2500
echo "=== ALL 3 SCALING CELLS DONE ==="
