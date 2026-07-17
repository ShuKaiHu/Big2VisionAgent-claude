#!/usr/bin/env bash
# Upgrade the 9-cell scaling grid to 300 games/cell.
# Order: the 6 completely-unrun cells first (fresh 300), then top up the 3
# already-100 cells to 300 (+200 each, combined with their old windows in analysis).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-Value"
cd "$HERE"

run_cell () {
  local P=$1 V=$2 ALREADY=$3
  local VCK="$VAL/checkpoints/VALUE_scale_${V}.pt"
  while [ ! -f "$VCK" ]; do sleep 60; done
  echo "=== CELL p${P}_v${V}: online -> 300 (already=$ALREADY) ==="
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
      BELIEF_CKPT="$PPO/ppo/checkpoints/belief_${P}_best.pt" \
      VALUE_CKPT="$VCK" \
      ./.venv/bin/python3 run_online_leg.py \
          --tag scale_p${P}_v${V}_300 \
          --ckpt "$PPO/ppo/checkpoints/policy_${P}_best.pt" \
          --goal 300 --already $ALREADY
  echo "=== CELL p${P}_v${V} DONE (300) ==="
}

# 6 unrun cells first (fresh 300)
for V in 1500 2500; do
  for P in 500 1500 2500; do
    run_cell $P $V 0
  done
done
# top up the 3 already-100 cells to 300
for P in 500 1500 2500; do
  run_cell $P 500 100
done
echo "=== ALL 9 CELLS AT 300 DONE ==="
