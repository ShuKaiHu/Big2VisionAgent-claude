#!/usr/bin/env bash
# Online-test all 9 data-scaling cells: (policy_P + belief_P) x value_V,
# P,V in {500,1500,2500}. Each = ISMCTS w/ that policy+belief+value, 100 games.
# Sequential (one account). Ordered by value size so value_500 cells run first
# (ready), while value_1500/2500 finish training in the background.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/value"
cd "$HERE"

run_cell () {
  local P=$1 V=$2
  local VCK="$VAL/checkpoints/VALUE_scale_${V}.pt"
  echo "=== CELL p${P}_v${V}: waiting for value $VCK ==="
  while [ ! -f "$VCK" ]; do sleep 60; done
  sleep 5
  echo "=== CELL p${P}_v${V}: online 100 (policy_$P + belief_$P + VALUE_scale_$V + ISMCTS) ==="
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
      BELIEF_CKPT="$PPO/ppo/checkpoints/belief_${P}_best.pt" \
      VALUE_CKPT="$VCK" \
      ./.venv/bin/python3 run_online_leg.py \
          --tag scale_p${P}_v${V} \
          --ckpt "$PPO/ppo/checkpoints/policy_${P}_best.pt" \
          --goal 100
  echo "=== CELL p${P}_v${V} DONE ==="
}

# value_500 group (ready) -> value_1500 group -> value_2500 group
for V in 500 1500 2500; do
  for P in 500 1500 2500; do
    run_cell $P $V
  done
done
echo "=== ALL 9 SCALING CELLS DONE ==="
