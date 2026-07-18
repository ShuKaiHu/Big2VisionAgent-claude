#!/usr/bin/env bash
# Finish the 9-cell scaling grid: fill the last 4 cells to 300 games each.
#
# Remaining after the interrupted run_scale_online_300.sh:
#   p2500_v2500  36 -> 300   (the cell that was mid-flight when we stopped)
#   p500_v500   100 -> 300
#   p1500_v500  100 -> 300
#   p2500_v500  100 -> 300
#
# Also writes MACHINE-READABLE cell boundaries to scale_cell_bounds.jsonl.
# The old analysis inferred cell start times from "Saved artifacts to ..." lines,
# but those are printed at the END of each launch -- so a cell's first launch got
# credited to the PREVIOUS cell (that's why p1500_v2500 read n=336 and
# p2500_v2500 read n=2). These explicit start/end stamps make attribution exact.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/value"
BOUNDS="$HERE/scale_cell_bounds.jsonl"
cd "$HERE"

# LOCAL time on purpose: reward_log.jsonl timestamps come from datetime.now(),
# so a UTC stamp here would sit 8h off and mis-slice every window.
stamp () { date +%Y-%m-%dT%H:%M:%S; }

run_cell () {
  local P=$1 V=$2 ALREADY=$3
  local VCK="$VAL/checkpoints/VALUE_scale_${V}.pt"
  local CELL="p${P}_v${V}"
  echo "{\"cell\":\"$CELL\",\"event\":\"start\",\"already\":$ALREADY,\"ts\":\"$(stamp)\"}" >> "$BOUNDS"
  echo "=== CELL ${CELL}: online -> 300 (already=$ALREADY) ==="
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
      BELIEF_CKPT="$PPO/ppo/checkpoints/belief_${P}_best.pt" \
      VALUE_CKPT="$VCK" \
      ./.venv/bin/python3 run_online_leg.py \
          --tag scale_${CELL}_300 \
          --ckpt "$PPO/ppo/checkpoints/policy_${P}_best.pt" \
          --goal 300 --already "$ALREADY"
  echo "{\"cell\":\"$CELL\",\"event\":\"end\",\"ts\":\"$(stamp)\"}" >> "$BOUNDS"
  echo "=== CELL ${CELL} DONE (300) ==="
}

# finish the interrupted cell first, then the three v500 top-ups
run_cell 2500 2500 44
run_cell 500  500  100
run_cell 1500 500  100
run_cell 2500 500  100

echo "{\"event\":\"all_done\",\"ts\":\"$(stamp)\"}" >> "$BOUNDS"
echo "=== ALL 9 CELLS AT 300 DONE ==="
