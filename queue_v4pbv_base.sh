#!/usr/bin/env bash
# Wait for the in-flight archpool_gen1_full online test (PID passed as $1) to
# finish, then run V4PBV base = PPO_V4 + BELIEF + VALUE_minplays + ISMCTS(200)
# under the fixed rule, 100 real games. One 神來也 account => must be sequential.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-Value"
cd "$HERE"
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
    echo "=== waiting for in-flight online test (pid $WAIT_PID) to finish ==="
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "=== in-flight test done; starting V4PBV_base_v2 ==="
fi

env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS \
    ISMCTS=1 BELIEF_SEARCH=1 VALUE_LEAF=1 ISMCTS_SIMS=200 CARDAWARE_BUDGET=1.0 \
    VALUE_CKPT="$VAL/checkpoints/VALUE_minplays.pt" \
    ./.venv/bin/python3 run_online_leg.py \
        --tag V4PBV_base_v2 \
        --ckpt "$PPO/ppo/checkpoints/saved/PPO_V4.pt" \
        --goal 100
echo "=== V4PBV_base_v2 DONE ==="
