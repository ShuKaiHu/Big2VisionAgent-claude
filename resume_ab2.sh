#!/usr/bin/env bash
# Resume the AB2 3-arm campaign from wherever each arm stands (pause-safe).
# Reads live counts from reward_log and tops arms up round-by-round (50-game legs,
# interleaved C,L,B) to 300 each. Checkpoints live in the consolidated repo.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
cd "$HERE"

CTRL="$PPO/ppo/checkpoints/policy_4500_best.pt"
LAST="$PPO/ppo/checkpoints/awbc_T40_last.pt"
BEST="$PPO/ppo/checkpoints/awbc_T40_best.pt"
GOAL=300

count () { grep -c "\"run_tag\": \"$1\"" artifacts/reward_log.jsonl 2>/dev/null || echo 0; }

leg () {   # $1 tag  $2 ckpt  $3 target
  local HAVE; HAVE=$(count "$1")
  [ "$HAVE" -ge "$3" ] && { echo "[$1] already $HAVE/$3 — skip"; return; }
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      -u ISMCTS -u ISMCTS_SIMS -u BELIEF_SEARCH -u VALUE_LEAF \
      -u BELIEF_CKPT -u VALUE_CKPT \
      CARDAWARE_BUDGET=1.0 \
      ./.venv/bin/python3 run_online_leg.py \
          --tag "$1" --ckpt "$2" --goal "$3" --already "$HAVE"
}

for TARGET in 200 250 300; do
  echo "=== ROUND -> $TARGET ==="
  leg ab2_ctrl_p4500   "$CTRL" "$TARGET"
  leg ab2_awbc_T40last "$LAST" "$TARGET"
  leg ab2_awbc_T40best "$BEST" "$TARGET"
done
echo "=== AB2 COMPLETE: $GOAL x 3 arms ==="
