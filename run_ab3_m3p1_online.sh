#!/usr/bin/env bash
# AB3: does the RL policy's offline stop-loss gain survive real humans?
#
#   arm A  ab3_m3p1_200   ppo_m3p1_200.pt  (anchored PPO, 200 updates)
#   arm B  ab3_m3p1_400   ppo_m3p1_400.pt  (same run continued to 400 updates)
#
# Both PURE POLICY (no ISMCTS/belief/value). 300 games each, interleaved 50-game
# legs. Contemporaneous reference = AB2's fresh ctrl arm (policy_4500, 300 games,
# finished hours earlier on the same platform).
#
# PRE-REGISTERED READING (before any data):
#   PRIMARY (powered at n=300/arm): the canonical last-chance 2-dump statistic
#   (P(play 2 | seat's LAST legal-2-play opportunity), actual-behavior version
#   computed from each arm's own action_log trajectories with the same engine
#   tables as probe_two_decisions --mode last-chance). Offline counterfactual
#   read: policy_4500 65.7% / m3p1_200 72.9% / m3p1_400 TBD; human anchor 85.7%.
#   Also: lead-5card rate, caged rate, disaster rate.
#   SECONDARY (NOT powered for <4.5 pts): avg_score with CI — non-inferiority
#   read vs AB2 ctrl; do not rank on it.
#   Exclusions: mechanistic only (bomb-dealt games via canonical detector,
#   no-choice games, (run_dir, game_uid) join failures).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
cd "$HERE"

A="$R/ppo/checkpoints/ppo_m3p1_200.pt"
B="$R/ppo/checkpoints/ppo_m3p1_400.pt"
GOAL=300
LEG=50

for CK in "$A" "$B"; do
  [ -f "$CK" ] || { echo "FATAL: missing checkpoint $CK"; exit 1; }
done

count () { grep -c "\"run_tag\": \"$1\"" artifacts/reward_log.jsonl 2>/dev/null || echo 0; }

leg () {   # $1 tag  $2 ckpt  $3 target — pure policy
  local HAVE; HAVE=$(count "$1")
  [ "$HAVE" -ge "$3" ] && { echo "[$1] already $HAVE/$3 — skip"; return; }
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      -u ISMCTS -u ISMCTS_SIMS -u BELIEF_SEARCH -u VALUE_LEAF \
      -u BELIEF_CKPT -u VALUE_CKPT \
      CARDAWARE_BUDGET=1.0 \
      ./.venv/bin/python3 run_online_leg.py \
          --tag "$1" --ckpt "$2" --goal "$3" --already "$HAVE"
}

for TARGET in $(seq $LEG $LEG $GOAL); do
  echo "=== ROUND -> $TARGET ==="
  leg ab3_m3p1_200 "$A" "$TARGET"
  leg ab3_m3p1_400 "$B" "$TARGET"
done
echo "=== AB3 COMPLETE: $GOAL x 2 arms ==="
