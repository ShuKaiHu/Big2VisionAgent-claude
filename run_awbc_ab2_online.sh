#!/usr/bin/env bash
# AB2: does AW-BC's human-like fingerprint TRANSFER to real humans online?
#
#   arm C  ab2_ctrl_p4500     policy_4500_best  (incumbent control)
#   arm L  ab2_awbc_T40last   awbc_T40_last     (strongest fingerprints: lead5 32.1%,
#                                                hoard 42.4%, humanness 85.1%, but -1.24
#                                                paired vs incumbent OFFLINE)
#   arm B  ab2_awbc_T40best   awbc_T40_best     (score-selected: +0.40 vs incumbent,
#                                                fingerprints unmoved)
#
# All three arms are PURE POLICY (no ISMCTS/belief/value) — M1 scope. 300 games each,
# interleaved C,L,B in 50-game legs so opponent-pool drift hits all arms equally.
#
# WHY THIS TEST EXISTS (decided after the M1 epoch sweep): offline training traces a
# MONOTONIC frontier — every extra epoch is more human-like AND weaker vs the bot
# incumbent; there is no offline sweet spot, and bot-incumbent score is known not to
# transfer online (RL2 lesson). Only real humans can say which direction pays.
#
# PRE-REGISTERED READING (before any data):
#   PRIMARY (powered at n=300/arm): online mechanism fingerprints, computed from
#   action_log with the CORRECT trick tracking (a lead = first play after 3
#   consecutive passes incl. server autos, or game start):
#     * lead-5card rate: L vs C expected +8pp if fingerprints transfer (~450 lead
#       decisions/arm -> powered)
#     * P(leave 2 unplayed | caged loss, dealt 2), caged rate, disaster rate
#   SECONDARY (NOT powered for <4.5 pts): avg_score. Report with CI; only a huge gap
#   is conclusive. Do NOT rank arms on avg_score alone at this n.
#   DECISION RULE:
#     * L's fingerprints move toward human online AND avg not clearly worse
#         -> human-likeness transfers; deploy direction = fingerprint-selected
#            checkpoints; M2 targets fingerprints with confidence.
#     * L's fingerprints move but avg clearly worse (>=4.5 sig)
#         -> humans punish the imitation drift; deploy B-direction; rethink.
#     * L's fingerprints DON'T move online despite offline shift
#         -> offline gate tool not predictive of online behavior — fix the yardstick
#            before any more training.
#   Exclusions (mechanistic, decided now): bomb-dealt games (canonical detector,
#   hand_features.has_bomb_rank_suit — 10 windows, NO J-Q-K-A-2/Q-K-A-2-3/K-A-2-3-4),
#   no-choice games, (run_dir, game_uid) join failures.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
cd "$HERE"

CTRL="$PPO/ppo/checkpoints/policy_4500_best.pt"
LAST="$PPO/ppo/checkpoints/awbc_T40_last.pt"
BEST="$PPO/ppo/checkpoints/awbc_T40_best.pt"
GOAL=300
LEG=50

for CK in "$CTRL" "$LAST" "$BEST"; do
  [ -f "$CK" ] || { echo "FATAL: missing checkpoint $CK"; exit 1; }
done

leg () {   # $1 tag  $2 ckpt  $3 already  $4 target — pure policy: strip all search env
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      -u ISMCTS -u ISMCTS_SIMS -u BELIEF_SEARCH -u VALUE_LEAF \
      -u BELIEF_CKPT -u VALUE_CKPT \
      CARDAWARE_BUDGET=1.0 \
      ./.venv/bin/python3 run_online_leg.py \
          --tag "$1" --ckpt "$2" --goal "$4" --already "$3"
}

DONE=0
while [ "$DONE" -lt "$GOAL" ]; do
  NEXT=$(( DONE + LEG )); [ "$NEXT" -gt "$GOAL" ] && NEXT=$GOAL
  echo "=== ROUND: -> $NEXT/$GOAL ==="
  leg ab2_ctrl_p4500   "$CTRL" "$DONE" "$NEXT"
  leg ab2_awbc_T40last "$LAST" "$DONE" "$NEXT"
  leg ab2_awbc_T40best "$BEST" "$DONE" "$NEXT"
  DONE=$NEXT
  echo "=== ROUND DONE: all arms at $DONE/$GOAL ==="
done
echo "=== AB2 COMPLETE: $GOAL games x 3 arms ==="
