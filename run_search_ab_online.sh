#!/usr/bin/env bash
# A/B: does the ISMCTS override actually beat the raw policy online, vs real humans?
#
#   arm A  ab1_ismcts_p4500      policy_4500 + belief_hist_sc4500 + VALUE_minplays + ISMCTS(200 sims)
#   arm B  ab1_policyonly_p4500  policy_4500 argmax, no search at all
#
# The two arms share the SAME policy checkpoint and the same legal-action code, and
# differ only where ISMCTS overrides the policy's argmax -- measured at 6.3% of
# decisions. Everything else about the agent is identical. (Verified: with ISMCTS
# unset, cardaware_wrapper falls through to `action_idx = int(legal[pos])`, the
# policy argmax; belief/value are then loaded-but-unused, so arm B drops them too --
# same play, less wasted compute. policy_4500_best.pt has arch="cardaware", so the
# history-policy guard at cardaware_wrapper.py:142 does not fire.)
#
# 300/ARM, INTERLEAVED -- AND WHAT THAT CAN AND CANNOT SHOW
#   Per-game score sd = 19.68 (measured over 7608 logged games; needs no labels).
#   Two-sample power at alpha=.05/power=.80 => n/arm = 15.7*sd^2/delta^2, so
#   300/arm resolves only a >=4.50 pt difference. For reference: >=2.95 pts needs
#   700/arm, 2 pts needs 1521/arm, 1 pt needs 6082/arm.
#   4.50 pts is larger than the entire best-to-worst spread of the 9-cell scaling
#   grid, and that spread was itself shown to be noise (permutation P=0.13). So a
#   null here is EXPECTED and means only "no effect >= ~4.5 pts" -- it is NOT
#   evidence that search does nothing. This run is a sanity read, not a verdict.
#   Legs alternate A,B,A,B... in 50-game blocks rather than running 300 of A then
#   300 of B: the opponent pool drifts over hours/days and a blocked design would
#   confound that drift with the arm. Interleaving makes the drift hit both arms
#   equally, and it also means a mid-run abort still leaves two comparable halves.
#
# KNOWN CONFOUND, NOT CONTROLLED: arm A spends ~1s/move searching, arm B answers
# instantly. Move latency is therefore perfectly correlated with the arm. If human
# opponents play differently against an obviously-instant bot, that effect lands
# entirely on arm B. Matching the think-time would fix it; it is deliberately not
# done here. Read any arm difference with this in mind.
#
# PRE-REGISTERED DECISION RULE (written before any data is collected -- do not
# revise it after seeing the numbers; that is how every previous null got
# over-read):
#   Compare arm A vs arm B by mean self_score over CLEAN games (exclusions below),
#   Welch two-sample t-test, two-sided alpha=0.05. Report both arms with SE.
#     * significant at this n -> the effect is >=4.5 pts, i.e. far bigger than
#                                anything this project has measured. Treat as a
#                                lead to confirm at 700+/arm, not as a result.
#     * not significant       -> the expected outcome. Concludes nothing about
#                                whether search helps; only rules out a huge effect.
#                                Do NOT read it as "ISMCTS is useless".
#   Exclusions, decided now, applied mechanically, all mechanistic (never
#   outcome-based): games where the agent held a bomb (the pre-2026-07-16
#   _build_legal_actions bug is fixed, but keep the exclusion symmetric with the
#   legacy analysis), games where the agent never had >1 legal option, and games
#   whose (run_dir, game_uid) join fails. Report every exclusion count per arm.
#
# GATE: run validate_run_provenance.py on the first leg BEFORE trusting any of this.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
VAL="/Users/shukaihu/Code_Project_Local/AlphaBig2-Value"
cd "$HERE"

POLICY="$PPO/ppo/checkpoints/policy_4500_best.pt"
BELIEF="$PPO/ppo/checkpoints/belief_hist_sc4500_best.pt"
VALUE="$VAL/checkpoints/VALUE_minplays.pt"

GOAL=300
LEG=50                       # 6 alternating rounds of 50 games per arm

for CK in "$POLICY" "$BELIEF" "$VALUE"; do
  [ -f "$CK" ] || { echo "FATAL: missing checkpoint $CK"; exit 1; }
done

leg_A () {   # ISMCTS arm
  local ALREADY=$1 TARGET=$2
  echo "=== ARM A (ismcts) -> $TARGET (already=$ALREADY) ==="
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      ISMCTS=1 ISMCTS_SIMS=200 BELIEF_SEARCH=1 VALUE_LEAF=1 CARDAWARE_BUDGET=1.0 \
      BELIEF_CKPT="$BELIEF" VALUE_CKPT="$VALUE" \
      ./.venv/bin/python3 run_online_leg.py \
          --tag ab1_ismcts_p4500 --ckpt "$POLICY" \
          --goal "$TARGET" --already "$ALREADY"
}

leg_B () {   # policy-only arm: no search, and therefore no belief/value either
  local ALREADY=$1 TARGET=$2
  echo "=== ARM B (policy-only) -> $TARGET (already=$ALREADY) ==="
  env -u CARDAWARE_SEARCH -u CARDAWARE_WORLDS -u POSTACTION_VALUE \
      -u ISMCTS -u ISMCTS_SIMS -u BELIEF_SEARCH -u VALUE_LEAF \
      -u BELIEF_CKPT -u VALUE_CKPT \
      CARDAWARE_BUDGET=1.0 \
      ./.venv/bin/python3 run_online_leg.py \
          --tag ab1_policyonly_p4500 --ckpt "$POLICY" \
          --goal "$TARGET" --already "$ALREADY"
}

DONE=0
while [ "$DONE" -lt "$GOAL" ]; do
  NEXT=$(( DONE + LEG )); [ "$NEXT" -gt "$GOAL" ] && NEXT=$GOAL
  leg_A "$DONE" "$NEXT"
  leg_B "$DONE" "$NEXT"
  DONE=$NEXT
  echo "=== ROUND DONE: both arms at $DONE/$GOAL ==="
done
echo "=== A/B COMPLETE: $GOAL games per arm ==="
