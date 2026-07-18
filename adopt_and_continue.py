#!/usr/bin/env python3
"""One-off: adopt the already-running policy_all_minloss process (launched
before run_online_leg.py existed), stop it via SIGTERM once it reaches its
goal (tracking real game_results.jsonl progress, not the CLI's session
count), then top up policy_all_minloss/policy_all/policy_all_vs1 to 100
individual games each using run_online_leg.py, sequentially.
"""
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_online_leg import count_since, run_leg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PPO = "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/ppo/checkpoints/saved"
GOAL = 100

ADOPT_PID = 84663
ADOPT_START_TS = "2026-07-06T07:53:03"
ADOPT_TAG = "policy_all_minloss"


def adopt_running_leg(pid, start_ts, tag, goal, poll_s=20):
    print(f"[{tag}] adopting running pid={pid}, tracking since {start_ts}", flush=True)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        got = count_since(start_ts)
        if got >= goal:
            print(f"[{tag}] adopted leg hit goal (+{got}) -- sending SIGTERM", flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                break
            for _ in range(9):
                time.sleep(10)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                print(f"[{tag}] still alive after 90s, SIGTERM again", flush=True)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            break
        time.sleep(poll_s)
    got = count_since(start_ts)
    print(f"[{tag}] adopted leg ended with +{got} games", flush=True)
    return got


def main():
    got = adopt_running_leg(ADOPT_PID, ADOPT_START_TS, ADOPT_TAG, GOAL)
    if got < GOAL:
        run_leg(ADOPT_TAG, os.path.join(PPO, "policy_all_minloss.pt"), GOAL, already=got)
    else:
        print(f"[{ADOPT_TAG}] DONE: reached {got}/{GOAL} games (no top-up needed)", flush=True)

    run_leg("policy_all", os.path.join(PPO, "policy_all.pt"), GOAL, already=64)
    run_leg("policy_all_vs1", os.path.join(PPO, "policy_all_vs1.pt"), GOAL, already=7)

    print("=== ALL THREE ONLINE CAMPAIGNS DONE (100 games each) ===", flush=True)


if __name__ == "__main__":
    main()
