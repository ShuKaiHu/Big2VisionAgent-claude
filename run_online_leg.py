#!/usr/bin/env python3
"""Run online games for a single checkpoint until GOAL individual game
RESULTS have accumulated -- not sessions. The CLI's --games flag actually
counts SESSIONS (each up to 4 rounds; ends early if a player goes broke), and
--timeout-seconds (default 7200s) is a hard safety-net that can fire long
before the session target is reached. Worse, lobby matchmaking can get stuck
in a bounce loop (observed directly this session: policy_all_vs1's first
attempt spent 1h50m of its 2h budget stuck re-clicking quick-play and
bouncing back to the lobby, producing only 7 games) -- so neither --games nor
--timeout-seconds reliably controls how many individual games you actually
get.

This script tracks real progress itself via artifacts/game_results.jsonl
(one line appended per completed round, globally across all runs) and stops
the underlying process with SIGTERM (routed through main.py's existing
graceful-shutdown handler -- same code path as a normal `kill <pid>`, saves
all artifacts) as soon as the goal is hit. If a launch exits early (timeout,
bounce, crash) before reaching the goal, it just launches again and keeps
accumulating until the goal is met.

    ./.venv/bin/python3 run_online_leg.py --tag policy_all --ckpt /path/to/policy_all.pt --goal 100 --already 64
"""
import argparse
import datetime
import glob
import json
import os
import signal
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "artifacts", "game_results.jsonl")
PPO_DIR = "/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
STUCK_BOUNCE_LIMIT = 6  # consecutive lobby bounces w/o a completed game -> treat as stuck


def count_since(ts: str) -> int:
    if not os.path.exists(RESULTS):
        return 0
    n = 0
    with open(RESULTS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("timestamp", "") >= ts:
                n += 1
    return n


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def find_run_log(launch_start_time: float) -> str | None:
    """Find this launch's artifacts/<timestamp>/autoplay_agent/run.log -- the
    directory is created a few seconds after the process starts, so this may
    return None on the first few polls."""
    candidates = glob.glob(os.path.join(HERE, "artifacts", "*", "autoplay_agent", "run.log"))
    candidates = [c for c in candidates if os.path.getmtime(os.path.dirname(c)) >= launch_start_time - 5]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def trailing_bounce_count(run_log_path: str) -> int:
    """Consecutive 'bounce' lines at the END of the log with no 'Game ...
    result' in between -- i.e. how long we've been stuck re-clicking quick-play
    without actually starting/finishing a round since the last real progress."""
    try:
        with open(run_log_path) as f:
            lines = f.readlines()
    except OSError:
        return 0
    n = 0
    for line in reversed(lines):
        if "bounce screenshot" in line:
            n += 1
        elif " result:" in line or "wait_for_game_scene -> GameScene" in line:
            break
    return n


def run_leg(tag, ckpt, goal, already=0, per_launch_sessions=60, per_launch_timeout=9600, poll_s=20):
    total = already
    launch_n = 0
    while total < goal:
        launch_n += 1
        need = goal - total
        start_ts = now_iso()
        env = os.environ.copy()
        env["BIG2_AGENT_COMMAND"] = os.path.join(HERE, "cardaware_wrapper.py")
        env["CARDAWARE_DIR"] = PPO_DIR
        env["CARDAWARE_CKPT"] = ckpt
        # Stamp every game this leg produces with its tag + checkpoints, so
        # reward_log rows are self-describing. Attribution used to be inferred
        # from timestamp windows, which mis-assigned a cell's first launch.
        env["RUN_TAG"] = tag
        cmd = [os.path.join(HERE, ".venv", "bin", "big2-agent"), "autoplay-agent",
               "--executor", "packet", "--games", str(per_launch_sessions),
               "--timeout-seconds", str(per_launch_timeout)]
        print(f"[{tag}] launch {launch_n}: have {total}/{goal}, need {need} more", flush=True)
        launch_t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=HERE, env=env)
        stopped_gracefully = False
        stuck = False
        while proc.poll() is None:
            time.sleep(poll_s)
            got = count_since(start_ts)
            if got >= need:
                print(f"[{tag}] launch {launch_n} hit target (+{got}) -- sending SIGTERM for graceful stop", flush=True)
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=90)
                    stopped_gracefully = True
                except subprocess.TimeoutExpired:
                    print(f"[{tag}] no exit after 90s, sending SIGTERM again", flush=True)
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(timeout=60)
                        stopped_gracefully = True
                    except subprocess.TimeoutExpired:
                        print(f"[{tag}] still alive, force kill", flush=True)
                        proc.kill()
                        proc.wait(timeout=30)
                break
            # Stuck-in-lobby watchdog: a known bounce loop (lobby popup blocking
            # quick-play) can eat the WHOLE per_launch_timeout for 0 games (seen
            # live, repeatedly, this session) -- don't wait that long to notice.
            run_log = find_run_log(launch_t0)
            if run_log is not None and trailing_bounce_count(run_log) >= STUCK_BOUNCE_LIMIT:
                print(f"[{tag}] launch {launch_n} looks stuck ({STUCK_BOUNCE_LIMIT}+ consecutive lobby "
                      f"bounces, no game progress) -- stopping early instead of waiting out the full timeout",
                      flush=True)
                stuck = True
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                    stopped_gracefully = True
                except subprocess.TimeoutExpired:
                    print(f"[{tag}] still alive, force kill", flush=True)
                    proc.kill()
                    proc.wait(timeout=30)
                break
        got = count_since(start_ts)
        total += got
        print(f"[{tag}] launch {launch_n} ended (exit={proc.returncode}, graceful_stop={stopped_gracefully}, "
              f"stuck={stuck}): +{got} games, total {total}/{goal}", flush=True)
    print(f"[{tag}] DONE: reached {total}/{goal} games", flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", type=int, default=100)
    ap.add_argument("--already", type=int, default=0)
    args = ap.parse_args()
    run_leg(args.tag, args.ckpt, args.goal, args.already)


if __name__ == "__main__":
    main()
