#!/usr/bin/env python3
"""Overnight autonomous pipeline:
1. Interleave short online bursts across V4 / V4PBV / V4PBV_Iter4 / V4PBV_Iter9
   (round-robin by biggest deficit) until each has >= TARGET games in the corpus,
   or a hard wall-clock deadline is hit.
2. Re-parse the corpus one last time, then train PPO_1000 (behavioral cloning on
   the FULL accumulated human corpus, same recipe as PPO_V4 -- ppo/train_bc.py)
   and save it as checkpoints/saved/PPO_{total_games}.pt.

Everything is logged to overnight_campaign.log so progress can be checked without
needing this process to be interactive.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

TARGET = 150
BURST_GAMES = 25
DEADLINE_HOURS = 11.0  # hard stop for the collection phase regardless of progress

VISION_DIR = "/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude"
PPO_DIR = "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
VALUE_DIR = "/Users/shukaihu/Code_Project_Local/AlphaBig2-claude/value"
CORPUS = f"{PPO_DIR}/ppo/data/online_games.jsonl"
PPO_VENV_PY = f"{PPO_DIR}/.venv/bin/python3"
LOG_PATH = f"{VISION_DIR}/overnight_campaign.log"

VERSIONS = {
    "V4": {
        "launch": ["./play_cardaware.sh", f"{PPO_DIR}/ppo/checkpoints/saved/PPO_V4.pt", str(BURST_GAMES)],
        "env": {},
    },
    "V4PBV": {
        "launch": ["./play_cardaware_ismcts.sh", str(BURST_GAMES)],
        "env": {"CARDAWARE_CKPT": f"{PPO_DIR}/ppo/checkpoints/saved/PPO_V4.pt"},
    },
    "V4PBV_Iter4": {
        "launch": ["./play_cardaware_ismcts.sh", str(BURST_GAMES)],
        "env": {"CARDAWARE_CKPT": f"{VALUE_DIR}/checkpoints/V4PBV_Iter4_policy.pt",
                "VALUE_CKPT": f"{VALUE_DIR}/checkpoints/V4PBV_Iter4_value.pt"},
    },
    "V4PBV_Iter9": {
        "launch": ["./play_cardaware_ismcts.sh", str(BURST_GAMES)],
        "env": {"CARDAWARE_CKPT": f"{VALUE_DIR}/checkpoints/V4PBV_Iter9_policy.pt",
                "VALUE_CKPT": f"{VALUE_DIR}/checkpoints/V4PBV_Iter9_value.pt"},
    },
}

_log_f = open(LOG_PATH, "a")


def p(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_f.write(line + "\n")
    _log_f.flush()


def any_agent_running():
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        return any("autoplay-agent" in line and "grep" not in line for line in out.splitlines())
    except Exception:
        return False


def parse_corpus():
    subprocess.run([sys.executable, "ppo/parse_online_games.py"], cwd=PPO_DIR,
                    check=False, capture_output=True, text=True)


def counts():
    c = Counter()
    if not os.path.exists(CORPUS):
        return c
    for line in open(CORPUS):
        line = line.strip()
        if not line:
            continue
        g = json.loads(line)
        if g.get("our_seat") is not None:
            c[g["seats"][g["our_seat"]]] += 1
    return c


def run_burst(name):
    spec = VERSIONS[name]
    env = os.environ.copy()
    env.update(spec["env"])
    log_path = f"{VISION_DIR}/logs_overnight_{name}_{int(time.time())}.txt"
    try:
        with open(log_path, "w") as f:
            subprocess.run(spec["launch"], cwd=VISION_DIR, env=env, stdout=f,
                            stderr=subprocess.STDOUT, timeout=3600)
    except subprocess.TimeoutExpired:
        p(f"  burst for {name} TIMED OUT after 1h, moving on")
    except Exception as e:
        p(f"  burst for {name} raised {e!r}, moving on")
    return log_path


def main():
    p("=== overnight campaign starting ===")

    if any_agent_running():
        p("an autoplay-agent process is already running -- waiting for it to finish first")
        while any_agent_running():
            time.sleep(30)
        p("previous process finished")

    t0 = time.time()
    round_n = 0
    while True:
        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h > DEADLINE_HOURS:
            p(f"hit hard deadline ({DEADLINE_HOURS}h), stopping collection regardless of progress")
            break
        parse_corpus()
        c = counts()
        p(f"counts: {dict(c)}")
        deficits = {v: TARGET - c.get(v, 0) for v in VERSIONS}
        remaining = {v: d for v, d in deficits.items() if d > 0}
        if not remaining:
            p("all versions reached target, stopping campaign")
            break
        round_n += 1
        pick = max(remaining, key=remaining.get)
        p(f"round {round_n} (t+{elapsed_h:.1f}h): launching burst for {pick} (deficit={remaining[pick]})")
        log_path = run_burst(pick)
        p(f"  burst for {pick} finished, log={log_path}")

    p("=== collection phase done, starting PPO_1000 training ===")
    parse_corpus()
    final_counts = counts()
    total_games = sum(1 for _ in open(CORPUS)) if os.path.exists(CORPUS) else 0
    p(f"final corpus size: {total_games} games, by-version: {dict(final_counts)}")

    train_log = f"{VISION_DIR}/overnight_train_ppo1000.log"
    rc = 1
    try:
        with open(train_log, "w") as f:
            r = subprocess.run([PPO_VENV_PY, "-m", "ppo.train_bc", "--target", "human",
                                 "--epochs", "30"],
                                cwd=PPO_DIR, stdout=f, stderr=subprocess.STDOUT, timeout=5 * 3600)
            rc = r.returncode
    except Exception as e:
        p(f"training raised {e!r}")
    p(f"training exit code: {rc}, log={train_log}")

    if rc == 0:
        src = f"{PPO_DIR}/ppo/checkpoints/ppo_bc_human_best.pt"
        dst = f"{PPO_DIR}/ppo/checkpoints/saved/PPO_{total_games}.pt"
        try:
            shutil.copy(src, dst)
            p(f"saved final checkpoint -> {dst}")
        except Exception as e:
            p(f"failed to copy checkpoint: {e!r}")
    else:
        p("training failed -- NOT saving a PPO_1000 checkpoint; inspect train_log")

    p("=== overnight pipeline complete ===")
    _log_f.close()


if __name__ == "__main__":
    main()
