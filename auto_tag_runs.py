#!/usr/bin/env python3
"""Companion to overnight_campaign.py -- does NOT touch that process or its
subprocesses (additive only: writes run_versions.json tags).

overnight_campaign.py launches bursts via `logs_overnight_{VERSION}_{unixtime}.txt`
but never tells run_versions.json which policy each resulting online run actually
used, so parse_online_games.py's _agent_label() falls back to inferring a generic
"cardaware"/"cardaware+ismcts" bucket for all four versions -- collapsing
V4/V4PBV/Iter4/Iter9 into one indistinguishable bucket and breaking the
deficit-based round-robin (it would never see per-version counts increase).

This script matches each untagged artifacts/*/autoplay_agent run dir to the
burst-log whose launch time precedes it most closely, tags it in
run_versions.json, and exits once overnight_campaign.py is no longer running
(after one final pass).
"""
import glob
import json
import os
import re
import time

VISION_DIR = "/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude"
ART = f"{VISION_DIR}/artifacts"
VERSIONS_PATH = "/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo/ppo/data/run_versions.json"
LOG_PATH = f"{VISION_DIR}/auto_tag_runs.log"

# manually-known runs from before this script existed
KNOWN = {
    "20260704-230343": "V4",  # the "Restart V4 burst with club3 fix active" run
}

_log_f = open(LOG_PATH, "a")


def p(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    _log_f.write(line + "\n")
    _log_f.flush()


def campaign_running():
    try:
        out = os.popen("ps aux").read()
        return any("overnight_campaign.py" in line and "grep" not in line
                    for line in out.splitlines())
    except Exception:
        return False


def burst_logs():
    """[(version, launch_unixtime), ...] sorted by time."""
    out = []
    for path in glob.glob(f"{VISION_DIR}/logs_overnight_*.txt"):
        m = re.match(r"logs_overnight_(.+)_(\d+)\.txt$", os.path.basename(path))
        if m:
            out.append((m.group(1), int(m.group(2))))
    out.sort(key=lambda x: x[1])
    return out


def run_dirs():
    """[(run_name, start_epoch), ...] sorted by time."""
    out = []
    for d in glob.glob(f"{ART}/*/autoplay_agent"):
        run = d.split("/artifacts/")[1].split("/")[0]
        m = re.match(r"(\d{8})-(\d{6})$", run)
        if not m:
            continue
        dt = time.strptime(run, "%Y%m%d-%H%M%S")
        out.append((run, time.mktime(dt)))
    out.sort(key=lambda x: x[1])
    return out


def tag_once():
    overrides = json.load(open(VERSIONS_PATH)) if os.path.exists(VERSIONS_PATH) else {}
    changed = False
    for run, label in KNOWN.items():
        if run not in overrides:
            overrides[run] = label
            changed = True
            p(f"tagged {run} -> {label} (manually-known)")

    logs = burst_logs()
    cutoff = time.time() - 24 * 3600  # only ever-green runs from the last 24h are candidates
    for run, start_epoch in run_dirs():
        if run in overrides or start_epoch < cutoff:
            continue
        # find the burst log with the closest launch time before this run started
        # (browser startup lag can be up to a couple minutes; allow up to 90 min
        # since a 25-game burst can take ~50min and we want the LAUNCH, not end)
        best = None
        for version, launch_ts in logs:
            if launch_ts <= start_epoch + 60 and (start_epoch - launch_ts) < 90 * 60:
                if best is None or launch_ts > best[1]:
                    best = (version, launch_ts)
        if best:
            overrides[run] = best[0]
            changed = True
            p(f"tagged {run} -> {best[0]} (matched burst log launched at {best[1]})")
        else:
            p(f"could not match {run} to any burst log yet (may appear next tick)")

    if changed:
        json.dump(overrides, open(VERSIONS_PATH, "w"), indent=2, ensure_ascii=False)
        p(f"wrote {VERSIONS_PATH}")
    return changed


def main():
    p("=== auto_tag_runs starting ===")
    while True:
        tag_once()
        if not campaign_running():
            p("overnight_campaign.py no longer running -- doing one final pass and exiting")
            time.sleep(5)
            tag_once()
            break
        time.sleep(120)
    p("=== auto_tag_runs done ===")
    _log_f.close()


if __name__ == "__main__":
    main()
