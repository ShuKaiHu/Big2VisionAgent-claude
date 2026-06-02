#!/usr/bin/env python3
"""Lightweight overnight score capturer.

Polls the running session's game_timeline.json every 180s and appends any NEW
self round_result scores (deduped by event seq) to a persistent JSONL, so we
keep ALL 100 games' rewards even if the timeline rolls/truncates.

Exits when run.log has been idle (unchanged) for >20 min (run finished), or
after a hard cap. Self-contained; negligible CPU.
"""
import os, sys, json, glob, time

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
SESS = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob(os.path.join(ART, "*", "autoplay_agent")))[-1]
CAP = os.path.join(SESS, "captured_self_scores.jsonl")
TL = os.path.join(SESS, "game_timeline.json")
RUNLOG = os.path.join(SESS, "run.log")

seen = set()
if os.path.exists(CAP):
    for l in open(CAP):
        try:
            seen.add(json.loads(l)["seq"])
        except Exception:
            pass

print(f"capturing from: {SESS}")
print(f"appending to: {CAP}")

POLL = 180
IDLE_LIMIT = 20 * 60       # run considered done after 20 min of no run.log writes
HARD_CAP = 12 * 3600       # 12h safety
t0 = time.time()

while True:
    new = 0
    if os.path.exists(TL):
        try:
            tl = json.load(open(TL))
            ev = tl if isinstance(tl, list) else tl.get("events", [])
            for e in ev:
                if e.get("event") == "round_result" and e.get("actor") == "self":
                    seq = e.get("seq")
                    if seq is not None and seq not in seen:
                        seen.add(seq)
                        rec = {"seq": seq, "score": e.get("score"),
                               "remaining": len(e.get("remaining_cards") or []),
                               "ts": e.get("ts")}
                        with open(CAP, "a") as f:
                            f.write(json.dumps(rec) + "\n")
                        new += 1
        except Exception:
            pass
    total = len(seen)
    print(f"[{time.strftime('%H:%M:%S')}] captured total={total} (+{new})", flush=True)

    # done detection
    idle = (time.time() - os.path.getmtime(RUNLOG)) if os.path.exists(RUNLOG) else 1e9
    if idle > IDLE_LIMIT:
        print(f"run.log idle {idle/60:.0f} min — run finished. total scores captured: {total}")
        break
    if time.time() - t0 > HARD_CAP:
        print("hard cap reached")
        break
    time.sleep(POLL)
