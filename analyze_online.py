#!/usr/bin/env python3
"""Analyze an online autoplay session (reward-focused).

Usage:
  python3 analyze_online.py [SESSION_DIR]
If SESSION_DIR omitted, uses the newest artifacts/*/autoplay_agent.

Reports:
  - Execution health (mcts notes, play_not_confirmed/forced_pass/exception,
    ok=False reasons)
  - Reward stats from timeline round_result scores (however many were retained)
  - Go-out rate + remaining-card stats + placement distribution for ALL games
    in this session (from game_results.jsonl, filtered by session start time)
"""
import sys, os, json, re, glob
from statistics import median, mean

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def newest_session():
    dirs = sorted(glob.glob(os.path.join(ART, "*", "autoplay_agent")))
    return dirs[-1] if dirs else None


def session_start_ts(sess_dir):
    # dir name like .../20260601-220650/autoplay_agent
    name = os.path.basename(os.path.dirname(sess_dir))
    m = re.match(r"(\d{8})-(\d{6})", name)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}"


def exec_health(run_log):
    txt = open(run_log).read() if os.path.exists(run_log) else ""
    notes = re.findall(r"ML: ([a-z_]+)", txt)
    from collections import Counter
    nc = Counter(notes)
    okF = re.findall(r"Executor result: ok=False reason=([a-z_]+)", txt)
    return {
        "mcts_plays": nc.get("mcts", 0),
        "only_legal_pass": nc.get("only_legal", 0),
        "play_not_confirmed": txt.count("play_not_confirmed"),
        "forced_pass": txt.count("forced_pass"),
        "exception": txt.count("Error during inference") + txt.count("fallback:exception"),
        "ok_false_reasons": dict(Counter(okF)),
    }


def reward_stats(timeline_json):
    if not os.path.exists(timeline_json):
        return None
    tl = json.load(open(timeline_json))
    ev = tl if isinstance(tl, list) else tl.get("events", [])
    scores = [float(e["score"]) for e in ev
              if e.get("event") == "round_result" and e.get("actor") == "self"
              and e.get("score") is not None]
    if not scores:
        return None
    s = sorted(scores)
    n = len(s)
    return {
        "n_scored": n,
        "avg": mean(s),
        "median": median(s),
        "p25": s[n // 4],
        "p75": s[(3 * n) // 4],
        "total": sum(s),
        "pos": sum(1 for x in s if x > 0),
        "neg": sum(1 for x in s if x < 0),
        "min": s[0], "max": s[-1],
    }


def results_stats(start_ts):
    path = os.path.join(ART, "game_results.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if start_ts:
        rows = [r for r in rows if r.get("timestamp", "") >= start_ts]
    if not rows:
        return None
    from collections import Counter
    places = Counter(r["placement"] for r in rows)
    rem = [r["my_remaining"] for r in rows]
    goout = sum(1 for r in rows if r["my_remaining"] == 0)
    return {
        "n_games": len(rows),
        "go_out_rate": goout / len(rows),
        "go_out_count": goout,
        "my_remaining_avg": mean(rem),
        "my_remaining_median": median(rem),
        "placements": {k: places.get(k, 0) for k in (1, 2, 3, 4)},
    }


def reward_log_stats(start_ts):
    """Authoritative per-game reward from reward_log.jsonl (server scores)."""
    path = os.path.join(ART, "reward_log.jsonl")
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if start_ts:
        rows = [r for r in rows if r.get("timestamp", "") >= start_ts]
    s = sorted(r["self_score"] for r in rows if r.get("self_score") is not None)
    if not s:
        return None
    n = len(s)
    return {
        "n": n, "avg": mean(s), "median": median(s),
        "p25": s[n // 4], "p75": s[(3 * n) // 4],
        "total": sum(s), "pos": sum(1 for x in s if x > 0),
        "neg": sum(1 for x in s if x < 0), "min": s[0], "max": s[-1],
        "go_out": sum(1 for r in rows if r.get("self_remaining") == 0),
    }


def main():
    sess = sys.argv[1] if len(sys.argv) > 1 else newest_session()
    if not sess:
        print("no session found"); return
    print(f"session: {sess}")
    start = session_start_ts(sess)
    print(f"session start ts filter: {start}\n")

    print("=== 執行健康度 ===")
    for k, v in exec_health(os.path.join(sess, "run.log")).items():
        print(f"  {k}: {v}")

    print("\n=== Reward (reward_log.jsonl, server 分數, 權威完整) ===")
    rl = reward_log_stats(start)
    if rl:
        print(f"  場數: {rl['n']}  出完: {rl['go_out']}")
        print(f"  avg={rl['avg']:+.2f}  median={rl['median']:+.1f}  "
              f"p25={rl['p25']:+.0f}  p75={rl['p75']:+.0f}")
        print(f"  total={rl['total']:+.0f}  賺={rl['pos']} 賠={rl['neg']}  "
              f"range=[{rl['min']:+.0f},{rl['max']:+.0f}]")
    else:
        print("  (尚無 reward_log.jsonl — 此功能在新一輪線上才會產生)")

    print("\n=== Reward (timeline 備援, 可能只含最近場次) ===")
    rs = reward_stats(os.path.join(sess, "game_timeline.json"))
    if rs:
        print(f"  有分數場數: {rs['n_scored']}")
        print(f"  avg={rs['avg']:+.2f}  median={rs['median']:+.1f}  "
              f"p25={rs['p25']:+.0f}  p75={rs['p75']:+.0f}")
        print(f"  total={rs['total']:+.0f}  賺={rs['pos']} 賠={rs['neg']}  "
              f"range=[{rs['min']:+.0f},{rs['max']:+.0f}]")
    else:
        print("  (timeline 無分數)")

    print("\n=== 全部 100 場 (game_results.jsonl, reward 代理指標) ===")
    gr = results_stats(start)
    if gr:
        print(f"  場數: {gr['n_games']}")
        print(f"  出完牌率 (=賺分率): {gr['go_out_rate']:.1%} ({gr['go_out_count']}/{gr['n_games']})")
        print(f"  我剩牌 avg={gr['my_remaining_avg']:.2f}  median={gr['my_remaining_median']}")
        print(f"  名次分布: {gr['placements']}")
    else:
        print("  (無 game_results)")


if __name__ == "__main__":
    main()
