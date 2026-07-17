"""dashboard_writer.py — 看板狀態的「單一真相」建構與寫入（純標準庫）。

main.py（專案環境）與 alpha_big2_wrapper.py（AlphaBig2 venv）都 import 這個模組，
確保兩個寫入者產生的 dashboard_state.json 結構完全一致——座位映射、撲克牌符號、
players/events/constraint 的欄位都來自這裡，不會兩邊漂移。

職責分工：
  - main.py：每次偵測到任何一家出牌/PASS，呼叫 write_live() 即時更新
    「四家牌況 + 事件紀錄 + 桌面狀態 + 輪到誰」。算不出 AI 決策與對手推測手牌，
    所以這些區塊從現有檔案保留。
  - wrapper：輪到我做決策時，呼叫 write_full() 寫完整版（額外帶 AI 決策選項、
    對手推測手牌、本場統計）。

零第三方依賴（不 import pydantic/numpy/torch），AlphaBig2 venv 也能載入。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

# ── 座位：出牌順序 self → right → top → left（伺服器 actor_index 遞增方向，
# 已由對局封包驗證）。AlphaBig2 的 player 1→2→3→4 即出牌順序。 ─────────────────
SEAT_INT = {"self": 1, "right": 2, "top": 3, "left": 4}   # 座位字串 → player index
SEAT_ID = {1: "self", 2: "right", 3: "top", 4: "left"}    # player index → 座位字串
SEAT_ZH = {"self": "自己", "right": "下家", "top": "對家", "left": "上家"}

_SUIT_SYM = {"1": "♠", "2": "♥", "3": "♦", "4": "♣"}
_RANK_DISP = {
    "1": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "10", "J": "J", "Q": "Q", "K": "K",
}
COMBO_ZH = {
    "single": "單張", "pair": "對子", "straight": "順子",
    "full_house": "葫蘆", "four_of_kind": "四條", "four_of_a_kind": "四條",
    "straight_flush": "同花順", "pass": "PASS",
}

# 牌力排序：點數 3<4<…<K<A<2；同點數花色 ♣<♦<♥<♠
_RANK_ORDER = {"3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
               "T": 10, "J": 11, "Q": 12, "K": 13, "1": 14, "2": 15}
_SUIT_ORDER = {"4": 1, "3": 2, "2": 3, "1": 4}  # C<D<H<S


def code_to_symbol(code: str) -> str:
    """BV 牌碼（suit_digit + rank_char，如 '13'）→ 顯示符號（如 '♠3'）。"""
    if not code or len(code) < 2:
        return code or "?"
    return _SUIT_SYM.get(code[0], code[0]) + _RANK_DISP.get(code[1], code[1:])


def card_sort_key(code: str):
    if not code or len(code) < 2:
        return (99, 99)
    return (_RANK_ORDER.get(code[1], 99), _SUIT_ORDER.get(code[0], 99))


def _sorted_symbols(codes) -> list[str]:
    return [code_to_symbol(c) for c in sorted(codes, key=card_sort_key)]


def _fmt_ts(ts) -> str:
    """epoch 毫秒 → 本地 HH:MM:SS。無 ts 時用現在時間。"""
    try:
        if isinstance(ts, (int, float)) and ts > 0:
            return datetime.fromtimestamp(ts / 1000).strftime("%H:%M:%S")
    except (OSError, ValueError, OverflowError):
        pass
    return datetime.now().strftime("%H:%M:%S")


# ── 各區塊建構 ─────────────────────────────────────────────────────────────────

def build_players(obs: dict, opp_hands: dict | None = None) -> list[dict]:
    """四家牌況。剩餘張數一律精確（13 - 已出，play_history 完整無遺漏）。

    opp_hands（可選）：{player_index: [(bv_code, conf_level), ...]} 對手推測手牌
    與信心等級（'high'/'mid'/'low'），由 wrapper 的 belief 估計提供。main.py 不提供，
    對手「手牌/推測」欄留空。每位玩家額外輸出 hand_conf（與 hand 平行的信心等級）。
    """
    play_history = obs.get("play_history", []) or []
    # 各座位已出牌碼（依出牌順序累積，actor 為座位字串）
    played_codes: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: []}
    for ev in play_history:
        if ev.get("action") != "play":
            continue
        p = SEAT_INT.get(ev.get("actor"))
        if p is not None:
            played_codes[p].extend(ev.get("card_codes", []))

    # UI 讀到的剩餘張數（交叉驗證用）
    ui_remaining = {}
    for opp in obs.get("opponents", []):
        p = SEAT_INT.get(opp.get("seat"))
        if p is not None:
            ui_remaining[p] = opp.get("remaining_count")

    self_codes = [c.get("code") for c in obs.get("self_hand", []) if c.get("code")]

    players = []
    for seat in (1, 2, 3, 4):
        seat_id = SEAT_ID[seat]
        known_played = len(played_codes[seat])
        played_syms = _sorted_symbols(played_codes[seat])
        hand_conf: list[str] = []

        if seat == 1:
            remaining = obs.get("hand_count", len(self_codes))
            played_total = 13 - remaining
            hand_syms = _sorted_symbols(self_codes)   # 自己手牌已知，無信心標示
            is_estimated = False
        else:
            # 精確：13 - 已出。永不為 None。
            remaining = 13 - known_played
            played_total = known_played
            ui = ui_remaining.get(seat)
            if ui is not None and ui != remaining:
                # UI 是遊戲真實顯示，視為 ground truth（理論上追蹤完整時不會發生）。
                remaining = ui
                played_total = 13 - ui
            if opp_hands and seat in opp_hands and opp_hands[seat]:
                # opp_hands[seat] = [(bv_code, conf_level), ...] → 依牌力排序
                items = sorted(opp_hands[seat], key=lambda t: card_sort_key(t[0]))
                hand_syms = [code_to_symbol(c) for c, _ in items]
                hand_conf = [lvl for _, lvl in items]
                is_estimated = True
            else:
                hand_syms = []
                is_estimated = False

        players.append({
            "seat": seat,
            "seat_id": seat_id,
            "seat_zh": SEAT_ZH[seat_id],
            "remaining": remaining,
            "played": played_syms,
            "played_total": played_total,
            "hand": hand_syms,
            "hand_conf": hand_conf,
            "is_estimated": is_estimated,
        })
    return players


def build_events(obs: dict, limit: int = 30) -> list[dict]:
    """事件紀錄：直接從完整 play_history 算（含每一家的出牌與 PASS）。"""
    play_history = obs.get("play_history", []) or []
    events = []
    for ev in play_history:
        actor = ev.get("actor")
        if actor not in SEAT_ZH:
            continue
        zh = SEAT_ZH[actor]
        ts = _fmt_ts(ev.get("ts"))
        if ev.get("action") == "play":
            syms = _sorted_symbols(ev.get("card_codes", []))
            czh = COMBO_ZH.get(ev.get("combo_type") or "", "")
            cs = " ".join(syms)
            msg = f"{zh} 出牌：{cs}（{czh}）" if czh else f"{zh} 出牌：{cs}"
            events.append({"ts": ts, "type": "play", "actor": actor,
                           "cards": syms, "combo": ev.get("combo_type"), "msg": msg})
        else:
            events.append({"ts": ts, "type": "pass", "actor": actor,
                           "cards": [], "combo": None, "msg": f"{zh} PASS"})
    return events[-limit:]


def build_constraint(obs: dict) -> dict:
    con = obs.get("constraint", {}) or {}
    last = con.get("last_played_cards", []) or []
    return {
        "lead_actor": con.get("lead_actor"),
        "last_played": _sorted_symbols([c.get("code") for c in last if c.get("code")]),
        "combo_type": con.get("required_combo_type"),
        "passes_since_last_play": con.get("passes_since_last_play", 0),
    }


def base_state(obs: dict, opp_hands: dict | None = None) -> dict:
    """牌況核心（players + events + constraint + 局/手序號 + turn）。"""
    return {
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
        "game_index": obs.get("game_index"),
        "trick_index": obs.get("trick_index"),
        "turn": obs.get("turn"),
        "constraint": build_constraint(obs),
        "players": build_players(obs, opp_hands),
        "events": build_events(obs),
    }


# ── 原子寫入與合併 ─────────────────────────────────────────────────────────────

def atomic_write(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read_existing(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_live(path: str, obs: dict) -> None:
    """main.py 用：即時更新牌況，保留 wrapper 寫的 AI 區塊與對手推測手牌。

    對手回合 main.py 算不出 AI 決策與對手手牌估計，所以從現有檔案保留：
      - last_decision / legal_actions / session（AI 決策區塊）
      - 對手 players[].hand（上次的推測，畫面連續）
    """
    prev = _read_existing(path)
    state = base_state(obs)

    # 保留對手上次的推測手牌與信心（main.py 算不出，避免清空造成閃爍）
    prev_hand_by_seat = {}
    for pp in prev.get("players", []) or []:
        if pp.get("seat") and pp.get("is_estimated"):
            prev_hand_by_seat[pp["seat"]] = (pp.get("hand", []), pp.get("hand_conf", []))
    for pl in state["players"]:
        if pl["seat"] != 1 and not pl["hand"] and pl["seat"] in prev_hand_by_seat:
            pl["hand"], pl["hand_conf"] = prev_hand_by_seat[pl["seat"]]
            pl["is_estimated"] = True

    # 保留 AI 決策區塊與統計
    for key in ("last_decision", "legal_actions", "session"):
        if key in prev:
            state[key] = prev[key]

    atomic_write(path, state)
