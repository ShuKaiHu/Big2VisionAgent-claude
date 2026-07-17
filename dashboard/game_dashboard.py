#!/usr/bin/env python3
"""game_dashboard.py — 大老二 AI 即時看板（HTTP + SSE 版）

由 big2-agent autoplay-agent 自動啟動並開啟瀏覽器。
也可手動執行：
    python3 dashboard/game_dashboard.py [--port 7373]

架構：
  - /        → HTML 看板頁面
  - /state   → 最新 JSON 狀態（一次性讀取）
  - /events  → Server-Sent Events 串流；state 檔案一有變動立即推送
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DASH_STATE  = PROJECT_DIR / "state" / "dashboard_state.json"

# ── SSE 狀態 ──────────────────────────────────────────────────────────────────

_clients: list[queue.Queue] = []   # 每個連線一個 Queue
_clients_lock = threading.Lock()
_last_mtime: float = 0.0
_last_bytes: bytes = b""


def _watcher() -> None:
    """每 50ms 檢查一次 dashboard_state.json，有變動就推給所有 SSE 客戶端。"""
    global _last_mtime, _last_bytes
    while True:
        try:
            mtime = DASH_STATE.stat().st_mtime
            if mtime != _last_mtime:
                data = DASH_STATE.read_bytes()
                if data != _last_bytes:
                    _last_mtime = mtime
                    _last_bytes = data
                    with _clients_lock:
                        dead = []
                        for q in _clients:
                            try:
                                q.put_nowait(data)
                            except queue.Full:
                                dead.append(q)
                        for q in dead:
                            _clients.remove(q)
        except FileNotFoundError:
            pass
        time.sleep(0.05)


threading.Thread(target=_watcher, daemon=True).start()

# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>大老二 AI 即時看板</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--panel:#161b22;--border:#30363d;
  --text:#e6edf3;--dim:#7d8590;
  --green:#3fb950;--yellow:#d29922;--red:#f85149;--cyan:#58a6ff;--purple:#bc8cff;
  --orange:#f0883e;--card-red:#ff7b72;
}
body{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;padding:14px;font-size:23px;line-height:1.45;max-width:800px;margin:0 auto}

header{text-align:center;margin-bottom:14px}
header h1{font-size:1.2em;color:var(--cyan);letter-spacing:3px;margin-bottom:2px}
#game-info{color:var(--dim);font-size:.84em}
.blink{animation:blink .7s step-end infinite;color:var(--green);font-weight:bold}
@keyframes blink{50%{opacity:0}}

.panel{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:12px}
.panel h2{font-size:.62em;color:var(--cyan);text-transform:uppercase;letter-spacing:3px;padding-bottom:6px;margin-bottom:10px;border-bottom:1px solid var(--border)}

/* 四家牌況表格 — 固定欄寬，避免已出牌過多撐爆表格寬度 */
.ptable{width:100%;border-collapse:collapse;table-layout:fixed}
.ptable th{font-size:.62em;color:var(--dim);font-weight:normal;text-transform:uppercase;letter-spacing:1px;padding:4px 6px;text-align:left;border-bottom:1px solid var(--border)}
.ptable td{padding:7px 6px;vertical-align:top;border-bottom:1px solid #21262d}
.ptable tr:last-child td{border-bottom:none}
.ptable th:nth-child(1),.ptable td:nth-child(1){width:11%}
.ptable th:nth-child(2),.ptable td:nth-child(2){width:17%}
.ptable th:nth-child(3),.ptable td:nth-child(3){width:36%}
.ptable th:nth-child(4),.ptable td:nth-child(4){width:36%}
.seat-cell{white-space:nowrap}
.seat-name{font-weight:bold;font-size:.78em}
.seat-name.self{color:var(--green)}
.seat-name.opp{color:var(--yellow)}
.rem-cell{white-space:nowrap}
.rem-count{font-weight:bold;font-size:.92em;margin-bottom:4px}
.bar{background:#21262d;border-radius:3px;height:8px;overflow:hidden;width:100%;max-width:90px}
.bar-fill{height:100%;border-radius:3px;background:var(--yellow);transition:width .2s ease}
.bar-fill.self{background:var(--green)}

/* 撲克牌（大）— 尺寸固定 28x38，字體用 px 不隨 body 縮放 */
.card{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:28px;height:38px;background:#fff;border-radius:4px;font-weight:bold;box-shadow:0 1px 4px #0008;user-select:none;margin:1px;gap:0}
.card .s{font-size:22px;line-height:1.0;color:#222}
.card .r{font-size:18px;line-height:1.0;color:#222}
.card.red .s,.card.red .r{color:var(--card-red)}
.card.chosen{box-shadow:0 0 0 2px var(--green),0 1px 6px #0009}

/* 撲克牌（小）— 同尺寸 28x38，字體同大牌 */
.card-sm{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:28px;height:38px;background:#fff;border-radius:3px;font-weight:bold;box-shadow:0 1px 3px #0006;user-select:none;margin:1px;gap:0}
.card-sm .s{font-size:22px;line-height:1.0;color:#222}
.card-sm .r{font-size:18px;line-height:1.0;color:#222}
.card-sm.red .s,.card-sm.red .r{color:var(--card-red)}
.card-sm.est{opacity:.65;border:1px dashed #888}
/* 推測手牌信心框：綠=高把握、黃=中等、紅=接近隨機 */
.card-sm.conf-high{box-shadow:0 0 0 2px var(--green),0 1px 4px #0008}
.card-sm.conf-mid{box-shadow:0 0 0 2px var(--yellow),0 1px 4px #0008}
.card-sm.conf-low{box-shadow:0 0 0 2px var(--red),0 1px 4px #0008;opacity:.82}

/* 信心圖例 */
.conf-legend{display:flex;gap:14px;align-items:center;font-size:.6em;color:var(--dim);margin-top:10px}
.conf-legend .lg{display:inline-flex;align-items:center;gap:4px}
.conf-legend .sw{width:13px;height:13px;border-radius:3px}
.conf-legend .sw.h{box-shadow:0 0 0 2px var(--green)}
.conf-legend .sw.m{box-shadow:0 0 0 2px var(--yellow)}
.conf-legend .sw.l{box-shadow:0 0 0 2px var(--red)}

.cards-wrap{display:flex;flex-wrap:wrap;gap:2px;align-items:center;min-height:22px}
.est-label{font-size:.6em;color:var(--dim);margin-left:3px;vertical-align:top}
.partial-label{font-size:.6em;color:var(--red);margin-left:4px;vertical-align:middle}
.empty{color:var(--dim);font-style:italic;font-size:.82em}

/* 版面：中段兩欄（800px 寬維持並排；極窄才單欄）*/
.mid-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:600px){.mid-grid{grid-template-columns:1fr}}

/* 桌面狀態 */
.lead-by{color:var(--dim);font-size:.82em;margin-bottom:4px}
.lead-by strong{color:var(--yellow)}
.pass-info{color:var(--dim);font-size:.78em;margin-top:4px}
.badge{display:inline-block;background:var(--border);border-radius:3px;padding:1px 5px;font-size:.72em;margin-left:4px;vertical-align:middle}

/* AI 選項列表 */
.ai-option{padding:5px 8px;border-radius:5px;margin-bottom:5px;border:1px solid var(--border)}
.ai-option.chosen{border-color:var(--green);background:#0d2318}
.ai-opt-header{display:flex;align-items:center;gap:6px;margin-bottom:4px}
.opt-combo{font-size:.78em;font-weight:bold;min-width:40px;color:var(--yellow)}
.opt-combo.pass{color:var(--dim)}
.opt-cards{display:flex;flex-wrap:wrap;gap:1px;flex:1;align-items:center}
.chosen-mark{font-size:.7em;color:var(--green);font-weight:bold;white-space:nowrap}
.bar-row{display:flex;align-items:center;gap:6px;font-size:.72em;margin-bottom:2px}
.bar-lbl{color:var(--dim);width:42px;white-space:nowrap;flex-shrink:0}
.bar-val{width:40px;text-align:right;color:var(--text)}
.score-bar{flex:1;background:#21262d;border-radius:2px;height:6px;overflow:hidden}
.score-bar-fill{height:100%;border-radius:2px;transition:width .2s}
.fill-policy{background:var(--cyan)}
.fill-mcts{background:var(--orange)}
.mode-chip{display:inline-block;border-radius:4px;padding:1px 6px;font-size:.72em;font-weight:bold}
.m-mcts,.m-ismcts,.m-search{background:#0d2744;color:var(--cyan)}
.m-greedy{background:#2d1e00;color:var(--yellow)}
.m-fallback{background:#2d0c0c;color:var(--red)}
.m-forced,.m-unknown{background:#1c2128;color:var(--dim)}
.ai-header-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}

/* 事件紀錄 */
.event-log{max-height:380px;overflow-y:auto;font-size:.78em;font-family:'SF Mono','Fira Code',monospace}
.ev-row{display:flex;gap:8px;padding:3px 0;border-bottom:1px solid #1c2128}
.ev-row:last-child{border-bottom:none}
.ev-ts{color:var(--dim);white-space:nowrap;flex-shrink:0}
.ev-msg{flex:1}
.ev-play{color:var(--text)}
.ev-pass{color:var(--dim)}
.ev-self{color:var(--green)}

/* 統計（橫向一行） */
.stats-row{display:flex;flex-wrap:wrap;gap:14px;font-size:.78em}
.stat-item .lbl{color:var(--dim);margin-right:3px}
.stat-item .val{font-weight:bold}
.warn{color:var(--red)}
.ok{color:var(--green)}

footer{display:flex;justify-content:space-between;font-size:.7em;color:var(--dim);margin-top:8px}
#conn{transition:color .3s}
#conn.live{color:var(--green)}
#conn.wait{color:var(--yellow)}
</style>
</head>
<body>
<header>
  <h1>🎴 大老二 AI 即時看板</h1>
  <div id="game-info">等待遊戲開始…</div>
</header>

<!-- 四家牌況 -->
<div class="panel">
  <h2>四家牌況</h2>
  <table class="ptable">
    <thead><tr>
      <th>座位</th>
      <th>剩餘張數</th>
      <th>已出牌</th>
      <th>手牌 / 推測</th>
    </tr></thead>
    <tbody id="players-tbody"></tbody>
  </table>
  <div class="conf-legend">
    <span>推測信心：</span>
    <span class="lg"><span class="sw h"></span>高（集中）</span>
    <span class="lg"><span class="sw m"></span>中</span>
    <span class="lg"><span class="sw l"></span>低（接近隨機）</span>
  </div>
</div>

<!-- 中段：桌面 + AI選項 ｜ 事件紀錄 -->
<div class="mid-grid">
  <div>
    <div class="panel">
      <h2>桌面狀態</h2>
      <div id="table-state"><span class="empty">—</span></div>
    </div>
    <div class="panel">
      <h2>AI 決策選項</h2>
      <div id="ai-options"><span class="empty">等待中…</span></div>
    </div>
  </div>
  <div class="panel" style="margin-bottom:0">
    <h2>即時事件紀錄</h2>
    <div class="event-log" id="event-log"><span class="empty">等待事件…</span></div>
  </div>
</div>

<!-- 統計 -->
<div class="panel">
  <h2>本次執行統計</h2>
  <div class="stats-row" id="stats"></div>
</div>

<footer>
  <span id="updated">—</span>
  <span id="conn" class="wait">● 連線中…</span>
</footer>

<script>
const RED_SUITS = new Set(['♥','♦']);
// 出牌順序 self→right→top→left：right=下家, left=上家
const SEAT_ZH = {self:'自己', right:'下家', top:'對家', left:'上家'};
const COMBO_ZH = {single:'單張',pair:'對子',straight:'順子',full_house:'葫蘆',
                  four_of_kind:'四條',four_of_a_kind:'四條',straight_flush:'同花順',pass:'PASS'};
const MODE_ZH = {mcts:'MCTS 搜尋',ismcts:'ISMCTS 搜尋',search:'PIMC 搜尋',greedy:'貪婪策略',
                 forced:'唯一合法',fallback:'備用策略',unknown:'未知'};

const $ = id => document.getElementById(id);

function cardHtml(sym, cls='card', chosen=false) {
  const suit=sym[0]||'', rank=sym.slice(1);
  const red = RED_SUITS.has(suit) ? ' red' : '';
  const ch  = chosen ? ' chosen' : '';
  return `<div class="${cls}${red}${ch}"><span class="s">${suit}</span><span class="r">${rank}</span></div>`;
}
const card   = (sym, chosen=false) => cardHtml(sym, 'card', chosen);
const cardSm = (sym, est=false)    => cardHtml(sym, 'card-sm'+(est?' est':''));
// 推測手牌：依信心等級（high/mid/low）上色框
const cardConf = (sym, level) => cardHtml(sym, 'card-sm conf-'+(level||'low'));

function renderPlayers(players) {
  if (!players || !players.length) {
    $('players-tbody').innerHTML = '<tr><td colspan="4" class="empty">—</td></tr>';
    return;
  }
  $('players-tbody').innerHTML = players.map(p => {
    const isSelf = p.seat_id === 'self';
    const rem = p.remaining ?? 0;
    const pct = Math.min(100, Math.round(rem / 13 * 100));
    const nameClass = isSelf ? 'self' : 'opp';
    const barClass  = isSelf ? 'self' : '';

    // 已出牌：追蹤到的 vs. 實際總數
    const knownPlayed  = p.played ? p.played.length : 0;
    const totalPlayed  = p.played_total ?? null;  // 伺服器推算的實際出牌總數
    const isPartial    = !isSelf && totalPlayed !== null && knownPlayed < totalPlayed;
    const missingCount = isPartial ? (totalPlayed - knownPlayed) : 0;

    let playedHtml = '';
    if (knownPlayed > 0) {
      const partialNote = isPartial
        ? `<span class="partial-label">+${missingCount} 張未追蹤</span>`
        : '';
      playedHtml = `<div class="cards-wrap">${p.played.map(c=>cardSm(c)).join('')}${partialNote}</div>`;
    } else if (isPartial) {
      playedHtml = `<span class="partial-label">${totalPlayed} 張未追蹤</span>`;
    } else {
      playedHtml = `<span class="empty">—</span>`;
    }

    let handHtml = '';
    if (isSelf) {
      handHtml = p.hand && p.hand.length
        ? `<div class="cards-wrap">${p.hand.map(c=>cardSm(c)).join('')}</div>`
        : `<span class="empty">—</span>`;
    } else {
      if (p.hand && p.hand.length) {
        const conf = p.hand_conf || [];
        const cardsHtml = p.hand.map((c,i)=>cardConf(c, conf[i])).join('');
        handHtml = `<div class="cards-wrap">${cardsHtml}</div>`;
      } else {
        handHtml = `<span class="empty">未知</span>`;
      }
    }

    return `<tr>
      <td class="seat-cell"><span class="seat-name ${nameClass}">${p.seat_zh}</span></td>
      <td class="rem-cell">
        <div class="rem-count">${p.remaining ?? '?'}</div>
        <div class="bar"><div class="bar-fill ${barClass}" style="width:${pct}%"></div></div>
      </td>
      <td class="cards-cell">${playedHtml}</td>
      <td class="hand-cell">${handHtml}</td>
    </tr>`;
  }).join('');
}

function renderTableState(con) {
  const lp = con.last_played || [];
  const passes = con.passes_since_last_play || 0;
  if (!lp.length) {
    $('table-state').innerHTML = '<span class="empty">自由出牌（桌面無牌）</span>';
    return;
  }
  const lead = con.lead_actor ? SEAT_ZH[con.lead_actor]||con.lead_actor : '?';
  const combo = con.combo_type ? `<span class="badge">${COMBO_ZH[con.combo_type]||con.combo_type}</span>` : '';
  const cardsHtml = `<div class="cards-wrap" style="margin:5px 0">${lp.map(c=>card(c)).join('')}${combo}</div>`;
  const passInfo = passes ? `<div class="pass-info">已 PASS：${passes} 人</div>` : '';
  $('table-state').innerHTML = `<div class="lead-by"><strong>${lead}</strong> 出牌</div>${cardsHtml}${passInfo}`;
}

function renderAiOptions(legalActions, lastDecision) {
  if (!legalActions || !legalActions.length) {
    $('ai-options').innerHTML = '<span class="empty">—</span>';
    return;
  }

  // 計算最大值供 bar 縮放
  const maxPolicy = Math.max(...legalActions.map(a=>a.policy_pct??0), 0.01);
  const maxVisit  = Math.max(...legalActions.map(a=>a.visit_pct??0), 0.01);
  const hasVisits = legalActions.some(a=>a.visits!=null);
  const hasPol    = legalActions.some(a=>a.policy_pct!=null);

  const dec = lastDecision || {};
  const mode = dec.mode || 'unknown';
  const sims = dec.mcts_sims, elapsed = dec.mcts_time_s;

  let header = `<div class="ai-header-row">
    <span class="mode-chip m-${mode}">${MODE_ZH[mode]||mode}</span>`;
  if (sims!=null && elapsed!=null)
    header += `<span style="font-size:.75em;color:var(--dim)">${sims.toLocaleString()} 次模擬 · ${elapsed.toFixed(2)}s</span>`;
  header += '</div>';

  const rows = legalActions.map(a => {
    const isChosen = a.chosen;
    const isPass   = a.action === 'pass';
    const comboLabel = a.combo_zh || (isPass ? 'PASS' : COMBO_ZH[a.combo_type]||a.combo_type||'');
    const cardsHtml = a.cards && a.cards.length
      ? a.cards.map(c=>card(c, isChosen)).join('')
      : '';

    let bars = '';
    if (hasPol && a.policy_pct != null) {
      const w = (a.policy_pct / maxPolicy * 100).toFixed(1);
      bars += `<div class="bar-row">
        <span class="bar-lbl">策略</span>
        <div class="score-bar"><div class="score-bar-fill fill-policy" style="width:${w}%"></div></div>
        <span class="bar-val">${a.policy_pct.toFixed(1)}%</span>
      </div>`;
    }
    if (hasVisits && a.visits != null) {
      const w = (a.visit_pct / maxVisit * 100).toFixed(1);
      bars += `<div class="bar-row">
        <span class="bar-lbl">MCTS</span>
        <div class="score-bar"><div class="score-bar-fill fill-mcts" style="width:${w}%"></div></div>
        <span class="bar-val">${a.visit_pct!=null?a.visit_pct.toFixed(1)+'%':'—'}</span>
      </div>`;
    }

    return `<div class="ai-option${isChosen?' chosen':''}">
      <div class="ai-opt-header">
        <span class="opt-combo${isPass?' pass':''}">${comboLabel}</span>
        <div class="opt-cards">${cardsHtml}</div>
        ${isChosen?'<span class="chosen-mark">✔ 選擇</span>':''}
      </div>
      ${bars}
    </div>`;
  }).join('');

  $('ai-options').innerHTML = header + rows;
}

function renderEvents(events) {
  if (!events || !events.length) {
    $('event-log').innerHTML = '<span class="empty">等待事件…</span>';
    return;
  }
  const reversed = [...events].reverse();
  $('event-log').innerHTML = reversed.map(ev => {
    const isSelf = ev.actor === 'self';
    const msgClass = ev.type === 'pass' ? 'ev-pass' : 'ev-play';
    return `<div class="ev-row">
      <span class="ev-ts">${ev.ts}</span>
      <span class="ev-msg ${msgClass}${isSelf?' ev-self':''}">${ev.msg}</span>
    </div>`;
  }).join('');
}

function render(state) {
  // 遊戲尚未開始（wrapper 剛啟動）
  if (state.__status === 'waiting') {
    $('game-info').innerHTML = '<span style="color:var(--dim)">等待遊戲開始…</span>';
    $('players-tbody').innerHTML = '<tr><td colspan="4" class="empty">—</td></tr>';
    $('table-state').innerHTML = '<span class="empty">—</span>';
    $('ai-options').innerHTML  = '<span class="empty">—</span>';
    $('event-log').innerHTML   = '<span class="empty">—</span>';
    $('stats').innerHTML = '';
    $('updated').textContent = '—';
    $('conn').className='live'; $('conn').textContent='● 即時';
    return;
  }

  const gi=state.game_index??'?', ti=state.trick_index??'?';
  const myTurn = state.turn==='self';
  $('game-info').innerHTML = `第 ${gi} 局 &nbsp;·&nbsp; 第 ${ti} 手`
    + (myTurn ? ' &nbsp;<span class="blink">◀ 輪到你！</span>' : '');

  renderPlayers(state.players);
  renderTableState(state.constraint||{});
  renderAiOptions(state.legal_actions, state.last_decision);
  renderEvents(state.events);

  const s=state.session||{};
  const d=s.decisions??0, pl=s.plays??0, pa=s.passes??0, fb=s.fallbacks??0;
  const ts=s.mcts_total_sims??0, tt=s.mcts_total_time_s??0;
  const avg = pl>0 ? Math.round(ts/pl) : 0;
  $('stats').innerHTML = [
    ['決策',d],['出牌',pl],['PASS',pa],
    [`備用<span class="${fb>0?'warn':'ok'}">${fb}</span>`,''],
    ['平均模擬',avg.toLocaleString()],
    ['MCTS 時間',tt.toFixed(1)+'s'],
  ].map(([l,v])=>`<span class="stat-item"><span class="lbl">${l}</span>${v!==''?`<span class="val">${v}</span>`:''}</span>`).join('');

  $('updated').textContent = `更新：${state.updated_at||'?'}`;
  $('conn').className='live'; $('conn').textContent='● 即時';
}

function connect() {
  const es = new EventSource('/events');
  es.onmessage = e => { try { render(JSON.parse(e.data)); } catch(_){} };
  es.onerror   = () => {
    $('conn').className='wait'; $('conn').textContent='● 重新連線…';
    es.close();
    setTimeout(connect, 2000);
  };
}
connect();
</script>
</body>
</html>
"""

# ── HTTP Handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = _HTML.encode()
            self._respond(200, "text/html; charset=utf-8", body)

        elif self.path == "/state":
            try:
                body = DASH_STATE.read_bytes()
                self._respond(200, "application/json; charset=utf-8", body,
                              extra=[("Cache-Control", "no-cache")])
            except FileNotFoundError:
                self._respond(404, "text/plain", b"not found yet")

        elif self.path == "/events":
            self._sse()

        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, code: int, ct: str, body: bytes,
                 extra: list[tuple[str, str]] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=20)
        with _clients_lock:
            _clients.append(q)

        # 先送一次現有狀態（讓頁面開啟就有資料）
        if _last_bytes:
            try:
                self.wfile.write(b"data: " + _last_bytes + b"\n\n")
                self.wfile.flush()
            except OSError:
                with _clients_lock:
                    if q in _clients:
                        _clients.remove(q)
                return

        try:
            while True:
                try:
                    data = q.get(timeout=25)
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                except queue.Empty:
                    # Heartbeat 讓連線不中斷
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except OSError:
            pass
        finally:
            with _clients_lock:
                if q in _clients:
                    _clients.remove(q)

    def log_message(self, *_) -> None:
        pass  # 靜音 HTTP log


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="大老二 AI 看板伺服器")
    parser.add_argument("--port", type=int, default=7373)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("localhost", args.port), _Handler)
    print(f"大老二 AI 看板：http://localhost:{args.port}  (Ctrl+C 停止)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
