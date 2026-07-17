#!/usr/bin/env bash
# 一鍵啟動「belief/dominance 啟發式 v1」線上對局(Big2-Belief-Policy 的 bot.choose_action)。
# 與 play.sh 不同:這支不載入任何 ML 模型,所以沒有 ALPHA_BIG2_CKPT,純 CPU 啟發式。
#
# 背景:整條探索(AlphaZero value+MCTS、CFR、出完計畫/留牌)離線收斂的結論 = 簡單積極丟牌的
# v1 啟發式最強,任何留牌/計畫層都降分(見 AlphaBig2-claude memory)。唯一剩的裁判是線上對真人,
# 這支就是把 v1 推到真人面前測 reward(avg_score)。
#
# 用法:
#   ./play_heuristic.sh [局數]        # 預設 30 局
#
# 開跑後驗證:tail artifacts/heuristic_moves.jsonl 看 "agent":"heuristic_v1" 逐手紀錄。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAMES="${1:-30}"

export BIG2_AGENT_COMMAND="$HERE/heuristic_wrapper.py"
# bot 決策核心所在(可用 BIG2_BOT_DIR 覆寫);預設 Big2-Belief-Policy。
export BIG2_BOT_DIR="${BIG2_BOT_DIR:-/Users/shukaihu/Code_Project_Local/Big2-Belief-Policy}"

echo "[play] agent = 啟發式 v1 (heuristic_wrapper.py,無模型)"
echo "[play] BIG2_AGENT_COMMAND = $BIG2_AGENT_COMMAND"
echo "[play] BIG2_BOT_DIR       = $BIG2_BOT_DIR"
echo "[play] 局數 = $GAMES"
echo "[play] 驗證:tail artifacts/heuristic_moves.jsonl 看逐手 (agent=heuristic_v1)"
exec uv run big2-agent autoplay-agent --executor packet --games "$GAMES"
