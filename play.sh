#!/usr/bin/env bash
# 一鍵啟動線上對局，自動設好兩個必要環境變數，避免「忘了設 → 測錯模型/跑成笨 agent」。
#   BIG2_AGENT_COMMAND  → 指向 ML wrapper（沒設就會退回內建笨 FallbackDecisionAgent）
#   ALPHA_BIG2_CKPT     → 指向要測的模型（沒設就用預設 best.pt = V9a）
#
# 用法:
#   ./play.sh <模型> [局數]
#   ./play.sh v9c_combo_strongopp_deploy.pt 30   # 測 V9c，30 局
#   ./play.sh v9a_fullinfo_deploy.pt 30          # 測 V9a baseline
#   ./play.sh                                    # 預設 best.pt (V9a)，30 局
#
# 模型參數接受:絕對路徑 / checkpoints 下相對路徑 / 純檔名(自動找 saved/)。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB2="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
MODEL="${1:-}"
GAMES="${2:-30}"

export BIG2_AGENT_COMMAND="$HERE/alpha_big2_wrapper.py"

if [ -n "$MODEL" ]; then
  if   [ -f "$MODEL" ];                                   then CKPT="$MODEL"
  elif [ -f "$AB2/engine/checkpoints/$MODEL" ];           then CKPT="$AB2/engine/checkpoints/$MODEL"
  elif [ -f "$AB2/engine/checkpoints/saved/$MODEL" ];     then CKPT="$AB2/engine/checkpoints/saved/$MODEL"
  else echo "[play] ✗ 找不到模型: $MODEL" >&2; exit 1; fi
  export ALPHA_BIG2_CKPT="$CKPT"
  echo "[play] 模型 = $CKPT"
else
  echo "[play] 模型 = 預設 best.pt (V9a)"
fi

echo "[play] BIG2_AGENT_COMMAND = $BIG2_AGENT_COMMAND"
echo "[play] 局數 = $GAMES"
echo "[play] 開跑後驗證:tail mcts_moves.jsonl 看 \"ckpt\" 欄位是否為目標模型(V9c 應為 ...s312+v)"
exec uv run big2-agent autoplay-agent --executor packet --games "$GAMES"
