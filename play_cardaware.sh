#!/usr/bin/env bash
# 一鍵啟動線上對局，用 cardaware PPO 模型(非 MCTS)。
# 跟 play.sh 同款,只是把決策腦袋換成 cardaware_wrapper.py。
#
# 用法:
#   ./play_cardaware.sh [模型檔] [局數]
#   ./play_cardaware.sh                         # 預設 ppo_cardaware_best.pt,30 局
#   ./play_cardaware.sh ppo_cardaware_latest.pt 50
#   ./play_cardaware.sh /abs/path/to.pt 30
#
# 模型檔接受:絕對路徑 / AlphaBig2-ppo/ppo/checkpoints 下的純檔名。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
CKPT_DIR="$PPO/ppo/checkpoints"
MODEL="${1:-ppo_cardaware_best.pt}"
GAMES="${2:-30}"

export BIG2_AGENT_COMMAND="$HERE/cardaware_wrapper.py"
export CARDAWARE_DIR="$PPO"

if   [ -f "$MODEL" ];            then CKPT="$MODEL"
elif [ -f "$CKPT_DIR/$MODEL" ]; then CKPT="$CKPT_DIR/$MODEL"
else echo "[play] ✗ 找不到 cardaware 模型: $MODEL" >&2; exit 1; fi
export CARDAWARE_CKPT="$CKPT"

echo "[play] 模型(cardaware) = $CKPT"
echo "[play] BIG2_AGENT_COMMAND = $BIG2_AGENT_COMMAND"
echo "[play] 局數 = $GAMES"
echo "[play] 驗證:run.log 的 note 應為 'cardaware p=.. v=.. L=..'(不是 mcts)"
exec uv run big2-agent autoplay-agent --executor packet --games "$GAMES"
