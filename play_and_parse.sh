#!/usr/bin/env bash
# 一鍵:上線對局 → 記錄版本 → 自動 parse 累積真人資料集。
#
# 用法:
#   ./play_and_parse.sh <模型> <版本標籤> [search 0/1] [局數]
#   ./play_and_parse.sh PPO_V4.pt V4              # V4 純 policy
#   ./play_and_parse.sh PPO_V4.pt V4 1            # V4 + 1s search
#   ./play_and_parse.sh PPO_V1.pt V1 0 15
# 模型:saved/ 下純檔名 或 絕對路徑。版本標籤會記進每局 seats(我方座位)。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
MODEL="${1:?usage: ./play_and_parse.sh <model> <version-label> [search] [games]}"
LABEL="${2:?need a version label, e.g. V4}"
SEARCH="${3:-0}"
GAMES="${4:-15}"

if   [ -f "$MODEL" ];                                   then CKPT="$MODEL"
elif [ -f "$PPO/ppo/checkpoints/saved/$MODEL" ];        then CKPT="$PPO/ppo/checkpoints/saved/$MODEL"
elif [ -f "$PPO/ppo/checkpoints/$MODEL" ];              then CKPT="$PPO/ppo/checkpoints/$MODEL"
else echo "[play] ✗ 找不到模型: $MODEL" >&2; exit 1; fi

export BIG2_AGENT_COMMAND="$HERE/cardaware_wrapper.py"
export CARDAWARE_DIR="$PPO"
export CARDAWARE_CKPT="$CKPT"
if [ "$SEARCH" = "1" ]; then
  export CARDAWARE_SEARCH="1"; export CARDAWARE_BUDGET="${CARDAWARE_BUDGET:-1.0}"
fi
# BELIEF=1 -> belief-guided PIMC (importance-sample opponent hands from PPO_V6's
# belief head instead of uniform). Needs SEARCH=1. This is the V6 deploy mode.
if [ "${BELIEF:-0}" = "1" ]; then export BELIEF_SEARCH="1"; fi

START=$(date +%s)
echo "[play] $LABEL = $CKPT | search=$SEARCH | belief=${BELIEF:-0} | games=$GAMES"
uv run big2-agent autoplay-agent --executor packet --games "$GAMES" --timeout-seconds "${TIMEOUT:-2400}" || true

echo "[parse] 記錄版本 + 累積真人資料集…"
PYTHONPATH="$PPO" "$PPO/.venv/bin/python" -m ppo.record_run "$LABEL" "$START"
PYTHONPATH="$PPO" "$PPO/.venv/bin/python" -m ppo.parse_online_games
echo "[done] 資料集已更新(累積去重)。"
