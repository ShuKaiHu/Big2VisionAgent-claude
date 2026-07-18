#!/usr/bin/env bash
# PPO_V2 = PPO_V1 權重 + 推論期 PIMC search(determinized rollout)。
# 不是新模型檔,是「V1 + 搜尋」這個更強的 agent。search 預算預設 1 秒/手。
#
# 用法:
#   ./play_cardaware_v2.sh [局數(session)]            # 預設跑到 timeout
#   ./play_cardaware_v2.sh 15
# 可調:CARDAWARE_BUDGET(秒/手)、CARDAWARE_TOPM(候選動作數)、CARDAWARE_WORLDS(上限)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-claude"
GAMES="${1:-15}"

export BIG2_AGENT_COMMAND="$HERE/cardaware_wrapper.py"
export CARDAWARE_DIR="$PPO"
export CARDAWARE_CKPT="$PPO/ppo/checkpoints/saved/PPO_V1.pt"   # V2 = V1 weights + search
export CARDAWARE_SEARCH="1"
export CARDAWARE_BUDGET="${CARDAWARE_BUDGET:-1.0}"             # ~1s/move
export CARDAWARE_TOPM="${CARDAWARE_TOPM:-4}"

echo "[play] PPO_V2 = PPO_V1 + search (budget ${CARDAWARE_BUDGET}s/move, topM ${CARDAWARE_TOPM})"
echo "[play] CKPT = $CARDAWARE_CKPT"
echo "[play] 驗證:run.log note 應為 'cardaware+search worlds=.. q=..'"
exec uv run big2-agent autoplay-agent --executor packet --games "$GAMES" --timeout-seconds 2400
