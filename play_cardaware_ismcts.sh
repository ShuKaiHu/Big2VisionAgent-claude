#!/usr/bin/env bash
# 一鍵啟動線上對局 —— 未來主線的 3-model ISMCTS:
#   policy = PPO_V4.pt(真人資料訓練)
#   belief = BELIEF.pt(真人資料,82.8% P@count)→ 決定每個模擬的 determinized world
#   value  = VALUE_minplays.pt(公開資訊 + 最少出牌數,ablation 驗證的部署 value)
# 跟 play_cardaware.sh 同款,只是把三顆模型 + ISMCTS 全部打開。
#
# 用法:
#   ./play_cardaware_ismcts.sh [局數]
#   ./play_cardaware_ismcts.sh 50
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO="/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo"
GAMES="${1:-30}"

export BIG2_AGENT_COMMAND="$HERE/cardaware_wrapper.py"
export CARDAWARE_DIR="$PPO"
# 預設 policy = 人類 V4;要測其他版本(如 V4PBV_IterN)就 export CARDAWARE_CKPT=/abs/path.pt。
export CARDAWARE_CKPT="${CARDAWARE_CKPT:-$PPO/ppo/checkpoints/saved/PPO_V4.pt}"

# 3-model ISMCTS。VALUE_CKPT 預設就是 VALUE_minplays.pt(wrapper 內建),要換 value
# 就 export VALUE_CKPT=/abs/path.pt(extra_dim 會自動偵測)。
export BELIEF_SEARCH=1        # 用 BELIEF.pt 抽 determinized world
export VALUE_LEAF=1           # leaf = value net(而非 rollout 到底)
export ISMCTS=1              # 單棵 information-set 樹,每次模擬重抽世界
export ISMCTS_SIMS="${ISMCTS_SIMS:-200}"
export CARDAWARE_BUDGET="${CARDAWARE_BUDGET:-1.0}"   # 每手約 1s,配合線上計時器

echo "[play] 3-model ISMCTS"
echo "[play]   policy = $(basename "$CARDAWARE_CKPT")"
echo "[play]   belief = BELIEF.pt (BELIEF_SEARCH=1)"
echo "[play]   value  = ${VALUE_CKPT:-VALUE_minplays.pt} (VALUE_LEAF=1, extra_dim auto)"
echo "[play]   sims=$ISMCTS_SIMS budget=${CARDAWARE_BUDGET}s | 局數=$GAMES"
echo "[play] 驗證:run.log 的 note 應為 'cardaware+ismcts+belief+value sims=.. (..s)'"

# ── 看板伺服器:沒在跑就自動起一個(背景、detached,這個腳本結束後繼續留著），
# 這樣只要跑這支腳本，http://localhost:7373 就看得到，不用再開第二個 terminal。
DASH_PORT=7373
if ! lsof -i ":$DASH_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[play] 看板未啟動，背景啟動中…"
    nohup "$HERE/.venv/bin/python3" "$HERE/dashboard/game_dashboard.py" --port "$DASH_PORT" \
        > "$HERE/dashboard/server.log" 2>&1 &
    disown
    for _ in $(seq 1 20); do
        lsof -i ":$DASH_PORT" -sTCP:LISTEN -t >/dev/null 2>&1 && break
        sleep 0.3
    done
fi
if lsof -i ":$DASH_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[play] 看板 = http://localhost:$DASH_PORT/ (log: dashboard/server.log)"
else
    echo "[play] ⚠️ 看板啟動失敗，見 dashboard/server.log" >&2
fi

exec uv run big2-agent autoplay-agent --executor packet --games "$GAMES"
