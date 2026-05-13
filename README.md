# Big2 Vision Agent

Playwright-based browser automation for playing [神來也大老二](https://www.gamesofa.com/bigtwo/#) (Gamesofa Big2). It launches a real Chromium, navigates from the lobby into a game, and drives turns through an agent interface — decoupled so you can plug in any decision model.

## Architecture

```
Perception → Decision → Action
```

**Perception** — injects a JS hook that captures every WebSocket packet into `window.__big2NetworkLog`, decodes card codes, classifies combo types, and produces a structured `AgentObservation` (hand cards, turn, last play, legal actions).

**Decision** — `DecisionAgent` protocol with two built-in implementations:
- `FallbackDecisionAgent` — greedy: play the first legal non-pass move
- `ExternalCommandAgent` — pipes `AgentObservation` JSON to `$BIG2_AGENT_COMMAND` on stdin, reads `AgentDecision` JSON on stdout (**ML model hook**)

**Action** — two executor backends:
- `gui` — Cocos sprite-toggle API (no pixel clicks; avoids off-by-one overlap issues)
- `packet` — raw WebSocket `send` / `pass` commands (faster, no UI dependency)

## Project layout

```
src/big2_vision_agent/
├── main.py            # CLI entry point (all subcommands)
├── config.py          # Settings via env vars
├── agent_schema.py    # Pydantic: AgentCard / AgentDecision / AgentObservation
├── agent_runtime.py   # DecisionAgent protocol + Fallback + External impls
├── action_executor.py # execute_agent_decision — the hot path
├── packet_state.py    # builds AgentObservation from WS timeline
├── network_parser.py  # WS payload decoder + combo classifier
└── browser/
    ├── session.py     # Playwright launch + WS network hook
    ├── actions.py     # Cocos JS evals: game state, sprite toggles, ws_send
    ├── inspector.py   # Scene tree / page summary dumpers
    └── selectors.py   # Text-based selectors (rarely used)
```

Per-run artifacts saved to `artifacts/<timestamp>/autoplay_agent/`:
`run.log`, `action_log.json`, `network_log.json`, `game_timeline.json`, screenshots, optional video.

## Setup

```bash
uv sync
uv run playwright install chromium

# One-time interactive login (saves profile to state/)
uv run big2-agent login
```

## Usage

```bash
# Play 1 game then stop (default)
uv run big2-agent autoplay-agent --executor packet

# Play 3 games
uv run big2-agent autoplay-agent --executor packet --games 3

# Record video
uv run big2-agent autoplay-agent --executor packet --record-video

# Open the Cocos lobby and wait for input (diagnostic / popup inspection)
uv run big2-agent lobby-wait

# One-shot game state dump
uv run big2-agent game-state

# Capture WebSocket traffic for 30 s
uv run big2-agent network-capture --seconds 30

# Dump Cocos scene tree
uv run big2-agent scene-dump

# Inspect a specific node
uv run big2-agent node-probe --name QuickJoinButton
```

## Plugging in an ML model

Set `BIG2_AGENT_COMMAND=/path/to/your/model_wrapper`. The wrapper reads an `AgentObservation` JSON from stdin and writes an `AgentDecision` JSON to stdout. Schema is in `agent_schema.py`. No code changes needed in this repo.

```python
# AgentObservation (stdin)
{
  "self_hand": [{"code": "1A", "display": "SA", "rank": "A", "suit": "S"}, ...],
  "hand_count": 13,
  "turn": "self",
  "constraint": {"required_combo_type": "single", "last_played_cards": [...], ...},
  "legal_actions": [{"action": "play", "cards": [...], "combo_type": "single"}, ...]
}

# AgentDecision (stdout)
{"action": "play", "card_codes": ["1A"], "combo_type": "single"}
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `BIG2_TARGET_URL` | gamesofa URL | Target page URL |
| `BIG2_HEADLESS` | `false` | Run browser headless |
| `BIG2_TIMEOUT_MS` | `30000` | Page operation timeout |
| `BIG2_STATE_PATH` | `state/storage_state.json` | Login session path |
| `BIG2_PROFILE_DIR` | `state/browser-profile` | Persistent Chromium profile |
| `BIG2_ARTIFACT_DIR` | `artifacts` | Output root directory |
| `BIG2_AGENT_COMMAND` | _(unset)_ | External decision model command |
