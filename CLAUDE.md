# CLAUDE.md — Big2 Vision Agent

Context for Claude Code sessions. Read this first.

## What this project does

Playwright-based browser automation for playing **神來也大老二** (Gamesofa Big2)
at `https://www.gamesofa.com/bigtwo/#`. Launches a real Chromium, navigates
from lobby into a game, then drives turns either via a built-in fallback
agent or through an external decision agent (intended for plugging in an ML
model).

Python 3.12+, uses `uv`, depends on `playwright`, `pydantic`.

## Architecture: Perception → Decision → Action

Three layers, deliberately decoupled so you can swap the brain without
touching perception or action:

**Perception** — `src/big2_vision_agent/browser/session.py` injects a JS hook
(`NETWORK_HOOK_SCRIPT`) that captures every WebSocket / fetch / XHR into
`window.__big2NetworkLog`. `network_parser.py` decodes card codes and
classifies combo types from captured traffic. `packet_state.py` plus
`browser/actions.py::read_big2_game_state` produce an `AgentObservation`
(`agent_schema.py`) — hand cards, turn, last play, legal action options.

**Decision** — `agent_runtime.py` defines a `DecisionAgent` protocol.
- `FallbackDecisionAgent` — "能不 pass 就不 pass", picks the first legal play.
- `ExternalCommandAgent` — pipes the `AgentObservation` as JSON on stdin to
  the program named in env var `BIG2_AGENT_COMMAND`, expects an
  `AgentDecision` JSON on stdout. **This is the ML model hook.**

**Action** — `action_executor.py::execute_agent_decision` translates a
decision into UI events. Order matters:
1. Clear any stale selection (`_clear_selected_cards`).
2. For plays: click the combo-type button (單張/一對/順子…). The game both
   filters and auto-suggests; the click also acts as an implicit reset.
3. Explicitly clear the auto-suggestion that step 2 produced.
4. Sprite-toggle each target card via `toggle_my_card_by_sprite`
   (Cocos-API, NOT pixel clicks — see "Lessons learned").
5. Click Play.
6. Wait 800 ms (+ one 800 ms retry) then verify `hand_count` / `turn` changed.

## Where things live

```
src/big2_vision_agent/
├── main.py            # argparse CLI with all subcommands
├── config.py          # Settings (env vars)
├── agent_schema.py    # Pydantic: AgentCard / AgentDecision / AgentObservation
├── agent_runtime.py   # DecisionAgent protocol + Fallback + External impls
├── action_executor.py # execute_agent_decision — the hot path
├── packet_state.py    # builds AgentObservation from timeline + runtime state
├── network_parser.py  # WS payload decoder + combo classification
└── browser/
    ├── session.py     # Playwright launch + WS network hook
    ├── actions.py     # Cocos JS evals: read_big2_game_state, sprite toggles, click helpers
    ├── inspector.py   # Page summary / scene tree dumpers
    └── selectors.py   # Text hints only (rarely used)
```

Per-run artifacts: `artifacts/<timestamp>/autoplay_agent/` — `run.log`,
`action_log.json` (state+decision+result tuples), `network_log.json`,
`game_timeline.json`, `parsed_events.json`, `turn_summary.json`, screenshots
at every key step, and optionally `video/` (with `--record-video`).

Login state: `state/storage_state.json` + persistent Chromium profile at
`state/browser-profile/`. Run `uv run big2-agent login` once, manually
complete login + click 開始遊戲 in the browser, hit Enter in the terminal.

## Common commands

```bash
uv sync && uv run playwright install chromium               # first-time setup
uv run big2-agent login                                     # interactive login
uv run big2-agent autoplay-agent --timeout-seconds 300 --record-video
uv run big2-agent autoplay-random --timeout-seconds 240     # legacy — see below
uv run big2-agent game-state                                # one-shot state dump
uv run big2-agent network-capture --seconds 30
uv run big2-agent parse-network-log --path artifacts/.../network_log.json
uv run big2-agent scene-dump                                # dump Cocos scene tree
uv run big2-agent node-probe --name <NodeName>              # inspect a Cocos node
```

**`autoplay-agent` is the canonical entry point.** `autoplay-random` has the
naive pixel-click selection that hits the off-by-one bug — keep for reference
only, do not use for real plays.

## Lessons learned (do NOT re-discover these)

**Sprite-toggle, not pixel clicks.** The hand on canvas has heavily
overlapping cards (~70% overlap). Pixel-clicking card center hits the next
card; offset clicks overshoot the other way. `autoplay-random` exhibits this
as deterministic off-by-one ("亂舉牌"). `autoplay-agent` avoids it by going
through `toggle_my_card_by_sprite`, which finds the card by sprite frame name
via Cocos JS and selects it via `cardComponent.setSelect(true)`.

**The combo-type button click cannot be removed.** When the executor clicks
單張/一對/順子 to declare the combo type, the game (a) filters playable
cards, (b) auto-suggests a candidate combo, and (c) **clears the previous
selection as a side effect**. (c) is load-bearing — without it, selection
accumulates across failed attempts and you get `selection_mismatch` errors.
We tried removing this click; selection_mismatch jumped from 0 to 7+ within a
single game. The executor now clicks the combo button, explicitly clears the
auto-suggestion, then sprite-toggles its real targets.

**Verifier needs ≥ 800 ms wait.** Click Play → immediately read state =
false-negative `play_not_confirmed`, because Cocos animation + WS round-trip
hasn't landed yet. `execute_agent_decision` waits 800 ms, then retries once
with another 800 ms if no change. This took false-negative rate from
"almost every play" to 0.

**Cancel button (取消) is unreliable.** It's reported as `active` and clickable
but often doesn't actually deselect (root cause unconfirmed — may be stale
state read, may be a different button semantics than we assumed). Both
`clear_selected_cards` (main.py) and `_clear_selected_cards`
(action_executor.py) now fall through to per-card `toggle_my_card_by_sprite`
deselection if cancel didn't reduce `my_selected_count`.

**`_card_click_point` was a latent typo.** The correct symbol is
`_card_click_points` (plural, list-returning). The singular form was only
reached when cancel failed AND a follow-up path ran, then NameError-crashed.
Already fixed; mentioned here because it was a real production failure.

## WebSocket protocol (reverse-engineered from logs)

Text, space-separated. `cards_blob` is a sequence of 2-char codes
`<suit_digit><rank_char>` where suits are `1=S 2=H 3=D 4=C` and ranks include
`T` for 10 and `1` for Ace.

Outgoing (client → server):

| Command | Args | Meaning |
|---|---|---|
| `send` | `<code> <cards_blob>` | Play cards. `<code>` looks like a request correlation id; exact semantics not fully nailed down. |
| `pass` | — | Pass. |

Incoming (server → client):

| Command | Args | Meaning |
|---|---|---|
| `play` | `<actor_index> <mode> <candidate_mask> <cards_blob>` | Turn / hand snapshot. `mode=1` ⇒ "this is your hand revealed". |
| `plsend` | `<actor_index> <cards_blob>` | A player played cards. |
| `plpass` | `<actor_index>` | A player passed. |
| `send` | `<code> <cards_blob>` | Server response to our outgoing `send`. |
| `showScore` | `<actor> <score> <remaining_cards>` | Round result. |
| `newRoom` / `sJS2` | room state | Lobby / room events. |
| `startGame` / `gsstart` / `finish1` / `finish2` / `finish3` | — | Game lifecycle. |

Going packet-level (skipping GUI clicks entirely) is a strong candidate for
the next major change. The hook already captures every send/recv; to drive
plays via WS we'd need to:
1. Stash a reference to the live WebSocket inside the hook
   (`window.__big2GameWebSocket = ws`)
2. Confirm `target_code` semantics by inspecting a real run's outgoing `send`s
3. Add a `PacketActionExecutor` that replaces GUI clicks with
   `ws.send("send <code> <blob>")` / `ws.send("pass")`

## Plugging in an ML model

Set `BIG2_AGENT_COMMAND=/path/to/your/model_wrapper`. The wrapper reads a
JSON `AgentObservation` from stdin and writes a JSON `AgentDecision` to
stdout. Schema in `agent_schema.py`. **No code changes needed in this repo.**

Recommended data collection: run `autoplay-agent --record-video` repeatedly,
mine `game_timeline.json` + `action_log.json` for state-action pairs. The WS
protocol gives ground truth for every player's plays (including opponents'),
not just the agent's.

## Open issues (as of 2026-05-12)

1. **Lobby selector returns `text=None`.** `read_lobby_selector` resolves the
   node path (`LobbyLayer > BottomPanel > AreaSettingGroup > NormalSetting >
   MatchSettingNode > RuleNode > ContentNode > ContentLabel`) and finds the
   node, but the label's `.string` is null on the first lobby visit. Result:
   `ensure_normal_rule` and `ensure_min_amount` return None and quick-play
   fires with whatever defaults are on screen (often 換牌局 + 20元 instead
   of the desired 正常局 + min amount). `wait_for_lobby_settings_ready` polls
   for "selector exists" but should also wait for "text is non-empty"; even
   then the underlying mystery (why text stays null) wants a `scene-dump` of
   the live lobby to verify path correctness.

2. **Cancel button doesn't reliably clear selection.** See "Lessons learned".
   Fallback exists; root cause TBD.

3. **Game canvas auto-rescales** between scene transitions ("呼大呼小"). Does
   not affect sprite-toggle but breaks any pixel-based interaction (combo
   buttons, Play button). Reading geometry fresh before each click partly
   compensates; locking viewport size would be cleaner.

4. **`autoplay-random` pixel-click path has off-by-one.** Don't fix — just
   don't use it for real plays.

5. **In-game popups (e.g. "房間已滿") can stall the agent** if it lands on a
   full room. `maybe_clear_lobby_popup` handles some but not all.

## House rules

- Prefer `autoplay-agent` over `autoplay-random` for everything except
  pixel-click regression testing.
- Always read the latest `run.log` before guessing why something failed —
  the script logs aggressively.
- When adding new code paths, double-check NameErrors aren't silently
  swallowed (see the `_card_click_point` lesson).
- The `BIG2_RULE_TARGET` we *want* in the lobby is the "no card swap"
  variant. The game's actual label for that is **`正常局`** (despite the
  intuitive name being "不換牌局" — that string does not exist in the lobby
  cycle). `DEFAULT_RULE_TARGET` in `main.py` is the knob.
- `DEFAULT_AMOUNT_TARGET` is now superseded by `ensure_min_amount` cycling
  to find the numerically smallest; the constant is mostly cosmetic.

## Suggested next steps (in rough order)

1. Fix the lobby `text=None` issue (`scene-dump` + structural verification).
2. Decide: keep iterating on GUI automation, or move to packet-level. The
   data needed to do the latter is already in any `network_log.json`.
3. Plug in a real decision model via `BIG2_AGENT_COMMAND` and start
   collecting self-play data.
