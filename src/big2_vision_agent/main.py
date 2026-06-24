from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from big2_vision_agent.browser.actions import (
    click_named_node,
    click_canvas_design_position,
    click_design_point,
    deselect_all_selected_cards,
    invoke_node,
    invoke_named_node,
    probe_nodes,
    probe_nodes_by_name,
    read_big2_game_state,
    read_current_scene,
    resolve_live_page,
    inspect_my_card_by_sprite,
    toggle_my_card_by_sprite,
)
from big2_vision_agent.browser.inspector import dump_scene_tree, inspect_page, save_network_log
from big2_vision_agent.browser.session import BrowserSession
from big2_vision_agent.config import Settings
from big2_vision_agent.agent_runtime import build_decision_agent, sample_random_decision
from big2_vision_agent.packet_state import build_agent_observation, build_live_agent_observation
from big2_vision_agent.action_executor import _card_click_points, execute_agent_decision, execute_packet_decision
from big2_vision_agent.decision_review import write_markdown_report
from big2_vision_agent.session_review import write_markdown_report as write_session_review
from big2_vision_agent.training_export import write_training_export
from big2_vision_agent.network_parser import (
    build_sprite_card_mapping_report,
    format_turn_summary_text,
    build_game_timeline,
    load_parse_and_build_timeline,
    parse_network_entries,
    summarize_turns,
)
from big2_vision_agent.ml_state import build_ml_state
from big2_vision_agent import dashboard_writer

# 看板狀態檔（與 alpha_big2_wrapper.py 的 _DASH_STATE_PATH 同一路徑）
_DASH_STATE_PATH = str(Path(__file__).resolve().parent.parent.parent / "state" / "dashboard_state.json")

DEFAULT_QUICK_PLAY_X = 885.0
DEFAULT_QUICK_PLAY_Y = 948.0
DEFAULT_POPUP_CONFIRM_X = 864.0
DEFAULT_POPUP_CONFIRM_Y = 741.0
DEFAULT_EVENT_CLOSE_X = 1695.0
DEFAULT_EVENT_CLOSE_Y = 68.0
DEFAULT_MODAL_CLOSE_X = 1308.0
DEFAULT_MODAL_CLOSE_Y = 257.0
DEFAULT_AMOUNT_TARGET = "10元"
DEFAULT_RULE_TARGET = "不換牌"
IDLE_POLL_MS = 180
POST_ACTION_WAIT_MS = 220


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gamesofa Big2 browser scaffold.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login", help="Open the game page and save storage state.")
    subparsers.add_parser("inspect", help="Capture screenshot, HTML, and page summary.")
    click_parser = subparsers.add_parser(
        "click-design",
        help="Click the canvas using Cocos design-resolution coordinates.",
    )
    click_parser.add_argument("--x", type=float, required=True, help="Design-space x.")
    click_parser.add_argument("--y", type=float, required=True, help="Design-space y.")
    click_parser.add_argument(
        "--wait-ms",
        type=int,
        default=1500,
        help="Wait time after click before finishing.",
    )
    quick_play_parser = subparsers.add_parser(
        "quick-play",
        help="Click the lobby quick-play button and capture the resulting state.",
    )
    quick_play_parser.add_argument(
        "--x",
        type=float,
        default=DEFAULT_QUICK_PLAY_X,
        help="Quick-play button design-space x.",
    )
    quick_play_parser.add_argument(
        "--y",
        type=float,
        default=DEFAULT_QUICK_PLAY_Y,
        help="Quick-play button design-space y.",
    )
    quick_play_parser.add_argument(
        "--wait-ms",
        type=int,
        default=5000,
        help="Wait time after clicking quick-play.",
    )
    quick_play_scan_parser = subparsers.add_parser(
        "quick-play-scan",
        help="Scan a small coordinate grid around the lobby quick-play button.",
    )
    quick_play_scan_parser.add_argument(
        "--center-x",
        type=float,
        default=DEFAULT_QUICK_PLAY_X,
        help="Center x in design-space.",
    )
    quick_play_scan_parser.add_argument(
        "--center-y",
        type=float,
        default=DEFAULT_QUICK_PLAY_Y,
        help="Center y in design-space.",
    )
    quick_play_scan_parser.add_argument(
        "--delta-x",
        type=float,
        default=36.0,
        help="Horizontal scan step in design-space.",
    )
    quick_play_scan_parser.add_argument(
        "--delta-y",
        type=float,
        default=24.0,
        help="Vertical scan step in design-space.",
    )
    quick_play_scan_parser.add_argument(
        "--wait-ms",
        type=int,
        default=3500,
        help="Wait time after each click.",
    )
    subparsers.add_parser(
        "scene-dump",
        help="Dump the current Cocos scene tree to an artifact directory.",
    )
    node_probe_parser = subparsers.add_parser(
        "node-probe",
        help="Probe Cocos nodes by exact node name.",
    )
    node_probe_parser.add_argument("--name", required=True, help="Exact node name.")
    click_node_parser = subparsers.add_parser(
        "click-node",
        help="Click the center of an active Cocos node by exact node name.",
    )
    click_node_parser.add_argument("--name", required=True, help="Exact node name.")
    click_node_parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="Active-match occurrence index.",
    )
    click_node_parser.add_argument(
        "--wait-ms",
        type=int,
        default=5000,
        help="Wait time after clicking the node.",
    )
    invoke_node_parser = subparsers.add_parser(
        "invoke-node",
        help="Invoke a Cocos node via button clickEvents or emitted events.",
    )
    invoke_node_parser.add_argument("--name", required=True, help="Exact node name.")
    invoke_node_parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="Active-match occurrence index.",
    )
    invoke_node_parser.add_argument(
        "--wait-ms",
        type=int,
        default=5000,
        help="Wait time after invoking the node.",
    )
    subparsers.add_parser(
        "invoke-quick-join",
        help="Find the currently active quick-join node and invoke it.",
    )
    quick_play_timeline_parser = subparsers.add_parser(
        "quick-play-timeline",
        help="Click quick-play, then save one screenshot per second.",
    )
    quick_play_timeline_parser.add_argument(
        "--x",
        type=float,
        default=DEFAULT_QUICK_PLAY_X,
        help="Quick-play button design-space x.",
    )
    quick_play_timeline_parser.add_argument(
        "--y",
        type=float,
        default=DEFAULT_QUICK_PLAY_Y,
        help="Quick-play button design-space y.",
    )
    quick_play_timeline_parser.add_argument(
        "--seconds",
        type=int,
        default=15,
        help="How many seconds to capture after clicking.",
    )
    popup_timeline_parser = subparsers.add_parser(
        "popup-quick-play-timeline",
        help="Clear the in-game popup, then click quick-play and save one screenshot per second.",
    )
    popup_timeline_parser.add_argument(
        "--popup-x",
        type=float,
        default=DEFAULT_POPUP_CONFIRM_X,
        help="Popup confirm button design-space x.",
    )
    popup_timeline_parser.add_argument(
        "--popup-y",
        type=float,
        default=DEFAULT_POPUP_CONFIRM_Y,
        help="Popup confirm button design-space y.",
    )
    popup_timeline_parser.add_argument(
        "--quick-play-x",
        type=float,
        default=DEFAULT_QUICK_PLAY_X,
        help="Quick-play button design-space x.",
    )
    popup_timeline_parser.add_argument(
        "--quick-play-y",
        type=float,
        default=DEFAULT_QUICK_PLAY_Y,
        help="Quick-play button design-space y.",
    )
    popup_timeline_parser.add_argument(
        "--seconds",
        type=int,
        default=15,
        help="How many seconds to capture after clicking quick-play.",
    )
    subparsers.add_parser(
        "game-state",
        help="Print parsed Big2 game state from the current runtime.",
    )
    autoplay_parser = subparsers.add_parser(
        "autoplay-random",
        help="Enter a game and randomly play until your hand is empty or timeout.",
    )
    autoplay_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=240,
        help="Overall timeout for the autoplay loop.",
    )
    autoplay_parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record a Playwright video for the autoplay run.",
    )
    autoplay_agent_parser = subparsers.add_parser(
        "autoplay-agent",
        help="Enter a game and drive turns through the agent interface.",
    )
    autoplay_agent_parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of complete games to play before stopping.",
    )
    autoplay_agent_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="Hard safety-net timeout in seconds; the games limit takes priority.",
    )
    autoplay_agent_parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record a Playwright video for the autoplay run.",
    )
    autoplay_agent_parser.add_argument(
        "--executor",
        choices=["gui", "packet"],
        default="gui",
        help="Action executor: 'gui' uses Cocos sprite-toggle clicks (default); 'packet' sends raw WebSocket messages.",
    )
    control_probe_parser = subparsers.add_parser(
        "control-probe",
        help="Run a single-step control probe for cancel/select/play with state dumps.",
    )
    control_probe_parser.add_argument(
        "--mode",
        choices=["cancel", "select-cancel", "select-play"],
        default="select-cancel",
        help="Atomic control flow to test.",
    )
    control_probe_parser.add_argument(
        "--card-code",
        help="Specific card code to test, e.g. 4J. Defaults to the first playable card.",
    )
    control_probe_parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record a Playwright video for the probe run.",
    )
    network_parser = subparsers.add_parser(
        "network-capture",
        help="Open the page, wait, and save WebSocket/XHR/fetch logs.",
    )
    network_parser.add_argument(
        "--seconds",
        type=int,
        default=30,
        help="How long to observe network activity before saving.",
    )
    parse_network_parser = subparsers.add_parser(
        "parse-network-log",
        help="Parse a saved network_log.json into structured events.",
    )
    parse_network_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to network_log.json",
    )
    observation_parser = subparsers.add_parser(
        "build-agent-observation",
        help="Build an agent-ready observation from a saved network_log.json.",
    )
    observation_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to network_log.json",
    )
    ml_state_parser = subparsers.add_parser(
        "build-ml-state",
        help="Build an ML-ready game state from a saved network_log.json.",
    )
    ml_state_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to network_log.json",
    )
    decision_parser = subparsers.add_parser(
        "build-agent-decision",
        help="Build a decision from a saved agent_observation.json.",
    )
    decision_parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to agent_observation.json",
    )
    decision_parser.add_argument(
        "--mode",
        choices=["fallback", "random", "external"],
        default="fallback",
        help="How to build the decision.",
    )
    subparsers.add_parser(
        "lobby-wait",
        help="Open the lobby and wait for user input, then dump scene tree + screenshot.",
    )
    return parser


async def wait_for_scene(page, expected_scene: str, timeout_ms: int = 15000) -> str | None:
    deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)
    last_scene = None
    while asyncio.get_running_loop().time() < deadline:
        page = resolve_active_page(page)
        try:
            last_scene = await read_current_scene(page)
        except Exception:
            last_scene = None
        if last_scene == expected_scene:
            return last_scene
        try:
            await page.wait_for_timeout(500)
        except Exception:
            await asyncio.sleep(0.5)
    return last_scene


async def safe_read_current_scene(page) -> str | None:
    try:
        return await read_current_scene(page)
    except Exception:
        return None


async def classify_page_stage(page) -> str:
    """
    Return a human-readable stage string that combines Cocos scene detection
    with URL-pattern fallback.  The main loop uses this everywhere instead of
    bare ``safe_read_current_scene`` so it always knows where it is.

    Return values
    -------------
    "GameScene"        Cocos game in progress
    "LobbyScene"       Cocos lobby (matchmaking / settings)
    "<OtherScene>"     Any other Cocos scene name
    "GameCanvas"       Game-canvas URL loaded but Cocos not yet initialised
    "GamesofaHome"     Gamesofa website main page (no canvas yet)
    "GamesofaLogin"    Gamesofa login page  (?op=login_all)
    "FacebookConsent"  Facebook OAuth / GDPR consent popup
    "Unknown"          None of the above
    """
    # Priority 1: Cocos scene (most specific; only available inside the canvas)
    cocos_scene = await safe_read_current_scene(page)
    if cocos_scene:
        return cocos_scene

    # Priority 2: URL-based classification (canvas not ready yet)
    try:
        url = page.url
    except Exception:
        return "Unknown"

    if "facebook.com" in url:
        return "FacebookConsent"
    if "gamesofa.com/bigtwo/html5" in url or "lobby.php" in url:
        return "GameCanvas"          # canvas page loaded, waiting for Cocos
    if "op=login_all" in url:
        return "GamesofaLogin"
    if "gamesofa.com/bigtwo" in url or "gamesofa.com" in url:
        return "GamesofaHome"
    return "Unknown"


def resolve_active_page(page):
    try:
        if not page.is_closed():
            return page
    except Exception:
        pass
    try:
        context = page.context
        live_pages = [candidate for candidate in context.pages if not candidate.is_closed()]
        if live_pages:
            return live_pages[-1]
    except Exception:
        return page
    return page


async def safe_click_design_position(
    page,
    x: float,
    y: float,
    attempts: int = 3,
    wait_ms: int = 1000,
) -> dict | None:
    for _ in range(attempts):
        try:
            result = await click_canvas_design_position(page, x, y)
            await page.wait_for_timeout(wait_ms)
            return result
        except Exception:
            await page.wait_for_timeout(wait_ms)
    return None


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{timestamp}] {message}\n")


class SingleInstanceGuard:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.acquired = False

    def acquire(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing_pid = None
            try:
                existing_pid = int(self.lock_path.read_text(encoding="utf-8").strip())
            except Exception:
                existing_pid = None
            if existing_pid and _pid_exists(existing_pid):
                raise RuntimeError(
                    f"Another autoplay-agent is already running (pid={existing_pid}). "
                    "Stop it before starting a new one."
                )
            self.lock_path.unlink(missing_ok=True)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        self.lock_path.unlink(missing_ok=True)
        self.acquired = False

    def __enter__(self) -> "SingleInstanceGuard":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def normalize_turn_actor(turn: str | None) -> str | None:
    if turn in {"self", "left", "top", "right"}:
        return turn
    return None


def is_self_actionable_turn(state: dict) -> bool:
    if state.get("turn") != "self":
        return False
    if not state.get("my_clock_active"):
        return False
    action_buttons = state.get("action_buttons", {})
    pass_button = action_buttons.get("pass", {})
    play_button = action_buttons.get("play", {})
    if pass_button.get("active") or play_button.get("active"):
        return True
    # 遊戲開始持有梅花三時，pass/play 按鈕都是灰的（不能 PASS、尚未選牌），
    # 但牌型按鈕（單張/一對/順子…）是 active 的。
    # 用牌型按鈕是否有 active 來補足這個情境。
    card_type_buttons = state.get("card_type_buttons", {})
    return any(
        isinstance(btn, dict) and btn.get("active")
        for btn in card_type_buttons.values()
    )


def summarize_table_play(state: dict) -> dict | None:
    sets = list(state.get("visible_show_card_sets", []))
    if not sets:
        return None

    chosen = max(
        sets,
        key=lambda item: (
            item.get("card_count", 0),
            1 if item.get("card_type_sprite_active") else 0,
            round((item.get("center") or {}).get("y", 0), 1),
        ),
    )
    cards = [card.get("sprite_frame") for card in chosen.get("cards", []) if card.get("sprite_frame")]
    card_count = chosen.get("card_count", 0)
    required_type = state.get("current_required_type")
    if required_type is None:
        if card_count == 1:
            required_type = "single"
        elif card_count == 2:
            required_type = "pair"
        elif card_count == 5:
            required_type = "five_card"

    center = chosen.get("center") or {}
    return {
        "card_count": card_count,
        "cards": cards,
        "required_type": required_type,
        "card_type_sprite_frame": chosen.get("card_type_sprite_frame"),
        "center": {
            "x": round(center.get("x", 0), 1),
            "y": round(center.get("y", 0), 1),
        },
    }


def table_play_signature(state: dict) -> tuple | None:
    summary = summarize_table_play(state)
    if summary is None:
        return None
    return (
        summary.get("card_count"),
        summary.get("required_type"),
        summary.get("card_type_sprite_frame"),
        tuple(summary.get("cards", [])),
        summary["center"]["x"],
        summary["center"]["y"],
    )


def maybe_record_turn_event(
    previous_state: dict | None,
    current_state: dict,
    action_log: list[dict[str, object]],
    logger: RunLogger,
) -> None:
    if previous_state is None:
        return

    previous_turn = normalize_turn_actor(previous_state.get("turn"))
    current_turn = normalize_turn_actor(current_state.get("turn"))
    if previous_turn is None or previous_turn == current_turn:
        return

    previous_signature = table_play_signature(previous_state)
    current_signature = table_play_signature(current_state)
    current_summary = summarize_table_play(current_state)

    event_type = "pass"
    if current_signature != previous_signature and current_summary is not None:
        event_type = "played"

    event = {
        "step": "turn_event",
        "actor": previous_turn,
        "event": event_type,
        "round": current_state.get("round"),
        "table": current_summary,
    }
    action_log.append(event)

    if event_type == "played":
        logger.log(
            "Turn event: "
            f"{previous_turn} played type={current_summary.get('required_type')} "
            f"count={current_summary.get('card_count')} cards={current_summary.get('cards')}"
        )
    else:
        logger.log(f"Turn event: {previous_turn} passed")


async def read_lobby_selector(page, node_name: str) -> dict | None:
    return await page.evaluate(
        """
        (targetNodeName) => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          const view = ccGlobal && ccGlobal.view && ccGlobal.view.getDesignResolutionSize
            ? ccGlobal.view.getDesignResolutionSize()
            : null;
          if (!scene || !view) {
            return null;
          }

          const names = [
            'LobbyLayer',
            'BottomPanel',
            'AreaSettingGroup',
            'NormalSetting',
            'MatchSettingNode',
            targetNodeName,
          ];

          let node = scene;
          for (const name of names) {
            node = node && node.children ? node.children.find((child) => child.name === name) : null;
            if (!node) {
              return null;
            }
          }

          const labelNode = node.children.find((child) => child.name === 'ContentNode');
          const contentLabel = labelNode && labelNode.children
            ? labelNode.children.find((child) => child.name === 'ContentLabel')
            : null;
          const nextButton = node.children.find((child) => child.name === 'NextButton');
          const prevButton = node.children.find((child) => child.name === 'PrevButton');

          function center(node) {
            if (!node || !node.getBoundingBoxToWorld) {
              return null;
            }
            const box = node.getBoundingBoxToWorld();
            return {
              x: box.x + box.width / 2,
              y: view.height - (box.y + box.height / 2),
            };
          }

          let text = null;
          try {
            const label = contentLabel && contentLabel.getComponent && (
              contentLabel.getComponent('cc.Label') || contentLabel.getComponent(ccGlobal.Label)
            );
            text = label && typeof label.string === 'string' ? label.string : null;
          } catch (error) {}

          return {
            text,
            next_center: center(nextButton),
            prev_center: center(prevButton),
          };
        }
        """,
        node_name,
    )


async def invoke_lobby_selector_button(page, node_name: str, button_name: str) -> bool:
    return await page.evaluate(
        """
        ({ targetNodeName, buttonName }) => {
          const ccGlobal = window.cc;
          const scene = ccGlobal && ccGlobal.director && ccGlobal.director.getScene
            ? ccGlobal.director.getScene()
            : null;
          if (!scene) {
            return false;
          }

          function byName(node, name) {
            return (node && Array.isArray(node.children) ? node.children : []).find((child) => child.name === name) || null;
          }

          const names = [
            'LobbyLayer',
            'BottomPanel',
            'AreaSettingGroup',
            'NormalSetting',
            'MatchSettingNode',
            targetNodeName,
            buttonName,
          ];

          let node = scene;
          for (const name of names) {
            node = byName(node, name);
            if (!node) {
              return false;
            }
          }

          let invoked = false;
          try {
            const button = node.getComponent && (node.getComponent('cc.Button') || node.getComponent(ccGlobal.Button));
            if (button && Array.isArray(button.clickEvents) && button.clickEvents.length > 0) {
              ccGlobal.Component && ccGlobal.Component.EventHandler.emitEvents(button.clickEvents, { type: 'click' });
              invoked = true;
            }
          } catch (error) {}

          if (!invoked) {
            try {
              node.emit('click');
              node.emit('touchend');
              invoked = true;
            } catch (error) {}
          }
          return invoked;
        }
        """,
        {"targetNodeName": node_name, "buttonName": button_name},
    )


async def ensure_lobby_selector(
    page,
    node_name: str,
    target_text: str,
    max_attempts: int = 8,
) -> str | None:
    last_text = None
    for _ in range(max_attempts):
        selector = await read_lobby_selector(page, node_name)
        if selector is None:
            return None
        last_text = selector.get("text")
        if last_text == target_text:
            return last_text

        next_center = selector.get("next_center")
        prev_center = selector.get("prev_center")
        if next_center is None and prev_center is None:
            return last_text

        if next_center is not None:
            invoked = await invoke_lobby_selector_button(page, node_name, "NextButton")
            if not invoked:
                await click_canvas_design_position(page, next_center["x"], next_center["y"])
            await page.wait_for_timeout(1000)
            selector = await read_lobby_selector(page, node_name)
            if selector and selector.get("text") == target_text:
                return target_text
            if selector and selector.get("text") != last_text:
                last_text = selector.get("text")
                continue

        if prev_center is not None:
            invoked = await invoke_lobby_selector_button(page, node_name, "PrevButton")
            if not invoked:
                await click_canvas_design_position(page, prev_center["x"], prev_center["y"])
            await page.wait_for_timeout(1000)
            selector = await read_lobby_selector(page, node_name)
            if selector and selector.get("text") == target_text:
                return target_text
            if selector:
                last_text = selector.get("text")

    return last_text


async def wait_for_lobby_settings_ready(page, timeout_ms: int = 12000) -> bool:
    """Poll until the lobby BottomPanel rule selector is readable *and* has text.

    The Cocos LobbyScene flag flips to "LobbyScene" before the BottomPanel
    finishes rendering, so a naive read of read_lobby_selector returns None on
    the first attempt and ensure_lobby_selector then bails out without setting
    rule/amount.  Even once the node path resolves, label.string can still be
    null for a frame or two after FB login — so we require text to be a
    non-empty string before returning True."""
    import asyncio as _asyncio

    deadline = _asyncio.get_running_loop().time() + (timeout_ms / 1000)
    while _asyncio.get_running_loop().time() < deadline:
        selector = await read_lobby_selector(page, "RuleNode")
        if selector is not None and selector.get("text"):
            return True
        try:
            await page.wait_for_timeout(400)
        except Exception:
            await _asyncio.sleep(0.4)
    return False


async def _cycle_lobby_options(
    page,
    node_name: str,
    max_cycles: int = 16,
) -> list[str]:
    """Cycle a lobby selector through every distinct option, returning the texts
    in the order they were observed. Stops as soon as we see a repeat."""
    seen: list[str] = []
    for _ in range(max_cycles):
        selector = await read_lobby_selector(page, node_name)
        if selector is None:
            await page.wait_for_timeout(400)
            continue
        text = selector.get("text")
        if text is None:
            break
        if text in seen:
            break
        seen.append(text)
        invoked = await invoke_lobby_selector_button(page, node_name, "NextButton")
        if not invoked:
            next_center = selector.get("next_center")
            if next_center is None:
                break
            await click_canvas_design_position(page, next_center["x"], next_center["y"])
        await page.wait_for_timeout(700)
    return seen


async def ensure_normal_rule(page, max_attempts: int = 12) -> str | None:
    """Pick the "no card swap" rule from whatever the lobby cycle actually offers.

    Order of preference:
      1. Exact match for DEFAULT_RULE_TARGET (so user can override).
      2. Any option whose label contains "不換".
      3. Any option whose label contains "正常" (game's no-swap variant).
      4. Whatever the cycle landed on last as a final fallback.
    """
    if not await wait_for_lobby_settings_ready(page):
        return None

    options = await _cycle_lobby_options(page, "RuleNode", max_cycles=max_attempts)
    if not options:
        return None

    target = None
    if DEFAULT_RULE_TARGET in options:
        target = DEFAULT_RULE_TARGET
    if target is None:
        for opt in options:
            if "不換" in opt:
                target = opt
                break
    if target is None:
        for opt in options:
            if "正常" in opt:
                target = opt
                break
    if target is None:
        target = options[-1]

    return await ensure_lobby_selector(page, "RuleNode", target, max_attempts=max_attempts * 2)


async def ensure_min_amount(page, max_attempts: int = 12) -> str | None:
    """Cycle through all amount options and settle on the numerically smallest.

    Previously this hard-coded DEFAULT_AMOUNT_TARGET ("10元") and silently fell
    back to whatever option the cycle landed on if no exact match was found —
    which meant the agent could end up at an arbitrarily expensive room.
    """
    import re

    if not await wait_for_lobby_settings_ready(page):
        return None

    options = await _cycle_lobby_options(page, "BaseTaiNode", max_cycles=max_attempts)
    if not options:
        return None

    def parse_amount(text: str) -> float:
        if not text:
            return float("inf")
        match = re.search(r"\d+", text)
        return float(match.group(0)) if match else float("inf")

    smallest_text = min(options, key=parse_amount)
    return await ensure_lobby_selector(page, "BaseTaiNode", smallest_text, max_attempts=max_attempts * 2)


async def run_login(settings: Settings) -> None:
    async with BrowserSession(settings) as session:
        print("Opening browser and attempting auto-login…")
        page = await session.new_page()
        await page.goto(settings.target_url, wait_until="domcontentloaded")
        await session._ensure_home_authenticated(page)

        if await session._home_has_start_button(page):
            state_path = await session.save_storage_state()
            print(f"Already logged in. Saved storage state to: {state_path}")
            return

        print("Not logged in — trying Facebook auto-login…")
        logged_in = await session.auto_facebook_login(page)
        if logged_in:
            state_path = await session.save_storage_state()
            print(f"Auto-login succeeded. Saved storage state to: {state_path}")
            return

        print()
        print("Auto-login did not complete (see [FB-login] lines above for details).")
        print("A browser window should be open. Please log in manually,")
        print("wait until you see the 開始遊戲 button, then press Enter here.")
        await asyncio.to_thread(input)
        state_path = await session.save_storage_state()
        print(f"Saved storage state to: {state_path}")


async def run_inspect(settings: Settings) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        artifact_dir = await inspect_page(page, settings)
        print(f"Saved page artifacts to: {artifact_dir}")


async def run_click_design(settings: Settings, x: float, y: float, wait_ms: int) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        result = await click_canvas_design_position(page, x, y)
        await page.wait_for_timeout(wait_ms)
        await session.save_storage_state()
        print(
            "Clicked canvas at design=({design_x:.1f}, {design_y:.1f}) "
            "screen=({screen_x:.1f}, {screen_y:.1f})".format(**result)
        )


async def run_quick_play(settings: Settings, x: float, y: float, wait_ms: int) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        rule_text = await ensure_normal_rule(page)
        amount_text = await ensure_min_amount(page)
        before_scene = await read_current_scene(page)
        click_result = await click_canvas_design_position(page, x, y)
        await page.wait_for_timeout(wait_ms)
        after_scene = await read_current_scene(page)
        artifact_dir = await inspect_page(page, settings)
        await session.save_storage_state()
        print(f"Rule before quick-play settled to: {rule_text}")
        print(f"Amount before quick-play settled to: {amount_text}")
        print(f"Scene before click: {before_scene}")
        print(f"Scene after click: {after_scene}")
        print(
            "Clicked quick-play at design=({design_x:.1f}, {design_y:.1f}) "
            "screen=({screen_x:.1f}, {screen_y:.1f})".format(**click_result)
        )
        print(f"Saved page artifacts to: {artifact_dir}")


async def _run_single_quick_play_probe(
    settings: Settings,
    x: float,
    y: float,
    wait_ms: int,
) -> tuple[str | None, str | None, str]:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        await ensure_normal_rule(page)
        await ensure_min_amount(page)
        before_scene = await read_current_scene(page)
        await click_canvas_design_position(page, x, y)
        await page.wait_for_timeout(wait_ms)
        after_scene = await read_current_scene(page)
        artifact_dir = await inspect_page(page, settings)
        await session.save_storage_state()
        return before_scene, after_scene, str(artifact_dir)


async def run_quick_play_scan(
    settings: Settings,
    center_x: float,
    center_y: float,
    delta_x: float,
    delta_y: float,
    wait_ms: int,
) -> None:
    probes = [
        (center_x, center_y),
        (center_x - delta_x, center_y),
        (center_x + delta_x, center_y),
        (center_x, center_y - delta_y),
        (center_x, center_y + delta_y),
        (center_x - delta_x, center_y - delta_y),
        (center_x + delta_x, center_y - delta_y),
        (center_x - delta_x, center_y + delta_y),
        (center_x + delta_x, center_y + delta_y),
    ]

    for index, (x, y) in enumerate(probes, start=1):
        before_scene, after_scene, artifact_dir = await _run_single_quick_play_probe(
            settings, x, y, wait_ms
        )
        print(
            f"[{index}/{len(probes)}] design=({x:.1f}, {y:.1f}) "
            f"scene: {before_scene} -> {after_scene} artifact: {artifact_dir}"
        )


async def run_scene_dump(settings: Settings) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        await page.wait_for_timeout(1000)
        artifact_dir = await inspect_page(page, settings)
        scene_tree = await dump_scene_tree(page)
        scene_tree_path = artifact_dir / "scene_tree.json"
        scene_tree_path.write_text(
            json.dumps(scene_tree, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved page artifacts to: {artifact_dir}")
        print(f"Saved scene tree to: {scene_tree_path}")


async def run_node_probe(settings: Settings, name: str) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        await page.wait_for_timeout(1000)
        matches = await probe_nodes_by_name(page, name)
        print(json.dumps(matches, ensure_ascii=False, indent=2))


async def run_click_node(
    settings: Settings,
    name: str,
    occurrence: int,
    wait_ms: int,
) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        before_scene = await read_current_scene(page)
        result = await click_named_node(page, name, occurrence)
        await page.wait_for_timeout(wait_ms)
        after_scene = await read_current_scene(page)
        artifact_dir = await inspect_page(page, settings)
        await session.save_storage_state()
        print(f"Scene before click: {before_scene}")
        print(f"Scene after click: {after_scene}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved page artifacts to: {artifact_dir}")


async def run_invoke_node(
    settings: Settings,
    name: str,
    occurrence: int,
    wait_ms: int,
) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        before_scene = await read_current_scene(page)
        result = await invoke_named_node(page, name, occurrence)
        await page.wait_for_timeout(wait_ms)
        after_scene = await read_current_scene(page)
        artifact_dir = await inspect_page(page, settings)
        await session.save_storage_state()
        print(f"Scene before invoke: {before_scene}")
        print(f"Scene after invoke: {after_scene}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved page artifacts to: {artifact_dir}")


async def run_invoke_quick_join(settings: Settings) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        before_scene = await read_current_scene(page)
        candidates = await probe_nodes(page, "QuickJoin", exact=False)
        active_candidates = [candidate for candidate in candidates if candidate.get("active")]
        if not active_candidates:
            raise RuntimeError("No active quick-join nodes were found.")

        result = await invoke_node(page, "QuickJoin", occurrence=0, exact=False)
        await page.wait_for_timeout(5000)
        after_scene = await read_current_scene(page)
        artifact_dir = await inspect_page(page, settings)
        await session.save_storage_state()
        print(f"Scene before invoke: {before_scene}")
        print(f"Scene after invoke: {after_scene}")
        print(json.dumps(active_candidates, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved page artifacts to: {artifact_dir}")


async def run_quick_play_timeline(
    settings: Settings,
    x: float,
    y: float,
    seconds: int,
) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "timeline"
        output_dir.mkdir(parents=True, exist_ok=True)

        rule_text = await ensure_normal_rule(page)
        amount_text = await ensure_min_amount(page)
        before_scene = await read_current_scene(page)
        click_result = await click_canvas_design_position(page, x, y)

        frames: list[dict[str, object]] = []
        for second in range(1, seconds + 1):
            page = resolve_live_page(page)
            await page.wait_for_timeout(1000)
            scene = await read_current_scene(page)
            filename = f"t{second:02d}.png"
            await page.screenshot(path=str(output_dir / filename), full_page=True)
            frames.append(
                {
                    "second": second,
                    "scene": scene,
                    "file": filename,
                }
            )

        summary = {
            "rule_text": rule_text,
            "amount_text": amount_text,
            "before_scene": before_scene,
            "click": click_result,
            "frames": frames,
        }
        (output_dir / "index.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await session.save_storage_state()
        print(f"Saved timeline to: {output_dir}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


async def run_popup_quick_play_timeline(
    settings: Settings,
    popup_x: float,
    popup_y: float,
    quick_play_x: float,
    quick_play_y: float,
    seconds: int,
) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await wait_for_scene(page, "LobbyScene")
        output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "timeline"
        output_dir.mkdir(parents=True, exist_ok=True)

        await page.screenshot(path=str(output_dir / "before_popup_clear.png"), full_page=True)
        popup_click = await click_canvas_design_position(page, popup_x, popup_y)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(output_dir / "after_popup_clear.png"), full_page=True)

        rule_text = await ensure_normal_rule(page)
        amount_text = await ensure_min_amount(page)
        before_scene = await read_current_scene(page)
        quick_play_click = await click_canvas_design_position(page, quick_play_x, quick_play_y)

        frames: list[dict[str, object]] = []
        for second in range(1, seconds + 1):
            await page.wait_for_timeout(1000)
            scene = await read_current_scene(page)
            filename = f"t{second:02d}.png"
            await page.screenshot(path=str(output_dir / filename), full_page=True)
            frames.append(
                {
                    "second": second,
                    "scene": scene,
                    "file": filename,
                }
            )

        summary = {
            "popup_click": popup_click,
            "rule_text": rule_text,
            "amount_text": amount_text,
            "before_scene": before_scene,
            "quick_play_click": quick_play_click,
            "frames": frames,
        }
        (output_dir / "index.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await session.save_storage_state()
        print(f"Saved timeline to: {output_dir}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


async def run_game_state(settings: Settings) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        await page.wait_for_timeout(3000)
        state = await read_big2_game_state(page)
        print(json.dumps(state, ensure_ascii=False, indent=2))


async def run_control_probe(
    settings: Settings,
    mode: str,
    card_code: str | None,
    record_video: bool = False,
) -> None:
    lock_path = settings.lock_dir / "autoplay_agent.lock"
    with SingleInstanceGuard(lock_path):
        output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "control_probe"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir / "video" if record_video else None
        logger = RunLogger(output_dir / "run.log")

        async with BrowserSession(settings, record_video_dir=video_dir) as session:
            page = await session.goto_target()
            page = resolve_active_page(page)
            scene = await safe_read_current_scene(page)
            logger.log(f"Initial scene={scene}")
            if scene != "GameScene":
                page, scene = await ensure_game_scene_from_lobby(page, logger, attempts=3)

            state = await wait_until_self_actionable_turn(page, timeout_seconds=60)
            if not state or not is_self_actionable_turn(state):
                logger.log("Did not reach a self actionable turn in time")
                await save_issue_snapshot(page, output_dir, "self_actionable_not_reached")
                print(f"Saved control probe artifacts to: {output_dir}")
                return

            await save_probe_step(output_dir, page, state, "00_before")
            state = await clear_selected_cards(page, state, [], logger)
            await save_probe_step(output_dir, page, state, "01_after_clear")

            if mode == "cancel":
                logger.log("Cancel probe complete")
                print(f"Saved control probe artifacts to: {output_dir}")
                return

            chosen_code = card_code
            if chosen_code is None:
                playable_indexes = state.get("my_playable_indexes") or []
                if not playable_indexes:
                    logger.log("No playable indexes available for probe")
                    await save_issue_snapshot(page, output_dir, "no_playable_indexes")
                    print(f"Saved control probe artifacts to: {output_dir}")
                    return
                chosen_index = playable_indexes[0]
                chosen_code = str(state["my_cards"][chosen_index]["sprite_frame"])[1:]

            logger.log(f"Probe selecting card_code={chosen_code}")
            inspect = await inspect_my_card_by_sprite(page, f"c{chosen_code}")
            (output_dir / "target_card_component.json").write_text(
                json.dumps(inspect, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            selected_ok, state = await probe_select_card(page, state, chosen_code)
            logger.log(
                f"Probe selection result: ok={selected_ok} selected={selected_card_codes(state)}"
            )
            await save_probe_step(output_dir, page, state, "02_after_select")

            if mode == "select-cancel":
                state = await clear_selected_cards(page, state, [], logger)
                await save_probe_step(output_dir, page, state, "03_after_cancel")
                print(f"Saved control probe artifacts to: {output_dir}")
                return

            play_button = state.get("action_buttons", {}).get("play", {})
            center = play_button.get("center")
            if not (selected_ok and play_button.get("active") and center):
                logger.log("Play probe unavailable after selection")
                await save_issue_snapshot(page, output_dir, "play_probe_unavailable")
                print(f"Saved control probe artifacts to: {output_dir}")
                return

            before_count = state.get("my_hand_count")
            await click_design_point(page, center["x"], center["y"])
            await page.wait_for_timeout(POST_ACTION_WAIT_MS)
            state = await read_big2_game_state(page)
            logger.log(
                "Probe play result: "
                f"hand_count={before_count}->{state.get('my_hand_count')} turn={state.get('turn')}"
            )
            await save_probe_step(output_dir, page, state, "03_after_play")
            print(f"Saved control probe artifacts to: {output_dir}")


async def run_network_capture(settings: Settings, seconds: int) -> None:
    async with BrowserSession(settings) as session:
        page = await session.goto_target()
        output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "network"
        output_dir.mkdir(parents=True, exist_ok=True)
        await page.wait_for_timeout(max(1000, seconds * 1000))
        log_path = await save_network_log(page, output_dir)
        await page.screenshot(path=str(output_dir / "page.png"), full_page=True)
        print(f"Saved network capture to: {output_dir}")
        print(f"Saved network log to: {log_path}")


async def run_parse_network_log(path: Path) -> None:
    parsed, timeline = load_parse_and_build_timeline(path)
    output_path = path.with_name("parsed_events.json")
    timeline_path = path.with_name("game_timeline.json")
    turn_summary_path = path.with_name("turn_summary.json")
    turn_summary_text_path = path.with_name("turn_summary.txt")
    sprite_mapping_path = path.with_name("sprite_card_mapping.json")
    ml_state_path = path.with_name("ml_state.json")
    output_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    turn_summary = summarize_turns(timeline)
    turn_summary_path.write_text(json.dumps(turn_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    turn_summary_text_path.write_text(format_turn_summary_text(turn_summary), encoding="utf-8")
    ml_state_path.write_text(
        json.dumps(build_ml_state(timeline), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    action_log_path = path.with_name("action_log.json")
    if action_log_path.exists():
        action_log = json.loads(action_log_path.read_text(encoding="utf-8"))
        sprite_mapping = build_sprite_card_mapping_report(action_log, timeline)
        sprite_mapping_path.write_text(
            json.dumps(sprite_mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Parsed {len(parsed)} ws events")
    print(f"Built {len(timeline)} timeline events")
    print(f"Saved parsed events to: {output_path}")
    print(f"Saved game timeline to: {timeline_path}")
    print(f"Saved turn summary to: {turn_summary_path}")
    print(f"Saved readable turn summary to: {turn_summary_text_path}")
    print(f"Saved ML state to: {ml_state_path}")
    if action_log_path.exists():
        print(f"Saved sprite mapping to: {sprite_mapping_path}")


async def run_build_agent_observation(path: Path) -> None:
    _, timeline = load_parse_and_build_timeline(path)
    observation = build_agent_observation(timeline)
    output_path = path.with_name("agent_observation.json")
    output_path.write_text(
        json.dumps(observation.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved agent observation to: {output_path}")


async def run_build_ml_state(path: Path) -> None:
    _, timeline = load_parse_and_build_timeline(path)
    ml_state = build_ml_state(timeline)
    output_path = path.with_name("ml_state.json")
    output_path.write_text(
        json.dumps(ml_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved ML state to: {output_path}")


async def run_build_agent_decision(path: Path, mode: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    from big2_vision_agent.agent_schema import AgentObservation

    observation = AgentObservation.model_validate(payload)
    if mode == "random":
        decision = sample_random_decision(observation)
    elif mode == "external":
        decision = build_decision_agent().decide(observation)
    else:
        decision = build_decision_agent().decide(observation)

    output_path = path.with_name("agent_decision.json")
    output_path.write_text(
        json.dumps(decision.model_dump() if decision else None, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved agent decision to: {output_path}")


async def _build_live_observation(page, runtime_state: dict):
    from big2_vision_agent.browser.inspector import read_network_log

    entries = await read_network_log(page)
    parsed_events = parse_network_entries(entries)
    timeline = build_game_timeline(parsed_events)
    return build_live_agent_observation(timeline, runtime_state), parsed_events, timeline


async def _update_dashboard_live(page, runtime_state: dict, logger) -> None:
    """對手回合即時推進看板牌況（四家牌況 + 事件紀錄 + 桌面 + 輪到誰）。

    wrapper 只在輪到我時被呼叫，對手出牌/PASS 時看板靠這裡更新。AI 決策選項與
    對手推測手牌算不出來，write_live 會保留 wrapper 上次寫入的那些區塊。"""
    try:
        observation, _, _ = await _build_live_observation(page, runtime_state)
        dashboard_writer.write_live(_DASH_STATE_PATH, observation.model_dump())
    except Exception as exc:  # 看板更新失敗絕不可影響對局主流程
        logger.log(f"dashboard live update skipped: {exc}")


async def _read_timeline(page):
    """Fetch the current WS-derived game timeline (round_result events carry the
    authoritative SERVER scores, unlike the Cocos-derived hand counts)."""
    from big2_vision_agent.browser.inspector import read_network_log
    entries = await read_network_log(page)
    return build_game_timeline(parse_network_entries(entries))


def _latest_round_scores(timeline):
    """Return {actor: {'score':int,'remaining':int,'seq':int}} for the most
    recent round_result batch (one per actor), or None. Authoritative reward."""
    rr = [e for e in (timeline or []) if e.get("event") == "round_result"]
    if not rr:
        return None
    max_seq = max(e.get("seq", 0) for e in rr)
    out = {}
    for e in rr:
        if e.get("seq", 0) < max_seq - 8:   # keep only the trailing batch (~4 events)
            continue
        a = e.get("actor")
        if a and (a not in out or e.get("seq", 0) > out[a]["seq"]):
            out[a] = {"score": e.get("score"),
                      "remaining": len(e.get("remaining_cards") or []),
                      "seq": e.get("seq", 0)}
    return out or None


async def save_autoplay_snapshot(page, output_dir, index: int, label: str) -> None:
    filename = f"{index:03d}_{label}.png"
    await page.screenshot(path=str(output_dir / filename), full_page=True)


async def save_issue_snapshot(page, output_dir: Path, label: str) -> str | None:
    path = output_dir / f"issue_{label}.png"
    try:
        await page.screenshot(path=str(path), full_page=True)
        return str(path)
    except Exception:
        return None


async def try_select_playable_card(
    page,
    state: dict,
    action_log: list[dict[str, object]],
    logger: RunLogger,
    max_attempts: int = 3,
) -> tuple[int | None, dict]:
    playable_indexes = list(state.get("my_playable_indexes", []))
    if not playable_indexes:
        return None, state

    random.shuffle(playable_indexes)
    attempts = min(max_attempts, len(playable_indexes))
    for chosen in playable_indexes[:attempts]:
        center = state["my_cards"][chosen]["center"]
        await click_design_point(page, center["x"], center["y"])
        action_log.append({"step": "select_play_card", "index": chosen})
        logger.log(f"Selected playable card index={chosen}")
        await page.wait_for_timeout(500)

        refreshed = await read_big2_game_state(page)
        selected_count = refreshed.get("my_selected_count", 0)
        selected_indexes = [
            index
            for index, card in enumerate(refreshed.get("my_cards", []))
            if card.get("selected")
        ]
        if selected_count > 0:
            logger.log(
                f"Card selection confirmed: selected_count={selected_count}, "
                f"selected_indexes={selected_indexes}"
            )
            return chosen, refreshed

        logger.log(f"Card index={chosen} did not stay selected; trying another card")

    return None, await read_big2_game_state(page)


async def verify_play_succeeded(
    page,
    previous_hand_count: int,
    logger: RunLogger,
    wait_ms: int = 1200,
) -> tuple[bool, dict]:
    await page.wait_for_timeout(wait_ms)
    refreshed = await read_big2_game_state(page)
    new_hand_count = refreshed.get("my_hand_count", previous_hand_count)
    new_turn = refreshed.get("turn")
    success = new_hand_count < previous_hand_count or new_turn != "self"
    logger.log(
        "Verified play result: "
        f"success={success}, hand_count={previous_hand_count}->{new_hand_count}, turn={new_turn}"
    )
    return success, refreshed


async def clear_selected_cards(
    page,
    state: dict,
    action_log: list[dict[str, object]],
    logger: RunLogger,
) -> dict:
    selected_count = state.get("my_selected_count", 0)
    if selected_count <= 0:
        return state

    logger.log(f"Clearing stale selection before acting: selected_count={selected_count}")

    # Primary path: call setSelect(false) on all selected cards via Cocos API.
    # toggle_my_card_by_sprite always calls setSelect(true) and CANNOT deselect;
    # pixel clicks are unreliable on overlapping cards (~70% overlap).
    await deselect_all_selected_cards(page)
    action_log.append({"step": "clear_selection_via_deselect_all"})
    refreshed = await read_big2_game_state(page)
    logger.log(f"Selection after card reset: selected_count={refreshed.get('my_selected_count', 0)}")
    if refreshed.get("my_selected_count", 0) == 0:
        return refreshed

    # Fallback: cancel button if Cocos API deselect didn't fully clear.
    cancel_button = refreshed.get("action_buttons", {}).get("cancel", {})
    if cancel_button.get("active") and cancel_button.get("center"):
        center = cancel_button["center"]
        await click_design_point(page, center["x"], center["y"])
        action_log.append({"step": "clear_selection_via_cancel"})
        await page.wait_for_timeout(400)
        refreshed = await read_big2_game_state(page)
        logger.log(f"Selection after cancel: selected_count={refreshed.get('my_selected_count', 0)}")

    return refreshed


def selected_card_codes(state: dict) -> list[str]:
    return [
        str(card.get("sprite_frame"))[1:]
        for card in state.get("my_cards", [])
        if card.get("selected") and isinstance(card.get("sprite_frame"), str) and card.get("sprite_frame").startswith("c")
    ]


async def wait_until_self_actionable_turn(page, timeout_seconds: int = 45) -> dict | None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    latest_state = None
    while asyncio.get_running_loop().time() < deadline:
        latest_state = await read_big2_game_state(page)
        if is_self_actionable_turn(latest_state):
            return latest_state
        await page.wait_for_timeout(IDLE_POLL_MS)
    return latest_state


async def probe_select_card(
    page,
    state: dict,
    card_code: str,
) -> tuple[bool, dict]:
    target = f"c{card_code}"
    invoke_result = await toggle_my_card_by_sprite(page, target)
    latest_state = await read_big2_game_state(page)
    if card_code in selected_card_codes(latest_state):
        return True, latest_state

    for index, card in enumerate(state.get("my_cards", [])):
        if card.get("sprite_frame") != target:
            continue
        center = card.get("center")
        if not center:
            continue
        for point in _card_click_points(state.get("my_cards", []), index) or [center]:
            await click_design_point(page, point["x"], point["y"])
            latest_state = await read_big2_game_state(page)
            if card_code in selected_card_codes(latest_state):
                return True, latest_state
            if latest_state.get("my_selected_count", 0) > 0:
                cancel_button = latest_state.get("action_buttons", {}).get("cancel", {})
                cancel_center = cancel_button.get("center")
                if cancel_button.get("active") and cancel_center:
                    await click_design_point(page, cancel_center["x"], cancel_center["y"])
                    latest_state = await read_big2_game_state(page)
            state = latest_state
        break
    return False, await read_big2_game_state(page)


async def save_probe_step(output_dir: Path, page, state: dict, step: str) -> None:
    (output_dir / f"{step}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    await page.screenshot(path=str(output_dir / f"{step}.png"), full_page=True)


async def wait_for_game_scene(page, timeout_seconds: int = 60) -> str | None:
    """Poll until Cocos scene becomes GameScene.

    Clears lobby popups every ~5 seconds while waiting so that ads appearing
    during matchmaking do not block the transition to GameScene."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_popup_clear = asyncio.get_running_loop().time()
    last_scene = None
    while asyncio.get_running_loop().time() < deadline:
        page = resolve_active_page(page)
        last_scene = await safe_read_current_scene(page)
        if last_scene == "GameScene":
            return last_scene
        now = asyncio.get_running_loop().time()
        if now - last_popup_clear >= 5:
            await maybe_clear_lobby_popup(page)
            last_popup_clear = now
        try:
            await page.wait_for_timeout(1000)
        except Exception:
            await asyncio.sleep(1)
    return last_scene


async def maybe_clear_lobby_popup(page) -> bool:
    """
    Attempt to dismiss any lobby popup.

    Strategy -1: coordinate-based preemptive click on event banner X button
                 (e.g. 賭城傳奇 full-screen banner).  Always runs FIRST so
                 that later strategies (which might return early) don't skip it.
    Strategy 0: JS walk — find any active button whose label text is one of
                the known dismiss strings (確定, OK, 關閉, ×, …) and emit a
                Cocos touch event on it.  Works regardless of node name.
    Strategy 1: invoke known dialog close button node names (exact match).
                New names should be added here as they are discovered via
                scene-dump / lobby-wait.
    Strategy 2: broad partial-name scan for active close-button-style nodes;
                picks the candidate closest to the top-right corner of the
                canvas (most likely a popup X, not a settings-panel button).
    Strategy 3: coordinate-based fallback for modal dialogs.
    """
    cleared = False

    # --- Strategy -1a: JS invoke — target known event/ad panel close nodes ---
    # ADPanel (賭城傳奇, daily login, etc.) close buttons have predictable node
    # names.  Invoke via cc.Component.EventHandler.emitEvents first so we don't
    # rely on pixel-perfect coordinates.
    AD_PANEL_CLOSE_NODES = [
        "CloseAreaButton",   # ADPanel banner × (confirmed ADPanel node name)
        "CloseButton",       # common event panel close
        "BtnClose",
        "btnClose",
        "close_btn",
    ]
    for _ad_node in AD_PANEL_CLOSE_NODES:
        if cleared:
            break
        try:
            result = await invoke_node(page, _ad_node, exact=True)
            if result.get("invoked"):
                await asyncio.sleep(0.6)
                cleared = True
        except Exception:
            pass

    # --- Strategy -1b: coordinate-based click on event banner X button ---
    # Full-screen event banners (e.g. 賭城傳奇) have an X button near the
    # top-right corner at approximately design (1695, 68).
    # Always runs even if Strategy -1a already fired, to handle banners whose
    # close node isn't in the list above.  Clicking empty space is harmless.
    try:
        event_result = await safe_click_design_position(
            page,
            DEFAULT_EVENT_CLOSE_X,
            DEFAULT_EVENT_CLOSE_Y,
            attempts=1,
            wait_ms=600,
        )
        if event_result is not None:
            cleared = True
    except Exception:
        pass

    # --- Strategy 0: JS walk — click by label text ---
    # Handles dialogs like 房間已滿 that have a 確定 button but no X node.
    # Also handles banners whose X button has '×' as its label text.
    # Uses the same cc.Component.EventHandler.emitEvents mechanism as
    # invoke_node (known to work), but searches by Label.string instead of
    # node name.
    try:
        clicked_text = await page.evaluate("""
        (() => {
            const DISMISS_TEXTS = ['確定', 'OK', '關閉', 'Close', '知道了', '确定', '×', 'X', '✕', '✗',
                                   '下次吧！', '下次吧', '下次', '取消', '跳過', '略過', 'Skip', 'Cancel'];
            const cc = window.cc;
            if (!cc || !cc.director) return null;
            const scene = cc.director.getScene();
            if (!scene) return null;

            let result = null;

            function walk(node) {
                if (result) return;
                if (!node || !node.active) return;

                // Check whether this node itself carries a dismiss label
                const label = node.getComponent &&
                    (node.getComponent('cc.Label') || node.getComponent(cc.Label));
                if (label && typeof label.string === 'string' &&
                        DISMISS_TEXTS.includes(label.string.trim())) {
                    // Walk up ≤3 levels to find a cc.Button ancestor
                    let candidate = node;
                    for (let i = 0; i < 3; i++) {
                        const btn = candidate.getComponent &&
                            (candidate.getComponent('cc.Button') ||
                             candidate.getComponent(cc.Button));
                        if (btn && Array.isArray(btn.clickEvents) &&
                                btn.clickEvents.length > 0) {
                            // Invoke using the same path as invoke_node
                            for (const eh of btn.clickEvents) {
                                try {
                                    cc.Component.EventHandler.emitEvents(
                                        [eh], { type: 'click' }
                                    );
                                } catch(e) {}
                            }
                            result = label.string.trim();
                            return;
                        }
                        if (candidate.parent) {
                            candidate = candidate.parent;
                        } else {
                            break;
                        }
                    }
                    // Fallback: emit on the direct parent
                    if (!result && node.parent) {
                        try {
                            node.parent.emit('click');
                            node.parent.emit('touchend');
                            result = label.string.trim();
                        } catch(e) {}
                    }
                }

                for (const child of (node.children || [])) {
                    walk(child);
                }
            }

            walk(scene);
            return result;
        })()
        """)
        if clicked_text:
            await asyncio.sleep(0.6)
            cleared = True
    except Exception:
        pass

    if cleared:
        return cleared

    # --- Strategy 1: exact node names (confirmed via scene-dump) ---
    # Add new names here when new popup types are discovered.
    KNOWN_CLOSE_NODES = [
        "dialog_close_button",   # CommonDialog X button (神幣不足, etc.)
        "CloseAreaButton",       # ADPanel ad banner close
        # 確定/OK buttons for dialogs without an X (e.g. 房間已滿).
        # These are tried in order; only the first hit is used.
        # Exact node names to be confirmed via lobby-wait scene-dump;
        # common Cocos naming conventions listed here as best-effort:
        "BtnConfirm",
        "btn_confirm",
        "BtnOK",
        "btn_ok",
        "OkButton",
        "OKButton",
        "ConfirmButton",
        "ButtonConfirm",
        "ButtonOK",
    ]
    for node_name in KNOWN_CLOSE_NODES:
        if cleared:
            break
        try:
            result = await invoke_node(page, node_name, exact=True)
            if result.get("invoked"):
                await asyncio.sleep(0.5)
                cleared = True
        except Exception:
            pass

    if cleared:
        return cleared

    # --- Strategy 2: broad partial-name scan, most-top-right active node ---
    seen_paths: set[str] = set()
    candidates: list[tuple[float, float, dict]] = []
    # Patterns ordered from most specific to least specific
    PARTIAL_PATTERNS = (
        "close_button", "CloseButton",
        "CloseBtn", "BtnClose", "close_btn", "btn_close", "btnClose",
    )
    for pattern in PARTIAL_PATTERNS:
        try:
            nodes = await probe_nodes(page, pattern, exact=False)
        except Exception:
            nodes = []
        for node in nodes:
            path = node.get("path", "")
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not node.get("active"):
                continue
            rect = node.get("rect")
            if not rect:
                continue
            cx = rect.get("center_x", 0.0)
            cy_top = rect.get("center_y_top_left", 9999.0)
            candidates.append((cx, cy_top, rect))

    if candidates:
        # Most top-right: smallest cy_top (nearest screen top) then largest cx (nearest right)
        candidates.sort(key=lambda c: (c[1], -c[0]))
        best = candidates[0][2]
        try:
            await click_canvas_design_position(page, best["center_x"], best["center_y_top_left"])
            await asyncio.sleep(0.5)
            cleared = True
        except Exception:
            pass

    # --- Strategy 3: coordinate-based fallback for modal dialogs ---
    # (Event banner X was already handled by Strategy -1 above.)
    modal_close_result = await safe_click_design_position(
        page,
        DEFAULT_MODAL_CLOSE_X,
        DEFAULT_MODAL_CLOSE_Y,
        attempts=2,
        wait_ms=800,
    )
    cleared = cleared or (modal_close_result is not None)

    return cleared


async def maybe_decline_rematch_dialog(page) -> bool:
    """結算/再來一局畫面出現時，主動點擊「否」按鈕拒絕加入下一局。

    使用 Cocos JS walk 在場景中尋找 Label 文字為「否」的 active 按鈕並觸發
    click event。若找到並成功觸發則回傳 True。

    只應在 in_scoring_wait=True 期間呼叫，以免在遊戲中誤觸。
    """
    try:
        clicked = await page.evaluate("""
        (() => {
            const DECLINE_TEXTS = ['否', '取消', 'No', 'Cancel'];
            const cc = window.cc;
            if (!cc || !cc.director) return null;
            const scene = cc.director.getScene();
            if (!scene) return null;

            let result = null;

            function walk(node) {
                if (result) return;
                if (!node || !node.active) return;

                const label = node.getComponent &&
                    (node.getComponent('cc.Label') || node.getComponent(cc.Label));
                if (label && typeof label.string === 'string' &&
                        DECLINE_TEXTS.includes(label.string.trim())) {
                    // Walk up ≤3 levels to find a cc.Button ancestor
                    let candidate = node;
                    for (let i = 0; i < 3; i++) {
                        const btn = candidate.getComponent &&
                            (candidate.getComponent('cc.Button') ||
                             candidate.getComponent(cc.Button));
                        if (btn && Array.isArray(btn.clickEvents) &&
                                btn.clickEvents.length > 0) {
                            for (const eh of btn.clickEvents) {
                                try {
                                    cc.Component.EventHandler.emitEvents(
                                        [eh], { type: 'click' }
                                    );
                                } catch(e) {}
                            }
                            result = label.string.trim();
                            return;
                        }
                        if (candidate.parent) {
                            candidate = candidate.parent;
                        } else {
                            break;
                        }
                    }
                    // Fallback: emit on the direct parent
                    if (!result && node.parent) {
                        try {
                            node.parent.emit('click');
                            node.parent.emit('touchend');
                            result = label.string.trim();
                        } catch(e) {}
                    }
                }

                for (const child of (node.children || [])) {
                    walk(child);
                }
            }

            walk(scene);
            return result;
        })()
        """)
        if clicked:
            await asyncio.sleep(0.6)
            return True
    except Exception:
        pass
    return False


async def ensure_game_scene_from_lobby(
    page,
    logger: RunLogger,
    attempts: int = 3,
) -> tuple[object, str | None]:
    page = resolve_active_page(page)
    scene = await safe_read_current_scene(page)
    if scene == "GameScene":
        return page, scene

    for attempt in range(1, attempts + 1):
        page = resolve_active_page(page)
        lobby_scene = await wait_for_scene(page, "LobbyScene", timeout_ms=30000)
        logger.log(f"enter_game attempt={attempt} wait_for_scene(LobbyScene) -> {lobby_scene}")
        if lobby_scene == "GameScene":
            # Matchmaking placed us into a game while we were waiting.
            # Log it but don't treat it as a clean entry — settings may not
            # have been applied.  Accept it and move on; settings are already
            # baked into the room at this point.
            logger.log(
                f"enter_game attempt={attempt} landed in GameScene via queue "
                f"(settings from previous attempt may apply)"
            )
            return page, lobby_scene
        # Clear ALL popups before touching lobby settings.
        # Loop until the lobby settings panel is confirmed readable, so that
        # event banners (e.g. 賭城傳奇) and other overlays are fully dismissed
        # before we try to read/set rule/amount.
        MAX_POPUP_CLEARS = 6
        for clear_iter in range(MAX_POPUP_CLEARS):
            popup_cleared = await maybe_clear_lobby_popup(page)
            logger.log(
                f"enter_game attempt={attempt} popup_clear_iter={clear_iter + 1} cleared={popup_cleared}"
            )
            # Check if lobby settings panel is now accessible
            selector = await read_lobby_selector(page, "RuleNode")
            if selector is not None and selector.get("text"):
                logger.log(
                    f"enter_game attempt={attempt} lobby panel accessible after {clear_iter + 1} clear(s)"
                )
                break
            if clear_iter < MAX_POPUP_CLEARS - 1:
                await asyncio.sleep(1.0)
        else:
            logger.log(
                f"enter_game attempt={attempt} lobby panel still blocked after {MAX_POPUP_CLEARS} clears; proceeding anyway"
            )

        # Block on lobby BottomPanel becoming readable before reading rule/amount;
        # otherwise the very first attempt fires before the panel has rendered and
        # we end up entering a game with whatever the default settings were.
        ready = await wait_for_lobby_settings_ready(page, timeout_ms=12000)
        logger.log(f"enter_game attempt={attempt} lobby_settings_ready={ready}")

        # Use a small max_attempts so lobby setup finishes quickly.
        # The text=None lobby bug can cause full 12-cycle traversals (8+ s each);
        # capping at 4 cycles (≤ 2.8 s each) keeps the total lobby time under
        # ~6 s and gives the game a chance to start before tricks 1-2 are done.
        rule_text = await ensure_normal_rule(page, max_attempts=4)
        amount_text = await ensure_min_amount(page, max_attempts=4)
        logger.log(
            f"enter_game attempt={attempt} settings rule={rule_text}, amount={amount_text}"
        )

        # Safety check: if settings are unreadable or we ended up on a card-swap
        # game (換牌局), do NOT click quick play — skip this attempt and retry.
        if rule_text is None or amount_text is None:
            logger.log(
                f"enter_game attempt={attempt} lobby settings unreadable; "
                f"skipping quick play to avoid wrong game type"
            )
            if attempt < attempts:
                await asyncio.sleep(4)
            continue

        # "換牌局" must NOT appear unless "不換" is present (not-swap variant)
        if rule_text and "換" in rule_text and "不換" not in rule_text:
            logger.log(
                f"enter_game attempt={attempt} wrong rule={rule_text!r} (card-swap game); "
                f"skipping quick play"
            )
            if attempt < attempts:
                await asyncio.sleep(2)
            continue

        quick_play_result = await safe_click_design_position(
            page,
            DEFAULT_QUICK_PLAY_X,
            DEFAULT_QUICK_PLAY_Y,
            attempts=3,
            wait_ms=1000,
        )
        logger.log(f"enter_game attempt={attempt} clicked_quick_play={quick_play_result is not None}")
        page = resolve_active_page(page)
        scene = await wait_for_game_scene(page, timeout_seconds=45)
        page = resolve_active_page(page)
        logger.log(f"enter_game attempt={attempt} wait_for_game_scene -> {scene}")
        if scene == "GameScene":
            return page, scene
        # Bounce diagnosis: capture the lobby so we can tell an unrecognized ad
        # popup apart from a genuinely empty/no-match lobby.
        try:
            shot = logger.path.parent / f"lobby_bounce_attempt{attempt}.png"
            await page.screenshot(path=str(shot), full_page=True)
            logger.log(f"enter_game attempt={attempt} bounce screenshot -> {shot.name}")
        except Exception as _shot_err:
            logger.log(f"enter_game attempt={attempt} bounce screenshot failed: {_shot_err}")
        # Failed to enter game — clear any popup that might have appeared
        # (e.g. "房間已滿") and wait before the next attempt.
        if attempt < attempts:
            popup_cleared = await maybe_clear_lobby_popup(page)
            logger.log(
                f"enter_game attempt={attempt} returned to {scene}; "
                f"popup_cleared={popup_cleared}; waiting 4s before retry"
            )
            await asyncio.sleep(4)
    return page, scene


async def run_autoplay_random(settings: Settings, timeout_seconds: int, record_video: bool = False) -> None:
    output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "autoplay"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "video" if record_video else None
    async with BrowserSession(settings, record_video_dir=video_dir) as session:
        page = await session.goto_target()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger = RunLogger(output_dir / "run.log")

        action_log: list[dict[str, object]] = []
        snapshot_index = 0
        last_change_confirm_at = -9999.0
        last_turn = None
        last_hand_count = None
        last_polled_state: dict | None = None

        scene = await classify_page_stage(page)
        logger.log(f"Initial stage={scene}")
        if scene != "GameScene":
            lobby_scene = await wait_for_scene(page, "LobbyScene", timeout_ms=30000)
            logger.log(f"wait_for_scene(LobbyScene) -> {lobby_scene}")
            if lobby_scene != "LobbyScene":
                logger.log("LobbyScene was not reached; continuing with best-effort flow")
                issue_path = await save_issue_snapshot(page, output_dir, "lobby_not_reached")
                logger.log(f"Saved issue snapshot: {issue_path}")
            await save_autoplay_snapshot(page, output_dir, snapshot_index, "lobby_before_popup")
            snapshot_index += 1
            popup_cleared = await maybe_clear_lobby_popup(page)
            logger.log(f"Tried clearing lobby popup -> {popup_cleared}")
            if not popup_cleared:
                issue_path = await save_issue_snapshot(page, output_dir, "popup_not_cleared")
                logger.log(f"Saved issue snapshot: {issue_path}")
            await save_autoplay_snapshot(page, output_dir, snapshot_index, "lobby_after_popup")
            snapshot_index += 1

            # Same fix as in ensure_game_scene_from_lobby: wait for the lobby
            # BottomPanel to be readable before reading rule/amount, and retry
            # the settings pass if it first returns None.
            ready = await wait_for_lobby_settings_ready(page, timeout_ms=12000)
            logger.log(f"Lobby settings ready: {ready}")
            rule_text = None
            amount_text = None
            for settings_try in range(3):
                rule_text = await ensure_normal_rule(page)
                amount_text = await ensure_min_amount(page)
                if rule_text is not None and amount_text is not None:
                    break
                logger.log(
                    f"settings retry {settings_try + 1}: rule={rule_text}, amount={amount_text}"
                )
                await page.wait_for_timeout(800)
            logger.log(f"Lobby settings settled: rule={rule_text}, amount={amount_text}")
            if rule_text is None or amount_text is None:
                issue_path = await save_issue_snapshot(page, output_dir, "lobby_settings_not_set")
                logger.log(f"Saved issue snapshot: {issue_path}")
            action_log.append(
                {
                    "step": "ensure_lobby_settings",
                    "rule_text": rule_text,
                    "amount_text": amount_text,
                }
            )
            quick_play_result = await safe_click_design_position(
                page,
                DEFAULT_QUICK_PLAY_X,
                DEFAULT_QUICK_PLAY_Y,
                attempts=3,
                wait_ms=1000,
            )
            action_log.append({"step": "click_quick_play", "result": quick_play_result})
            logger.log(f"Clicked quick-play -> {quick_play_result is not None}")
            if quick_play_result is None:
                issue_path = await save_issue_snapshot(page, output_dir, "quick_play_click_failed")
                logger.log(f"Saved issue snapshot: {issue_path}")
            await save_autoplay_snapshot(page, output_dir, snapshot_index, "after_quick_play")
            snapshot_index += 1
            scene = await wait_for_game_scene(page)
            logger.log(f"wait_for_game_scene -> {scene}")
            if scene != "GameScene":
                issue_path = await save_issue_snapshot(page, output_dir, "game_scene_not_reached")
                logger.log(f"Saved issue snapshot: {issue_path}")

        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            scene = await safe_read_current_scene(page)
            if scene != "GameScene":
                logger.log(f"Scene not ready for play: {scene}")
                await page.wait_for_timeout(1000)
                continue

            state = await read_big2_game_state(page)
            action_log.append({"step": "poll", "state": state})
            maybe_record_turn_event(last_polled_state, state, action_log, logger)
            last_polled_state = state

            my_hand_count = state.get("my_hand_count", 0)
            turn = state.get("turn")
            round_text = state.get("round")
            if my_hand_count != last_hand_count or turn != last_turn:
                logger.log(
                    "Poll state: "
                    f"round={round_text}, hand_count={my_hand_count}, turn={turn}, "
                    f"required_type={state.get('current_required_type')}, "
                    f"table_count={state.get('visible_table_card_count')}, "
                    f"change_three_active={state.get('change_three_active')}"
                )
                last_hand_count = my_hand_count
                last_turn = turn
            if my_hand_count == 0:
                logger.log("No hand cards visible right now; likely dealing or round transition")

            if state.get("change_three_active"):
                if my_hand_count == 0:
                    logger.log("Change-three phase detected but no cards visible yet; waiting")
                    await page.wait_for_timeout(1000)
                    continue
                now = asyncio.get_running_loop().time()
                if now - last_change_confirm_at < 4:
                    logger.log("Recently confirmed change-three; waiting for phase transition")
                    await page.wait_for_timeout(1000)
                    continue
                selectable = list(range(min(my_hand_count, len(state.get("my_cards", [])))))
                chosen_indexes = random.sample(selectable, k=min(3, len(selectable)))
                logger.log(f"Selecting change-three cards: {chosen_indexes}")
                for chosen in chosen_indexes:
                    center = state["my_cards"][chosen]["center"]
                    await click_design_point(page, center["x"], center["y"])
                    action_log.append({"step": "select_change_card", "index": chosen})
                    await page.wait_for_timeout(400)
                state = await read_big2_game_state(page)
                confirm = state.get("change_confirm", {})
                if confirm.get("active") and confirm.get("center"):
                    center = confirm["center"]
                    await click_design_point(page, center["x"], center["y"])
                    action_log.append({"step": "confirm_change_three"})
                    last_change_confirm_at = asyncio.get_running_loop().time()
                    logger.log("Clicked confirm for change-three")
                    await save_autoplay_snapshot(page, output_dir, snapshot_index, "after_change_three")
                    snapshot_index += 1
                    await page.wait_for_timeout(1500)
                else:
                    issue_path = await save_issue_snapshot(page, output_dir, "change_three_no_confirm")
                    logger.log(f"Change-three confirm unavailable. Saved issue snapshot: {issue_path}")
                continue

            if my_hand_count > 0 and state.get("turn") == "self":
                state = await clear_selected_cards(page, state, action_log, logger)
                my_hand_count = state.get("my_hand_count", my_hand_count)
                chosen, refreshed = await try_select_playable_card(
                    page,
                    state,
                    action_log,
                    logger,
                )
                if chosen is not None:
                    play_button = refreshed.get("action_buttons", {}).get("play", {})
                    if play_button.get("active") and play_button.get("center"):
                        center = play_button["center"]
                        await click_design_point(page, center["x"], center["y"])
                        play_succeeded, after_play_state = await verify_play_succeeded(
                            page,
                            my_hand_count,
                            logger,
                        )
                        action_log.append(
                            {
                                "step": "click_play",
                                "success": play_succeeded,
                                "state": after_play_state,
                            }
                        )
                        if play_succeeded:
                            logger.log("Clicked Play")
                            await save_autoplay_snapshot(page, output_dir, snapshot_index, "after_play")
                            snapshot_index += 1
                            await page.wait_for_timeout(300)
                            continue
                        issue_path = await save_issue_snapshot(page, output_dir, "play_had_no_effect")
                        logger.log(f"Clicked Play but state did not advance. Saved issue snapshot: {issue_path}")
                    issue_path = await save_issue_snapshot(page, output_dir, "selected_card_but_play_unavailable")
                    logger.log(f"Selected card but could not click Play. Saved issue snapshot: {issue_path}")
                else:
                    logger.log("Could not get any playable card to stay selected")

                pass_button = state.get("action_buttons", {}).get("pass", {})
                if pass_button.get("active") and pass_button.get("center"):
                    center = pass_button["center"]
                    await click_design_point(page, center["x"], center["y"])
                    action_log.append({"step": "click_pass"})
                    logger.log("Clicked PASS")
                    await save_autoplay_snapshot(page, output_dir, snapshot_index, "after_pass")
                    snapshot_index += 1
                    await page.wait_for_timeout(1500)
                    continue
                issue_path = await save_issue_snapshot(page, output_dir, "self_turn_no_play_or_pass")
                logger.log(f"Self turn but no actionable button. Saved issue snapshot: {issue_path}")

            await page.wait_for_timeout(1000)

        (output_dir / "action_log.json").write_text(
            json.dumps(action_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        timeout_path = await save_issue_snapshot(page, output_dir, "autoplay_timeout")
        logger.log(f"Saved final snapshot: {timeout_path}")
        network_path = await save_network_log(page, output_dir)
        logger.log(f"Saved network log: {network_path}")
        parsed_events, timeline = load_parse_and_build_timeline(network_path)
        parsed_path = output_dir / "parsed_events.json"
        timeline_path = output_dir / "game_timeline.json"
        turn_summary_path = output_dir / "turn_summary.json"
        turn_summary_text_path = output_dir / "turn_summary.txt"
        sprite_mapping_path = output_dir / "sprite_card_mapping.json"
        parsed_path.write_text(json.dumps(parsed_events, ensure_ascii=False, indent=2), encoding="utf-8")
        timeline_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        turn_summary = summarize_turns(timeline)
        turn_summary_path.write_text(json.dumps(turn_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        turn_summary_text_path.write_text(format_turn_summary_text(turn_summary), encoding="utf-8")
        sprite_mapping = build_sprite_card_mapping_report(action_log, timeline)
        sprite_mapping_path.write_text(
            json.dumps(sprite_mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.log(f"Saved parsed events: {parsed_path}")
        logger.log(f"Saved game timeline: {timeline_path}")
        logger.log(f"Saved turn summary: {turn_summary_path}")
        logger.log(f"Saved readable turn summary: {turn_summary_text_path}")
        logger.log(f"Saved sprite mapping: {sprite_mapping_path}")
        video_path = None
        if record_video and page.video is not None:
            await page.close()
            video_path = await page.video.path()
            logger.log(f"Saved autoplay video: {video_path}")
        await session.save_storage_state()
        logger.log(f"Saved autoplay artifacts to {output_dir}")
        print(f"Saved autoplay artifacts to: {output_dir}")
        if video_path is not None:
            print(f"Saved autoplay video to: {video_path}")


async def run_autoplay_agent(settings: Settings, timeout_seconds: int, record_video: bool = False, executor: str = "gui", games_to_play: int = 1) -> None:
    lock_path = settings.lock_dir / "autoplay_agent.lock"
    with SingleInstanceGuard(lock_path):
        output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "autoplay_agent"
        output_dir.mkdir(parents=True, exist_ok=True)
        video_dir = output_dir / "video" if record_video else None
        os.environ["BIG2_AGENT_DEBUG_DIR"] = str(output_dir.resolve())
        agent = build_decision_agent()

        async with BrowserSession(settings, record_video_dir=video_dir) as session:
            page = await session.goto_target()
            page = resolve_active_page(page)
            logger = RunLogger(output_dir / "run.log")
            action_log: list[dict[str, object]] = []

            scene = await classify_page_stage(page)
            if scene not in ("GameScene", "LobbyScene"):
                # Not in Cocos yet — wait up to 8 s for the canvas to initialise
                settled_scene = await wait_for_game_scene(page, timeout_seconds=8)
                if settled_scene is not None:
                    scene = settled_scene
                else:
                    scene = await classify_page_stage(page)
                page = resolve_active_page(page)
            logger.log(f"Initial stage={scene}")
            if scene != "GameScene":
                page, scene = await ensure_game_scene_from_lobby(page, logger, attempts=3)

            loop = asyncio.get_running_loop()
            hard_deadline = loop.time() + timeout_seconds
            games_played = 0            # complete sessions (lobby returns); stop at games_to_play
            game_number = 0             # individual games (局) finished within current session
            was_in_game_scene = False
            last_failed_signature = None
            skip_count: int = 0         # consecutive skips for current failed signature
            last_finish3_seq: int = -1  # highest finish3 seq already counted
            idle_poll_count: int = 0    # counts idle polls; used to throttle finish3 checks
            game_has_started: bool = False  # True once ≥5 cards seen this game
            in_scoring_wait: bool = False   # True after a game ends, until fresh cards dealt
            last_decline_time: float = 0.0  # throttle decline calls (avoid log spam)
            play_fail_count: int = 0    # consecutive play_not_confirmed; triggers forced pass
            _last_acted_hand: int | None = None  # our hand count as of last sync; if it DROPS
                                        # while it's not our turn, the server auto-played for us
                                        # (turn missed) — caught even though no executor ran.
            game_results: list[dict] = []   # per-game results for this run

            # Persistent results log: all runs append to the same file so
            # long-term win/loss stats accumulate across sessions.
            results_log_path = settings.artifact_dir / "game_results.jsonl"
            # Authoritative reward log: per-game SERVER round_result scores
            # (reliable, unlike Cocos hand counts). Append-only, robust to the
            # in-memory timeline rolling.
            reward_log_path = settings.artifact_dir / "reward_log.jsonl"
            last_logged_round_seq: int = -1

            def _any_player_emptied(state: dict) -> bool:
                """Return True when any player's hand has dropped to 0 this game."""
                nonlocal game_has_started
                if not game_has_started:
                    return False
                # Check our own hand
                if state.get("my_hand_count", -1) == 0:
                    return True
                # Check each visible enemy profile
                for profile in state.get("enemy_profiles", []):
                    text = profile.get("remain_text", "")
                    try:
                        if int(text) == 0:
                            return True
                    except (ValueError, TypeError):
                        pass
                return False

            async def _check_finish3() -> bool:
                """Read WS log; return True if a new finish3 event is detected.

                finish3 signals the end of one individual game (局) within the
                current session.  It does NOT end the session — the session only
                ends when the Cocos scene returns to LobbyScene.  This function
                just updates the per-session game counter for logging purposes.
                """
                nonlocal last_finish3_seq, last_failed_signature, skip_count
                from big2_vision_agent.browser.inspector import read_network_log as _rn
                try:
                    entries = await _rn(page)
                    parsed = parse_network_entries(entries)
                    tl = build_game_timeline(parsed)
                except Exception:
                    return False
                f3 = [e for e in tl if e.get("type") == "finish3"]
                if not f3:
                    return False
                max_seq = max(e.get("seq", 0) for e in f3)
                if max_seq <= last_finish3_seq:
                    return False
                last_finish3_seq = max_seq
                last_failed_signature = None
                skip_count = 0
                logger.log(
                    f"finish3 detected (seq={max_seq}); game {game_number} in session ended"
                )
                return True

            while loop.time() < hard_deadline:
                page = resolve_active_page(page)
                scene = await classify_page_stage(page)
                # Only attempt popup-clear when we are not in the game canvas
                # (popup clicks at game-canvas coordinates cause ghost card lifts).
                if scene != "GameScene":
                    # 注意：不在這裡呼叫 maybe_decline_rematch_dialog。
                    # 「再來一局/是否」對話框只在 GameScene 的 in_scoring_wait 期間處理。
                    # 在大廳呼叫它可能誤點大廳按鈕造成意外加入牌局。
                    await maybe_clear_lobby_popup(page)

                if scene == "GameScene":
                    was_in_game_scene = True
                elif was_in_game_scene and scene == "LobbyScene":
                    # ── Session ended ──────────────────────────────────────────
                    # The game only returns to lobby after all games in the
                    # session are done OR a player went bankrupt.  Either way
                    # this is the definitive end of one session.
                    games_played += 1
                    was_in_game_scene = False
                    game_has_started = False
                    in_scoring_wait = False
                    last_failed_signature = None
                    skip_count = 0
                    play_fail_count = 0

                    # ── Session summary ──────────────────────────────────────────
                    _session_results = [r for r in game_results
                                        if r["session"] == games_played]
                    _wins = sum(1 for r in _session_results if r["won"])
                    _total = len(_session_results)
                    _placements = [r["placement"] for r in _session_results]
                    logger.log(
                        f"Session {games_played}/{games_to_play} complete "
                        f"({game_number} game(s) played; returned to lobby) "
                        f"| wins={_wins}/{_total} placements={_placements}"
                    )
                    game_number = 0
                    if games_played >= games_to_play:
                        break

                if scene != "GameScene":
                    if scene == "LobbyScene" and games_played < games_to_play:
                        page, scene = await ensure_game_scene_from_lobby(page, logger, attempts=2)
                    elif scene not in ("LobbyScene", "GameCanvas"):
                        logger.log(f"Waiting for game canvas (stage={scene}, url={page.url!r})")
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue

                state = await read_big2_game_state(page)

                # Update game-started flag.
                # Conditions: our hand has >= 5 cards AND all readable enemy
                # counts are also > 0 (ensures new cards have been dealt and we
                # are not still in the between-game scoring phase where one
                # enemy's count is still 0 from the game just ended).
                #
                # While in_scoring_wait is True (after a game ended, before
                # new cards are dealt) we require a much higher threshold —
                # all players must have ≥ 10 cards — to confirm a fresh deal.
                # This prevents spurious re-triggers caused by the scoring
                # overlay changing how card counts are displayed (which can
                # make some enemy counts temporarily unparseable so that
                # all(c > 0) becomes vacuously true for the remaining values).
                enemy_counts = []
                for _ep in state.get("enemy_profiles", []):
                    try:
                        enemy_counts.append(int(_ep.get("remain_text", "")))
                    except (ValueError, TypeError):
                        pass
                my_count = state.get("my_hand_count", 0)
                if in_scoring_wait:
                    # Accept "game started" only when all players clearly have
                    # a fresh hand (≥ 10 cards each — new deal = 13 cards).
                    if (my_count >= 10
                            and enemy_counts
                            and all(c >= 10 for c in enemy_counts)):
                        in_scoring_wait = False
                        last_decline_time = 0.0
                        game_has_started = True
                else:
                    if (my_count >= 5
                            and enemy_counts
                            and all(c > 0 for c in enemy_counts)):
                        game_has_started = True

                # ── Scoring phase (between games within a session) ─────────────
                # A player emptied their hand → this individual game is over.
                # Cards will be re-dealt for the next game automatically; we
                # just wait for the scoring screen to clear.
                # game_has_started will not become True again until all players
                # have fresh cards, so this block fires exactly once per game.
                if _any_player_emptied(state):
                    game_number += 1
                    game_has_started = False
                    in_scoring_wait = True
                    last_failed_signature = None
                    skip_count = 0
                    play_fail_count = 0

                    # ── 記錄本局輸贏 ────────────────────────────────────────────
                    _my_remaining = state.get("my_hand_count", -1)
                    _enemy_remaining: list[int] = []
                    for _ep in state.get("enemy_profiles", []):
                        try:
                            _enemy_remaining.append(int(_ep.get("remain_text", -1)))
                        except (ValueError, TypeError):
                            _enemy_remaining.append(-1)

                    # Placement: 1st if I emptied (0 cards).
                    # Otherwise count enemies with fewer or equal remaining cards.
                    if _my_remaining == 0:
                        _placement = 1
                    else:
                        _placement = 1 + sum(
                            1 for c in _enemy_remaining if c >= 0 and c < _my_remaining
                        )
                        # If any enemy has 0, they definitively beat me
                        _placement = max(_placement,
                                         1 + sum(1 for c in _enemy_remaining if c == 0))

                    _game_result = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "session": games_played + 1,
                        "game_in_session": game_number,
                        "placement": _placement,
                        "won": _placement == 1,
                        "my_remaining": _my_remaining,
                        "enemy_remaining": _enemy_remaining,
                    }
                    game_results.append(_game_result)
                    # Append to persistent results log
                    try:
                        with results_log_path.open("a", encoding="utf-8") as _rf:
                            _rf.write(json.dumps(_game_result, ensure_ascii=False) + "\n")
                    except Exception as _e:
                        logger.log(f"Warning: could not write results log: {_e}")

                    # ── Authoritative reward log (SERVER round_result scores) ──────
                    # Cocos hand counts above are unreliable; the WS round_result
                    # carries the true per-player score. Capture it NOW (fresh,
                    # before the in-memory timeline rolls), deduped by round seq.
                    try:
                        _tl = await _read_timeline(page)
                        _rs = _latest_round_scores(_tl)
                        if _rs and "self" in _rs:
                            _seq = _rs["self"]["seq"]
                            if _seq != last_logged_round_seq and _rs["self"]["score"] is not None:
                                last_logged_round_seq = _seq
                                _reward_rec = {
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                    "session": games_played + 1,
                                    "game_in_session": game_number,
                                    "round_seq": _seq,
                                    "self_score": _rs["self"]["score"],
                                    "self_remaining": _rs["self"]["remaining"],
                                    "scores": {a: v["score"] for a, v in _rs.items()},
                                    "remaining": {a: v["remaining"] for a, v in _rs.items()},
                                }
                                with reward_log_path.open("a", encoding="utf-8") as _rf:
                                    _rf.write(json.dumps(_reward_rec, ensure_ascii=False) + "\n")
                                logger.log(f"Reward logged: self_score={_rs['self']['score']:+d} "
                                           f"(seq={_seq})")
                    except Exception as _e:
                        logger.log(f"Warning: could not write reward log: {_e}")

                    _result_str = "WIN 🎉" if _placement == 1 else f"{_placement}nd/rd/th"
                    logger.log(
                        f"Game {game_number} result: {_result_str} "
                        f"(placement={_placement}, my_remaining={_my_remaining}, "
                        f"enemies={_enemy_remaining})"
                    )
                    await page.wait_for_timeout(3000)
                    continue

                # ── 計分過渡期間絕對不出手，主動拒絕再來一局 ──────────────────
                # in_scoring_wait=True 時計分畫面上的按鈕可能被 is_self_actionable_turn
                # 誤判為可出牌，導致 agent 點到「再來一局」等按鈕而意外加入下一局。
                # 必須等到所有玩家都有 ≥10 張牌（in_scoring_wait reset）才恢復行動。
                # 同時主動偵測並點擊「否/取消」，避免任何重新加入對局的確認視窗
                # 因使用者或 agent 誤觸而確認。
                if in_scoring_wait:
                    now = loop.time()
                    # 每 3 秒嘗試一次拒絕對話框，避免每個 poll 都觸發（log 連打）
                    if now - last_decline_time >= 3.0:
                        declined = await maybe_decline_rematch_dialog(page)
                        if declined:
                            logger.log("Declined rematch dialog during scoring wait")
                            last_decline_time = now
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue

                if state.get("change_three_active"):
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue

                if not is_self_actionable_turn(state):
                    # Turn-miss / server auto-play detector: when it is NOT our
                    # turn our hand must stay constant (only opponents act). If it
                    # SHRANK since last sync, the server played for us — the
                    # "turn came, timeout, random single" case, which leaves no
                    # executor record. Flag it loudly, then resync.
                    _cur_hc = state.get("my_hand_count")
                    if isinstance(_cur_hc, int):
                        if (_last_acted_hand is not None and _cur_hc < _last_acted_hand
                                and not in_scoring_wait):
                            logger.log(
                                f"⚠️ AUTO-PLAY DETECTED: our hand {_last_acted_hand}->{_cur_hc} "
                                f"with NO agent decision (server timed us out / turn missed)"
                            )
                        _last_acted_hand = _cur_hc
                    # Missed-turn root-cause probe: the screen says it's OUR turn
                    # (turn==self) yet is_self_actionable_turn() returned False, so
                    # the loop idles while the server clock runs down → timeout →
                    # auto-play (the user's "our turn but waited to timeout"). Log
                    # WHICH field failed so we can fix the parser, not guess.
                    # Throttled to ~every 2 s so a real miss is visible w/o spam.
                    if state.get("turn") == "self" and idle_poll_count % 11 == 0:
                        _ab = state.get("action_buttons", {})
                        _ctb = state.get("card_type_buttons") or {}
                        logger.log(
                            "⚠️ turn=self but NOT actionable — "
                            f"clock_active={state.get('my_clock_active')} "
                            f"play_btn={_ab.get('play', {}).get('active')} "
                            f"pass_btn={_ab.get('pass', {}).get('active')} "
                            f"cardtype_any_active="
                            f"{any(isinstance(b, dict) and b.get('active') for b in _ctb.values())} "
                            f"hand={state.get('my_hand_count')} selected={state.get('my_selected_count')}"
                        )
                    if state.get("my_selected_count", 0) > 0:
                        logger.log("Clearing stale selection outside my actionable turn")
                        state = await clear_selected_cards(page, state, action_log, logger)
                    # Check finish3 every ~2 s of idle time (throttled; for logging only)
                    idle_poll_count += 1
                    # 對手回合也即時更新看板牌況（每 2 次 poll ≈ 360 ms）。
                    # wrapper 只在輪到我時被呼叫，對手出牌時要靠這裡推進看板。
                    if idle_poll_count % 2 == 0:
                        await _update_dashboard_live(page, state, logger)
                    if idle_poll_count % 11 == 0:  # 11 × 180 ms ≈ 2 s
                        await _check_finish3()
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue

                latest_state = await read_big2_game_state(page)
                if not is_self_actionable_turn(latest_state):
                    logger.log("State changed before acting; no longer my actionable turn")
                    if latest_state.get("my_selected_count", 0) > 0:
                        latest_state = await clear_selected_cards(page, latest_state, action_log, logger)
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue
                state = latest_state
                state = await clear_selected_cards(page, state, action_log, logger)
                observation, parsed_events, timeline = await _build_live_observation(page, state)
                (output_dir / "agent_observation.json").write_text(
                    json.dumps(observation.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # Check finish3 directly from the timeline we just built —
                # this catches game-over that arrived while we were in an
                # actionable turn, before the idle-poll throttle fires.
                # finish3 ends an individual game (局) within the session;
                # it does NOT end the session (lobby return does that).
                f3_in_timeline = [e for e in timeline if e.get("type") == "finish3"]
                if f3_in_timeline:
                    max_f3_seq = max(e.get("seq", 0) for e in f3_in_timeline)
                    if max_f3_seq > last_finish3_seq:
                        last_finish3_seq = max_f3_seq
                        last_failed_signature = None
                        skip_count = 0
                        logger.log(
                            f"finish3 in observation timeline (seq={max_f3_seq}); "
                            f"game {game_number} in session ended"
                        )
                        await page.wait_for_timeout(3000)
                        continue

                # Latency clock: from "our actionable turn is confirmed" to the
                # executor finishing. If this approaches the game's turn timer the
                # server will AUTO-PLAY for us (looks like "timeout, played a random
                # single") — log it so slow moves are visible, not silent.
                _turn_t0 = loop.time()

                # ── Forced pass after consecutive play failures ────────────────
                # If play_not_confirmed fired ≥ 2 times in a row (regardless of
                # which card was tried), the game state is likely diverged or the
                # game is near-over. Override the ML decision with a pass to
                # unstick the loop rather than flailing with different cards.
                if play_fail_count >= 2:
                    pass_action = next(
                        (a for a in observation.legal_actions if a.action == "pass"),
                        None,
                    )
                    if pass_action is not None:
                        logger.log(
                            f"play_fail_count={play_fail_count} → forcing pass to unstick"
                        )
                        from big2_vision_agent.agent_schema import AgentDecision as _AD
                        decision = _AD(action="pass", note="forced_pass:play_fail")
                        play_fail_count = 0
                    else:
                        # No pass available (我方有控制權且必須出牌) → let ML decide
                        decision = agent.decide(observation)
                else:
                    decision = agent.decide(observation)
                if decision is None:
                    logger.log("Agent returned no decision; waiting")
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue

                # source_seq is intentionally excluded from the signature:
                # a new WS packet (e.g. plsend echo) can bump source_seq even
                # when the Cocos state hasn't updated yet after a play, which
                # would otherwise invalidate the "skip this repeated decision"
                # guard and cause a duplicate WS send → 牌型錯誤.
                decision_signature = (
                    tuple(card.code for card in observation.self_hand),
                    decision.action,
                    tuple(decision.card_codes),
                    decision.combo_type,
                )
                if decision_signature == last_failed_signature:
                    skip_count += 1
                    if skip_count == 1:
                        logger.log("Skipping repeated failed decision until state changes")
                    await page.wait_for_timeout(IDLE_POLL_MS)
                    continue

                if skip_count > 1:
                    logger.log(f"State changed after {skip_count} skipped poll(s); resuming")
                    skip_count = 0

                (output_dir / "agent_decision.json").write_text(
                    json.dumps(decision.model_dump(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                _note_str = f" | ML: {decision.note}" if decision.note else ""
                logger.log(
                    f"Agent decision: action={decision.action}, cards={decision.card_codes}, combo={decision.combo_type} executor={executor}{_note_str}"
                )
                if executor == "packet":
                    result = await execute_packet_decision(page, state, decision)
                else:
                    result = await execute_agent_decision(page, state, decision)
                action_log.append(
                    {
                        "step": "agent_decision",
                        "observation": observation.model_dump(),
                        "decision": decision.model_dump(),
                        "result": result,
                    }
                )
                _turn_dt = loop.time() - _turn_t0
                _slow = " ⚠️SLOW" if _turn_dt > 5.0 else ""
                logger.log(
                    f"Executor result: ok={result.get('ok')} reason={result.get('reason')} "
                    f"latency={_turn_dt:.1f}s{_slow}"
                )
                if result.get("ok"):
                    last_failed_signature = None
                    skip_count = 0
                    play_fail_count = 0
                    # Resync the turn-miss baseline to our post-action hand so our
                    # OWN play isn't mistaken for a server auto-play next idle poll.
                    _res_state = result.get("state") or {}
                    if isinstance(_res_state.get("my_hand_count"), int):
                        _last_acted_hand = _res_state["my_hand_count"]
                    # ── Brief settle after our own play ────────────────────────
                    # Right after our play the Cocos UI still reports our turn as
                    # actionable for a few hundred ms before the server updates. A
                    # stateless / instant agent (the built-in fallback) would race
                    # back and fire a DUPLICATE decision on the stale state → the
                    # game rejects it ("牌型錯誤") / play_confirmation_timeout.
                    # Wait until the turn leaves us — BUT break immediately if it
                    # stays ours as a fresh LEAD (winning a trick = all others
                    # passed = we act again right away; the pass button goes
                    # inactive when we lead). Without that check we used to stall
                    # the full budget every time we won a trick. Short cap keeps
                    # it from ever feeling like a hang.
                    if decision.action == "play":
                        for _ in range(3):  # ≤ ~0.66 s (3 × POST_ACTION_WAIT_MS)
                            await page.wait_for_timeout(POST_ACTION_WAIT_MS)
                            settle_state = await read_big2_game_state(page)
                            if not is_self_actionable_turn(settle_state):
                                break
                            pass_btn = settle_state.get("action_buttons", {}).get("pass", {})
                            if not pass_btn.get("active"):  # our turn again as a fresh lead
                                break
                elif result.get("reason") in ("play_not_confirmed", "play_confirmation_timeout"):
                    play_fail_count += 1
                    if decision_signature != last_failed_signature:
                        skip_count = 0
                    last_failed_signature = decision_signature
                elif result.get("reason") == "selection_mismatch":
                    if decision_signature != last_failed_signature:
                        skip_count = 0
                    last_failed_signature = decision_signature
                if result.get("reason") == "selection_mismatch":
                    logger.log(
                        "Selection mismatch: "
                        f"expected={result.get('card_codes')} actual={result.get('selected_card_codes')}"
                    )
                (output_dir / "action_log.json").write_text(
                    json.dumps(action_log, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                await page.wait_for_timeout(POST_ACTION_WAIT_MS)

            (output_dir / "action_log.json").write_text(
                json.dumps(action_log, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            network_path = await save_network_log(page, output_dir)
            parsed_events, timeline = load_parse_and_build_timeline(network_path)
            (output_dir / "parsed_events.json").write_text(
                json.dumps(parsed_events, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / "game_timeline.json").write_text(
                json.dumps(timeline, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / "ml_state.json").write_text(
                json.dumps(build_ml_state(timeline), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            turn_summary = summarize_turns(timeline)
            (output_dir / "turn_summary.json").write_text(
                json.dumps(turn_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (output_dir / "turn_summary.txt").write_text(
                format_turn_summary_text(turn_summary),
                encoding="utf-8",
            )
            review_path = write_markdown_report(output_dir)
            session_review_path = write_session_review(output_dir)
            training_dataset_path, training_summary_path = write_training_export(output_dir)
            logger.log(f"Saved decision review: {review_path}")
            logger.log(f"Saved session review: {session_review_path}")
            logger.log(f"Saved training dataset: {training_dataset_path}")
            logger.log(f"Saved training summary: {training_summary_path}")
            logger.log(f"Saved autoplay-agent artifacts to {output_dir}")
            print(f"Saved autoplay-agent artifacts to: {output_dir}")
            # 關閉長駐 wrapper process
            if hasattr(agent, "close"):
                agent.close()


async def run_lobby_wait(settings: Settings) -> None:
    """Open the Cocos game lobby and wait for user to press Enter, then dump scene tree + screenshot."""
    async with BrowserSession(settings) as session:
        # goto_game() opens the Cocos canvas (new tab); goto_home() only reaches the website main page
        page = await session.goto_game()
        scene = await wait_for_scene(page, "LobbyScene", timeout_ms=20000)
        print()
        print("=" * 60)
        print(f"Game canvas loaded (scene={scene}). URL: {page.url}")
        print("Check the browser for any popup ads — do NOT close them manually.")
        print("Press Enter here when you are ready to capture...")
        print("=" * 60)
        await asyncio.to_thread(input)

        output_dir = settings.artifact_dir / datetime.now().strftime("%Y%m%d-%H%M%S") / "lobby_wait"
        output_dir.mkdir(parents=True, exist_ok=True)

        screenshot_path = output_dir / "lobby.png"
        await page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"Screenshot saved: {screenshot_path}")

        scene_tree = await dump_scene_tree(page)
        scene_tree_path = output_dir / "scene_tree.json"
        scene_tree_path.write_text(
            json.dumps(scene_tree, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Scene tree saved: {scene_tree_path}")


def _launch_dashboard_window() -> None:
    """啟動 HTML 看板伺服器（背景）並開啟瀏覽器。"""
    import atexit, socket, time, webbrowser

    project_dir = Path(__file__).resolve().parent.parent.parent
    dashboard   = project_dir / "dashboard" / "game_dashboard.py"
    port = 7373
    url  = f"http://localhost:{port}"

    if not dashboard.exists():
        print(f"[dashboard] 找不到 {dashboard}，略過")
        return

    # 若同 port 已有伺服器，直接開瀏覽器即可
    try:
        s = socket.create_connection(("localhost", port), timeout=0.3)
        s.close()
        webbrowser.open(url)
        print(f"[dashboard] 看板已開啟：{url}")
        return
    except OSError:
        pass

    # 啟動看板伺服器（跟著主程式一起結束）
    proc = subprocess.Popen(
        [sys.executable, str(dashboard), "--port", str(port)],
        cwd=str(project_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)

    # 等待伺服器就緒（最多 3 秒）
    for _ in range(30):
        try:
            s = socket.create_connection(("localhost", port), timeout=0.1)
            s.close()
            break
        except OSError:
            time.sleep(0.1)

    webbrowser.open(url)
    print(f"[dashboard] 看板已開啟：{url}  (PID {proc.pid})")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "login":
        asyncio.run(run_login(settings))
        return
    if args.command == "inspect":
        asyncio.run(run_inspect(settings))
        return
    if args.command == "click-design":
        asyncio.run(run_click_design(settings, args.x, args.y, args.wait_ms))
        return
    if args.command == "quick-play":
        asyncio.run(run_quick_play(settings, args.x, args.y, args.wait_ms))
        return
    if args.command == "quick-play-scan":
        asyncio.run(
            run_quick_play_scan(
                settings,
                args.center_x,
                args.center_y,
                args.delta_x,
                args.delta_y,
                args.wait_ms,
            )
        )
        return
    if args.command == "scene-dump":
        asyncio.run(run_scene_dump(settings))
        return
    if args.command == "node-probe":
        asyncio.run(run_node_probe(settings, args.name))
        return
    if args.command == "click-node":
        asyncio.run(run_click_node(settings, args.name, args.occurrence, args.wait_ms))
        return
    if args.command == "invoke-node":
        asyncio.run(run_invoke_node(settings, args.name, args.occurrence, args.wait_ms))
        return
    if args.command == "invoke-quick-join":
        asyncio.run(run_invoke_quick_join(settings))
        return
    if args.command == "quick-play-timeline":
        asyncio.run(run_quick_play_timeline(settings, args.x, args.y, args.seconds))
        return
    if args.command == "popup-quick-play-timeline":
        asyncio.run(
            run_popup_quick_play_timeline(
                settings,
                args.popup_x,
                args.popup_y,
                args.quick_play_x,
                args.quick_play_y,
                args.seconds,
            )
        )
        return
    if args.command == "game-state":
        asyncio.run(run_game_state(settings))
        return
    if args.command == "control-probe":
        asyncio.run(run_control_probe(settings, args.mode, args.card_code, args.record_video))
        return
    if args.command == "network-capture":
        asyncio.run(run_network_capture(settings, args.seconds))
        return
    if args.command == "parse-network-log":
        asyncio.run(run_parse_network_log(args.path))
        return
    if args.command == "build-agent-observation":
        asyncio.run(run_build_agent_observation(args.path))
        return
    if args.command == "build-ml-state":
        asyncio.run(run_build_ml_state(args.path))
        return
    if args.command == "build-agent-decision":
        asyncio.run(run_build_agent_decision(args.path, args.mode))
        return
    if args.command == "autoplay-random":
        asyncio.run(run_autoplay_random(settings, args.timeout_seconds, args.record_video))
        return
    if args.command == "autoplay-agent":
        _launch_dashboard_window()
        asyncio.run(run_autoplay_agent(settings, args.timeout_seconds, args.record_video, args.executor, args.games))
        return
    if args.command == "lobby-wait":
        asyncio.run(run_lobby_wait(settings))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
