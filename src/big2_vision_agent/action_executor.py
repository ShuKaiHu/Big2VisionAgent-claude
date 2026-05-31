from __future__ import annotations

from big2_vision_agent.agent_schema import AgentActionOption, AgentDecision
from big2_vision_agent.browser.actions import click_design_point, deselect_all_selected_cards, read_big2_game_state, toggle_my_card_by_sprite, ws_send_raw

WS_SEND_TARGET_CODE = "9"

COMBO_BUTTON_KEYS = {
    "single": "single",
    "pair": "pair",
    "straight": "straight",
    "full_house": "full_house",
    "four_of_a_kind": "four_kind",
    "four_of_kind": "four_kind",   # wrapper returns this spelling
    "straight_flush": "straight_flush",
}


class ActionExecutor:
    def choose_fallback_action(self, legal_actions: list[AgentActionOption]) -> AgentDecision | None:
        play_actions = [action for action in legal_actions if action.action == "play"]
        if play_actions:
            action = play_actions[0]
            return AgentDecision(
                action="play",
                card_codes=[card.code for card in action.cards],
                combo_type=action.combo_type,
                note="fallback:first_play",
            )
        pass_actions = [action for action in legal_actions if action.action == "pass"]
        if pass_actions:
            return AgentDecision(action="pass", note="fallback:pass")
        return None


async def _clear_selected_cards(page, state: dict) -> dict:
    if state.get("my_selected_count", 0) <= 0:
        return state

    # Primary path: call setSelect(false) on all selected cards via Cocos API.
    # This is the only reliable deselection path — toggle_my_card_by_sprite
    # calls setSelect(true) and cannot deselect; pixel clicks are unreliable on
    # overlapping cards (~70% overlap).
    await deselect_all_selected_cards(page)
    refreshed = await read_big2_game_state(page)
    if refreshed.get("my_selected_count", 0) == 0:
        return refreshed

    # Fallback: cancel button if Cocos API deselect didn't fully clear.
    cancel_button = refreshed.get("action_buttons", {}).get("cancel", {})
    center = cancel_button.get("center")
    if cancel_button.get("active") and center:
        await click_design_point(page, center["x"], center["y"])
        refreshed = await read_big2_game_state(page)

    return refreshed


def _selected_card_codes(state: dict) -> list[str]:
    return [
        str(card.get("sprite_frame"))[1:]
        for card in state.get("my_cards", [])
        if card.get("selected") and isinstance(card.get("sprite_frame"), str) and card.get("sprite_frame").startswith("c")
    ]


def _sorted_card_indexes(cards: list[dict]) -> list[int]:
    return sorted(
        range(len(cards)),
        key=lambda idx: (
            (cards[idx].get("center") or {}).get("x", 0),
            idx,
        ),
    )


def _card_click_points(cards: list[dict], index: int) -> list[dict[str, float]]:
    if index < 0 or index >= len(cards):
        return []
    card = cards[index]
    box = card.get("box") or {}
    center = card.get("center")
    if not center:
        return []

    left = box.get("left")
    right = box.get("right")
    width = box.get("width")
    top = box.get("top")
    height = box.get("height")
    if not all(isinstance(value, (int, float)) for value in (left, right, width, top, height)):
        return [center]

    sorted_indexes = _sorted_card_indexes(cards)
    sorted_pos = sorted_indexes.index(index)
    prev_idx = sorted_indexes[sorted_pos - 1] if sorted_pos > 0 else None
    next_idx = sorted_indexes[sorted_pos + 1] if sorted_pos < len(sorted_indexes) - 1 else None
    prev_center = cards[prev_idx].get("center") if prev_idx is not None else None
    next_center = cards[next_idx].get("center") if next_idx is not None else None

    upper_band_y = top + min(max(height * 0.32, 46), 72)
    mid_band_y = top + min(max(height * 0.42, 62), 92)

    points: list[dict[str, float]] = []
    if next_center and isinstance(next_center.get("x"), (int, float)):
        gap = max(1.0, next_center["x"] - center["x"])
        primary_x = min(right - 10, center["x"] + max(8.0, min(gap * 0.34, 22.0)))
        seam_x = min(right - 10, center["x"] + max(4.0, min(gap * 0.18, 12.0)))
        points.append({"x": center["x"], "y": upper_band_y})
        points.append({"x": primary_x, "y": upper_band_y})
        points.append({"x": seam_x, "y": upper_band_y})
        points.append({"x": center["x"], "y": mid_band_y})
        points.append({"x": primary_x, "y": mid_band_y})
    else:
        points.append({"x": center["x"], "y": upper_band_y})
        points.append({"x": center["x"], "y": mid_band_y})

    if prev_center and isinstance(prev_center.get("x"), (int, float)):
        gap = max(1.0, center["x"] - prev_center["x"])
        secondary_x = max(left + 10, center["x"] - min(gap * 0.12, 10.0))
        points.append({"x": secondary_x, "y": upper_band_y})

    points.append({"x": center["x"], "y": upper_band_y})
    points.append({"x": center["x"], "y": mid_band_y})
    deduped: list[dict[str, float]] = []
    seen = set()
    for point in points:
        key = (round(point["x"], 1), round(point["y"], 1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


async def execute_agent_decision(
    page,
    state: dict,
    decision: AgentDecision,
) -> dict[str, object]:
    state = await _clear_selected_cards(page, state)

    if decision.action == "pass":
        pass_button = state.get("action_buttons", {}).get("pass", {})
        center = pass_button.get("center")
        if not (pass_button.get("active") and center):
            return {"ok": False, "reason": "pass_unavailable"}
        await click_design_point(page, center["x"], center["y"])
        refreshed = await read_big2_game_state(page)
        return {"ok": True, "action": "pass", "state": refreshed}

    selected_indexes = []
    combo_key = COMBO_BUTTON_KEYS.get(decision.combo_type)
    if combo_key:
        # Clicking the combo-type button (單張/一對/順子…) makes the game both
        # CLEAR any previous selection and auto-suggest a candidate combo. We
        # rely on the clear-side-effect here — without this click, stale cards
        # from earlier failed plays stay selected across attempts and we end up
        # with selection_mismatch errors. The brief auto-suggestion is then
        # overwritten by the explicit sprite-toggle below.
        button_state = (state.get("card_type_buttons") or {}).get(combo_key) or {}
        center = button_state.get("center")
        if button_state.get("active") and center:
            await click_design_point(page, center["x"], center["y"])
            await page.wait_for_timeout(150)
            state = await read_big2_game_state(page)
            # Whatever the auto-suggestion picked is not necessarily what we
            # want; clear it before we lay down our own selection.
            if state.get("my_selected_count", 0) > 0:
                state = await _clear_selected_cards(page, state)

    remaining_targets = [f"c{code}" for code in decision.card_codes]
    while remaining_targets:
        target = remaining_targets.pop(0)
        match = None
        for index, card in enumerate(state.get("my_cards", [])):
            if card.get("sprite_frame") != target:
                continue
            if card.get("selected"):
                match = None
                break
            center = card.get("center")
            if not center:
                continue
            match = (index, center)
            break
        if match is None:
            refreshed_state = await read_big2_game_state(page)
            already_selected = {
                card.get("sprite_frame")
                for card in refreshed_state.get("my_cards", [])
                if card.get("selected")
            }
            if target in already_selected:
                state = refreshed_state
                continue
            return {
                "ok": False,
                "reason": "target_card_not_found",
                "card_code": target[1:],
                "selected_indexes": selected_indexes,
                "state": refreshed_state,
            }

        index, center = match
        selected = False
        invoke_result = await toggle_my_card_by_sprite(page, target)
        state = await read_big2_game_state(page)
        selected_codes = set(_selected_card_codes(state))
        if target[1:] in selected_codes:
            selected = True

        for click_point in ([] if selected else (_card_click_points(state.get("my_cards", []), index) or [center])):
            await click_design_point(page, click_point["x"], click_point["y"])
            selected_indexes.append(index)
            state = await read_big2_game_state(page)
            selected_codes = set(_selected_card_codes(state))
            if target[1:] in selected_codes:
                selected = True
                break
            if selected_codes:
                state = await _clear_selected_cards(page, state)
        if not selected:
            return {
                "ok": False,
                "reason": "target_click_failed",
                "card_code": target[1:],
                "invoke_result": invoke_result,
                "selected_indexes": selected_indexes,
                "state": state,
            }

    refreshed = state
    selected_codes = sorted(_selected_card_codes(refreshed))
    expected_codes = sorted(decision.card_codes)
    if selected_codes != expected_codes:
        refreshed = await _clear_selected_cards(page, refreshed)
        return {
            "ok": False,
            "reason": "selection_mismatch",
            "card_codes": decision.card_codes,
            "selected_card_codes": selected_codes,
            "selected_indexes": selected_indexes,
            "state": refreshed,
        }
    play_button = refreshed.get("action_buttons", {}).get("play", {})
    center = play_button.get("center")
    if not (play_button.get("active") and center):
        return {
            "ok": False,
            "reason": "play_unavailable",
            "selected_indexes": selected_indexes,
            "state": refreshed,
        }
    await click_design_point(page, center["x"], center["y"])
    # Give Cocos animation + WebSocket round-trip enough time to land before verifying.
    # Without this, hand_count/turn often hasn't updated yet and we incorrectly mark
    # the play as play_not_confirmed even though it actually went through.
    await page.wait_for_timeout(800)
    after_play = await read_big2_game_state(page)
    before_count = state.get("my_hand_count")
    after_count = after_play.get("my_hand_count")
    # Absolute check: did the turn leave "self"?  More reliable than comparing
    # against the stale `state` snapshot which may predate an auto-win event.
    turn_left_self = after_play.get("turn") != "self"
    hand_decreased = (
        isinstance(before_count, int)
        and isinstance(after_count, int)
        and after_count < before_count
    )
    # Retry up to 2 more times before giving up (server can take ~3 s to respond).
    for _extra_wait in (800, 1500):
        if hand_decreased or turn_left_self:
            break
        await page.wait_for_timeout(_extra_wait)
        after_play = await read_big2_game_state(page)
        after_count = after_play.get("my_hand_count")
        turn_left_self = after_play.get("turn") != "self"
        hand_decreased = (
            isinstance(before_count, int)
            and isinstance(after_count, int)
            and after_count < before_count
        )
    system_messages = after_play.get("system_messages") or {}
    blocked_by_error = any(
        system_messages.get(key)
        for key in ("card_type_error", "no_bigger_card", "cant_lock")
    )
    ok = bool(hand_decreased or (turn_left_self and not blocked_by_error))
    return {
        "ok": ok,
        "action": "play",
        "card_codes": decision.card_codes,
        "selected_indexes": selected_indexes,
        "reason": None if ok else "play_not_confirmed",
        "state": after_play,
    }


async def execute_packet_decision(
    page,
    state: dict,
    decision: AgentDecision,
) -> dict[str, object]:
    """Send a play or pass directly over WebSocket, bypassing all GUI interaction.

    Outgoing protocol (reverse-engineered from network logs):
      play  → "send 9 <cards_blob>"   (target_code 9 is constant across sessions)
      pass  → "pass"
    """
    if decision.action == "pass":
        ok = await ws_send_raw(page, "pass")
        if not ok:
            return {"ok": False, "reason": "ws_unavailable", "action": "pass"}
        await page.wait_for_timeout(800)
        after = await read_big2_game_state(page)
        turn_left_self = after.get("turn") != "self"
        if not turn_left_self:
            await page.wait_for_timeout(800)
            after = await read_big2_game_state(page)
            turn_left_self = after.get("turn") != "self"
        return {
            "ok": turn_left_self,
            "action": "pass",
            "reason": None if turn_left_self else "pass_not_confirmed",
            "state": after,
        }

    # Pre-execution check: MCTS takes ~1 s; auto-win or one-card-rule enforcement
    # may have already played cards for us during that time.  If the turn has
    # already left "self" before we even send, there is nothing to do — return
    # ok=True so the caller doesn't retry or force-pass unnecessarily.
    pre_state = await read_big2_game_state(page)
    if pre_state.get("turn") != "self":
        return {
            "ok": True,
            "action": "play",
            "card_codes": decision.card_codes,
            "reason": None,
            "note": "auto_advanced_before_send",
            "state": pre_state,
        }
    # Use the freshly read count as the reference so that any auto-play that
    # already happened (and reduced our hand) is accounted for.
    before_count = pre_state.get("my_hand_count")

    cards_blob = "".join(decision.card_codes)
    message = f"send {WS_SEND_TARGET_CODE} {cards_blob}"
    ok = await ws_send_raw(page, message)
    if not ok:
        return {
            "ok": False,
            "reason": "ws_unavailable",
            "action": "play",
            "card_codes": decision.card_codes,
        }

    await page.wait_for_timeout(800)
    after = await read_big2_game_state(page)
    after_count = after.get("my_hand_count")
    # Use an absolute check ("turn left self") instead of a relative comparison
    # against the stale `state` snapshot.  This correctly handles auto-win and
    # one-card-rule auto-play where the turn moves on without our explicit card.
    turn_left_self = after.get("turn") != "self"
    hand_decreased = (
        isinstance(before_count, int)
        and isinstance(after_count, int)
        and after_count < before_count
    )
    # Retry up to 2 more times (800 ms each) before giving up.
    # The game server can silently hold a request for up to ~3 s before either
    # confirming or sending self_play_request_unconfirmed + self_pass_request.
    # Giving up after 1.6 s causes unnecessary play_not_confirmed errors.
    for _extra_wait in (800, 1500):
        if hand_decreased or turn_left_self:
            break
        await page.wait_for_timeout(_extra_wait)
        after = await read_big2_game_state(page)
        after_count = after.get("my_hand_count")
        turn_left_self = after.get("turn") != "self"
        hand_decreased = (
            isinstance(before_count, int)
            and isinstance(after_count, int)
            and after_count < before_count
        )
    system_messages = after.get("system_messages") or {}
    blocked_by_error = any(
        system_messages.get(key)
        for key in ("card_type_error", "no_bigger_card", "cant_lock")
    )
    play_ok = bool(hand_decreased or (turn_left_self and not blocked_by_error))
    return {
        "ok": play_ok,
        "action": "play",
        "card_codes": decision.card_codes,
        "ws_message": message,
        "reason": None if play_ok else "play_not_confirmed",
        "state": after,
    }
