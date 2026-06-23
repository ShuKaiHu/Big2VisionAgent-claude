from __future__ import annotations

from big2_vision_agent.agent_schema import AgentActionOption, AgentDecision
from big2_vision_agent.browser.actions import click_design_point, deselect_all_selected_cards, read_big2_game_state, toggle_my_card_by_sprite, ws_send_raw

WS_SEND_TARGET_CODE = "9"
PACKET_CONFIRM_DELAYS_MS = (350, 500, 700, 900, 1200)
PACKET_REJECTION_MESSAGE_KEYS = ("card_type_error", "no_bigger_card", "cant_lock")

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


def _is_self_actionable_turn(state: dict) -> bool:
    if state.get("turn") != "self":
        return False
    if not state.get("my_clock_active"):
        return False
    action_buttons = state.get("action_buttons", {})
    pass_button = action_buttons.get("pass", {})
    play_button = action_buttons.get("play", {})
    return bool(pass_button.get("active") or play_button.get("active"))


def _hand_sprite_codes(state: dict) -> set[str]:
    return {
        str(card.get("sprite_frame"))
        for card in state.get("my_cards", [])
        if isinstance(card.get("sprite_frame"), str)
    }


def _packet_targets_gone(before: dict, after: dict, decision: AgentDecision) -> bool:
    before_codes = _hand_sprite_codes(before)
    after_codes = _hand_sprite_codes(after)
    targets = {f"c{code}" for code in decision.card_codes}
    if not targets:
        return False
    if not targets & before_codes:
        return False
    return not (targets & after_codes)


def _packet_play_confirmed(before: dict, after: dict, decision: AgentDecision) -> bool:
    before_count = before.get("my_hand_count")
    after_count = after.get("my_hand_count")
    hand_decreased = (
        isinstance(before_count, int)
        and isinstance(after_count, int)
        and after_count < before_count
    )
    targets_gone = _packet_targets_gone(before, after, decision)
    return bool(hand_decreased or targets_gone)


def _packet_pass_confirmed(before: dict, after: dict) -> bool:
    turn_changed = after.get("turn") != before.get("turn")
    lost_actionable_turn = _is_self_actionable_turn(before) and not _is_self_actionable_turn(after)
    table_cleared = (
        isinstance(before.get("visible_table_card_count"), int)
        and isinstance(after.get("visible_table_card_count"), int)
        and after.get("visible_table_card_count") < before.get("visible_table_card_count")
    )
    return bool(turn_changed or lost_actionable_turn or table_cleared)


def _packet_rejection_reason(before: dict, after: dict) -> str | None:
    before_messages = before.get("system_messages") or {}
    after_messages = after.get("system_messages") or {}
    for key in PACKET_REJECTION_MESSAGE_KEYS:
        if after_messages.get(key) and not before_messages.get(key):
            return f"play_rejected_{key}"
    return None


def _compact_packet_confirmation_state(state: dict, decision: AgentDecision | None = None) -> dict[str, object]:
    targets = {f"c{code}" for code in decision.card_codes} if decision else set()
    hand_codes = _hand_sprite_codes(state)
    return {
        "turn": state.get("turn"),
        "my_clock_active": state.get("my_clock_active"),
        "my_hand_count": state.get("my_hand_count"),
        "my_selected_count": state.get("my_selected_count"),
        "self_actionable": _is_self_actionable_turn(state),
        "target_cards_present": sorted(targets & hand_codes),
        "system_messages": {
            key: bool((state.get("system_messages") or {}).get(key))
            for key in PACKET_REJECTION_MESSAGE_KEYS
        },
    }


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
        preflight = await read_big2_game_state(page)
        confirmation_states = [_compact_packet_confirmation_state(preflight)]
        if not _is_self_actionable_turn(preflight):
            return {
                "ok": False,
                "sent": False,
                "reason": "state_changed_before_send",
                "action": "pass",
                "state": preflight,
                "state_before": preflight,
                "confirmation_states": confirmation_states,
            }
        ok = await ws_send_raw(page, "pass")
        if not ok:
            return {"ok": False, "sent": False, "reason": "ws_unavailable", "action": "pass"}
        after = preflight
        confirmed = False
        for delay_ms in PACKET_CONFIRM_DELAYS_MS:
            await page.wait_for_timeout(delay_ms)
            after = await read_big2_game_state(page)
            confirmation_states.append(_compact_packet_confirmation_state(after))
            confirmed = _packet_pass_confirmed(preflight, after)
            if confirmed:
                break
        return {
            "ok": confirmed,
            "sent": True,
            "action": "pass",
            "reason": None if confirmed else "pass_confirmation_timeout",
            "state": after,
            "state_before": preflight,
            "confirmation_states": confirmation_states,
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

    cards_blob = "".join(decision.card_codes)
    message = f"send {WS_SEND_TARGET_CODE} {cards_blob}"
    confirmation_states = [_compact_packet_confirmation_state(pre_state, decision)]
    if not _is_self_actionable_turn(pre_state):
        return {
            "ok": False,
            "sent": False,
            "reason": "state_changed_before_send",
            "action": "play",
            "card_codes": decision.card_codes,
            "ws_message": message,
            "state": pre_state,
            "state_before": pre_state,
            "confirmation_states": confirmation_states,
        }
    ok = await ws_send_raw(page, message)
    if not ok:
        return {
            "ok": False,
            "sent": False,
            "reason": "ws_unavailable",
            "action": "play",
            "card_codes": decision.card_codes,
        }

    after = pre_state
    rejection_reason: str | None = None
    for delay_ms in PACKET_CONFIRM_DELAYS_MS:
        await page.wait_for_timeout(delay_ms)
        after = await read_big2_game_state(page)
        confirmation_states.append(_compact_packet_confirmation_state(after, decision))
        if _packet_play_confirmed(pre_state, after, decision):
            # Diagnostic: the hand should shrink by EXACTLY the number of cards we
            # intended. If it shrank by a different amount the confirmation passed
            # on a loose 'hand_decreased' — almost always the game AUTO-PLAYED a
            # different (usually 1-card) move because we were too slow / missed the
            # turn. ok stays True (don't change retry behaviour), but surface the
            # mismatch in `reason` so it shows up in run.log instead of hiding.
            _bc = pre_state.get("my_hand_count")
            _ac = after.get("my_hand_count")
            _intended = len(decision.card_codes) if decision.card_codes else 0
            _mismatch = (
                isinstance(_bc, int) and isinstance(_ac, int)
                and _intended > 0 and (_bc - _ac) != _intended
            )
            return {
                "ok": True,
                "sent": True,
                "action": "play",
                "card_codes": decision.card_codes,
                "ws_message": message,
                "reason": (f"count_mismatch:played{_bc - _ac}/intended{_intended}"
                           "(likely game auto-play)") if _mismatch else None,
                "state": after,
                "state_before": pre_state,
                "confirmation_states": confirmation_states,
            }
        rejection_reason = _packet_rejection_reason(pre_state, after)
        if rejection_reason:
            break
    return {
        "ok": False,
        "sent": True,
        "action": "play",
        "card_codes": decision.card_codes,
        "ws_message": message,
        "reason": rejection_reason or "play_confirmation_timeout",
        "state": after,
        "state_before": pre_state,
        "confirmation_states": confirmation_states,
    }
