from __future__ import annotations


RANK_SEQUENCE = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
ACTOR_TO_PLAYER = {
    "self": 1,
    "right": 2,
    "top": 3,
    "left": 4,
}
PLAYER_TO_KEY = {player: str(player) for player in range(1, 5)}


def card_code_to_ml_id(code: str) -> int:
    """Convert Gamesofa card code to the 1..52 id used by the ML record.

    Gamesofa code format is <suit><rank>, e.g. 11 = spade ace. The ML id is
    rank-major and suit-minor: 11 -> 1, 21 -> 2, 12 -> 5, 1K -> 49.
    """
    if len(code) != 2:
        raise ValueError(f"Invalid card code: {code!r}")
    suit_code = code[0]
    rank_code = code[1]
    if suit_code not in {"1", "2", "3", "4"} or rank_code not in RANK_SEQUENCE:
        raise ValueError(f"Invalid card code: {code!r}")
    return RANK_SEQUENCE.index(rank_code) * 4 + int(suit_code)


def card_codes_to_ml_ids(codes: list[str]) -> list[int]:
    return [card_code_to_ml_id(code) for code in codes]


def build_ml_state(timeline: list[dict[str, object]]) -> dict[str, object]:
    state = _empty_state()
    started = False
    awaiting_new_deal = False

    for item in timeline:
        event = item.get("event")

        if event == "self_hand_snapshot":
            cards = [code for code in item.get("cards", []) if isinstance(code, str)]
            if cards and awaiting_new_deal and len(cards) >= 10:
                state = _empty_state()
                awaiting_new_deal = False
            state["my_hand"] = card_codes_to_ml_ids(cards)
            started = True
            continue

        if not started:
            continue

        if event == "player_play":
            _apply_play(state, item)
            continue

        if event == "player_pass":
            _apply_pass(state, item)
            continue

        if event == "round_result":
            _apply_round_result(state, item)
            continue

        if event == "finish3":
            state["current_player"] = state["perspective_player"]
            state["control"] = True
            state["last_hand"] = None
            state["last_player"] = None
            _reset_passed(state)
            awaiting_new_deal = True
            continue

    return state


def _empty_state() -> dict[str, object]:
    return {
        "my_hand": [],
        "perspective_player": 1,
        "current_player": 1,
        "opponent_counts": {
            "2": 13,
            "3": 13,
            "4": 13,
        },
        "played_cards": [],
        "played_cards_by_player": {},
        "last_hand": None,
        "last_player": None,
        "control": True,
        "passed": {
            "1": False,
            "2": False,
            "3": False,
            "4": False,
        },
        "action_history": [],
    }


def _apply_play(state: dict[str, object], item: dict[str, object]) -> None:
    player = _player_from_item(item)
    if player is None:
        return

    cards = [code for code in item.get("cards", []) if isinstance(code, str)]
    hand = card_codes_to_ml_ids(cards)
    state["action_history"].append(
        {
            "player": player,
            "hand": hand,
            "pass": False,
            "forced_skip": False,
            "control_break": False,
            "passed_snapshot": _passed_snapshot(state),
        }
    )

    state["played_cards"].extend(hand)
    played_by_player = state["played_cards_by_player"].setdefault(str(player), [])
    played_by_player.extend(hand)

    if player == 1:
        _remove_cards_from_my_hand(state, hand)
    else:
        key = PLAYER_TO_KEY[player]
        state["opponent_counts"][key] = max(0, int(state["opponent_counts"].get(key, 13)) - len(hand))

    state["last_hand"] = hand
    state["last_player"] = player
    state["control"] = False
    _reset_passed(state)
    state["current_player"] = _next_player(player)


def _apply_pass(state: dict[str, object], item: dict[str, object]) -> None:
    player = _player_from_item(item)
    if player is None:
        return

    passed_snapshot = _passed_snapshot(state)
    state["passed"][str(player)] = True
    control_break = _trick_closed_by_pass(state)

    state["action_history"].append(
        {
            "player": player,
            "hand": None,
            "pass": True,
            "forced_skip": False,
            "control_break": control_break,
            "passed_snapshot": passed_snapshot,
        }
    )

    if control_break:
        next_player = int(state["last_player"])
        state["current_player"] = next_player
        state["control"] = True
        state["last_hand"] = None
        state["last_player"] = None
        _reset_passed(state)
    else:
        state["current_player"] = _next_player(player)


def _apply_round_result(state: dict[str, object], item: dict[str, object]) -> None:
    player = _player_from_item(item)
    if player is None:
        return
    remaining = [code for code in item.get("remaining_cards", []) if isinstance(code, str)]
    if player == 1:
        state["my_hand"] = card_codes_to_ml_ids(remaining)
        return
    state["opponent_counts"][str(player)] = len(remaining)


def _player_from_item(item: dict[str, object]) -> int | None:
    actor = item.get("actor")
    if isinstance(actor, str):
        return ACTOR_TO_PLAYER.get(actor)
    return None


def _next_player(player: int) -> int:
    return 1 if player == 4 else player + 1


def _reset_passed(state: dict[str, object]) -> None:
    for key in state["passed"]:
        state["passed"][key] = False


def _passed_snapshot(state: dict[str, object]) -> list[bool]:
    return [bool(state["passed"][str(player)]) for player in range(1, 5)]


def _trick_closed_by_pass(state: dict[str, object]) -> bool:
    last_player = state.get("last_player")
    if not isinstance(last_player, int):
        return False
    return all(
        bool(state["passed"][str(player)])
        for player in range(1, 5)
        if player != last_player
    )


def _remove_cards_from_my_hand(state: dict[str, object], cards: list[int]) -> None:
    remaining = list(state["my_hand"])
    for card in cards:
        try:
            remaining.remove(card)
        except ValueError:
            pass
    state["my_hand"] = remaining
