from __future__ import annotations

from collections import Counter
from itertools import combinations

from big2_vision_agent.agent_schema import (
    AgentActionOption,
    AgentCard,
    AgentObservation,
    OpponentState,
    TurnConstraint,
)
from big2_vision_agent.network_parser import classify_cards


RANK_STRENGTH = {
    "3": 0,
    "4": 1,
    "5": 2,
    "6": 3,
    "7": 4,
    "8": 5,
    "9": 6,
    "10": 7,
    "J": 8,
    "Q": 9,
    "K": 10,
    "A": 11,
    "2": 12,
}

SUIT_STRENGTH = {
    "C": 0,
    "D": 1,
    "H": 2,
    "S": 3,
}

FIVE_CARD_HIERARCHY = {
    "straight": 0,
    "full_house": 1,
    "four_of_a_kind": 2,
    "straight_flush": 3,
}

STRAIGHT_RANK_PATTERNS = [
    ("A", "2", "3", "4", "5"),
    ("3", "4", "5", "6", "7"),
    ("4", "5", "6", "7", "8"),
    ("5", "6", "7", "8", "9"),
    ("6", "7", "8", "9", "10"),
    ("7", "8", "9", "10", "J"),
    ("8", "9", "10", "J", "Q"),
    ("9", "10", "J", "Q", "K"),
    ("10", "J", "Q", "K", "A"),
    ("2", "3", "4", "5", "6"),
]


def build_agent_observation(timeline: list[dict[str, object]]) -> AgentObservation:
    game_index = 1
    current_hand: list[str] = []
    trick_index: int | None = None
    turn = "unknown"
    lead_actor = None
    required_combo_type = None
    last_played_cards: list[dict[str, object]] = []
    last_played_by = None
    passes_since_last_play = 0
    remaining_counts: dict[str, int | None] = {
        "left": None,
        "top": None,
        "right": None,
    }
    source_seq = None

    for item in timeline:
        source_seq = item.get("seq")
        event = item.get("event")

        if event == "self_hand_snapshot":
            current_hand = list(item.get("cards", []))
            continue

        if event == "player_play":
            actor = item.get("actor")
            if passes_since_last_play >= 3 or lead_actor is None:
                trick_index = 1 if trick_index is None else trick_index + 1
                lead_actor = actor
            required_combo_type = (item.get("combo") or {}).get("type")
            last_played_cards = list(item.get("decoded_cards", []))
            last_played_by = actor
            passes_since_last_play = 0
            turn = _next_actor(actor)
            continue

        if event == "player_pass":
            actor = item.get("actor")
            passes_since_last_play += 1
            turn = _next_actor(actor)
            if passes_since_last_play >= 3:
                required_combo_type = None
                last_played_cards = []
                last_played_by = None
            continue

        if event == "round_result":
            actor = item.get("actor")
            remaining_cards = item.get("remaining_cards") or []
            if actor == "self":
                current_hand = list(remaining_cards)
            elif actor in remaining_counts:
                remaining_counts[actor] = len(remaining_cards)
            continue

        if event == "finish3":
            game_index += 1
            trick_index = None
            lead_actor = None
            required_combo_type = None
            last_played_cards = []
            last_played_by = None
            passes_since_last_play = 0
            turn = "unknown"
            continue

    hand_cards = [_agent_card_from_decoded(_decode_or_stub(code)) for code in current_hand]
    last_cards = [_agent_card_from_decoded(card) for card in last_played_cards]
    legal_actions = _build_legal_actions(
        hand_cards,
        required_combo_type,
        last_cards,
        allow_pass=True,
    )

    return AgentObservation(
        game_index=game_index,
        trick_index=trick_index,
        self_hand=hand_cards,
        hand_count=len(hand_cards),
        turn=turn if turn in {"self", "left", "top", "right"} else "unknown",
        constraint=TurnConstraint(
            lead_actor=lead_actor if lead_actor in {"self", "left", "top", "right"} else None,
            required_combo_type=required_combo_type,
            last_played_cards=last_cards,
            last_played_by=last_played_by if last_played_by in {"self", "left", "top", "right"} else None,
            passes_since_last_play=passes_since_last_play,
        ),
        opponents=[
            OpponentState(seat="left", remaining_count=remaining_counts["left"]),
            OpponentState(seat="top", remaining_count=remaining_counts["top"]),
            OpponentState(seat="right", remaining_count=remaining_counts["right"]),
        ],
        legal_actions=legal_actions,
        source_seq=source_seq if isinstance(source_seq, int) else None,
    )


def build_live_agent_observation(
    timeline: list[dict[str, object]],
    runtime_state: dict,
) -> AgentObservation:
    observation = build_agent_observation(timeline)

    runtime_hand_codes = [
        card.get("sprite_frame")[1:]
        for card in runtime_state.get("my_cards", [])
        if isinstance(card.get("sprite_frame"), str) and card.get("sprite_frame").startswith("c")
    ]
    observed_codes = [card.code for card in observation.self_hand]
    if runtime_hand_codes and (
        not observation.self_hand
        or len(runtime_hand_codes) != observation.hand_count
        or Counter(runtime_hand_codes) != Counter(observed_codes)
    ):
        observation.self_hand = [_agent_card_from_decoded(_decode_or_stub(code)) for code in runtime_hand_codes]
        observation.hand_count = len(observation.self_hand)

    turn = runtime_state.get("turn")
    if turn in {"self", "left", "top", "right"}:
        observation.turn = turn

    required_combo_type = runtime_state.get("current_required_type") or observation.constraint.required_combo_type
    observation.constraint.required_combo_type = required_combo_type
    _normalize_self_lead_constraint(observation)

    enemy_profiles = runtime_state.get("enemy_profiles", [])
    if enemy_profiles:
        remaining_by_seat = {}
        for enemy in enemy_profiles:
            seat = enemy.get("seat") or _infer_enemy_seat(enemy.get("center") or {})
            count = enemy.get("remaining_count")
            if not isinstance(count, int):
                count = _parse_remaining_count(enemy.get("remain_text"))
            if seat in {"left", "top", "right"}:
                remaining_by_seat[seat] = count
        observation.opponents = [
            OpponentState(seat="left", remaining_count=remaining_by_seat.get("left")),
            OpponentState(seat="top", remaining_count=remaining_by_seat.get("top")),
            OpponentState(seat="right", remaining_count=remaining_by_seat.get("right")),
        ]

    observation.legal_actions = _build_runtime_legal_actions(
        runtime_state,
        observation.self_hand,
        observation.constraint.required_combo_type,
        observation.constraint.last_played_cards,
        observation.opponents,
    )
    return observation


def _parse_remaining_count(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value.isdigit():
        return None
    return int(value)


def _infer_enemy_seat(center: dict[str, object]) -> str | None:
    x = center.get("x")
    if not isinstance(x, (int, float)):
        return None
    if x < 500:
        return "left"
    if x > 1200:
        return "right"
    return "top"


def _normalize_self_lead_constraint(observation: AgentObservation) -> None:
    constraint = observation.constraint
    if observation.turn != "self":
        return
    if constraint.last_played_by != "self":
        return
    # Only treat this as a free lead when all 3 opponents have already passed
    # (passes_since_last_play >= 3), meaning the trick is fully resolved.
    #
    # Without this guard, a Cocos UI transitional state can prematurely set
    # turn="self" after we auto-play (e.g. the opening C3 combo) while top/left
    # have not yet responded.  That causes _build_runtime_legal_actions to return
    # all possible plays instead of [pass], so MCTS tries to play into a round
    # that is not yet over → server rejects with play_not_confirmed.
    if constraint.passes_since_last_play < 3:
        return
    # If runtime says it is our turn again and the last recognized play was also ours,
    # the previous trick has already been collected and we are effectively leading a new trick.
    constraint.required_combo_type = None
    constraint.last_played_cards = []
    constraint.last_played_by = None
    constraint.passes_since_last_play = 0
    constraint.lead_actor = "self"


def _next_actor(actor: str | None) -> str:
    order = ["self", "right", "top", "left"]
    if actor not in order:
        return "unknown"
    return order[(order.index(actor) + 1) % 4]


def _agent_card_from_decoded(decoded: dict[str, object]) -> AgentCard:
    return AgentCard(
        code=str(decoded.get("code")),
        display=str(decoded.get("display")),
        rank=str(decoded.get("rank_label")),
        suit=str(decoded.get("suit_label")),
    )


def _decode_or_stub(code: str) -> dict[str, object]:
    from big2_vision_agent.network_parser import decode_card_code

    return decode_card_code(code)


def _build_runtime_legal_actions(
    runtime_state: dict,
    hand_cards: list[AgentCard],
    required_combo_type: str | None,
    last_played_cards: list[AgentCard],
    opponents: list[OpponentState] | None = None,
) -> list[AgentActionOption]:
    allow_pass = bool(runtime_state.get("action_buttons", {}).get("pass", {}).get("active"))
    actions = _build_legal_actions(hand_cards, required_combo_type, last_played_cards, allow_pass=allow_pass)
    actions = _filter_right_one_single_rule(actions, opponents)
    playable_indexes = list(runtime_state.get("my_playable_indexes") or [])

    if required_combo_type is not None and not playable_indexes:
        return [action for action in actions if action.action == "pass"]

    playable_codes = {
        card.get("sprite_frame")[1:]
        for index, card in enumerate(runtime_state.get("my_cards", []))
        if index in playable_indexes
        and isinstance(card.get("sprite_frame"), str)
        and card.get("sprite_frame").startswith("c")
    }
    if not playable_codes:
        return actions

    filtered = [action for action in actions if action.action == "pass"]
    for action in actions:
        if action.action != "play":
            continue
        if all(card.code in playable_codes for card in action.cards):
            filtered.append(action)
    return _filter_right_one_single_rule(filtered, opponents)


def _right_opponent_has_one_card(opponents: list[OpponentState] | None) -> bool:
    for opponent in opponents or []:
        if opponent.seat == "right" and opponent.remaining_count == 1:
            return True
    return False


def _filter_right_one_single_rule(
    actions: list[AgentActionOption],
    opponents: list[OpponentState] | None,
) -> list[AgentActionOption]:
    """When next player has one card, single-card plays must use our highest single."""
    if not _right_opponent_has_one_card(opponents):
        return actions

    single_actions = [
        action
        for action in actions
        if action.action == "play" and action.combo_type == "single" and len(action.cards) == 1
    ]
    if len(single_actions) <= 1:
        return actions

    max_single = max(single_actions, key=lambda action: _combo_signature(action.cards, "single"))
    max_code = max_single.cards[0].code
    return [
        action
        for action in actions
        if not (
            action.action == "play"
            and action.combo_type == "single"
            and len(action.cards) == 1
            and action.cards[0].code != max_code
        )
    ]


def _build_legal_actions(
    hand_cards: list[AgentCard],
    required_combo_type: str | None,
    last_played_cards: list[AgentCard],
    allow_pass: bool,
) -> list[AgentActionOption]:
    actions = [AgentActionOption(action="pass")] if allow_pass else []
    if not hand_cards:
        return actions

    sorted_hand = _sorted_cards(hand_cards)
    if required_combo_type in {None, "single"}:
        singles = [
            AgentActionOption(action="play", cards=[card], combo_type="single")
            for card in sorted_hand
        ]
        actions.extend(_filter_against_last_play(singles, "single", last_played_cards))
        if required_combo_type == "single":
            return actions

    if required_combo_type in {None, "pair"}:
        pairs = [
            AgentActionOption(action="play", cards=pair, combo_type="pair")
            for pair in _find_pairs(sorted_hand)
        ]
        actions.extend(_filter_against_last_play(pairs, "pair", last_played_cards))
        if required_combo_type == "pair":
            return actions

    valid_five_card_types = set(FIVE_CARD_HIERARCHY)
    allow_five_card_actions = (
        required_combo_type is None
        or required_combo_type == "five_card"
        or required_combo_type in valid_five_card_types
    )

    if allow_five_card_actions:
        five_actions: list[AgentActionOption] = []
        for combo_cards in combinations(sorted_hand, 5):
            combo_type = _classify_agent_cards(list(combo_cards))
            if combo_type not in valid_five_card_types:
                continue
            five_actions.append(
                AgentActionOption(action="play", cards=list(combo_cards), combo_type=combo_type)
            )
        actions.extend(_filter_against_last_play(five_actions, required_combo_type, last_played_cards))

    if required_combo_type in {None, "dragon"}:
        dragon_actions = [
            AgentActionOption(action="play", cards=sorted_hand, combo_type="dragon")
            for combo_cards in [sorted_hand]
            if len(combo_cards) == 13 and _classify_agent_cards(combo_cards) == "dragon"
        ]
        actions.extend(_filter_against_last_play(dragon_actions, required_combo_type, last_played_cards))

    return actions


def _find_pairs(hand_cards: list[AgentCard]) -> list[list[AgentCard]]:
    grouped: dict[str, list[AgentCard]] = {}
    for card in hand_cards:
        grouped.setdefault(card.rank, []).append(card)
    pairs: list[list[AgentCard]] = []
    for cards in grouped.values():
        if len(cards) >= 2:
            for pair in combinations(cards, 2):
                pairs.append(list(pair))
    return pairs


def _filter_against_last_play(
    actions: list[AgentActionOption],
    required_combo_type: str | None,
    last_played_cards: list[AgentCard],
) -> list[AgentActionOption]:
    if not last_played_cards or required_combo_type is None:
        return actions

    last_type = _classify_agent_cards(last_played_cards)
    if required_combo_type in {"single", "pair", "dragon"} and required_combo_type != last_type:
        return []

    if required_combo_type in set(FIVE_CARD_HIERARCHY) | {"five_card"} and last_type not in FIVE_CARD_HIERARCHY:
        return []

    return [action for action in actions if _combo_beats(action.cards, action.combo_type, last_played_cards, last_type)]


def _sorted_cards(cards: list[AgentCard]) -> list[AgentCard]:
    return sorted(cards, key=lambda card: (RANK_STRENGTH[card.rank], SUIT_STRENGTH[card.suit]))


def _classify_agent_cards(cards: list[AgentCard]) -> str:
    return classify_cards([card.code for card in cards]).get("type", "unknown")


def _combo_signature(cards: list[AgentCard], combo_type: str | None) -> tuple:
    if not combo_type:
        return tuple()

    sorted_cards = _sorted_cards(cards)
    rank_counts = Counter(card.rank for card in sorted_cards)
    rank_values_desc = sorted((RANK_STRENGTH[rank] for rank in rank_counts), reverse=True)
    suit_values_desc = sorted((SUIT_STRENGTH[card.suit] for card in sorted_cards), reverse=True)

    if combo_type == "single":
        card = sorted_cards[-1]
        return (RANK_STRENGTH[card.rank], SUIT_STRENGTH[card.suit])

    if combo_type == "pair":
        return (
            RANK_STRENGTH[sorted_cards[-1].rank],
            max(SUIT_STRENGTH[card.suit] for card in sorted_cards),
        )

    if combo_type in {"straight", "straight_flush"}:
        straight_info = _straight_info(sorted_cards)
        if straight_info is None:
            return tuple()
        high_rank = straight_info["high_rank"]
        return (
            straight_info["pattern_index"],
            max(SUIT_STRENGTH[card.suit] for card in sorted_cards if card.rank == high_rank),
        )

    if combo_type == "full_house":
        triple_rank = next(rank for rank, count in rank_counts.items() if count == 3)
        pair_rank = next(rank for rank, count in rank_counts.items() if count == 2)
        return (RANK_STRENGTH[triple_rank], RANK_STRENGTH[pair_rank])

    if combo_type == "four_of_a_kind":
        quad_rank = next(rank for rank, count in rank_counts.items() if count == 4)
        kicker_rank = next(rank for rank, count in rank_counts.items() if count == 1)
        return (RANK_STRENGTH[quad_rank], RANK_STRENGTH[kicker_rank])

    if combo_type == "dragon":
        is_flush = len({card.suit for card in sorted_cards}) == 1
        return (1 if is_flush else 0,)

    return tuple(rank_values_desc + suit_values_desc)


def _combo_beats(
    cards: list[AgentCard],
    combo_type: str | None,
    last_played_cards: list[AgentCard],
    last_type: str | None,
) -> bool:
    if not combo_type or not last_type:
        return True

    if combo_type == "dragon":
        if last_type != "dragon":
            return True
        return _combo_signature(cards, combo_type) > _combo_signature(last_played_cards, last_type)

    if combo_type in {"single", "pair"}:
        if combo_type != last_type:
            return False
        return _combo_signature(cards, combo_type) > _combo_signature(last_played_cards, last_type)

    if combo_type in FIVE_CARD_HIERARCHY:
        if last_type == "dragon":
            return False
        if last_type in FIVE_CARD_HIERARCHY and FIVE_CARD_HIERARCHY[combo_type] != FIVE_CARD_HIERARCHY[last_type]:
            return FIVE_CARD_HIERARCHY[combo_type] > FIVE_CARD_HIERARCHY[last_type]
        if last_type in FIVE_CARD_HIERARCHY:
            return _combo_signature(cards, combo_type) > _combo_signature(last_played_cards, last_type)
        return False

    return False


def _straight_info(cards: list[AgentCard]) -> dict[str, object] | None:
    if len(cards) != 5:
        return None
    ranks = {card.rank for card in cards}
    if len(ranks) != 5:
        return None
    rank_tuple = tuple(sorted(ranks, key=lambda rank: RANK_STRENGTH[rank]))
    pattern_lookup = {
        tuple(sorted(pattern, key=lambda rank: RANK_STRENGTH[rank])): index
        for index, pattern in enumerate(STRAIGHT_RANK_PATTERNS)
    }
    pattern_index = pattern_lookup.get(rank_tuple)
    if pattern_index is None:
        return None
    pattern = STRAIGHT_RANK_PATTERNS[pattern_index]
    return {
        "pattern_index": pattern_index,
        "high_rank": pattern[-1],
    }
