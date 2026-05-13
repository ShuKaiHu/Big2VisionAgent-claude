from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


RANK_LABELS = {
    "1": "A",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "T": "10",
    "J": "J",
    "Q": "Q",
    "K": "K",
}

RANK_ORDER = {
    "3": 0,
    "4": 1,
    "5": 2,
    "6": 3,
    "7": 4,
    "8": 5,
    "9": 6,
    "T": 7,
    "J": 8,
    "Q": 9,
    "K": 10,
    "1": 11,
    "2": 12,
}

SUIT_LABELS = {
    "1": "S",
    "2": "H",
    "3": "D",
    "4": "C",
}

STRAIGHT_RANK_PATTERNS = [
    ("1", "2", "3", "4", "5"),
    ("3", "4", "5", "6", "7"),
    ("4", "5", "6", "7", "8"),
    ("5", "6", "7", "8", "9"),
    ("6", "7", "8", "9", "T"),
    ("7", "8", "9", "T", "J"),
    ("8", "9", "T", "J", "Q"),
    ("9", "T", "J", "Q", "K"),
    ("T", "J", "Q", "K", "1"),
    ("2", "3", "4", "5", "6"),
]


def split_card_codes(cards: str) -> list[str]:
    if not cards:
        return []
    return [cards[index : index + 2] for index in range(0, len(cards), 2) if cards[index : index + 2]]


def decode_card_code(code: str) -> dict[str, object]:
    suit_code = code[0] if len(code) >= 1 else None
    rank_code = code[1] if len(code) >= 2 else None
    rank_label = RANK_LABELS.get(rank_code, rank_code)
    suit_label = SUIT_LABELS.get(suit_code, suit_code)
    return {
        "code": code,
        "suit_code": suit_code,
        "rank_code": rank_code,
        "rank_label": rank_label,
        "suit_label": suit_label,
        "display": f"{suit_label}{rank_label}" if suit_label and rank_label else code,
        "display_zh": f"花色{suit_code}{rank_label}" if suit_code and rank_label else code,
        "decode_note": "suit_label is inferred; rank_label is based on Big2 order where 1=A and 2=2",
    }


def decode_cards(codes: list[str]) -> list[dict[str, object]]:
    return [decode_card_code(code) for code in codes]


def classify_cards(cards: list[str]) -> dict[str, object]:
    decoded = decode_cards(cards)
    ranks = [item["rank_code"] for item in decoded if item.get("rank_code")]
    suits = [item["suit_code"] for item in decoded if item.get("suit_code")]
    count = len(cards)

    result: dict[str, object] = {
        "card_count": count,
        "rank_codes": ranks,
        "rank_labels": [item["rank_label"] for item in decoded if item.get("rank_label")],
        "decoded_cards": decoded,
        "type": "unknown",
    }

    if count == 0:
        result["type"] = "empty"
        return result

    rank_counter = Counter(ranks)
    rank_occurrences = sorted(rank_counter.values(), reverse=True)
    straight_info = _straight_info(ranks)
    is_flush = len(set(suits)) == 1
    is_straight = straight_info is not None

    if count == 1:
        result["type"] = "single"
        return result
    if count == 2:
        result["type"] = "pair" if len(rank_counter) == 1 else "invalid_pair"
        return result
    if count == 5:
        if is_straight and is_flush:
            result["type"] = "straight_flush"
        elif rank_occurrences == [4, 1]:
            result["type"] = "four_of_a_kind"
        elif rank_occurrences == [3, 2]:
            result["type"] = "full_house"
        elif is_straight:
            result["type"] = "straight"
        else:
            result["type"] = "five_card_unknown"
        return result
    if count == 13 and len(rank_counter) == 13:
        result["type"] = "dragon"
        return result

    result["type"] = f"{count}_card_unknown"
    return result


def _straight_info(ranks: list[str]) -> dict[str, object] | None:
    if len(ranks) != 5 or len(set(ranks)) != 5:
        return None

    rank_tuple = tuple(sorted(ranks, key=lambda rank: RANK_ORDER.get(rank, -1)))
    pattern_lookup = {
        tuple(sorted(pattern, key=lambda rank: RANK_ORDER[rank])): index
        for index, pattern in enumerate(STRAIGHT_RANK_PATTERNS)
    }
    pattern_index = pattern_lookup.get(rank_tuple)
    if pattern_index is None:
        return None

    pattern = STRAIGHT_RANK_PATTERNS[pattern_index]
    return {
        "pattern_index": pattern_index,
        "pattern": pattern,
        "high_rank": pattern[-1],
    }


def parse_ws_payload(payload: str, direction: str = "incoming") -> dict[str, object]:
    command, _, rest = payload.partition(" ")
    rest = rest.strip()

    parser = {
        "newRoom": _parse_new_room,
        "sJS2": _parse_room_snapshot,
        "startGame": _parse_simple_event,
        "gsstart": _parse_simple_event,
        "play": _parse_play,
        "plpass": _parse_pass,
        "plsend": _parse_player_send,
        "showScore": _parse_show_score,
        "autoplay": _parse_autoplay,
        "send": _parse_send_cards,
        "pass": _parse_self_pass,
        "finish1": _parse_simple_event,
        "finish2": _parse_simple_event,
        "finish3": _parse_simple_event,
    }.get(command, _parse_unknown)

    return parser(command, rest, payload, direction)


def parse_network_entries(entries: list[dict]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for entry in entries:
        kind = entry.get("kind")
        if kind not in {"ws_message", "ws_send"}:
            continue
        payload = entry.get("payload")
        if not isinstance(payload, str) or not payload:
            continue
        direction = "outgoing" if kind == "ws_send" else "incoming"
        event = parse_ws_payload(payload, direction=direction)
        event["seq"] = entry.get("seq")
        event["ts"] = entry.get("ts")
        event["direction"] = direction
        parsed.append(event)
    _apply_relative_seat_labels(parsed)
    return parsed


def build_game_timeline(events: list[dict[str, object]]) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    last_self_hand: list[str] | None = None
    pending_self_request: dict[str, object] | None = None
    recent_self_confirmed_cards: Counter[str] | None = None

    for event in events:
        seq = event.get("seq")
        ts = event.get("ts")
        parsed_event = event.get("event")

        if parsed_event == "room_snapshot":
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": "room_snapshot",
                    "room_id": event.get("room_id"),
                    "empty_seats": event.get("empty_seats"),
                    "base": event.get("base"),
                    "map_name": event.get("map_name"),
                    "players": event.get("players"),
                }
            )
            continue

        if parsed_event in {"startGame", "gsstart", "finish1", "finish2", "finish3"}:
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": parsed_event,
                }
            )
            continue

        if parsed_event == "self_hand_snapshot":
            current_hand = list(event.get("cards", []))
            timeline_event = {
                "seq": seq,
                "ts": ts,
                "event": "self_hand_snapshot",
                "hand_count": len(current_hand),
                "cards": current_hand,
                "decoded_cards": decode_cards(current_hand),
            }
            if last_self_hand is None:
                timeline_event["snapshot_type"] = "initial"
                timeline.append(timeline_event)
            else:
                removed = _subtract_cards(last_self_hand, current_hand)
                added = _subtract_cards(current_hand, last_self_hand)
                if removed or added:
                    timeline_event["snapshot_type"] = "delta"
                    timeline_event["removed_cards"] = removed
                    timeline_event["added_cards"] = added
                    timeline.append(timeline_event)

                    removed_counter = Counter(removed)
                    if recent_self_confirmed_cards == removed_counter:
                        recent_self_confirmed_cards = None
                    elif pending_self_request and removed_counter == Counter(
                        pending_self_request.get("cards", [])
                    ):
                        timeline.append(
                            {
                                "seq": seq,
                                "ts": ts,
                                "event": "self_play_confirmed",
                                "cards": removed,
                                "decoded_cards": decode_cards(removed),
                                "combo": classify_cards(removed),
                                "request_seq": pending_self_request.get("seq"),
                            }
                        )
                        pending_self_request = None
            last_self_hand = current_hand
            continue

        if parsed_event == "self_send_cards":
            pending_self_request = event
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": "self_play_request",
                    "request_code": event.get("target_code"),
                    "cards": event.get("cards"),
                    "decoded_cards": event.get("decoded_cards"),
                    "combo": event.get("combo"),
                }
            )
            continue

        if parsed_event == "send_response":
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": "send_response",
                    "response_code": event.get("response_code"),
                    "cards": event.get("cards"),
                    "decoded_cards": event.get("decoded_cards"),
                    "combo": event.get("combo"),
                }
            )
            continue

        if parsed_event == "self_pass_request":
            if pending_self_request is not None:
                timeline.append(
                    {
                        "seq": seq,
                        "ts": ts,
                        "event": "self_play_request_unconfirmed",
                        "request_seq": pending_self_request.get("seq"),
                        "cards": pending_self_request.get("cards"),
                        "decoded_cards": pending_self_request.get("decoded_cards"),
                        "combo": pending_self_request.get("combo"),
                    }
                )
                pending_self_request = None
            timeline.append({"seq": seq, "ts": ts, "event": "self_pass_request"})
            continue

        if parsed_event == "player_play":
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": "player_play",
                    "actor": event.get("actor"),
                    "actor_index": event.get("actor_index"),
                    "cards": event.get("cards"),
                    "decoded_cards": event.get("decoded_cards"),
                    "combo": event.get("combo"),
                }
            )
            if event.get("actor") == "self":
                recent_self_confirmed_cards = Counter(event.get("cards", []))
                if pending_self_request and recent_self_confirmed_cards == Counter(
                    pending_self_request.get("cards", [])
                ):
                    pending_self_request = None
            continue

        if parsed_event == "pass":
            if event.get("actor") == "self" and pending_self_request is not None:
                timeline.append(
                    {
                        "seq": seq,
                        "ts": ts,
                        "event": "self_play_request_unconfirmed",
                        "request_seq": pending_self_request.get("seq"),
                        "cards": pending_self_request.get("cards"),
                        "decoded_cards": pending_self_request.get("decoded_cards"),
                        "combo": pending_self_request.get("combo"),
                    }
                )
                pending_self_request = None
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": "player_pass",
                    "actor": event.get("actor"),
                    "actor_index": event.get("actor_index"),
                }
            )
            continue

        if parsed_event == "show_score":
            timeline.append(
                {
                    "seq": seq,
                    "ts": ts,
                    "event": "round_result",
                    "actor": event.get("actor"),
                    "actor_index": event.get("actor_index"),
                    "score": event.get("score"),
                    "remaining_cards": event.get("remaining_cards"),
                    "remaining_decoded_cards": event.get("decoded_remaining_cards"),
                    "remaining_combo": classify_cards(list(event.get("remaining_cards", []))),
                }
            )
            continue

    return timeline


def summarize_turns(timeline: list[dict[str, object]]) -> list[dict[str, object]]:
    tricks: list[dict[str, object]] = []
    current_trick: dict[str, object] | None = None
    passes_since_last_play = 0
    game_index = 1

    for item in timeline:
        event = item.get("event")

        if event in {"startGame", "gsstart", "finish1", "finish2", "finish3", "round_result"}:
            current_trick = None
            passes_since_last_play = 0
            if event == "finish3":
                game_index += 1
            continue

        if event == "player_play":
            if current_trick is None or passes_since_last_play >= 3:
                current_trick = {
                    "trick_index": len(tricks) + 1,
                    "game_index": game_index,
                    "lead_actor": item.get("actor"),
                    "required_type": (item.get("combo") or {}).get("type"),
                    "actions": [],
                }
                tricks.append(current_trick)
            elif current_trick.get("required_type") is None:
                current_trick["required_type"] = (item.get("combo") or {}).get("type")

            current_trick["actions"].append(
                {
                    "seq": item.get("seq"),
                    "actor": item.get("actor"),
                    "action": "play",
                    "cards": item.get("cards"),
                    "decoded_cards": item.get("decoded_cards"),
                    "combo_type": (item.get("combo") or {}).get("type"),
                }
            )
            passes_since_last_play = 0
            continue

        if event == "player_pass" and current_trick is not None:
            current_trick["actions"].append(
                {
                    "seq": item.get("seq"),
                    "actor": item.get("actor"),
                    "action": "pass",
                }
            )
            passes_since_last_play += 1
            continue

    for trick in tricks:
        actions = trick.get("actions", [])
        play_count = sum(1 for action in actions if action.get("action") == "play")
        pass_count = sum(1 for action in actions if action.get("action") == "pass")
        trick["play_count"] = play_count
        trick["pass_count"] = pass_count
        trick["closed"] = pass_count >= 3

    return tricks


def build_sprite_card_mapping_report(
    action_log: list[dict[str, object]],
    timeline: list[dict[str, object]],
) -> dict[str, object]:
    hand_snapshots = [
        item for item in timeline if item.get("event") == "self_hand_snapshot" and item.get("cards")
    ]
    poll_states = [
        item.get("state")
        for item in action_log
        if isinstance(item, dict) and item.get("step") == "poll" and isinstance(item.get("state"), dict)
    ]

    mapping_samples: list[dict[str, object]] = []
    direct_prefix_matches = 0

    for timeline_snapshot in hand_snapshots:
        hand_cards = list(timeline_snapshot.get("cards", []))
        hand_count = len(hand_cards)
        hand_set = Counter(hand_cards)

        for state in poll_states:
            my_cards = state.get("my_cards") or []
            sprite_frames = [card.get("sprite_frame") for card in my_cards if card.get("sprite_frame")]
            if len(sprite_frames) != hand_count:
                continue

            normalized = [frame[1:] if isinstance(frame, str) and frame.startswith("c") else frame for frame in sprite_frames]
            if Counter(normalized) != hand_set:
                continue

            sample_pairs = []
            for sprite_frame, code in zip(sprite_frames, normalized):
                pair = {"sprite_frame": sprite_frame, "card_code": code}
                sample_pairs.append(pair)
                if isinstance(sprite_frame, str) and sprite_frame == f"c{code}":
                    direct_prefix_matches += 1

            mapping_samples.append(
                {
                    "timeline_seq": timeline_snapshot.get("seq"),
                    "hand_count": hand_count,
                    "pairs": sample_pairs,
                }
            )
            break

    unique_pairs = {}
    for sample in mapping_samples:
        for pair in sample["pairs"]:
            unique_pairs[pair["sprite_frame"]] = pair["card_code"]

    inferred_rule = None
    if unique_pairs and all(sprite == f"c{code}" for sprite, code in unique_pairs.items()):
        inferred_rule = "sprite_frame = 'c' + card_code"

    return {
        "sample_count": len(mapping_samples),
        "unique_mapping_count": len(unique_pairs),
        "direct_prefix_match_count": direct_prefix_matches,
        "inferred_rule": inferred_rule,
        "unique_pairs": [
            {"sprite_frame": sprite_frame, "card_code": card_code}
            for sprite_frame, card_code in sorted(unique_pairs.items())
        ],
        "samples": mapping_samples,
    }


def format_turn_summary_text(tricks: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for trick in tricks:
        trick_index = trick.get("trick_index")
        lead_actor = trick.get("lead_actor")
        required_type = trick.get("required_type")
        closed = "是" if trick.get("closed") else "否"
        lines.append(f"第 {trick_index} 墩")
        lines.append(f"lead: {lead_actor}")
        lines.append(f"牌型: {required_type}")
        lines.append(f"是否收墩: {closed}")
        lines.append("行動:")

        for action in trick.get("actions", []):
            actor = action.get("actor")
            if action.get("action") == "pass":
                lines.append(f"- {actor}: pass")
                continue

            cards = action.get("cards") or []
            combo_type = action.get("combo_type")
            decoded_cards = action.get("decoded_cards") or []
            displays = [item.get("display") for item in decoded_cards if item.get("display")]
            cards_display = " ".join(displays) if displays else " ".join(cards)
            lines.append(f"- {actor}: 出牌 {combo_type} [{cards_display}]")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def load_and_parse_network_log(path: Path) -> list[dict[str, object]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return parse_network_entries(entries)


def load_parse_and_build_timeline(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events = load_and_parse_network_log(path)
    return events, build_game_timeline(events)


def _parse_simple_event(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    return {
        "command": command,
        "event": command,
        "raw": payload,
    }


def _parse_new_room(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    return {
        "command": command,
        "event": "new_room",
        "room_state": rest or None,
        "raw": payload,
    }


def _parse_room_snapshot(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    parts = rest.split(" ", 10)
    room_id = parts[0] if len(parts) > 0 else None
    empty_seats = parts[1] if len(parts) > 1 else None
    seat_mask = parts[2] if len(parts) > 2 else None
    room_mode = parts[3] if len(parts) > 3 else None
    room_rule = parts[4] if len(parts) > 4 else None
    map_id = parts[5] if len(parts) > 5 else None
    base = parts[6] if len(parts) > 6 else None
    map_name = parts[7] if len(parts) > 7 else None
    unknown_a = parts[8] if len(parts) > 8 else None
    unknown_b = parts[9] if len(parts) > 9 else None
    players_blob = parts[10] if len(parts) > 10 else ""
    players = []
    for player_blob in players_blob.split("|"):
        if not player_blob:
            continue
        fields = player_blob.split(",")
        decoded_fields = [unquote(field) for field in fields]
        players.append(
            {
                "userid": decoded_fields[0] if len(decoded_fields) > 0 else None,
                "nickname": decoded_fields[1] if len(decoded_fields) > 1 else None,
                "level": _maybe_int(decoded_fields[2] if len(decoded_fields) > 2 else None),
                "gmoney": _maybe_int(decoded_fields[3] if len(decoded_fields) > 3 else None),
                "title": decoded_fields[4] if len(decoded_fields) > 4 else None,
                "wins": _maybe_int(decoded_fields[5] if len(decoded_fields) > 5 else None),
                "losses": _maybe_int(decoded_fields[6] if len(decoded_fields) > 6 else None),
                "score": _maybe_int(decoded_fields[7] if len(decoded_fields) > 7 else None),
                "avatar_url": decoded_fields[10] if len(decoded_fields) > 10 else None,
                "gender": decoded_fields[11] if len(decoded_fields) > 11 else None,
                "badge_1": decoded_fields[12] if len(decoded_fields) > 12 else None,
                "badge_2": decoded_fields[13] if len(decoded_fields) > 13 else None,
                "raw_fields": decoded_fields,
            }
        )
    return {
        "command": command,
        "event": "room_snapshot",
        "room_id": room_id,
        "empty_seats": _maybe_int(empty_seats),
        "seat_mask": seat_mask,
        "room_mode": room_mode,
        "room_rule": room_rule,
        "map_id": _maybe_int(map_id),
        "base": _maybe_int(base),
        "map_name": unquote(map_name) if map_name else None,
        "unknown_a": unknown_a,
        "unknown_b": unknown_b,
        "players": players,
        "raw": payload,
    }


def _parse_play(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    parts = rest.split(" ")
    actor = parts[0] if len(parts) > 0 else None
    mode = parts[1] if len(parts) > 1 else None
    candidate_mask = parts[2] if len(parts) > 2 else None
    cards_blob = parts[3] if len(parts) > 3 else ""
    cards = split_card_codes(cards_blob)

    if mode == "1" and cards:
        event = "hand_snapshot"
    elif cards:
        event = "play_state"
    else:
        event = "turn_prompt"

    return {
        "command": command,
        "event": event,
        "actor_index": actor,
        "mode": mode,
        "candidate_mask": candidate_mask,
        "cards_blob": cards_blob or None,
        "cards": cards,
        "decoded_cards": decode_cards(cards),
        "combo": classify_cards(cards) if cards else None,
        "raw": payload,
    }


def _parse_pass(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    actor = rest.split(" ")[0] if rest else None
    return {
        "command": command,
        "event": "pass",
        "actor_index": actor,
        "raw": payload,
    }


def _parse_player_send(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    parts = rest.split(" ")
    actor = parts[0] if len(parts) > 0 else None
    cards_blob = parts[1] if len(parts) > 1 else ""
    cards = split_card_codes(cards_blob)
    return {
        "command": command,
        "event": "player_play",
        "actor_index": actor,
        "cards_blob": cards_blob or None,
        "cards": cards,
        "decoded_cards": decode_cards(cards),
        "combo": classify_cards(cards),
        "raw": payload,
    }


def _parse_show_score(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    parts = rest.split(" ")
    actor = parts[0] if len(parts) > 0 else None
    score = parts[1] if len(parts) > 1 else None
    cards_blob = parts[2] if len(parts) > 2 else ""
    remaining_cards = split_card_codes(cards_blob)
    return {
        "command": command,
        "event": "show_score",
        "actor_index": actor,
        "score": int(score) if score and score.lstrip("-").isdigit() else score,
        "remaining_cards_blob": cards_blob or None,
        "remaining_cards": remaining_cards,
        "decoded_remaining_cards": decode_cards(remaining_cards),
        "raw": payload,
    }


def _parse_autoplay(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    return {
        "command": command,
        "event": "autoplay_status",
        "value": rest or None,
        "raw": payload,
    }


def _parse_send_cards(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    parts = rest.split(" ")
    code = parts[0] if len(parts) > 0 else None
    cards_blob = parts[1] if len(parts) > 1 else ""
    cards = split_card_codes(cards_blob)
    if direction == "outgoing":
        return {
            "command": command,
            "event": "self_send_cards",
            "target_code": code,
            "cards_blob": cards_blob or None,
            "cards": cards,
            "decoded_cards": decode_cards(cards),
            "combo": classify_cards(cards),
            "raw": payload,
        }
    return {
        "command": command,
        "event": "send_response",
        "response_code": code,
        "cards_blob": cards_blob or None,
        "cards": cards,
        "decoded_cards": decode_cards(cards),
        "combo": classify_cards(cards),
        "raw": payload,
    }


def _parse_self_pass(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    return {
        "command": command,
        "event": "self_pass_request",
        "raw": payload,
    }


def _parse_unknown(command: str, rest: str, payload: str, direction: str) -> dict[str, object]:
    return {
        "command": command,
        "event": "raw",
        "body": rest or None,
        "raw": payload,
    }


def _build_seat_map(self_index: str | None) -> dict[str, str]:
    if self_index is None or not self_index.isdigit():
        return {}
    seat = int(self_index)
    return {
        str(seat): "self",
        str((seat + 1) % 4): "right",
        str((seat + 2) % 4): "top",
        str((seat + 3) % 4): "left",
    }


def _apply_relative_seat_labels(events: list[dict[str, object]]) -> None:
    self_index = None
    for event in events:
        if event.get("command") == "play" and event.get("event") == "hand_snapshot":
            actor_index = event.get("actor_index")
            if isinstance(actor_index, str):
                self_index = actor_index
                break

    seat_map = _build_seat_map(self_index)
    for event in events:
        actor_index = event.get("actor_index")
        if isinstance(actor_index, str):
            event["actor"] = seat_map.get(actor_index, actor_index)
        if event.get("event") == "hand_snapshot":
            event["event"] = "self_hand_snapshot"
            event["actor"] = "self"
            event["self_index"] = self_index


def _subtract_cards(before: list[str], after: list[str]) -> list[str]:
    counter = Counter(before)
    counter.subtract(after)
    return [card for card, count in counter.items() for _ in range(max(count, 0))]


def _maybe_int(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value.lstrip("-").isdigit():
        return int(value)
    return value
