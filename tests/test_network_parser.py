from big2_vision_agent.network_parser import (
    build_sprite_card_mapping_report,
    build_game_timeline,
    classify_cards,
    decode_card_code,
    parse_network_entries,
    parse_ws_payload,
    summarize_turns,
)


def test_parse_room_snapshot_decodes_players():
    payload = (
        "sJS2 56fdvpo 2 8 0 1 500386 10 甜點山丘 0 1 "
        "oman01564@fb,oman01564,22,888,%E5%A4%A7%E8%80%81%E4%BA%8C%E7%89%8C%E9%B7%B9%E4%B8%8B%E5%93%81,"
        "1494,27,51556,0,87,https://example.com/a.png,m,%E7%84%A1,,"
        "|tzm46@fb,%E9%9B%A2%E5%AE%AE%E7%B4%97%E6%9B%89,16,368,%E5%A4%A7%E8%80%81%E4%BA%8C%E9%BB%91%E6%A1%83%E5%88%9D%E6%AE%B5,"
        "901,24,65732,0,56,https://example.com/b.jpeg,m,%E7%84%A1,,"
    )

    parsed = parse_ws_payload(payload)

    assert parsed["event"] == "room_snapshot"
    assert parsed["room_id"] == "56fdvpo"
    assert parsed["empty_seats"] == 2
    assert parsed["base"] == 10
    assert parsed["map_name"] == "甜點山丘"
    assert len(parsed["players"]) == 2
    assert parsed["players"][1]["nickname"] == "離宮紗曉"


def test_classify_cards_detects_pair_and_full_house():
    assert classify_cards(["19", "49"])["type"] == "pair"
    assert classify_cards(["18", "28", "38", "35", "45"])["type"] == "full_house"


def test_classify_cards_detects_official_straight_variants_and_dragon():
    assert classify_cards(["11", "22", "33", "44", "15"])["type"] == "straight"
    assert classify_cards(["2T", "3J", "4Q", "1K", "11"])["type"] == "straight"
    assert classify_cards(["12", "23", "34", "45", "16"])["type"] == "straight"
    assert classify_cards(
        ["13", "24", "35", "46", "17", "28", "39", "4T", "1J", "2Q", "3K", "41", "12"]
    )["type"] == "dragon"
    assert classify_cards(["45", "4T", "4J", "4Q", "41"])["type"] == "five_card_unknown"


def test_decode_card_code_maps_rank_and_display():
    decoded = decode_card_code("41")
    assert decoded["rank_label"] == "A"
    assert decoded["display"] == "CA"


def test_timeline_confirms_self_play_and_opponent_actions():
    entries = [
        {"kind": "ws_message", "payload": "play 0 1 0!1!2!3 411631", "seq": 1, "ts": 1000},
        {"kind": "ws_send", "payload": "send 9 41", "seq": 2, "ts": 1001},
        {"kind": "ws_message", "payload": "plsend 0 41", "seq": 3, "ts": 1002},
        {"kind": "ws_message", "payload": "play 0 1 0!1!2!3 1631", "seq": 4, "ts": 1003},
        {"kind": "ws_message", "payload": "plpass 1", "seq": 5, "ts": 1004},
        {"kind": "ws_message", "payload": "plsend 2 1J3J", "seq": 6, "ts": 1005},
    ]

    parsed = parse_network_entries(entries)
    timeline = build_game_timeline(parsed)

    assert [item["event"] for item in timeline].count("self_play_confirmed") == 0
    assert any(
        item["event"] == "player_play" and item.get("actor") == "self" and item.get("cards") == ["41"]
        for item in timeline
    )
    assert any(
        item["event"] == "player_pass" and item.get("actor") == "right"
        for item in timeline
    )
    assert any(
        item["event"] == "player_play"
        and item.get("actor") == "top"
        and item.get("combo", {}).get("type") == "pair"
        for item in timeline
    )


def test_summarize_turns_groups_actions_into_tricks():
    timeline = [
        {"seq": 1, "event": "player_play", "actor": "left", "cards": ["43"], "combo": {"type": "single"}},
        {"seq": 2, "event": "player_pass", "actor": "self"},
        {"seq": 3, "event": "player_pass", "actor": "right"},
        {"seq": 4, "event": "player_pass", "actor": "top"},
        {"seq": 5, "event": "player_play", "actor": "top", "cards": ["19", "49"], "combo": {"type": "pair"}},
    ]

    tricks = summarize_turns(timeline)

    assert len(tricks) == 2
    assert tricks[0]["lead_actor"] == "left"
    assert tricks[0]["required_type"] == "single"
    assert tricks[0]["closed"] is True
    assert tricks[1]["lead_actor"] == "top"
    assert tricks[1]["required_type"] == "pair"


def test_build_sprite_card_mapping_report_finds_c_prefix_rule():
    action_log = [
        {
            "step": "poll",
            "state": {
                "my_cards": [
                    {"sprite_frame": "c41"},
                    {"sprite_frame": "c16"},
                    {"sprite_frame": "c1T"},
                ]
            },
        }
    ]
    timeline = [
        {
            "event": "self_hand_snapshot",
            "seq": 11,
            "cards": ["41", "16", "1T"],
        }
    ]

    report = build_sprite_card_mapping_report(action_log, timeline)

    assert report["inferred_rule"] == "sprite_frame = 'c' + card_code"
    assert report["unique_mapping_count"] == 3
