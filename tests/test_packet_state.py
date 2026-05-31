from big2_vision_agent.packet_state import build_agent_observation, build_live_agent_observation


def test_build_agent_observation_from_timeline():
    timeline = [
        {
            "event": "self_hand_snapshot",
            "seq": 10,
            "cards": ["43", "44", "35", "46", "47", "1K"],
        },
        {
            "event": "player_play",
            "seq": 11,
            "actor": "top",
            "combo": {"type": "straight"},
            "decoded_cards": [
                {"code": "43", "display": "C3", "rank_label": "3", "suit_label": "C"},
            ],
        },
        {"event": "player_pass", "seq": 12, "actor": "left"},
        {"event": "player_pass", "seq": 13, "actor": "self"},
        {"event": "player_pass", "seq": 14, "actor": "right"},
        {"event": "player_play", "seq": 15, "actor": "left", "combo": {"type": "pair"}, "decoded_cards": []},
        {"event": "round_result", "seq": 16, "actor": "self", "remaining_cards": ["1K"]},
    ]

    observation = build_agent_observation(timeline)

    assert observation.game_index == 1
    assert observation.trick_index == 2
    assert observation.turn == "self"
    assert observation.hand_count == 1
    assert observation.constraint.required_combo_type == "pair"
    assert observation.constraint.last_played_by == "left"
    assert observation.legal_actions[0].action == "pass"


def test_build_live_agent_observation_uses_runtime_playable_cards():
    timeline = [
        {"event": "self_hand_snapshot", "seq": 1, "cards": ["43", "44", "35"]},
        {"event": "player_play", "seq": 2, "actor": "top", "combo": {"type": "single"}, "decoded_cards": []},
    ]
    runtime_state = {
        "turn": "self",
        "current_required_type": "single",
        "my_cards": [
            {"sprite_frame": "c43"},
            {"sprite_frame": "c44"},
            {"sprite_frame": "c35"},
        ],
        "my_playable_indexes": [1, 2],
        "action_buttons": {"pass": {"active": True}},
        "enemy_profiles": [],
    }

    observation = build_live_agent_observation(timeline, runtime_state)

    assert observation.turn == "self"
    assert observation.hand_count == 3
    assert [action.action for action in observation.legal_actions[:3]] == ["pass", "play", "play"]
    assert observation.legal_actions[1].cards[0].code == "44"


def test_build_live_agent_observation_returns_only_pass_when_runtime_has_no_playable_cards():
    timeline = [
        {"event": "self_hand_snapshot", "seq": 1, "cards": ["21", "32", "22"]},
        {
            "event": "player_play",
            "seq": 2,
            "actor": "left",
            "combo": {"type": "single"},
            "decoded_cards": [{"code": "1K", "display": "SK", "rank_label": "K", "suit_label": "S"}],
        },
    ]
    runtime_state = {
        "turn": "self",
        "current_required_type": "single",
        "my_cards": [
            {"sprite_frame": "c21"},
            {"sprite_frame": "c32"},
            {"sprite_frame": "c22"},
        ],
        "my_playable_indexes": [],
        "action_buttons": {"pass": {"active": True}, "play": {"interactable": True}},
        "enemy_profiles": [],
    }

    observation = build_live_agent_observation(timeline, runtime_state)

    assert observation.legal_actions == [observation.legal_actions[0]]
    assert observation.legal_actions[0].action == "pass"


def test_build_agent_observation_allows_higher_five_card_types():
    timeline = [
        {
            "event": "self_hand_snapshot",
            "seq": 1,
            "cards": ["43", "23", "33", "24", "14"],
        },
        {
            "event": "player_play",
            "seq": 2,
            "actor": "top",
            "combo": {"type": "straight"},
            "decoded_cards": [
                {"code": "13", "display": "S3", "rank_label": "3", "suit_label": "S"},
                {"code": "24", "display": "H4", "rank_label": "4", "suit_label": "H"},
                {"code": "35", "display": "D5", "rank_label": "5", "suit_label": "D"},
                {"code": "46", "display": "C6", "rank_label": "6", "suit_label": "C"},
                {"code": "17", "display": "S7", "rank_label": "7", "suit_label": "S"},
            ],
        },
    ]

    observation = build_agent_observation(timeline)

    play_actions = [action for action in observation.legal_actions if action.action == "play"]
    assert any(action.combo_type == "full_house" for action in play_actions)


def test_build_agent_observation_supports_dragon():
    timeline = [
        {
            "event": "self_hand_snapshot",
            "seq": 1,
            "cards": ["13", "24", "35", "46", "17", "28", "39", "4T", "1J", "2Q", "3K", "41", "12"],
        }
    ]

    observation = build_agent_observation(timeline)

    assert any(
        action.action == "play" and action.combo_type == "dragon"
        for action in observation.legal_actions
    )


def test_build_live_agent_observation_normalizes_self_lead_constraint():
    """After all 3 opponents pass (passes_since_last_play==3), the constraint
    should be cleared so we can freely lead the next trick."""
    timeline = [
        {"event": "self_hand_snapshot", "seq": 1, "cards": ["26", "47", "27", "48", "2T", "1J"]},
        {
            "event": "player_play",
            "seq": 2,
            "actor": "self",
            "combo": {"type": "single"},
            "decoded_cards": [{"code": "11", "display": "SA", "rank_label": "A", "suit_label": "S"}],
        },
        {"event": "player_pass", "seq": 3, "actor": "right"},
        {"event": "player_pass", "seq": 4, "actor": "top"},
        {"event": "player_pass", "seq": 5, "actor": "left"},
    ]
    # After all 3 opponents pass, Cocos no longer shows a required type
    runtime_state = {
        "turn": "self",
        "current_required_type": None,
        "my_cards": [
            {"sprite_frame": "c26"},
            {"sprite_frame": "c47"},
            {"sprite_frame": "c27"},
            {"sprite_frame": "c48"},
            {"sprite_frame": "c2T"},
            {"sprite_frame": "c1J"},
        ],
        "my_playable_indexes": [0, 1, 2, 3, 4, 5],
        "action_buttons": {"pass": {"active": True}},
        "enemy_profiles": [],
    }

    observation = build_live_agent_observation(timeline, runtime_state)

    assert observation.turn == "self"
    assert observation.constraint.required_combo_type is None
    assert observation.constraint.last_played_by is None
    assert observation.constraint.last_played_cards == []
    assert any(action.action == "play" for action in observation.legal_actions)


def test_build_live_agent_observation_no_premature_lead_after_auto_play():
    """When Cocos prematurely shows turn='self' after we auto-played but before
    opponents have responded (passes_since_last_play < 3), the constraint must
    NOT be cleared — legal_actions should be pass-only to prevent playing into
    an unresolved trick."""
    timeline = [
        {"event": "self_hand_snapshot", "seq": 1, "cards": ["26", "47", "27", "48", "2T", "1J"]},
        {
            "event": "player_play",
            "seq": 2,
            "actor": "self",
            "combo": {"type": "single"},
            "decoded_cards": [{"code": "11", "display": "SA", "rank_label": "A", "suit_label": "S"}],
        },
        # Only 1 pass — top and left haven't responded yet
        {"event": "player_pass", "seq": 3, "actor": "right"},
    ]
    runtime_state = {
        "turn": "self",  # Cocos UI glitch: shows our turn before trick resolves
        "current_required_type": "single",
        "my_cards": [
            {"sprite_frame": "c26"},
            {"sprite_frame": "c47"},
            {"sprite_frame": "c27"},
            {"sprite_frame": "c48"},
            {"sprite_frame": "c2T"},
            {"sprite_frame": "c1J"},
        ],
        "my_playable_indexes": [0, 1, 2, 3, 4, 5],
        "action_buttons": {"pass": {"active": True}},
        "enemy_profiles": [],
    }

    observation = build_live_agent_observation(timeline, runtime_state)

    assert observation.turn == "self"
    # Constraint must NOT be cleared — trick is not over yet
    assert observation.constraint.last_played_by == "self"
    # No card in hand can beat SA (hand has H6,C7,H7,C8,HT,SJ — no 2s)
    # so legal_actions should be pass-only
    assert all(action.action == "pass" for action in observation.legal_actions)


def test_build_live_agent_observation_parses_runtime_enemy_counts_and_seats():
    timeline = [
        {"event": "self_hand_snapshot", "seq": 1, "cards": ["26", "47", "27"]},
    ]
    runtime_state = {
        "turn": "self",
        "current_required_type": None,
        "my_cards": [
            {"sprite_frame": "c26"},
            {"sprite_frame": "c47"},
            {"sprite_frame": "c27"},
        ],
        "my_playable_indexes": [0, 1, 2],
        "action_buttons": {"pass": {"active": True}},
        "enemy_profiles": [
            {"center": {"x": 120}, "remain_text": "9"},
            {"center": {"x": 860}, "remain_text": "4"},
            {"center": {"x": 1500}, "remain_text": "1"},
        ],
    }

    observation = build_live_agent_observation(timeline, runtime_state)

    assert [(opp.seat, opp.remaining_count) for opp in observation.opponents] == [
        ("left", 9),
        ("top", 4),
        ("right", 1),
    ]


def test_build_live_agent_observation_filters_non_max_singles_when_right_has_one_card():
    timeline = [
        {"event": "self_hand_snapshot", "seq": 1, "cards": ["43", "44", "35", "36", "37", "38", "1K"]},
    ]
    runtime_state = {
        "turn": "self",
        "current_required_type": None,
        "my_cards": [
            {"sprite_frame": "c43"},
            {"sprite_frame": "c44"},
            {"sprite_frame": "c35"},
            {"sprite_frame": "c36"},
            {"sprite_frame": "c37"},
            {"sprite_frame": "c38"},
            {"sprite_frame": "c1K"},
        ],
        "my_playable_indexes": [0, 1, 2, 3, 4, 5, 6],
        "action_buttons": {"pass": {"active": False}},
        "enemy_profiles": [
            {"seat": "right", "remaining_count": 1},
            {"seat": "top", "remaining_count": 5},
            {"seat": "left", "remaining_count": 5},
        ],
    }

    observation = build_live_agent_observation(timeline, runtime_state)

    single_codes = [
        action.cards[0].code
        for action in observation.legal_actions
        if action.action == "play" and action.combo_type == "single"
    ]
    assert single_codes == ["1K"]
    assert any(action.action == "play" and action.combo_type == "straight" for action in observation.legal_actions)
