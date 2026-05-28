from big2_vision_agent.ml_state import build_ml_state, card_code_to_ml_id


def test_card_code_to_ml_id_uses_rank_major_suit_minor_order():
    assert card_code_to_ml_id("11") == 1
    assert card_code_to_ml_id("21") == 2
    assert card_code_to_ml_id("12") == 5
    assert card_code_to_ml_id("1K") == 49
    assert card_code_to_ml_id("4K") == 52


def test_build_ml_state_initial_snapshot_matches_model_shape():
    state = build_ml_state(
        [
            {
                "event": "self_hand_snapshot",
                "cards": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "1T", "1J", "1Q", "1K"],
            }
        ]
    )

    assert state == {
        "my_hand": [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49],
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


def test_build_ml_state_tracks_play_pass_and_control_break():
    state = build_ml_state(
        [
            {"event": "self_hand_snapshot", "cards": ["11", "12", "13"]},
            {"event": "player_play", "actor": "self", "cards": ["11"]},
            {"event": "player_pass", "actor": "right"},
            {"event": "player_pass", "actor": "top"},
            {"event": "player_pass", "actor": "left"},
        ]
    )

    assert state["my_hand"] == [5, 9]
    assert state["played_cards"] == [1]
    assert state["played_cards_by_player"] == {"1": [1]}
    assert state["current_player"] == 1
    assert state["control"] is True
    assert state["last_hand"] is None
    assert state["last_player"] is None
    assert state["passed"] == {"1": False, "2": False, "3": False, "4": False}
    assert state["action_history"][-1] == {
        "player": 4,
        "hand": None,
        "pass": True,
        "forced_skip": False,
        "control_break": True,
        "passed_snapshot": [False, True, True, False],
    }


def test_build_ml_state_tracks_opponent_play_counts():
    state = build_ml_state(
        [
            {"event": "self_hand_snapshot", "cards": ["11", "12", "13"]},
            {"event": "player_play", "actor": "right", "cards": ["45", "35"]},
        ]
    )

    assert state["opponent_counts"]["2"] == 11
    assert state["last_hand"] == [20, 19]
    assert state["last_player"] == 2
    assert state["action_history"] == [
        {
            "player": 2,
            "hand": [20, 19],
            "pass": False,
            "forced_skip": False,
            "control_break": False,
            "passed_snapshot": [False, False, False, False],
        }
    ]


def test_self_snapshot_after_play_updates_hand_without_resetting_history():
    state = build_ml_state(
        [
            {
                "event": "self_hand_snapshot",
                "cards": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "1T", "1J", "1Q", "1K"],
            },
            {"event": "player_play", "actor": "self", "cards": ["11"]},
            {
                "event": "self_hand_snapshot",
                "cards": ["12", "13", "14", "15", "16", "17", "18", "19", "1T", "1J", "1Q", "1K"],
            },
        ]
    )

    assert state["played_cards"] == [1]
    assert len(state["action_history"]) == 1
    assert state["my_hand"] == [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
