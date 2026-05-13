from unittest.mock import AsyncMock

import pytest

from big2_vision_agent.action_executor import _card_click_points, execute_agent_decision
from big2_vision_agent.agent_schema import AgentDecision


@pytest.mark.asyncio
async def test_execute_agent_decision_detects_unconfirmed_play(monkeypatch):
    state_before = {
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [{"sprite_frame": "c43", "center": {"x": 1, "y": 2}}],
        "action_buttons": {"play": {"active": True, "center": {"x": 10, "y": 20}}},
    }
    state_after_select = {
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [{"sprite_frame": "c43", "selected": True, "center": {"x": 1, "y": 2}}],
        "action_buttons": {"play": {"active": True, "center": {"x": 10, "y": 20}}},
    }
    state_after_play = {
        "my_hand_count": 5,
        "turn": "self",
        "system_messages": {
            "card_type_error": True,
            "no_bigger_card": False,
            "cant_lock": False,
        },
    }

    click_mock = AsyncMock()
    read_mock = AsyncMock(side_effect=[state_after_select, state_after_play])

    monkeypatch.setattr("big2_vision_agent.action_executor.click_design_point", click_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr(
        "big2_vision_agent.action_executor.toggle_my_card_by_sprite",
        AsyncMock(return_value={"invoked": False, "reason": "unsupported_test_page"}),
    )

    result = await execute_agent_decision(
        page=object(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is False
    assert result["reason"] == "play_not_confirmed"


def test_card_click_points_bias_to_upper_visible_region():
    points = _card_click_points(
        [
            {
                "center": {"x": 100, "y": 200},
                "box": {"left": 60, "right": 140, "top": 120, "width": 80, "height": 160},
            },
            {
                "center": {"x": 160, "y": 200},
                "box": {"left": 120, "right": 200, "top": 120, "width": 80, "height": 160},
            }
        ],
        0,
    )

    assert points
    assert points[0]["x"] >= 100
    assert 120 < points[0]["y"] < 180


@pytest.mark.asyncio
async def test_execute_agent_decision_clears_stale_selection_via_cancel(monkeypatch):
    state_before = {
        "my_selected_count": 1,
        "my_cards": [{"sprite_frame": "c43", "center": {"x": 100, "y": 200}}],
        "action_buttons": {
            "cancel": {"active": True, "center": {"x": 10, "y": 20}},
            "pass": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_cancel = {
        "my_selected_count": 0,
        "my_cards": [{"sprite_frame": "c43", "center": {"x": 100, "y": 200}}],
        "action_buttons": {
            "cancel": {"active": False, "center": {"x": 10, "y": 20}},
            "pass": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_pass = {
        "my_selected_count": 0,
        "action_buttons": {"pass": {"active": True, "center": {"x": 30, "y": 40}}},
    }

    click_mock = AsyncMock()
    read_mock = AsyncMock(side_effect=[state_after_cancel, state_after_pass])

    monkeypatch.setattr("big2_vision_agent.action_executor.click_design_point", click_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr(
        "big2_vision_agent.action_executor.toggle_my_card_by_sprite",
        AsyncMock(return_value={"invoked": False, "reason": "unsupported_test_page"}),
    )

    result = await execute_agent_decision(
        page=object(),
        state=state_before,
        decision=AgentDecision(action="pass"),
    )

    assert result["ok"] is True
    assert click_mock.await_args_list[0].args[1:] == (10, 20)
    assert click_mock.await_args_list[1].args[1:] == (30, 40)


@pytest.mark.asyncio
async def test_execute_agent_decision_rejects_selection_mismatch(monkeypatch):
    state_before = {
        "my_selected_count": 0,
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [
            {"sprite_frame": "c43", "center": {"x": 100, "y": 200}},
            {"sprite_frame": "c44", "center": {"x": 120, "y": 200}},
        ],
        "action_buttons": {
            "play": {"active": True, "center": {"x": 10, "y": 20}},
            "cancel": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_select = {
        "my_selected_count": 1,
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [
            {"sprite_frame": "c43", "selected": False, "center": {"x": 100, "y": 200}},
            {"sprite_frame": "c44", "selected": True, "center": {"x": 120, "y": 200}},
        ],
        "action_buttons": {
            "play": {"active": True, "center": {"x": 10, "y": 20}},
            "cancel": {"active": True, "center": {"x": 30, "y": 40}},
        },
    }
    state_after_clear = {
        "my_selected_count": 0,
        "my_hand_count": 5,
        "turn": "self",
        "my_cards": [
            {"sprite_frame": "c43", "selected": False, "center": {"x": 100, "y": 200}},
            {"sprite_frame": "c44", "selected": False, "center": {"x": 120, "y": 200}},
        ],
        "action_buttons": {
            "play": {"active": True, "center": {"x": 10, "y": 20}},
            "cancel": {"active": False, "center": {"x": 30, "y": 40}},
        },
    }

    click_mock = AsyncMock()
    read_mock = AsyncMock(side_effect=[state_after_select, state_after_clear])

    monkeypatch.setattr("big2_vision_agent.action_executor.click_design_point", click_mock)
    monkeypatch.setattr("big2_vision_agent.action_executor.read_big2_game_state", read_mock)
    monkeypatch.setattr(
        "big2_vision_agent.action_executor.toggle_my_card_by_sprite",
        AsyncMock(return_value={"invoked": False, "reason": "unsupported_test_page"}),
    )

    result = await execute_agent_decision(
        page=object(),
        state=state_before,
        decision=AgentDecision(action="play", card_codes=["43"], combo_type="single"),
    )

    assert result["ok"] is False
    assert result["reason"] == "target_click_failed"
    assert result["card_code"] == "43"
    assert len(click_mock.await_args_list) == 1
