from big2_vision_agent.agent_runtime import FallbackDecisionAgent, sample_random_decision
from big2_vision_agent.agent_schema import AgentActionOption, AgentCard, AgentObservation


def test_fallback_agent_prefers_play():
    observation = AgentObservation(
        game_index=1,
        hand_count=1,
        legal_actions=[
            AgentActionOption(action="pass"),
            AgentActionOption(
                action="play",
                combo_type="single",
                cards=[AgentCard(code="43", display="C3", rank="3", suit="C")],
            ),
        ],
    )

    decision = FallbackDecisionAgent().decide(observation)

    assert decision is not None
    assert decision.action == "play"
    assert decision.card_codes == ["43"]
    assert decision.note == "fallback:no_pass"


def test_random_decision_returns_pass_when_no_play():
    observation = AgentObservation(
        game_index=1,
        legal_actions=[AgentActionOption(action="pass")],
    )

    decision = sample_random_decision(observation)

    assert decision is not None
    assert decision.action == "pass"
