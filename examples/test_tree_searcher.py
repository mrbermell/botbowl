import pytest
from pytest import approx

from tests.util import get_custom_game_turn
from .TreeSearcher import remove_randomness, ActionNode, ActionSampler, expand_action, ChanceNode
import botbowl
from botbowl import Square, Action, ActionType


@pytest.mark.parametrize("data", [(Square(2, 2), [1.0]),
                                  (Square(4, 4), [4/6, 2/6]),
                                  (Square(3, 3), [4/6, 2/6])])
def test_expand_move(data):
    move_target, outcome_probs = data
    game, player, _, _ = get_custom_game_turn(player_positions=[(1, 1)],
                                              opp_player_positions=[(1, 3), (3, 1)],
                                              ball_position=(5, 5))
    game.config.pathfinding_enabled = True
    game.set_available_actions()
    game.enable_forward_model()
    game.step(Action(ActionType.START_MOVE, position=player.position))

    parent_action_node = ActionNode(step_nbr=game.get_step(), parent=None, steps=[], action_sampler=ActionSampler(game))

    action = Action(ActionType.MOVE, position=move_target)
    next_node = expand_action(game, action, parent_action_node)

    assert next_node.parent is parent_action_node

    if len(outcome_probs) == 1:
        assert sum(outcome_probs) == 1.0
        assert type(next_node) is ActionNode
    else:
        next_node: ChanceNode
        assert type(next_node) is ChanceNode
        assert sum(next_node.child_probability) == 1.0
        assert all(approx(x-y) == 0 for x, y in zip(sorted(next_node.child_probability), sorted(outcome_probs)))

