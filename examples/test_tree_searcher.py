import pytest
from pytest import approx

from tests.util import get_custom_game_turn
from .TreeSearcher import ActionNode, expand_action, ChanceNode, Node
# import botbowl
from botbowl import Square, Action, ActionType


@pytest.mark.parametrize("data", [(Square(2, 2), [1.0]),
                                  (Square(4, 4), [4/6, 2/6]),
                                  (Square(3, 3), [4/6, 2/6]),
                                  (Square(9, 9), [(4*5*5)/(6**3), 1-(4*5*5)/(6**3)])
                                  ])
def test_expand_move(data):
    move_target, outcome_probs = data
    assert sum(outcome_probs) == 1.0
    game, player, _, _ = get_custom_game_turn(player_positions=[(1, 1)],
                                              opp_player_positions=[(1, 3), (3, 1)],
                                              forward_model_enabled=True,
                                              pathfinding_enabled=True)

    game.step(Action(ActionType.START_MOVE, position=player.position))

    action = Action(ActionType.MOVE, position=move_target)
    parent_action_node: ActionNode = ActionNode(game, parent=None)
    next_node: Node = expand_action(game, action, parent_action_node)

    assert next_node.parent is parent_action_node

    if len(outcome_probs) == 1:
        assert type(next_node) is ActionNode
    else:
        next_node: ChanceNode
        assert type(next_node) is ChanceNode
        assert sum(next_node.child_probability) == 1.0
        assert all(y == approx(x, abs=1e-12) for x, y in zip(sorted(next_node.child_probability), sorted(outcome_probs)))


@pytest.mark.parametrize("data", [(Square(3, 3), [4/6, 2/6]) ])  #,  # Not marked ball
                                  #(Square(3, 2), [3/6, 3/6]),  # Marked one tz
                                  #(Square(4, 2), [4/6, 2/6])   # Marked two tz
                                  #])
def test_expand_pickup(data):
    ball_square, outcome_probs = data
    assert sum(outcome_probs) == 1.0
    game, player, _, _ = get_custom_game_turn(player_positions=[(1, 1)],
                                              opp_player_positions=[(3, 1), (5, 1)],
                                              ball_position=ball_square,
                                              forward_model_enabled=True,
                                              pathfinding_enabled=True)

    game.step(Action(ActionType.START_MOVE, position=player.position))
    action = Action(ActionType.MOVE, position=ball_square)
    parent_action_node: ActionNode = ActionNode(game, parent=None)
    next_node: Node = expand_action(game, action, parent_action_node)

    next_node: ChanceNode
    assert next_node.parent is parent_action_node
    assert type(next_node) is ChanceNode
    assert sum(next_node.child_probability) == 1.0
    assert all(y == approx(x, abs=1e-12) for x, y in zip(sorted(next_node.child_probability), sorted(outcome_probs)))
